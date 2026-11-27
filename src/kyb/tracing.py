"""OpenTelemetry setup.

Every run writes its spans to a local JSONL file. That file is the recording: the
Tavily spans carry the full search response, so `kyb replay` can rebuild the exact
evidence later. If OTEL_EXPORTER_OTLP_ENDPOINT is set, spans also go to that OTLP
backend (Arize Phoenix, Langfuse, Jaeger, Datadog, ...). Attribute names follow the
OpenInference conventions so those UIs render LLM and retriever spans natively.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import IO, Any

from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

SERVICE_NAME = "kyb-screen"

_provider: TracerProvider | None = None
_trace_file: IO[str] | None = None


def init_tracing(trace_path: Path) -> TracerProvider:
    """Start a tracer provider that writes to `trace_path` (and OTLP if configured)."""
    global _provider, _trace_file
    shutdown_tracing()

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    _trace_file = trace_path.open("a", encoding="utf-8")
    provider = TracerProvider(resource=Resource.create({
        "service.name": SERVICE_NAME,
        ResourceAttributes.PROJECT_NAME: SERVICE_NAME,  # Phoenix groups traces by project
    }))
    provider.add_span_processor(
        SimpleSpanProcessor(
            ConsoleSpanExporter(
                out=_trace_file,
                formatter=lambda span: span.to_json(indent=None) + "\n",
            )
        )
    )
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    OpenAIInstrumentor().instrument(tracer_provider=provider)
    _provider = provider
    return provider


def shutdown_tracing() -> None:
    global _provider, _trace_file
    if _provider is not None:
        _provider.force_flush()
        _provider.shutdown()
        OpenAIInstrumentor().uninstrument()
        _provider = None
    if _trace_file is not None:
        _trace_file.close()
        _trace_file = None


def get_tracer() -> trace.Tracer:
    provider = _provider or trace.get_tracer_provider()
    return provider.get_tracer("kyb")


def set_json(span: trace.Span, key: str, value: Any) -> None:
    span.set_attribute(key, json.dumps(value, ensure_ascii=False, default=str))


def set_io(span: trace.Span, input_value: Any = None, output_value: Any = None) -> None:
    """Set OpenInference input/output attributes as JSON."""
    if input_value is not None:
        set_json(span, SpanAttributes.INPUT_VALUE, input_value)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
    if output_value is not None:
        set_json(span, SpanAttributes.OUTPUT_VALUE, output_value)
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
