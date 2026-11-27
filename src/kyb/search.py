"""Search backends: live Tavily, or replay from a recorded trace.

Both go through `traced_search`, which records every request and full response on a
`tavily.search` span. A replay run therefore produces a trace that can itself be
replayed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from .plan import PlannedSearch, search_key
from .planner import PlanResult
from .schema import Company
from .tracing import get_tracer, set_io


class SearchBackend(Protocol):
    mode: str

    def search(self, params: dict) -> dict: ...


class TavilyBackend:
    mode = "live"

    def __init__(self, api_key: str | None = None, retries: int = 2) -> None:
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key or os.environ["TAVILY_API_KEY"])
        self._retries = retries

    def search(self, params: dict) -> dict:
        params = dict(params)
        query = params.pop("query")
        for attempt in range(self._retries + 1):
            try:
                return self._client.search(query, **params)
            except Exception as exc:  # tavily-python raises several error types
                if attempt == self._retries or not _transient(exc):
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")


def _transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("timeout", "429", "rate", "500", "502", "503", "connection"))


class ReplayMiss(KeyError):
    pass


@dataclass
class Recording:
    path: Path
    trace_id: str
    company: Company
    as_of: date
    lookback_years: int
    plan_version: str
    responses: dict[str, dict]
    plan: PlanResult | None = None  # the searches as planned; None for recordings made before planners
    credits: float = 0.0             # Tavily credits spent when this was recorded live

    @property
    def planner(self) -> str:
        return self.plan.planner if self.plan else "fixed"


def load_recording(path: Path, trace_id: str | None = None) -> Recording:
    """Read a trace file and pull out the screen settings and every Tavily response."""
    spans = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    roots = [s for s in spans if s["name"] == "kyb.screen"]
    if trace_id:
        roots = [s for s in roots if s["context"]["trace_id"] == trace_id]
    if not roots:
        raise ValueError(f"No kyb.screen span found in {path}")
    root = roots[0]
    tid = root["context"]["trace_id"]
    attrs = root["attributes"]

    responses: dict[str, dict] = {}
    plan: PlanResult | None = None
    credits = 0.0
    for span in spans:
        if span["context"]["trace_id"] != tid:
            continue
        a = span["attributes"]
        if span["name"] == "tavily.search" and SpanAttributes.OUTPUT_VALUE in a:
            responses[a["kyb.search_key"]] = json.loads(a[SpanAttributes.OUTPUT_VALUE])
            credits += float(a.get("tavily.credits", 0) or 0)
        elif span["name"] == "kyb.plan":
            plan = PlanResult(
                planner=a["kyb.planner"],
                searches=[PlannedSearch(s["category"], s["params"])
                          for s in json.loads(a[SpanAttributes.OUTPUT_VALUE])],
                fixes=json.loads(a.get("kyb.plan_fixes", "[]")),
                fell_back=json.loads(a.get("kyb.plan_fell_back", "[]")),
                raw=json.loads(a["kyb.plan_raw"]) if "kyb.plan_raw" in a else None,
                prompt_tokens=int(a.get("kyb.prompt_tokens", 0)),
                completion_tokens=int(a.get("kyb.completion_tokens", 0)),
            )

    return Recording(
        path=path,
        trace_id=tid,
        company=Company.model_validate_json(attrs["kyb.company"]),
        as_of=date.fromisoformat(attrs["kyb.as_of"]),
        lookback_years=int(attrs["kyb.lookback_years"]),
        plan_version=attrs["kyb.plan_version"],
        responses=responses,
        plan=plan,
        credits=credits,
    )


class ReplayBackend:
    mode = "replay"

    def __init__(self, recording: Recording) -> None:
        self.recording = recording

    def search(self, params: dict) -> dict:
        key = search_key(params)
        try:
            return self.recording.responses[key]
        except KeyError:
            raise ReplayMiss(
                f"Search not in recording {self.recording.path}: {params.get('query')!r}. "
                "The screening plan or its parameters changed since this was recorded."
            ) from None


def traced_search(backend: SearchBackend, category: str, params: dict) -> dict:
    tracer = get_tracer()
    with tracer.start_as_current_span("tavily.search") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value
        )
        span.set_attribute("kyb.category", category)
        span.set_attribute("kyb.search_key", search_key(params))
        span.set_attribute("kyb.replayed", backend.mode == "replay")
        response = backend.search(params)
        set_io(span, input_value=params, output_value=response)

        results = response.get("results", []) or []
        for i, r in enumerate(results):
            prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
            span.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_ID}", r.get("url", ""))
            span.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_CONTENT}", r.get("content", ""))
            if r.get("score") is not None:
                span.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_SCORE}", float(r["score"]))
        span.set_attribute("tavily.result_count", len(results))
        credits = (response.get("usage") or {}).get("credits")
        if credits is not None and backend.mode == "live":  # replays spend nothing
            span.set_attribute("tavily.credits", credits)
        if response.get("request_id"):
            span.set_attribute("tavily.request_id", response["request_id"])
        return response
