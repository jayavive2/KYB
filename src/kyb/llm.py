"""OpenAI-compatible chat client (Nebius Token Factory by default) with JSON validation.

We call the OpenAI SDK directly rather than through an agent framework: this is a
measurement harness, and we want the exact request, the exact token counts from the
provider, and nothing in between.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

T = TypeVar("T", bound=BaseModel)


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class ChatModel(Protocol):
    model: str

    def chat(self, messages: list[dict]) -> ChatResult: ...


class OpenAICompatibleModel:
    """Chat model behind an OpenAI-compatible endpoint. Asks for JSON mode at temperature 0;
    if a model rejects either parameter, it stops sending it (the JSON is still validated)."""

    # Optional request parameters, dropped for a model that rejects them.
    _OPTIONAL = ("response_format", "temperature")

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 8192) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.unsupported: set[str] = set()
        self._client = OpenAI(
            base_url=base_url or os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key or os.environ["NEBIUS_API_KEY"],
            max_retries=3,
        )

    @property
    def json_mode(self) -> bool:
        return "response_format" not in self.unsupported

    def _request(self, messages: list[dict]) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                        "response_format": {"type": "json_object"}, "temperature": self.temperature}
        for name in self.unsupported:
            kwargs.pop(name, None)
        return kwargs

    def chat(self, messages: list[dict]) -> ChatResult:
        from openai import BadRequestError

        for _ in range(len(self._OPTIONAL) + 1):
            # Decide from what *this* request sent: categories run in parallel, and another
            # thread may update `unsupported` while this request is in flight.
            kwargs = self._request(messages)
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except BadRequestError as exc:
                rejected = [p for p in self._OPTIONAL if p in kwargs and p in str(exc).lower()]
                if not rejected:
                    raise
                self.unsupported.update(rejected)
        else:
            raise RuntimeError(f"{self.model}: request rejected even without optional parameters")
        latency = time.perf_counter() - start
        usage = response.usage
        return ChatResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
        )


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_output(text: str, schema: type[T]) -> tuple[T | None, str | None]:
    """Parse model output into `schema`. Tolerates reasoning tags and code fences,
    nothing else: anything that still fails is a real schema failure."""
    cleaned = _THINK.sub("", text).strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None, "no JSON object found in the output"
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    try:
        return schema.model_validate(data), None
    except ValidationError as exc:
        return None, f"does not match the schema: {exc.errors(include_url=False)}"


@dataclass
class JsonCallResult:
    value: BaseModel | None
    attempts: int
    errors: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def first_try(self) -> bool:
        return self.value is not None and self.attempts == 1


def complete_json(model: ChatModel, messages: list[dict], schema: type[T],
                  max_attempts: int = 2) -> JsonCallResult:
    """Call the model; on invalid output, show it the error once and ask for a fix."""
    from .tracing import get_tracer

    result = JsonCallResult(value=None, attempts=0)
    convo = list(messages)
    for attempt in range(1, max_attempts + 1):
        # One span per model call, whatever the provider: the trace analysis reads these.
        # (kyb.* names so trace UIs don't double-count the instrumented LLM span beneath it.)
        with get_tracer().start_as_current_span("kyb.llm.call") as span:
            span.set_attribute("kyb.model", model.model)
            span.set_attribute("kyb.attempt", attempt)
            reply = model.chat(convo)
            value, error = parse_json_output(reply.text, schema)
            span.set_attribute("kyb.prompt_tokens", reply.prompt_tokens)
            span.set_attribute("kyb.completion_tokens", reply.completion_tokens)
            span.set_attribute("kyb.latency_s", reply.latency_s)
            span.set_attribute("kyb.parsed", value is not None)
            if error:
                span.set_attribute("kyb.parse_error", error[:500])
        result.attempts = attempt
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        result.latency_s += reply.latency_s
        if value is not None:
            result.value = value
            return result
        result.errors.append(error or "unknown error")
        convo = convo + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": f"Your output could not be used ({error}). "
                                        "Return only the corrected JSON object."},
        ]
    return result
