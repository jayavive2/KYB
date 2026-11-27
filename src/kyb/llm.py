"""Chat models through LiteLLM, with JSON validation.

A model is named by its LiteLLM string, so switching provider is a config edit:
`nebius/<Token Factory model ID>`, `openai/gpt-4.1-mini`, `anthropic/claude-haiku-4-5`,
`gemini/gemini-2.5-flash`, ... Each provider reads its usual key (NEBIUS_API_KEY,
OPENAI_API_KEY, ANTHROPIC_API_KEY, ...). We call `litellm.completion` directly rather than
through an agent framework: this is a measurement harness, and we want the exact request
and the provider's own token counts.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

# Use LiteLLM's bundled price map instead of fetching it from GitHub on import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

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


class LiteLLMModel:
    """Any chat model LiteLLM supports. Asks for JSON mode at temperature 0; if a model rejects
    either parameter, it stops sending it (the JSON is still validated)."""

    # Optional request parameters, dropped for a model that rejects them.
    _OPTIONAL = ("response_format", "temperature")

    def __init__(self, model: str, max_tokens: int = 8192, temperature: float = 0.0,
                 api_base: str | None = None, api_key: str | None = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        # LiteLLM still defaults nebius/ to the old AI Studio host; send Token Factory's.
        if api_base is None and model.startswith("nebius/"):
            api_base = os.getenv("NEBIUS_API_BASE") or os.getenv("NEBIUS_BASE_URL") or NEBIUS_BASE_URL
        self.api_base = api_base
        self.api_key = api_key
        self.unsupported: set[str] = set()

    @property
    def json_mode(self) -> bool:
        return "response_format" not in self.unsupported

    def _request(self, messages: list[dict]) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                        "response_format": {"type": "json_object"}, "temperature": self.temperature,
                        "drop_params": True,  # LiteLLM drops what it knows a provider doesn't take
                        "max_retries": 3}     # rate limits and 5xx only
        for name in self.unsupported:
            kwargs.pop(name, None)
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs

    def chat(self, messages: list[dict]) -> ChatResult:
        import litellm

        litellm.suppress_debug_info = True  # errors are reported by describe_error, not banners
        for _ in range(len(self._OPTIONAL) + 1):
            # Decide from what *this* request sent: categories run in parallel, and another
            # thread may update `unsupported` while this request is in flight.
            kwargs = self._request(messages)
            start = time.perf_counter()
            try:
                response = litellm.completion(**kwargs)
                break
            except litellm.BadRequestError as exc:  # ...and what it didn't know, we learn here
                rejected = [p for p in self._OPTIONAL if p in kwargs and p in str(exc).lower()]
                if not rejected:
                    raise
                self.unsupported.update(rejected)
        else:
            raise RuntimeError(f"{self.model}: request rejected even without optional parameters")
        latency = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        return ChatResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
        )


def missing_keys(model_ids: list[str]) -> dict[str, list[str]]:
    """Env vars each model's provider needs but doesn't have, per LiteLLM."""
    import litellm

    out: dict[str, list[str]] = {}
    for m in dict.fromkeys(model_ids):
        if not m or m == "fixed":
            continue
        try:
            missing = litellm.validate_environment(m).get("missing_keys") or []
        except Exception:  # unknown provider: describe_error explains it when the call fails
            missing = []
        if missing:
            out[m] = missing
    return out


def is_fatal(exc: BaseException) -> bool:
    """Errors that will fail every call to this model (credit, key, model name): stop, don't retry
    category by category."""
    if type(exc).__module__.startswith("tavily"):  # bad key, no credits, forbidden: every search fails
        return type(exc).__name__ in ("InvalidAPIKeyError", "MissingAPIKeyError", "UsageLimitExceededError",
                                      "ForbiddenError")
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status in (401, 402, 403, 404) or any(w in text for w in (
        "payment required", "exhausted your budget", "insufficient_quota", "llm provider not provided",
        "model not found", "does not exist"))


def describe_error(exc: BaseException, model: str | None = None) -> str:
    """One line a person can act on, for the app and the CLI."""
    name = type(exc).__name__
    if type(exc).__module__.startswith("tavily"):
        if name in ("InvalidAPIKeyError", "MissingAPIKeyError"):
            return ("Tavily rejected TAVILY_API_KEY. Check the value (it starts with tvly-, no quotes or "
                    "spaces); a Codespaces secret overrides .env.")
        if name == "UsageLimitExceededError":
            return "Tavily: this key is out of credits for the month. Add credits at app.tavily.com."
        return f"Tavily: {name}: {str(exc).strip()[:300]}"
    status = getattr(exc, "status_code", None)
    text = str(exc)
    low = text.lower()
    model = model or getattr(exc, "model", None) or "the model"
    if status == 402 or "payment required" in low or "budget" in low or "insufficient_quota" in low:
        return (f"{model}: the provider account is out of credit (402 Payment Required). Top up "
                "(Nebius: Top up in the Token Factory console) or switch to another provider's model "
                "in config/models.toml.")
    if status == 401 or "authentication" in low:
        return f"{model}: the API key was rejected or is missing. Check the provider's key (e.g. NEBIUS_API_KEY)."
    if "llm provider not provided" in low:
        return (f"'{model}' is not a LiteLLM model string. Prefix the provider: nebius/<Token Factory ID>, "
                "openai/<model>, anthropic/<model>, ...")
    if status == 404 or "not found" in low or "does not exist" in low:
        return (f"{model}: model not found. `uv run kyb models` lists Nebius models (use them as "
                "nebius/<id>).")
    first = text.strip().splitlines()[0] if text.strip() else ""
    return f"{type(exc).__name__}: {first[:300]}"


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
