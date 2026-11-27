"""Builds live models and the search backend. The CLI and the Streamlit app go through
here, so tests can swap in fakes in one place."""

from __future__ import annotations

from .llm import ChatModel, LiteLLMModel
from .search import SearchBackend, TavilyBackend

# Reasoning models spend tokens thinking before they answer; 8,192 was too few for GLM-5.3 as a
# reader and for DeepSeek judging a long report. Only tokens generated are billed, and a model
# that rejects a limit this high gets it halved automatically (see LiteLLMModel).
MAX_TOKENS = {"planner": 32768, "reader": 32768, "judge": 32768}


def make_model(model_id: str, role: str = "reader") -> ChatModel:
    """`model_id` is a LiteLLM model string, e.g. nebius/openai/gpt-oss-120b or openai/gpt-4.1-mini."""
    return LiteLLMModel(model_id, max_tokens=MAX_TOKENS.get(role, 32768))


def make_search_backend() -> SearchBackend:
    return TavilyBackend()
