"""Builds live models and the search backend. The CLI and the Streamlit app go through
here, so tests can swap in fakes in one place."""

from __future__ import annotations

from .llm import ChatModel, LiteLLMModel
from .search import SearchBackend, TavilyBackend

# Reasoning models spend tokens thinking before they answer; give planners more room.
MAX_TOKENS = {"planner": 16384, "reader": 8192, "judge": 8192}


def make_model(model_id: str, role: str = "reader") -> ChatModel:
    """`model_id` is a LiteLLM model string, e.g. nebius/openai/gpt-oss-120b or openai/gpt-4.1-mini."""
    return LiteLLMModel(model_id, max_tokens=MAX_TOKENS.get(role, 8192))


def make_search_backend() -> SearchBackend:
    return TavilyBackend()
