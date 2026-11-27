"""Exercise the real LiteLLM path (and its OpenInference instrumentation) against a local mock
of an OpenAI-compatible endpoint."""

import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fakes import FakeTavily

from kyb.llm import LiteLLMModel
from kyb.schema import Company
from kyb.screen import run_screen
from kyb.tracing import init_tracing, shutdown_tracing


class _Handler(BaseHTTPRequestHandler):
    reject_json_mode = False
    reject_temperature = False
    fail_category: str | None = None
    out_of_credit = False
    requests: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        if self.out_of_credit:  # what Nebius returns when the balance runs out
            self._send(402, {"detail": "Payment Required: You have exhausted your budget. "
                                       "Please add funds to continue using the API."})
            return
        if self.reject_json_mode and "response_format" in body:
            self._send(400, {"error": {"message": "response_format is not supported for this model"}})
            return
        if self.reject_temperature and "temperature" in body:
            self._send(400, {"error": {"message": "Unsupported value: temperature"}})
            return
        if self.fail_category and self.fail_category in body["messages"][0]["content"]:
            self._send(400, {"error": {"message": "context length exceeded"}})
            return
        user = body["messages"][1]["content"]
        first_id = user.split("[", 1)[1].split("]", 1)[0] if "[S" in user else "S1"
        content = json.dumps({"findings": [{"summary": "Fined.", "severity": "medium",
                                            "source_ids": [first_id]}]})
        self._send(200, {
            "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290},
        })

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    _Handler.requests = []
    _Handler.reject_json_mode = _Handler.reject_temperature = False
    _Handler.fail_category = None
    _Handler.out_of_credit = False
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1/"
    srv.shutdown()


@pytest.mark.parametrize("reject", [False, True])
def test_real_client_tokens_and_llm_spans(tmp_path, server, reject):
    _Handler.reject_json_mode = _Handler.reject_temperature = reject
    model = LiteLLMModel("openai/mock-model", api_base=server, api_key="test")
    trace = tmp_path / "trace.jsonl"
    init_tracing(trace)
    try:
        result = run_screen(Company(name="Acme Corp"), date(2026, 1, 15), 5, FakeTavily(), model,
                            session_id="test-session")
    finally:
        shutdown_tracing()

    calls = 7 + 1                                                # 7 categories + the duplicate check
    assert result.prompt_tokens == 1234 * calls and result.completion_tokens == 56 * calls
    assert all(r.parsed and r.first_try for r in result.categories.values())
    assert model.json_mode is (not reject)
    assert model.unsupported == ({"response_format", "temperature"} if reject else set())
    if not reject:
        assert all(r["temperature"] == 0.0 for r in _Handler.requests)

    spans = [json.loads(line) for line in trace.read_text().splitlines()]
    by_id = {s["context"]["span_id"]: s for s in spans}
    llm_all = [s for s in spans if s["attributes"].get("openinference.span.kind") == "LLM"]
    llm = [s for s in llm_all if s["status"]["status_code"] != "ERROR"]
    assert len(llm) == calls
    # Rejected JSON-mode requests are traced as errors, not hidden.
    assert (len(llm_all) > calls) is reject
    for s in llm:
        assert s["attributes"]["llm.token_count.prompt"] == 1234
        call = by_id[s["parent_id"]]                                # our provider-independent call span
        assert call["name"] == "kyb.llm.call" and call["attributes"]["kyb.prompt_tokens"] == 1234
        step = by_id[call["parent_id"]]["name"]
        assert step.startswith("kyb.extract.") or step == "kyb.merge"  # context crossed threads
        assert s["attributes"]["session.id"] == "test-session"
        assert json.loads(s["attributes"]["metadata"])["company"] == "Acme Corp"


def test_api_failure_marks_only_that_category(tmp_path, server):
    from kyb.report import render_markdown

    _Handler.fail_category = "Risk category: Sanctions and export controls"
    model = LiteLLMModel("openai/mock-model", api_base=server, api_key="test")
    init_tracing(tmp_path / "trace.jsonl")
    try:
        result = run_screen(Company(name="Acme Corp"), date(2026, 1, 15), 5, FakeTavily(), model)
    finally:
        shutdown_tracing()
    failed = [k for k, r in result.categories.items() if not r.parsed]
    assert failed == ["sanctions"]
    assert "BadRequestError" in result.categories["sanctions"].errors[0]
    assert "**Not assessed:** the model call failed" in render_markdown(result)


def test_out_of_credit_stops_the_screen_with_a_clear_message(tmp_path, server):
    from kyb.llm import describe_error, is_fatal

    _Handler.out_of_credit = True
    model = LiteLLMModel("openai/mock-model", api_base=server, api_key="test")
    init_tracing(tmp_path / "trace.jsonl")
    try:
        with pytest.raises(Exception) as info:
            run_screen(Company(name="Acme Corp"), date(2026, 1, 15), 5, FakeTavily(), model)
    finally:
        shutdown_tracing()
    assert is_fatal(info.value)
    assert "out of credit" in describe_error(info.value, model.model)
    assert len(_Handler.requests) <= 4 + 3  # stopped early: not 7 categories x retries


def test_describe_error_names_the_fix():
    from kyb.llm import describe_error

    class Fake(Exception):
        def __init__(self, msg, status):
            super().__init__(msg)
            self.status_code = status

    assert "API key" in describe_error(Fake("Unauthorized", 401), "nebius/x")
    assert "kyb models" in describe_error(Fake("The model does not exist", 404), "nebius/x")
    assert "LiteLLM model string" in describe_error(Exception("LLM Provider NOT provided. Pass in ..."), "x/y")


def test_nebius_models_go_to_token_factory(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_BASE", raising=False)
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    assert LiteLLMModel("nebius/openai/gpt-oss-120b").api_base == "https://api.tokenfactory.nebius.com/v1/"
    assert LiteLLMModel("openai/gpt-4.1-mini").api_base is None   # other providers: LiteLLM's default


def test_tavily_key_errors_are_one_line():
    from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

    from kyb.llm import describe_error, is_fatal

    assert is_fatal(InvalidAPIKeyError("Unauthorized"))
    assert "TAVILY_API_KEY" in describe_error(InvalidAPIKeyError("Unauthorized: missing or invalid API key."))
    assert "out of credits" in describe_error(UsageLimitExceededError("limit"))
