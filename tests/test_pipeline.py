import json
from datetime import date

import pytest
from fakes import FakeModel, FakeTavily

from kyb.evidence import EvidenceStore, normalize_url
from kyb.llm import complete_json, parse_json_output
from kyb.plan import CATEGORIES, build_plan, plan_version
from kyb.report import render_markdown
from kyb.schema import CategoryOutput, Company
from kyb.screen import run_screen
from kyb.search import ReplayBackend, ReplayMiss, load_recording
from kyb.tracing import init_tracing, shutdown_tracing

ACME = Company(name="Acme Corp", country="United States", website="acme.example")
AS_OF = date(2026, 1, 15)


def _spans(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _live(tmp_path, model=None):
    trace = tmp_path / "live" / "trace.jsonl"
    init_tracing(trace)
    try:
        result = run_screen(ACME, AS_OF, 5, FakeTavily(), model)
    finally:
        shutdown_tracing()
    return result, trace


def test_plan_is_fixed_and_dated():
    plan = build_plan(ACME, AS_OF, 5)
    assert len(plan) == sum(len(c.queries) for c in CATEGORIES) == 8
    for p in plan:
        assert p.params["query"].startswith('"Acme Corp"')
        assert p.params["exact_match"] is True
        assert p.params["start_date"] == "2021-01-15"
    assert sum(1 for p in plan if p.params.get("include_domains")) == 1
    assert plan_version() == plan_version()


def test_evidence_dedupes_urls_and_keeps_text_per_category():
    store = EvidenceStore()
    page = {"url": "https://x.com/a/?utm_source=y", "title": "A", "content": "one"}
    store.add_response("litigation", {"query": "q1", "results": [page]})
    store.add_response("regulatory", {"query": "q2", "results": [{**page, "content": "two"}]})
    assert len(store) == 1
    s = store.get("S1")
    assert s.text("litigation") == "one" and s.text("regulatory") == "two"
    assert normalize_url("HTTPS://X.com/a/?utm_source=y#frag") == "https://x.com/a"


def test_live_run_validates_citations(tmp_path):
    result, _ = _live(tmp_path, FakeModel())
    for run in result.categories.values():
        assert run.parsed and run.first_try
        assert len(run.findings) == 1             # the S999-only finding is dropped
        assert run.dropped_findings == 1
        assert run.invalid_citations == 2         # S999 cited twice
        assert run.findings[0].finding.severity == "high"  # normalized from "HIGH"
        for f in run.findings:
            assert all(result.evidence.get(sid) for sid in f.finding.source_ids)
    report = render_markdown(result)
    assert "S999" not in report


def test_replay_reproduces_evidence_prompts_and_date(tmp_path):
    live_model = FakeModel()
    live, trace = _live(tmp_path, live_model)

    recording = load_recording(trace)
    assert recording.company == ACME and recording.as_of == AS_OF
    assert recording.plan_version == plan_version()
    assert len(recording.responses) == 8

    replay_model = FakeModel("fake/other")
    replay_trace = tmp_path / "replay" / "trace.jsonl"
    init_tracing(replay_trace)
    try:
        replayed = run_screen(recording.company, recording.as_of, recording.lookback_years,
                              ReplayBackend(recording), replay_model)
    finally:
        shutdown_tracing()

    assert [s.url for s in replayed.evidence.all()] == [s.url for s in live.evidence.all()]
    by_cat = lambda calls: sorted(json.dumps(c) for c in calls)  # noqa: E731
    assert by_cat(replay_model.calls) == by_cat(live_model.calls)
    assert "Today's date is 2026-01-15" in replay_model.calls[0][0]["content"]

    tavily_spans = [s for s in _spans(replay_trace) if s["name"] == "tavily.search"]
    assert len(tavily_spans) == 8
    assert all(s["attributes"]["kyb.replayed"] for s in tavily_spans)
    assert all("tavily.credits" not in s["attributes"] for s in tavily_spans)
    # A replay trace is itself a valid recording.
    assert load_recording(replay_trace).responses == recording.responses


def test_trace_has_openinference_spans(tmp_path):
    _, trace = _live(tmp_path, FakeModel())
    spans = _spans(trace)
    kinds = {s["name"]: s["attributes"].get("openinference.span.kind") for s in spans}
    assert kinds["kyb.screen"] == "CHAIN"
    assert kinds["tavily.search"] == "RETRIEVER"
    search = next(s for s in spans if s["name"] == "tavily.search")
    assert search["attributes"]["tavily.credits"] == 2
    assert "retrieval.documents.0.document.id" in search["attributes"]
    root = next(s for s in spans if s["name"] == "kyb.screen")
    assert all(s["context"]["trace_id"] == root["context"]["trace_id"] for s in spans)


def test_replay_refuses_unrecorded_search(tmp_path):
    _, trace = _live(tmp_path)
    recording = load_recording(trace)
    recording.responses.pop(next(iter(recording.responses)))
    with pytest.raises(ReplayMiss):
        run_screen(recording.company, recording.as_of, recording.lookback_years,
                   ReplayBackend(recording), None)
    # A different lookback changes every query's start_date, so nothing matches.
    recording = load_recording(trace)
    with pytest.raises(ReplayMiss):
        run_screen(recording.company, recording.as_of, 3, ReplayBackend(recording), None)


def test_parse_json_output_tolerates_wrappers_only():
    ok, err = parse_json_output('<think>x</think>```json\n{"findings": []}\n```', CategoryOutput)
    assert ok is not None and err is None
    bad, err = parse_json_output('{"findings": [{"summary": "x"}]}', CategoryOutput)
    assert bad is None and "schema" in err
    none, err = parse_json_output("no json here", CategoryOutput)
    assert none is None and "no JSON" in err


def test_repair_retry_and_not_assessed(tmp_path):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "[S1] t"}]
    call = complete_json(FakeModel(bad_first=True), msgs, CategoryOutput)
    assert call.value is not None and call.attempts == 2 and not call.first_try

    result, _ = _live(tmp_path, FakeModel(always_bad=True))
    assert all(not r.parsed and r.attempts == 2 for r in result.categories.values())
    report = render_markdown(result)
    assert report.count("Not assessed") == len(CATEGORIES)
    assert "No adverse findings" not in report
