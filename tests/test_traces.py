from datetime import date

from fakes import FakeModel, FakePlannerModel, FakeTavily

from kyb import traces as T
from kyb.planner import make_planner
from kyb.schema import Company
from kyb.screen import run_screen
from kyb.tracing import init_tracing, shutdown_tracing


def _screen(tmp_path, reader):
    trace = tmp_path / "trace.jsonl"
    init_tracing(trace)
    try:
        run_screen(Company(name="Acme Corp"), date(2026, 1, 15), 5, FakeTavily(), reader,
                   planner=make_planner("fake/planner", lambda m, role: FakePlannerModel(m)))
    finally:
        shutdown_tracing()
    return T.load_spans(trace)


def test_llm_call_stats_by_role(tmp_path):
    spans = _screen(tmp_path, FakeModel("fake/reader", bad_first=True))
    calls = T.llm_calls(spans)
    assert set(calls["role"]) == {"planner", "reader"}
    stats = T.llm_stats(calls).set_index("role")
    assert stats.loc["planner", "calls"] == 1 and stats.loc["planner", "mean_completion_tokens"] == 1500
    reader = stats.loc["reader"]
    assert reader["calls"] == 16                    # 7 categories + the merge step, each repaired once
    assert reader["json_repairs"] == 8 and reader["unusable_output"] == 8
    assert reader["errors"] == 0 and reader["p50_latency_s"] > 0


def test_search_stats_stages_and_figures(tmp_path):
    spans = _screen(tmp_path, FakeModel("fake/reader"))
    s = T.search_stats(spans)
    assert s["searches"] == 10 and s["results_per_search"] == 3 and s["credits"] == 20
    trace_id = spans.loc[spans["name"] == "kyb.screen", "trace_id"].iloc[0]
    stages = T.stage_times(spans, trace_id)
    assert set(stages) == {"plan", "search", "read"}
    fig = T.waterfall(spans, trace_id)
    labels = list(fig.layout.yaxis.ticktext)
    assert labels[0] == "plan" and "↳ model call · planner" in labels[1]
    assert sum(lbl.startswith("search ·") for lbl in labels) == 10
    assert {t.name for t in fig.data} == {"plan", "search", "read"}   # live searches, not replayed
    assert T.latency_strip(T.llm_calls(spans), "reader").data
