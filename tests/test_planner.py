import json
from datetime import date

from fakes import FakeModel, FakePlannerModel, FakeTavily

from kyb.plan import CATEGORIES, REGULATOR_DOMAINS
from kyb.planner import MAX_TOTAL, FixedPlanner, LLMPlanner, make_planner
from kyb.schema import Company
from kyb.screen import run_screen
from kyb.search import ReplayBackend, load_recording
from kyb.tracing import init_tracing, shutdown_tracing

ACME = Company(name="Acme Corp", country="United States")
AS_OF = date(2026, 1, 15)


def test_llm_plan_is_validated_capped_and_filled():
    result = LLMPlanner(FakePlannerModel()).plan(ACME, AS_OF, 5)
    by_cat: dict[str, list] = {}
    for s in result.searches:
        by_cat.setdefault(s.category, []).append(s.params)
    assert set(by_cat) == {c.key for c in CATEGORIES}           # every category searched
    assert len(result.searches) <= MAX_TOTAL
    assert len(by_cat["data_breach"]) == 3                      # capped at 3 per category
    assert sorted(result.fell_back) == ["financial_crime", "financial_distress", "leadership", "sanctions"]
    assert all(p["start_date"] == "2021-01-15" for s in result.searches for p in [s.params])
    assert by_cat["regulatory"][1]["include_domains"] == list(REGULATOR_DOMAINS)
    assert by_cat["litigation"][0]["exact_match"] is False      # no quoted phrase
    assert by_cat["data_breach"][0]["topic"] == "news"
    fixes = " ".join(result.fixes)
    assert "unknown category 'esg'" in fixes and "duplicate" in fixes and "over budget" in fixes
    assert "'finance' -> 'general'" in fixes
    assert result.prompt_tokens == 800 and result.completion_tokens == 1500


def test_unusable_planner_output_falls_back_to_fixed_plan():
    result = LLMPlanner(FakePlannerModel(garbage=True)).plan(ACME, AS_OF, 5)
    fixed = FixedPlanner().plan(ACME, AS_OF, 5)
    assert [s.params for s in result.searches] == [s.params for s in fixed.searches]
    assert len(result.fell_back) == len(CATEGORIES) and "unusable" in result.fixes[-1]


def test_llm_plan_is_recorded_and_replayed_exactly(tmp_path):
    trace = tmp_path / "rec" / "trace.jsonl"
    init_tracing(trace)
    try:
        live = run_screen(ACME, AS_OF, 5, FakeTavily(), None,
                          planner=make_planner("fake/planner", lambda m, role: FakePlannerModel(m)))
    finally:
        shutdown_tracing()
    recording = load_recording(trace)
    assert recording.planner == "fake/planner"
    assert recording.credits == 2 * len(live.plan.searches)
    assert recording.plan.prompt_tokens == 800

    reader = FakeModel("fake/reader")
    init_tracing(tmp_path / "replay" / "trace.jsonl")
    try:
        replayed = run_screen(recording.company, recording.as_of, recording.lookback_years,
                              ReplayBackend(recording), reader, recorded_plan=recording.plan)
    finally:
        shutdown_tracing()
    assert [s.query for s in replayed.searches] == [s.query for s in live.searches]
    assert [s.url for s in replayed.evidence.all()] == [s.url for s in live.evidence.all()]
    assert replayed.planner == "fake/planner" and len(replayed.findings) == len(CATEGORIES)
    plan_spans = [json.loads(line) for line in (tmp_path / "replay" / "trace.jsonl").read_text().splitlines()]
    plan_span = next(s for s in plan_spans if s["name"] == "kyb.plan")
    assert plan_span["attributes"]["kyb.plan_replayed"] is True
    assert "kyb.prompt_tokens" not in plan_span["attributes"]      # planning cost isn't re-counted
