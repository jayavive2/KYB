import csv
import json
from pathlib import Path

import yaml
from fakes import FakeJudge, FakeModel, FakePlannerModel, FakeTavily
from typer.testing import CliRunner

from kyb import cli, factory
from kyb import evaluate as ev
from kyb.judge import load_rubric

DATASET = {
    "as_of": "2026-01-15",
    "lookback_years": 5,
    "cases": [
        {"id": "acme", "company": {"name": "Acme Corp", "country": "United States"},
         "expected_events": [
             {"id": "fine-2024", "category": "regulatory", "description": "Regulator fined Acme Corp $5M"},
             {"id": "missing", "category": "litigation", "description": "An event search never found"}]},
        {"id": "beta", "company": {"name": "Beta Inc"}, "expected_events": []},
    ],
}
RUBRIC = Path(__file__).parents[1] / "config" / "rubric.yaml"
PRICES = {"m/a": {"input": 1.0, "output": 2.0}, "fake/planner": {"input": 1.0, "output": 1.0}}


def fake_factory(model_id: str, role: str = "reader"):
    if model_id == "judge/x":
        return FakeJudge()
    if role == "planner":
        return FakePlannerModel(model_id)
    return FakeModel(model_id, always_bad=model_id == "m/bad")


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "companies.yaml"
    path.write_text(yaml.safe_dump(DATASET))
    return path


def _record(tmp_path, planners):
    data = ev.load_dataset(_dataset(tmp_path))
    root = tmp_path / "recordings"
    for p in planners:
        ev.record(data, p, root, FakeTavily, fake_factory, log=lambda *_: None)
    return data, root


def test_readers_experiment(tmp_path):
    data, root = _record(tmp_path, ["fixed"])
    assert ev.missing_recordings(data.cases, ["fixed", "fake/planner"], root) == [
        ("fake/planner", "acme"), ("fake/planner", "beta")]
    out = tmp_path / "results" / "readers"
    a, b = ev.run_experiment(data, "readers", ev.readers_experiment("fixed", ["m/a", "m/bad"]),
                             model_factory=fake_factory, judge=FakeJudge(), prices=PRICES,
                             recordings_root=root, out_dir=out, rubric=load_rubric(RUBRIC), log=lambda *_: None)
    assert (a.label, a.cases, a.findings, a.searches) == ("m/a", 2, 14, 16)
    assert a.events_total == 2 and a.events_in_evidence == 1 and a.events_matched == 1
    assert b.findings == 0 and b.categories_failed == 14
    assert (a.wins, a.ties, a.losses) == (2, 0, 0) and (b.wins, b.losses) == (0, 2)

    rows = {r["arm"]: r for r in csv.DictReader((out / "summary.csv").open())}
    assert rows["m/a"]["coverage"] == "50%" and rows["m/a"]["recall_given_evidence"] == "100%"
    assert rows["m/a"]["rubric_citation"] == "4.00" and rows["m/a"]["rubric_scope"] == "3.00"
    assert rows["m/a"]["rubric_calibration"] == "n/a"           # judge returned null
    # weighted over the criteria that were scored: (.25*4 + .25*3 + .2*5 + .15*4) / .85
    assert rows["m/a"]["quality"] == f"{(0.25 * 4 + 0.25 * 3 + 0.2 * 5 + 0.15 * 4) / 0.85:.2f}"
    assert rows["m/bad"]["rubric_citation"] == "n/a"            # no findings: only scope applies
    assert rows["m/bad"]["rubric_scope"] == "3.00" and rows["m/bad"]["quality"] == "3.00"
    metrics = {m["arm"]: m for m in json.loads((out / "metrics.json").read_text())}
    assert metrics["m/a"]["cost_usd_per_company"] == (7000 / 1e6 * 1.0 + 1400 / 1e6 * 2.0)
    assert metrics["m/bad"]["cost_usd_per_company"] is None and metrics["m/a"]["json_first_try"] == 1.0
    assert rows["m/a"]["tavily_credits"] == "32"               # 2 companies x 8 searches x 2
    assert rows["m/a"]["planner_usd_per_company"] == "0.0000"  # fixed plan costs nothing
    summary = (out / "summary.md").read_text()
    assert "# Reader comparison" in summary and "held when the order was swapped: 2 of 2" in summary
    pair = next(csv.DictReader((out / "pairwise.csv").open()))
    assert pair["winner"] == "m/a" and pair["verdict_first_as_A"] == "A" and pair["verdict_second_as_A"] == "B"
    rubric_rows = list(csv.DictReader((out / "rubric.csv").open()))
    assert "made_up_key" not in rubric_rows[0]
    assert json.loads((out / "experiment.json").read_text())["kind"] == "readers"
    assert (out / "cases" / "acme" / "m_a.md").exists()


def test_search_eval_against_known_events(tmp_path):
    data, root = _record(tmp_path, ["fixed", "fake/planner"])
    out = tmp_path / "results" / "search"
    fixed, llm = ev.search_eval(data, ["fixed", "fake/planner"], judge=FakeJudge(), recordings_root=root,
                                out_dir=out, log=lambda *_: None)
    # FakeJudge marks the first known event found (in the first source) and the second missed.
    assert (fixed["events"], fixed["found"], fixed["coverage"]) == (2, 1, 0.5)
    assert fixed["searches_per_company"] == 8 and llm["searches_per_company"] == 10
    assert fixed["tavily_credits"] == 16 and fixed["results_per_search"] == 3       # from the recording's trace
    assert llm["planner_calls"] == 1 and fixed["planner_calls"] == 0
    assert fixed["by_category"] == {"regulatory": {"found": 1, "events": 1}, "litigation": {"found": 0, "events": 1}}
    rows = list(csv.DictReader((out / "search_eval.csv").open()))
    hit = next(r for r in rows if r["planner"] == "fixed" and r["event_id"] == "fine-2024")
    assert hit["found"] == "True" and hit["source_urls"] and hit["found_by_queries"].startswith('"Acme Corp"')
    summary = (out / "summary.md").read_text()
    assert "# Search eval" in summary and "## Missed events" in summary and "missing" in summary
    assert json.loads((out / "experiment.json").read_text())["cases"] == ["acme"]   # beta has no known events


def test_estimate_and_judge_outage(tmp_path):
    assert ev.estimate_credits([("fixed", "a"), ("x/model", "a")]) == 8 * 2 + 12 * 2

    class DownJudge(FakeJudge):
        def chat(self, messages):
            raise ConnectionError("judge endpoint down")

    data, root = _record(tmp_path, ["fixed"])
    out = tmp_path / "results" / "down"
    (a,) = ev.run_experiment(data, "readers", ev.readers_experiment("fixed", ["m/a"]),
                             model_factory=fake_factory, judge=DownJudge(), prices={}, recordings_root=root,
                             out_dir=out, rubric=load_rubric(RUBRIC), log=lambda *_: None)
    assert a.findings == 14 and a.judged == 0
    assert "**Warning:**" in (out / "summary.md").read_text()


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setattr(factory, "make_model", fake_factory)
    monkeypatch.setattr(factory, "make_search_backend", FakeTavily)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.toml").write_text(
        '[defaults]\nplanner = "fixed"\nreader = "m/a"\n'
        '[eval]\njudge = "judge/x"\nplanners = ["fake/planner"]\nreaders = ["m/a", "m/b"]\n')
    (tmp_path / "config" / "rubric.yaml").write_text(RUBRIC.read_text())
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "companies.yaml").write_text(yaml.safe_dump(DATASET))
    runner = CliRunner()

    r = runner.invoke(cli.app, ["screen", "Acme Corp", "--planner", "fake/planner", "--as-of", "2026-01-15"])
    assert r.exit_code == 0, r.output
    assert "Plan fixes" in r.output
    trace = next(Path("runs").glob("*/trace.jsonl"))
    assert "kyb compare" in r.output
    r = runner.invoke(cli.app, ["compare", str(trace), "--readers", "m/a,m/b", "--no-pairwise"])
    assert r.exit_code == 0 and "Reader comparison" in r.output, r.output
    compare_dir = next(Path("results").glob("*-compare-Acme_Corp"))
    metrics = json.loads((compare_dir / "metrics.json").read_text())
    assert {m["reader"] for m in metrics} == {"m/a", "m/b"}
    assert metrics[0]["recall"] is not None        # Acme Corp is in evals/companies.yaml: known events attached
    r = runner.invoke(cli.app, ["replay", str(trace), "--reader", "m/other"])
    assert r.exit_code == 0, r.output
    replay_report = next(trace.parent.glob("replay-m_other-*/report.md")).read_text()
    assert "planner: fake/planner · reader: m/other" in replay_report

    r = runner.invoke(cli.app, ["eval", "readers"])
    assert r.exit_code == 1 and "--record-missing" in r.output   # no recordings yet, no silent spending
    r = runner.invoke(cli.app, ["eval", "record"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli.app, ["eval", "readers", "--cases", "acme"])
    assert r.exit_code == 0 and "Reader comparison" in r.output, r.output
    r = runner.invoke(cli.app, ["eval", "search", "--record-missing"])
    assert r.exit_code == 0 and "Search eval" in r.output, r.output
    r = runner.invoke(cli.app, ["eval", "readers", "--cases", "acme", "--no-pairwise"])
    assert "Model calls (from the trace)" in r.output
    judgments = next(Path("results").glob("*-readers/judgments.csv"))
    r = runner.invoke(cli.app, ["eval", "agreement", str(judgments)])
    assert r.exit_code == 0 and "supported: 0/0" in r.output


def test_benchmark_from_past_screens(tmp_path):
    """No dataset: several past screens become the reader eval; known events attach by name."""
    from datetime import date

    from kyb.planner import make_planner
    from kyb.schema import Company
    from kyb.screen import run_screen
    from kyb.tracing import init_tracing, shutdown_tracing

    traces = []
    for name in ("Acme Corp", "Beta Inc", "Acme Corp"):        # the same company twice gets two cases
        trace = tmp_path / "runs" / f"{len(traces)}" / "trace.jsonl"
        init_tracing(trace)
        try:
            run_screen(Company(name=name), date(2026, 1, 15), 5, FakeTavily(), FakeModel("m/a"),
                       planner=make_planner("fake/planner", fake_factory))
        finally:
            shutdown_tracing()
        traces.append(trace)
    key = _dataset(tmp_path)                                     # answer key for Acme Corp only

    screens = ev.load_screens(traces, key)
    assert [c.id for c in screens.cases] == ["Acme_Corp", "Beta_Inc", "Acme_Corp-2"]
    assert len(screens.cases[0].expected_events) == 2 and screens.cases[1].expected_events == []
    assert screens.planner_label == "fake/planner"

    out = tmp_path / "results" / "readers"
    a, _ = ev.compare_screens(traces, ["m/a", "m/b"], model_factory=fake_factory, judge=FakeJudge(),
                              prices=PRICES, out_dir=out, answer_key=key, rubric=load_rubric(RUBRIC),
                              pairwise=False, log=lambda *_: None)
    assert a.cases == 3 and a.events_total == 4                  # two Acme screens x two known events
    assert "## Model calls (from the trace)" in (out / "summary.md").read_text()
    calls = {r["role"] for r in csv.DictReader((out / "model_calls.csv").open())}
    assert calls == {"reader", "judge"}
