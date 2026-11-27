import csv
import json
from datetime import date
from pathlib import Path

import pytest
from fakes import FakeJudge, FakeModel, FakePlannerModel, FakeTavily
from typer.testing import CliRunner

from kyb import cli, factory
from kyb import evaluate as ev
from kyb.judge import load_rubric
from kyb.planner import make_planner
from kyb.schema import Company
from kyb.screen import run_screen
from kyb.tracing import init_tracing, shutdown_tracing

RUBRIC = Path(__file__).parents[1] / "config" / "rubric.yaml"
PRICES = {"m/a": {"input": 1.0, "output": 2.0}, "fake/planner": {"input": 1.0, "output": 1.0}}


def fake_factory(model_id: str, role: str = "reader"):
    if model_id == "judge/x":
        return FakeJudge()
    if role == "planner":
        return FakePlannerModel(model_id)
    return FakeModel(model_id, always_bad=model_id == "m/bad")


def screens(tmp_path: Path, names: list[str], planners: list[str] | None = None) -> list[Path]:
    """Past screens to replay: one trace per company, as `kyb screen` would write them."""
    traces = []
    for i, name in enumerate(names):
        trace = tmp_path / "runs" / f"{i}-{name}" / "trace.jsonl"
        init_tracing(trace)
        try:
            run_screen(Company(name=name), date(2026, 1, 15), 5, FakeTavily(), FakeModel("m/a"),
                       planner=make_planner((planners or ["fixed"] * len(names))[i], fake_factory))
        finally:
            shutdown_tracing()
        traces.append(trace)
    return traces


def test_reader_eval_on_past_screens(tmp_path):
    traces = screens(tmp_path, ["Acme Corp", "Beta Inc"])
    out = tmp_path / "results" / "readers"
    a, b = ev.compare_screens(traces, ["m/a", "m/bad"], model_factory=fake_factory, judge=FakeJudge(),
                              prices=PRICES, out_dir=out, rubric=load_rubric(RUBRIC), pairwise=True,
                              log=lambda *_: None)
    assert (a.label, a.cases, a.findings, a.events) == ("m/a", 2, 14, 2)   # 7 duplicates merge to 1 per screen
    assert b.findings == 0 and b.categories_failed == 14
    assert (a.wins, a.ties, a.losses) == (2, 0, 0) and (b.wins, b.losses) == (0, 2)

    rows = {r["arm"]: r for r in csv.DictReader((out / "summary.csv").open())}
    assert rows["m/a"]["rubric_citation"] == "4.00" and rows["m/a"]["rubric_scope"] == "3.00"
    assert rows["m/a"]["rubric_calibration"] == "n/a"           # judge returned null
    # weighted over the criteria that were scored: (.25*4 + .25*3 + .2*5 + .15*4) / .85
    assert rows["m/a"]["quality"] == f"{(0.25 * 4 + 0.25 * 3 + 0.2 * 5 + 0.15 * 4) / 0.85:.2f}"
    assert rows["m/bad"]["rubric_citation"] == "n/a"            # no findings: only scope applies
    assert rows["m/bad"]["rubric_scope"] == "3.00" and rows["m/bad"]["quality"] == "3.00"
    assert "coverage" not in rows["m/a"] and "recall" not in rows["m/a"]
    metrics = {m["arm"]: m for m in json.loads((out / "metrics.json").read_text())}
    # 7 category calls (1000 in, 200 out each) + the merge call (300 in, 30 out)
    assert metrics["m/a"]["cost_usd_per_company"] == pytest.approx(7300 / 1e6 * 1.0 + 1430 / 1e6 * 2.0)
    assert metrics["m/bad"]["cost_usd_per_company"] is None and metrics["m/a"]["json_first_try"] == 1.0
    summary = (out / "summary.md").read_text()
    assert "# Reader comparison" in summary and "held when the order was swapped: 2 of 2" in summary
    assert "## Model calls (from the trace)" in summary
    pair = next(csv.DictReader((out / "pairwise.csv").open()))
    assert pair["winner"] == "m/a" and pair["verdict_first_as_A"] == "A" and pair["verdict_second_as_A"] == "B"
    rubric_rows = list(csv.DictReader((out / "rubric.csv").open()))
    assert "made_up_key" not in rubric_rows[0]
    exp = json.loads((out / "experiment.json").read_text())
    assert exp["kind"] == "readers" and exp["source"] == "screens" and exp["arms"][0]["planner"] == "fixed"
    assert (out / "cases" / "Acme_Corp" / "m_a.md").exists()
    assert {r["role"] for r in csv.DictReader((out / "model_calls.csv").open())} == {"reader", "judge"}


def test_screens_become_cases(tmp_path):
    traces = screens(tmp_path, ["Acme Corp", "Beta Inc", "Acme Corp"], ["fixed", "fixed", "fake/planner"])
    cases = ev.load_screens(traces)
    assert [c.id for c in cases] == ["Acme_Corp", "Beta_Inc", "Acme_Corp-2"]   # same company twice: two cases
    out = tmp_path / "results" / "mixed"
    ev.compare_screens(traces, ["m/a"], model_factory=fake_factory, judge=FakeJudge(), prices={}, out_dir=out,
                       log=lambda *_: None)
    assert json.loads((out / "experiment.json").read_text())["arms"][0]["planner"] == ev.AS_RECORDED


def test_judge_outage(tmp_path):
    class DownJudge(FakeJudge):
        def chat(self, messages):
            raise ConnectionError("judge endpoint down")

    out = tmp_path / "results" / "down"
    (a,) = ev.compare_screens(screens(tmp_path, ["Acme Corp"]), ["m/a"], model_factory=fake_factory,
                              judge=DownJudge(), prices={}, out_dir=out, rubric=load_rubric(RUBRIC),
                              log=lambda *_: None)
    assert a.findings == 7 and a.judged == 0
    assert "**Warning:**" in (out / "summary.md").read_text()


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setattr(factory, "make_model", fake_factory)
    monkeypatch.setattr(factory, "make_search_backend", FakeTavily)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.toml").write_text(
        '[defaults]\nplanner = "fake/planner"\nreader = "m/a"\n'
        '[eval]\njudge = "judge/x"\nreaders = ["m/a", "m/b"]\n')
    (tmp_path / "config" / "rubric.yaml").write_text(RUBRIC.read_text())
    runner = CliRunner()

    r = runner.invoke(cli.app, ["screen", "Acme Corp", "--as-of", "2026-01-15"])   # configured planner, champion
    assert r.exit_code == 0, r.output
    assert "planner fake/planner · reader m/a" in r.output
    assert runner.invoke(cli.app, ["screen", "Acme Corp", "--reader", "m/b"]).exit_code != 0  # no per-screen pick
    for old in Path("runs").glob("*"):
        import shutil

        shutil.rmtree(old)
    r = runner.invoke(cli.app, ["screen", "Acme Corp", "--as-of", "2026-01-15"])
    assert r.exit_code == 0, r.output
    assert "Plan fixes" in r.output and "source check:" in r.output
    trace = next(Path("runs").glob("*/trace.jsonl"))
    assert "kyb compare" in r.output
    r = runner.invoke(cli.app, ["compare", str(trace), "--readers", "m/a,m/b"])
    assert r.exit_code == 0 and "Reader comparison" in r.output, r.output
    assert "Model calls (from the trace)" in r.output
    compare_dir = next(Path("results").glob("*-compare-Acme_Corp"))
    metrics = json.loads((compare_dir / "metrics.json").read_text())
    assert {m["reader"] for m in metrics} == {"m/a", "m/b"}
    r = runner.invoke(cli.app, ["replay", str(trace), "--reader", "m/other"])
    assert r.exit_code == 0, r.output
    replay_report = next(trace.parent.glob("replay-m_other-*/report.md")).read_text()
    assert "planner: fake/planner · reader: m/other" in replay_report
    r = runner.invoke(cli.app, ["agreement", str(compare_dir / "judgments.csv")])
    assert r.exit_code == 0 and "supported: 0/0" in r.output
    assert runner.invoke(cli.app, ["eval", "search"]).exit_code != 0        # the search eval is gone


def test_readers_run_in_parallel_with_the_same_results(tmp_path):
    import threading
    import time

    active, peak, lock = [0], [0], threading.Lock()

    class Slow(FakeModel):
        def chat(self, messages):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            try:
                return super().chat(messages)
            finally:
                with lock:
                    active[0] -= 1

    def slow_factory(model_id, role="reader"):
        return FakeJudge() if model_id == "judge/x" else Slow(model_id, always_bad=model_id == "m/bad")

    traces = screens(tmp_path, ["Acme Corp"])
    readers = ["m/a", "m/b", "m/c", "m/bad"]
    outs = {}
    for parallel in (1, 4):
        out = tmp_path / f"p{parallel}"
        active[0] = peak[0] = 0
        ev.compare_screens(traces, readers, model_factory=slow_factory, judge=FakeJudge(), prices=PRICES,
                           out_dir=out, rubric=load_rubric(RUBRIC), pairwise=True, log=lambda *_: None,
                           parallel=parallel)
        outs[parallel] = (out, peak[0])
    assert outs[1][1] <= 4 < outs[4][1]          # one reader's 4 category calls at most, vs several readers
    for name in ("judgments.csv", "pairwise.csv"):
        assert (outs[1][0] / name).read_text() == (outs[4][0] / name).read_text()   # same rows, same order
    slow = lambda path: [{k: v for k, v in r.items() if "sec" not in k}  # noqa: E731
                         for r in csv.DictReader((path / "summary.csv").open())]
    assert slow(outs[1][0]) == slow(outs[4][0])


def test_cut_off_answers_and_judge_failures_are_explained(tmp_path):
    from kyb.llm import ChatResult

    class Rambles(FakeModel):                      # a reasoning model that never gets to the JSON
        def chat(self, messages):
            self.calls.append(messages)
            return ChatResult("<think>" + "hmm " * 50, 1000, 8192, 0.01, finish_reason="length")

    class PickyJudge(FakeJudge):                   # fails the rubric for one reader only
        def chat(self, messages):
            report = messages[1]["content"].split("=====")[0]
            if "against a rubric" in messages[0]["content"] and "Regulator fined" in report:
                return ChatResult("", 100, 8192, 0.01, finish_reason="length")
            return super().chat(messages)

    rambler = Rambles("m/ramble")
    factory_ = lambda mid, role="reader": rambler if mid == "m/ramble" else FakeModel(mid)  # noqa: E731
    out = tmp_path / "out"
    by = {t.reader: t for t in ev.compare_screens(
        screens(tmp_path, ["Acme Corp"]), ["m/ramble", "m/a"], model_factory=factory_, judge=PickyJudge(),
        prices={}, out_dir=out, rubric=load_rubric(RUBRIC), log=lambda *_: None)}
    (problem,) = by["m/ramble"].problems
    assert "7 of 7 categories not assessed" in problem and "cut off at the" in problem
    # the retry after a cut-off asks again briefly instead of resending the cut-off text
    retries = [c for c in rambler.calls if "ran out of room" in c[-1]["content"]]   # categories run in parallel
    assert len(retries) == 7 and not any(m["role"] == "assistant" for c in retries for m in c)
    assert any("judge rubric:" in p and "cut off" in p for p in by["m/a"].problems)
    summary = (out / "summary.md").read_text()
    assert "## Problems" in summary and "**m/ramble**" in summary
    assert json.loads((out / "metrics.json").read_text())[0]["problems"]
