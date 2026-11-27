import pandas as pd
from fakes import FakeJudge, FakeTavily
from test_eval_and_cli import RUBRIC, _dataset, fake_factory

from kyb import dashboard as dash
from kyb import evaluate as ev
from kyb.judge import load_rubric

READERS = ["big/model-a", "mid/model-b", "mid/model-c", "small/model-d", "m/bad"]
PRICES = {"big/model-a": {"input": 2.0, "output": 10.0}, "mid/model-b": {"input": 0.3, "output": 1.2},
          "mid/model-c": {"input": 0.3, "output": 0.9}, "small/model-d": {"input": 0.1, "output": 0.3}}


def test_pareto_and_composite():
    df = pd.DataFrame({"model": list("abcd"), "quality": [4.5, 4.0, 3.5, 4.0],
                       "cost_usd_per_company": [0.03, 0.01, 0.005, 0.02], "latency_s_per_company": [40, 20, 10, 30]})
    assert list(dash.pareto_front(df, "cost_usd_per_company")) == [True, True, True, False]  # d: b is cheaper, as good
    s = dash.composite(df, 1, 0, 0)
    assert list(s["model"]) == ["a", "b", "d", "c"] and s["composite"].iloc[0] == (4.5 - 1) / 4
    s = dash.composite(df, 0, 1, 0)
    assert s.iloc[0]["model"] == "c" and s.set_index("model").loc["a", "composite"] == 0.005 / 0.03
    df.loc[1, "cost_usd_per_company"] = None                     # unpriced reader: scored on what it has
    assert dash.composite(df, 0.5, 0.5, 0).set_index("model").loc["b", "composite"] == (3 / 4)


def test_labels_and_ticks():
    assert dash.wrap("nemotron-3-super-120b-a12b") == "nemotron-3-super-<br>120b-a12b"
    assert dash.wrap("gpt-oss-120b") == "gpt-oss-120b"
    assert dash._nice_ticks(0.002, 0.031) == [0.002, 0.005, 0.01, 0.02]


def test_five_reader_sweep_feeds_the_dashboard(tmp_path):
    data = ev.load_dataset(_dataset(tmp_path))
    root = tmp_path / "recordings"
    ev.record(data, "fake/planner", root, FakeTavily, fake_factory, log=lambda *_: None)
    judge = FakeJudge()
    out = tmp_path / "results" / "sweep-readers"
    ev.run_experiment(data, "readers", ev.readers_experiment("fake/planner", READERS), model_factory=fake_factory,
                      judge=judge, prices=PRICES, recordings_root=root, out_dir=out,
                      rubric=load_rubric(RUBRIC), pairwise=False, log=lambda *_: None)

    rubric_calls = [c for c in judge.calls if "against a rubric" in c[0]["content"]]
    assert len(rubric_calls) == 2 * len(READERS)
    assert "Screening question:" in rubric_calls[0][0]["content"]
    assert "Evidence the author was given" in rubric_calls[0][1]["content"]

    df, per_case, exp = dash.load_sweep(out)
    assert len(df) == 5 and exp["rubric"] == ["citation", "scope", "relevance", "specificity", "calibration"]
    assert df.iloc[0]["quality"] >= df.iloc[-1]["quality"]          # sorted best first
    priced = df.dropna(subset=["cost_usd_per_company"])
    assert dash.pareto_front(priced, "cost_usd_per_company").sum() >= 1
    keys = exp["rubric"]
    p = dash.PALETTES["light"]
    figs = [dash.ranking_bar(dash.composite(df, 0.6, 0.25, 0.15), p),
            dash.parallel_coordinates(df, keys, p),
            dash.tradeoff_scatter(df, "cost_usd_per_company", "Reader cost ($ per company)", p, "$%{x}"),
            dash.tradeoff_scatter(df, "latency_s_per_company", "Reader latency (sec per company)", p, "%{x}"),
            dash.heatmap(dash.rubric_matrix(df, keys), "Rubric", p, "score"),
            dash.heatmap(dash.company_matrix(per_case, list(df["model"])), "By company", p, "quality")]
    assert all(f.data for f in figs)
    table = dash.runs_table(df, keys)
    assert table.columns[0] == "Reader" and "L2 scope" in table and "L3 $ / company" in table
