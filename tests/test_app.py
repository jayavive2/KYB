"""Drive the Streamlit app headlessly with fake Tavily and fake models."""

import json
from pathlib import Path

import yaml
from fakes import FakeTavily
from streamlit.testing.v1 import AppTest
from test_eval_and_cli import DATASET, RUBRIC, fake_factory

from kyb import factory

APP = Path(__file__).parents[1] / "app.py"


def _setup(tmp_path, monkeypatch, dataset=True):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setattr(factory, "make_model", fake_factory)
    monkeypatch.setattr(factory, "make_search_backend", FakeTavily)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.toml").write_text(
        '[eval]\njudge = "judge/x"\nplanners = ["fake/planner"]\nreaders = ["m/a", "m/bad"]\n'
        '[prices."m/a"]\ninput = 1.0\noutput = 2.0\n')
    (tmp_path / "config" / "rubric.yaml").write_text(RUBRIC.read_text())
    if dataset:
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "companies.yaml").write_text(yaml.safe_dump(DATASET))


def _app():
    return AppTest.from_file(str(APP), default_timeout=60).run()


def _text(at) -> str:
    """All markdown on the page (stat cards are markdown)."""
    return " ".join(m.value for m in at.markdown)


def _screen(at, name):
    at.text_input(key="screen_name").input(name)
    at.button(key="screen_submit").click().run()
    assert not at.exception, at.exception


def test_screen_shows_trace_then_compares_readers(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, dataset=False)               # no companies file needed for this flow
    at = _app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Screen", "Search eval", "Reader eval", "Results"]
    _screen(at, "Acme Corp")
    assert any("Screened **Acme Corp**" in s.value for s in at.success)
    assert "Search time" in _text(at) and "Read time" in _text(at)              # trace view of the screen
    assert at.multiselect(key="compare_readers").value == ["m/a", "m/bad"]

    at.button(key="compare_run").click().run()                  # the screen stays on the page across reruns
    assert not at.exception, at.exception
    run_dir = next(Path("results").glob("*-compare-Acme_Corp"))
    exp = json.loads((run_dir / "experiment.json").read_text())
    assert exp["source"] == "screens" and exp["companies"] == {"Acme_Corp": "Acme Corp"}
    assert "Best with your weights" in _text(at) and "Highest quality" in _text(at)
    assert at.selectbox(key="dash_run").value == run_dir        # listed in the Reader eval tab too


def test_search_eval(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    at = _app()
    assert at.multiselect(key="search_planners").value == ["fixed", "fake/planner"]
    assert at.multiselect(key="search_cases").value == ["acme"]   # only companies with known events
    assert at.button(key="search_run").disabled                   # recordings missing, not yet allowed
    at.checkbox(key="search_allow_record").check().run()
    at.button(key="search_run").click().run()
    assert not at.exception, at.exception
    run_dir = next(Path("results").glob("*-search"))
    assert at.selectbox(key="search_pick").value == run_dir
    assert "Coverage · fixed" in _text(at) and "Coverage · planner" in _text(at)
    assert any("Missed" in m.value for m in at.markdown)


def test_reader_eval_from_companies_file_with_trace_analysis(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    at = _app()
    assert at.button(key="run").disabled                          # no past screens yet
    at.radio(key="bench_source").set_value("evals/companies.yaml").run()
    at.checkbox(key="allow_record").check().run()
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    run_dir = next(Path("results").glob("*-readers"))
    assert at.selectbox(key="dash_run").value == run_dir
    k = f"dash_{run_dir.name}"
    assert at.selectbox(key=f"wf_reader_{k}").value in ("m/a", "m/bad")      # timeline pickers
    assert any("Trace analysis" in h.value for h in at.subheader)
    next(sl for sl in at.slider if sl.label == "Quality").set_value(1.0).run()
    assert not at.exception


def test_reader_eval_from_past_screens(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    at = _app()
    for name in ("Acme Corp", "Beta Inc"):
        _screen(at, name)
    at = _app()
    assert at.radio(key="bench_source").value == "Past screens"
    assert len(at.multiselect(key="bench_screens").value) == 2
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    exp = json.loads(next(Path("results").glob("*-readers/experiment.json")).read_text())
    assert exp["source"] == "screens" and sorted(exp["companies"].values()) == ["Acme Corp", "Beta Inc"]


def test_runs_are_blocked_without_keys(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.delenv("NEBIUS_API_KEY")
    at = _app()
    at.checkbox(key="search_allow_record").check().run()
    assert at.button(key="search_run").disabled
    assert any("NEBIUS_API_KEY" in e.value for e in at.error)
