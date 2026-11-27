"""Drive the Streamlit app headlessly with fake Tavily and fake models."""

import json
from pathlib import Path

from fakes import FakeTavily
from streamlit.testing.v1 import AppTest
from test_eval_and_cli import RUBRIC, fake_factory

from kyb import factory

APP = Path(__file__).parents[1] / "app.py"


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setattr(factory, "make_model", fake_factory)
    monkeypatch.setattr(factory, "make_search_backend", FakeTavily)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.toml").write_text(
        '[defaults]\nplanners = ["fake/planner"]\n'
        '[eval]\njudge = "judge/x"\nreaders = ["m/a", "m/bad"]\n'
        '[prices."m/a"]\ninput = 1.0\noutput = 2.0\n')
    (tmp_path / "config" / "rubric.yaml").write_text(RUBRIC.read_text())


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
    _setup(tmp_path, monkeypatch)
    at = _app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Screen", "Reader eval", "Results"]
    _screen(at, "Acme Corp")
    assert any("Screened **Acme Corp**" in s.value for s in at.success)
    assert "Search time" in _text(at) and "Read time" in _text(at)              # trace view of the screen
    assert "Source check time" in _text(at)
    assert "Confirmed by sources" in _text(at) and "Distinct events" in _text(at)
    saved = json.loads(next(Path("runs").glob("*-Acme_Corp/findings.json")).read_text())
    assert saved["source_check"]["model"] == "fake/judge" and saved["events"]   # the fake factory's judge
    assert at.multiselect(key="compare_readers").value == ["m/a", "m/bad"]

    at.button(key="compare_run").click().run()                  # the screen stays on the page across reruns
    assert not at.exception, at.exception
    run_dir = next(Path("results").glob("*-compare-Acme_Corp"))
    exp = json.loads((run_dir / "experiment.json").read_text())
    assert exp["source"] == "screens" and exp["companies"] == {"Acme_Corp": "Acme Corp"}
    assert "Best with your weights" in _text(at) and "Highest quality" in _text(at)
    assert at.selectbox(key="dash_run").value == run_dir        # listed in the Reader eval tab too


def test_reader_eval_from_past_screens(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    at = _app()
    for name in ("Acme Corp", "Beta Inc"):
        _screen(at, name)
    at = _app()
    assert len(at.multiselect(key="bench_screens").value) == 2
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    run_dir = next(Path("results").glob("*-readers"))
    exp = json.loads((run_dir / "experiment.json").read_text())
    assert exp["source"] == "screens" and sorted(exp["companies"].values()) == ["Acme Corp", "Beta Inc"]
    assert at.selectbox(key="dash_run").value == run_dir
    k = f"dash_{run_dir.name}"
    assert at.selectbox(key=f"wf_reader_{k}").value in ("m/a", "m/bad")      # timeline pickers
    assert any("Trace analysis" in h.value for h in at.subheader)
    next(sl for sl in at.slider if sl.label == "Quality").set_value(1.0).run()
    assert not at.exception


def test_runs_are_blocked_without_keys(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.delenv("NEBIUS_API_KEY")
    at = _app()
    at.text_input(key="judge").input("nebius/some-judge").run()   # LiteLLM: nebius/... needs NEBIUS_API_KEY
    assert at.button(key="run").disabled
    assert any("NEBIUS_API_KEY" in e.value for e in at.error)


def test_a_failed_screen_shows_one_actionable_line(tmp_path, monkeypatch):
    from tavily.errors import InvalidAPIKeyError

    class BadKey:
        mode = "live"

        def search(self, params):
            raise InvalidAPIKeyError("Unauthorized: missing or invalid API key.")

    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(factory, "make_search_backend", BadKey)
    at = _app()
    at.text_input(key="screen_name").input("Acme Corp")
    at.button(key="screen_submit").click().run()
    assert any("Tavily rejected TAVILY_API_KEY" in e.value for e in at.error)
    # The traceback is still there, folded into "Details", and the page kept rendering.
    assert [str(x.message) for x in at.exception] == ["Unauthorized: missing or invalid API key."]
    assert at.button(key="screen_submit") is not None and "screen_dir" not in at.session_state


def test_promote_the_leader_to_champion(tmp_path, monkeypatch):
    from kyb import champion as C

    _setup(tmp_path, monkeypatch)
    (tmp_path / "config" / "models.toml").write_text(       # the champion starts as the weaker reader
        '[defaults]\nreader = "m/bad"\n[eval]\njudge = "judge/x"\nreaders = ["m/a"]\n')
    at = _app()
    assert at.selectbox(key="screen_reader").value == "m/bad"
    _screen(at, "Acme Corp")
    at = _app()
    assert at.multiselect(key="readers").value == ["m/a", "m/bad"]     # the champion is always compared
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    run_dir = next(Path("results").glob("*-readers"))
    assert "Leader with your weights" in _text(at)
    at.button(key=f"promote_dash_{run_dir.name}").click().run()
    assert not at.exception, at.exception
    assert C.current({}) == "m/a" and at.selectbox(key="screen_reader").value == "m/a"
    (entry,) = C.history()
    assert entry["replaced"] == "m/bad" and entry["run"] == str(run_dir) and entry["weights"]["quality"] == 0.6
    assert entry["scores"]["m/a"]["quality"] > entry["scores"]["m/bad"]["quality"]
    assert "leads with your weights" in _text(at)                    # no button once the leader is champion
    assert not any(b.key == f"promote_dash_{run_dir.name}" for b in at.button)
