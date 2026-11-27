"""Command-line interface.

  kyb screen "Company"            live: plan, search, read; writes a report and a replayable trace
  kyb replay <trace.jsonl>        same plan and evidence, another reader
  kyb compare <trace.jsonl>...    reader eval on past screens (search held constant, judged)
  kyb models                      models your Nebius key can use
  kyb agreement <judgments.csv>   judge vs your hand labels
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from . import evaluate as ev
from . import factory
from .judge import load_rubric
from .llm import NEBIUS_BASE_URL
from .plan import CATEGORIES, plan_version
from .planner import FIXED, make_planner
from .report import render_markdown, to_json
from .schema import Company
from .screen import ScreenResult, run_screen
from .search import ReplayBackend, load_recording
from .tracing import init_tracing, shutdown_tracing

load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="KYB adverse-media screening on Tavily + Nebius, with trace replay.")
console = Console()

CONFIG = Path("config/models.toml")
RUBRIC = Path("config/rubric.yaml")

JudgeOpt = Annotated[Optional[str], typer.Option(help="Judge model ID (default: config)")]


def _config() -> dict:
    return ev.load_config(CONFIG) if CONFIG.exists() else {}


def _planner() -> str:
    """Fixed: [defaults] planner in config/models.toml."""
    return _config().get("defaults", {}).get("planner", FIXED)


def _champion() -> str:
    """Fixed until a promotion (config/champion.json); before the first, [defaults] reader."""
    from .champion import current

    cfg = _config()
    return current(cfg) or cfg["eval"]["readers"][0]


def _require(name: str, hint: str) -> None:
    if not os.getenv(name):
        console.print(f"[bold red]Missing {name}.[/bold red] {hint}")
        raise typer.Exit(1)


def _require_tavily() -> None:
    _require("TAVILY_API_KEY", "Create one at https://app.tavily.com and put it in .env")


def _require_nebius() -> None:
    _require("NEBIUS_API_KEY", "Create one at https://tokenfactory.nebius.com and put it in .env")


def _fail(exc: Exception) -> None:
    """Credit, key and model-name errors get one actionable line; anything else, the traceback."""
    from .llm import describe_error, is_fatal

    if not is_fatal(exc):
        raise exc
    console.print(f"[bold red]{describe_error(exc)}[/bold red]")
    raise typer.Exit(1) from exc


def _require_keys(*model_ids: str) -> None:
    """Each model's provider key, per LiteLLM (nebius/... needs NEBIUS_API_KEY, openai/... OPENAI_API_KEY)."""
    from .llm import missing_keys

    missing = missing_keys([m for m in model_ids if m and m != FIXED])
    if missing:
        for model, keys in missing.items():
            console.print(f"[bold red]Missing {', '.join(keys)}[/bold red] for {model}. Put it in .env.")
        raise typer.Exit(1)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ids(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _print_result(result: ScreenResult, run_dir: Path) -> None:
    if result.categories:
        table = Table(title=f"{result.company.name} · planner {result.planner} · reader {result.model}")
        for col in ("Category", "Searches", "Findings", "Sources", "Note"):
            table.add_column(col)
        for cat in CATEGORIES:
            run = result.categories[cat.key]
            note = "not assessed" if not run.parsed else (
                f"{run.invalid_citations} invalid citation(s) removed" if run.invalid_citations else "")
            table.add_row(cat.title, str(sum(s.category == cat.key for s in result.searches)),
                          str(len(run.findings)) if run.parsed else "—", str(len(run.source_ids)), note)
        console.print(table)
        from .events import build_events

        events, held = build_events(result, result.check)
        line = f"{len(events)} distinct events from {len(result.findings)} findings"
        if result.check is not None:
            verdicts = [result.check.confirmed(f.id) for f in result.findings]
            line += (f" · source check: {verdicts.count(True)} confirmed, {len(held)} held back, "
                     f"{verdicts.count(None)} not checked")
        console.print(line)
        console.print(f"Reader tokens: {result.prompt_tokens:,} in / {result.completion_tokens:,} out · "
                      f"{result.wall_time_s:.1f}s")
    else:
        console.print(f"Recorded {len(result.evidence)} sources from {len(result.searches)} searches.")
    if result.plan and result.plan.fixes:
        console.print(f"[yellow]Plan fixes:[/yellow] {'; '.join(result.plan.fixes)}")
    console.print(f"Report: {run_dir / 'report.md'}\nTrace:  {run_dir / 'trace.jsonl'}")


def _checker(judge_id: str):
    from .judge import check_sources

    judge = factory.make_model(judge_id, "judge")
    return lambda result: check_sources(judge, result)


def _write_outputs(result: ScreenResult, run_dir: Path) -> None:
    (run_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")
    (run_dir / "findings.json").write_text(json.dumps(to_json(result), indent=2), encoding="utf-8")


@app.command()
def screen(
    name: Annotated[str, typer.Argument(help="Company name, searched as an exact phrase")],
    country: Annotated[Optional[str], typer.Option(help="Country of incorporation or HQ")] = None,
    website: Annotated[Optional[str], typer.Option(help="Company website, e.g. td.com")] = None,
    industry: Annotated[Optional[str], typer.Option()] = None,
    alias: Annotated[Optional[list[str]], typer.Option(help="Other names; repeatable")] = None,
    notes: Annotated[Optional[str], typer.Option(help="Anything that tells it apart from namesakes")] = None,
    lookback_years: Annotated[int, typer.Option(help="Only events this recent")] = 5,
    as_of: Annotated[Optional[str], typer.Option(help="Screen date YYYY-MM-DD (default today)")] = None,
    search_only: Annotated[bool, typer.Option(help="Plan and search, no reader")] = False,
    out: Annotated[Path, typer.Option(help="Parent folder for run output")] = Path("runs"),
    check: Annotated[bool, typer.Option(help="Check every finding against its cited sources")] = True,
    judge: Annotated[Optional[str], typer.Option(help="Model for the source check")] = None,
) -> None:
    """Screen a company live with the configured planner and the champion reader; write a cited report
    plus a replayable trace."""
    planner, reader = _planner(), _champion()   # fixed; only a promotion changes the champion
    check = check and not search_only
    judge_id = judge or _config().get("eval", {}).get("judge", reader)
    _require_tavily()
    _require_keys(planner, None if search_only else reader, judge_id if check else None)
    company = Company(name=name, aliases=alias or [], country=country, website=website,
                      industry=industry, notes=notes)
    run_dir = out / f"{_stamp()}-{ev.slug(name)}"
    init_tracing(run_dir / "trace.jsonl")
    try:
        result = run_screen(
            company, date.fromisoformat(as_of) if as_of else date.today(), lookback_years,
            factory.make_search_backend(), None if search_only else factory.make_model(reader, "reader"),
            planner=make_planner(planner, factory.make_model), session_id=run_dir.name,
            checker=_checker(judge_id) if check else None)
    except Exception as exc:
        _fail(exc)
    finally:
        shutdown_tracing()
    _write_outputs(result, run_dir)
    _print_result(result, run_dir)
    if not search_only:
        console.print(f"Compare other readers on this screen: kyb compare {run_dir / 'trace.jsonl'}")


@app.command()
def replay(
    trace: Annotated[Path, typer.Argument(help="trace.jsonl from an earlier screen")],
    reader: Annotated[Optional[str], typer.Option(help="Reader model to run on the recorded evidence")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Output folder")] = None,
) -> None:
    """Re-run a recorded screen with another reader: same plan, same evidence, no Tavily calls."""
    reader = reader or _champion()
    _require_keys(reader)
    recording = load_recording(trace)
    if recording.plan is None and recording.plan_version != plan_version():
        console.print("[yellow]The fixed plan changed since this was recorded; "
                      "searches may not be found in the recording.[/yellow]")
    run_dir = out or trace.parent / f"replay-{ev.slug(reader)}-{_stamp()}"
    init_tracing(run_dir / "trace.jsonl")
    try:
        result = run_screen(recording.company, recording.as_of, recording.lookback_years,
                            ReplayBackend(recording), factory.make_model(reader, "reader"),
                            recorded_plan=recording.plan, session_id=run_dir.name)
    except Exception as exc:
        _fail(exc)
    finally:
        shutdown_tracing()
    _write_outputs(result, run_dir)
    _print_result(result, run_dir)


@app.command()
def compare(
    traces: Annotated[list[Path], typer.Argument(help="One or more trace.jsonl files from `kyb screen`")],
    readers: Annotated[Optional[str], typer.Option(help="Comma-separated readers (default: config)")] = None,
    judge: JudgeOpt = None,
    pairwise: Annotated[bool, typer.Option(help="Also compare reports two at a time")] = False,
    parallel: Annotated[int, typer.Option(help="Readers to run at once on each screen")] = 5,
) -> None:
    """Reader eval on past screens: plan and search results held constant, readers swapped, rubric-judged."""
    cfg = _config()
    judge_id = judge or cfg["eval"]["judge"]
    from .champion import current

    reader_ids = _ids(readers) or list(dict.fromkeys(cfg["eval"]["readers"] + [current(cfg)]))
    _require_keys(judge_id, *reader_ids)
    if judge_id in reader_ids:
        console.print("[yellow]The judge is also a reader; it will grade its own output.[/yellow]")
    names = {load_recording(t).company.name for t in traces}
    what = ev.slug(names.pop()) if len(names) == 1 else f"{len(traces)}-screens"
    out_dir = Path("results") / f"{_stamp()}-compare-{what}"
    ev.compare_screens(traces, reader_ids, model_factory=factory.make_model,
                       judge=factory.make_model(judge_id, "judge"), prices=cfg.get("prices", {}), out_dir=out_dir,
                       rubric=load_rubric(RUBRIC), pairwise=pairwise, log=console.print, parallel=parallel)
    console.print(Markdown((out_dir / "summary.md").read_text(encoding="utf-8")))
    console.print(f"\nResults: {out_dir}\nDashboard: uv run streamlit run app.py (Reader eval tab)")


@app.command()
def models() -> None:
    """List the Nebius Token Factory models your NEBIUS_API_KEY can use, as LiteLLM model strings."""
    _require_nebius()
    from openai import OpenAI

    client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL", NEBIUS_BASE_URL), api_key=os.environ["NEBIUS_API_KEY"])
    priced = _config().get("prices", {})
    for m in sorted(client.models.list(), key=lambda m: m.id):
        name = f"nebius/{m.id}"
        console.print(f"{name}{'  (priced in config)' if name in priced else ''}")
    console.print("Other providers work too, with their own key: openai/<model> (OPENAI_API_KEY), "
                  "anthropic/<model> (ANTHROPIC_API_KEY), ... See docs.litellm.ai/docs/providers.")


@app.command()
def agreement(
    judgments: Annotated[Path, typer.Argument(help="judgments.csv with human_* columns filled (y/n)")],
) -> None:
    """Report how often your labels agree with the judge's."""
    for key, (agree, n) in ev.agreement(judgments).items():
        pct = f"{100 * agree / n:.0f}%" if n else "n/a"
        console.print(f"{key}: {agree}/{n} agree ({pct})")


if __name__ == "__main__":
    app()
