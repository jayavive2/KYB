"""Command-line interface.

  kyb screen "Company"            live: plan, search, read; writes a report and a replayable trace
  kyb replay <trace.jsonl>        same plan and evidence, another reader
  kyb compare <trace.jsonl>...    reader eval on past screens (search held constant, rubric judge)
  kyb models                      models your Nebius key can use
  kyb eval search                 eval 1: did search find the known events in evals/companies.yaml?
  kyb eval readers                eval 2 on the companies file: readers on fixed search results
  kyb eval record                 record a planner's searches for the companies file
  kyb eval agreement              judge vs your hand labels
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
from .llm import DEFAULT_BASE_URL
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
eval_app = typer.Typer(no_args_is_help=True,
                       help="The two evals on evals/companies.yaml: search vs known events; readers on fixed search.")
app.add_typer(eval_app, name="eval")
console = Console()

CONFIG = Path("config/models.toml")
RUBRIC = Path("config/rubric.yaml")
DATASET = Path("evals/companies.yaml")
RECORDINGS = Path("recordings")


DatasetOpt = Annotated[Path, typer.Option(help="Eval cases YAML")]
CasesOpt = Annotated[Optional[str], typer.Option(help="Comma-separated case IDs (default: all)")]
JudgeOpt = Annotated[Optional[str], typer.Option(help="Judge model ID (default: config)")]
RecordOpt = Annotated[bool, typer.Option(help="Record missing planner searches live (uses Tavily credits)")]


def _config() -> dict:
    return ev.load_config(CONFIG) if CONFIG.exists() else {}


def _default(role: str, fallback: str) -> str:
    return os.getenv(f"KYB_{role.upper()}") or _config().get("defaults", {}).get(role, fallback)


def _require(name: str, hint: str) -> None:
    if not os.getenv(name):
        console.print(f"[bold red]Missing {name}.[/bold red] {hint}")
        raise typer.Exit(1)


def _require_tavily() -> None:
    _require("TAVILY_API_KEY", "Create one at https://app.tavily.com and put it in .env")


def _require_nebius() -> None:
    _require("NEBIUS_API_KEY", "Create one at https://tokenfactory.nebius.com and put it in .env")


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
        console.print(f"Reader tokens: {result.prompt_tokens:,} in / {result.completion_tokens:,} out · "
                      f"{result.wall_time_s:.1f}s")
    else:
        console.print(f"Recorded {len(result.evidence)} sources from {len(result.searches)} searches.")
    if result.plan and result.plan.fixes:
        console.print(f"[yellow]Plan fixes:[/yellow] {'; '.join(result.plan.fixes)}")
    console.print(f"Report: {run_dir / 'report.md'}\nTrace:  {run_dir / 'trace.jsonl'}")


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
    planner: Annotated[Optional[str], typer.Option(help="'fixed' or a model ID")] = None,
    reader: Annotated[Optional[str], typer.Option(help="Model that reads results and writes findings")] = None,
    lookback_years: Annotated[int, typer.Option(help="Only events this recent")] = 5,
    as_of: Annotated[Optional[str], typer.Option(help="Screen date YYYY-MM-DD (default today)")] = None,
    search_only: Annotated[bool, typer.Option(help="Plan and search, no reader")] = False,
    out: Annotated[Path, typer.Option(help="Parent folder for run output")] = Path("runs"),
) -> None:
    """Screen a company live and write a cited report plus a replayable trace."""
    planner = planner or _default("planner", FIXED)
    reader = reader or _default("reader", "openai/gpt-oss-120b")
    _require_tavily()
    if not search_only or planner != FIXED:
        _require_nebius()
    company = Company(name=name, aliases=alias or [], country=country, website=website,
                      industry=industry, notes=notes)
    run_dir = out / f"{_stamp()}-{ev.slug(name)}"
    init_tracing(run_dir / "trace.jsonl")
    try:
        result = run_screen(
            company, date.fromisoformat(as_of) if as_of else date.today(), lookback_years,
            factory.make_search_backend(), None if search_only else factory.make_model(reader, "reader"),
            planner=make_planner(planner, factory.make_model), session_id=run_dir.name)
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
    reader = reader or _default("reader", "openai/gpt-oss-120b")
    _require_nebius()
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
    answer_key: Annotated[Optional[Path], typer.Option(help="Known events, matched by company name")] = DATASET,
) -> None:
    """Reader eval on past screens: plan and search results held constant, readers swapped, rubric-judged."""
    _require_nebius()
    cfg = _config()
    judge_id = judge or cfg["eval"]["judge"]
    reader_ids = _ids(readers) or cfg["eval"]["readers"]
    if judge_id in reader_ids:
        console.print("[yellow]The judge is also a reader; it will grade its own output.[/yellow]")
    names = {load_recording(t).company.name for t in traces}
    what = ev.slug(names.pop()) if len(names) == 1 else f"{len(traces)}-screens"
    out_dir = Path("results") / f"{_stamp()}-compare-{what}"
    ev.compare_screens(traces, reader_ids, model_factory=factory.make_model,
                       judge=factory.make_model(judge_id, "judge"), prices=cfg.get("prices", {}), out_dir=out_dir,
                       answer_key=answer_key, rubric=load_rubric(RUBRIC), pairwise=pairwise, log=console.print)
    console.print(Markdown((out_dir / "summary.md").read_text(encoding="utf-8")))
    console.print(f"\nResults: {out_dir}\nDashboard: uv run streamlit run app.py (Reader eval tab)")


@app.command()
def models() -> None:
    """List the models available to your NEBIUS_API_KEY (or NEBIUS_BASE_URL endpoint)."""
    _require_nebius()
    from openai import OpenAI

    client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL), api_key=os.environ["NEBIUS_API_KEY"])
    priced = _config().get("prices", {})
    for m in sorted(client.models.list(), key=lambda m: m.id):
        console.print(f"{m.id}{'  (priced in config)' if m.id in priced else ''}")


def _record_missing(dataset: ev.Dataset, planners: list[str], case_ids: list[str] | None, allowed: bool) -> None:
    missing = ev.missing_recordings(dataset.select(case_ids), planners, RECORDINGS)
    if not missing:
        return
    credits = ev.estimate_credits(missing)
    if not allowed:
        console.print(f"[red]{len(missing)} recordings are missing[/red] (up to {credits} Tavily credits to "
                      "record). Rerun with --record-missing, or use `kyb eval record`.")
        raise typer.Exit(1)
    _require_tavily()
    console.print(f"Recording {len(missing)} missing (planner, company) pairs, up to {credits} Tavily credits.")
    for planner in sorted({p for p, _ in missing}):
        ids = [c for p, c in missing if p == planner]
        ev.record(dataset, planner, RECORDINGS, factory.make_search_backend, factory.make_model,
                  cases=dataset.select(ids), log=console.print)


def _run(kind: str, arms: list[ev.Arm], dataset_path: Path, judge: str | None, case_ids: list[str] | None,
         rubric: bool, pairwise: bool) -> None:
    cfg = _config()
    judge_id = judge or cfg["eval"]["judge"]
    if any(judge_id in (a.planner, a.reader) for a in arms):
        console.print("[yellow]The judge is also one of the arms; it will grade its own output.[/yellow]")
    out_dir = Path("results") / f"{_stamp()}-{kind}"
    ev.run_experiment(ev.load_dataset(dataset_path), kind, arms, model_factory=factory.make_model,
                      judge=factory.make_model(judge_id, "judge"), prices=cfg.get("prices", {}),
                      recordings_root=RECORDINGS, out_dir=out_dir,
                      rubric=load_rubric(RUBRIC) if rubric else None, pairwise=pairwise,
                      case_ids=case_ids, log=console.print)
    console.print(Markdown((out_dir / "summary.md").read_text(encoding="utf-8")))
    console.print(f"\nResults: {out_dir}")




@eval_app.command("record")
def eval_record(
    planner: Annotated[str, typer.Option(help="'fixed' or a model ID")] = FIXED,
    dataset: DatasetOpt = DATASET, cases: CasesOpt = None,
    force: Annotated[bool, typer.Option(help="Re-record cases that already have a recording")] = False,
) -> None:
    """Plan and search live, once per company. Experiments then replay these recordings."""
    _require_tavily()
    if planner != FIXED:
        _require_nebius()
    data = ev.load_dataset(dataset)
    ev.record(data, planner, RECORDINGS, factory.make_search_backend, factory.make_model,
              cases=data.select(_ids(cases)), force=force, log=console.print)


@eval_app.command("readers")
def eval_readers(
    planner: Annotated[Optional[str], typer.Option(help="Planner whose recordings every reader gets "
                                                   "(default: config sweep_planner)")] = None,
    readers: Annotated[Optional[str], typer.Option(help="Comma-separated reader IDs (default: config)")] = None,
    dataset: DatasetOpt = DATASET, cases: CasesOpt = None, judge: JudgeOpt = None,
    rubric: bool = True, pairwise: bool = True, record_missing: RecordOpt = False,
) -> None:
    """Eval 2: swap the reader. The planner and its recorded search results stay fixed."""
    _require_nebius()
    ids = _ids(readers) or _config()["eval"]["readers"]
    planner = planner or _config()["eval"].get("sweep_planner", FIXED)
    _record_missing(ev.load_dataset(dataset), [planner], _ids(cases), record_missing)
    _run("readers", ev.readers_experiment(planner, ids), dataset, judge, _ids(cases), rubric, pairwise)


@eval_app.command("search")
def eval_search(
    planners: Annotated[Optional[str], typer.Option(help="Comma-separated: 'fixed' and/or model IDs "
                                                    "(default: fixed + config planners)")] = None,
    dataset: DatasetOpt = DATASET, cases: CasesOpt = None, judge: JudgeOpt = None,
    record_missing: RecordOpt = False,
) -> None:
    """Eval 1: did search find the known events in the companies file? No reader runs."""
    _require_nebius()
    cfg = _config()
    ids = _ids(planners) or [FIXED] + cfg["eval"]["planners"]
    data = ev.load_dataset(dataset)
    _record_missing(data, ids, _ids(cases), record_missing)
    out_dir = Path("results") / f"{_stamp()}-search"
    ev.search_eval(data, ids, judge=factory.make_model(judge or cfg["eval"]["judge"], "judge"),
                   recordings_root=RECORDINGS, out_dir=out_dir, case_ids=_ids(cases), log=console.print)
    console.print(Markdown((out_dir / "summary.md").read_text(encoding="utf-8")))
    console.print(f"\nResults: {out_dir}")


@eval_app.command("agreement")
def eval_agreement(
    judgments: Annotated[Path, typer.Argument(help="judgments.csv with human_* columns filled (y/n)")],
) -> None:
    """Report how often your labels agree with the judge's."""
    for key, (agree, n) in ev.agreement(judgments).items():
        pct = f"{100 * agree / n:.0f}%" if n else "n/a"
        console.print(f"{key}: {agree}/{n} agree ({pct})")


if __name__ == "__main__":
    app()
