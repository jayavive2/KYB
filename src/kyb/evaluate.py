"""The reader eval.

Past screens are the cases: each screen's trace already holds the company, the plan and every
Tavily response. Those are held constant, each reader replays them, and a judge scores every
report (rubric, a support check on every finding, optional pairwise). Model-call statistics come
from the run's own trace (see traces.py).
"""

from __future__ import annotations

import contextvars
import csv
import json
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import pandas as pd

from . import judge as J
from .llm import ChatModel, describe_error
from .planner import FIXED
from .report import render_markdown, to_json
from .schema import Company
from .screen import ScreenResult, run_screen
from .search import Recording, ReplayBackend, load_recording
from .tracing import init_tracing, shutdown_tracing

ModelFactory = Callable[[str, str], ChatModel]  # (model id, role) -> model

AS_RECORDED = "as recorded"  # planner label when the chosen screens were made by different planners


def load_config(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def cost_usd(prices: dict, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """From config/models.toml; for a model not priced there, LiteLLM's own price list."""
    p = prices.get(model)
    if p:
        return prompt_tokens / 1e6 * p["input"] + completion_tokens / 1e6 * p["output"]
    if not model or model == FIXED or "/" not in model:
        return None
    try:
        import litellm

        litellm.suppress_debug_info = True
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    except Exception:  # not in LiteLLM's list either
        return None
    return prompt_cost + completion_cost


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in text)


# ---------------------------------------------------------------- cases: past screens

@dataclass
class Case:
    id: str
    company: Company
    recording: Recording


def load_screens(traces: list[Path]) -> list[Case]:
    """One case per screen trace; IDs are the company name, numbered if a company repeats."""
    cases: list[Case] = []
    seen: set[str] = set()
    for trace in traces:
        rec = load_recording(trace)
        cid, n = slug(rec.company.name), 2
        while cid in seen:
            cid, n = f"{slug(rec.company.name)}-{n}", n + 1
        seen.add(cid)
        cases.append(Case(cid, rec.company, rec))
    return cases


# ---------------------------------------------------------------- the eval

@dataclass
class ArmTotals:
    """One reader's totals across the screens."""
    label: str
    planner: str
    reader: str
    cases: int = 0
    sources: int = 0
    findings: int = 0
    events: int = 0
    judged: int = 0
    supported_on_target: int = 0
    namesake_errors: int = 0
    unsupported: int = 0
    invalid_citations: int = 0
    categories_called: int = 0
    categories_first_try: int = 0
    categories_failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    errors: int = 0
    problems: list[str] = field(default_factory=list)  # why a run, a category or a judge call failed
    # judge
    rubric_sum: dict[str, int] = field(default_factory=dict)
    rubric_n: dict[str, int] = field(default_factory=dict)
    quality_sum: float = 0.0
    quality_n: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0


def _pct(n: float, d: float) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def _money(v: float | None, n: int) -> str:
    return "n/a" if v is None or not n else f"{v / n:.4f}"


def compare_screens(traces: list[Path], readers: list[str], *, model_factory: ModelFactory, judge: ChatModel,
                    prices: dict, out_dir: Path, rubric: J.Rubric | None = None, pairwise: bool = False,
                    log=print, parallel: int = 5) -> list[ArmTotals]:
    """Replay each screen's plan and search results through every reader (the screen's own reader
    included, so all reports are produced and judged the same way) and judge the reports.

    Up to `parallel` readers run at once on each screen, each with its own judging; judge calls are
    capped separately (judge.JUDGE_CONCURRENCY). `log` is only called from this thread, so it can
    write to a UI. Results and CSV rows come out in reader order whatever finishes first."""
    cases = load_screens(traces)
    planners = {c.recording.planner for c in cases}
    planner = planners.pop() if len(planners) == 1 else AS_RECORDED

    out_dir.mkdir(parents=True, exist_ok=True)
    J.JUDGE_FAILURES.clear()
    session = out_dir.name
    models = {r: model_factory(r, "reader") for r in readers}
    totals = {r: ArmTotals(r, planner, r) for r in readers}
    rows: dict[str, list[dict]] = {k: [] for k in ("judgments", "rubric", "pairwise")}

    def run_arm(reader: str, case: Case) -> tuple[ScreenResult | None, dict[str, list[dict]], str | None]:
        """One reader on one screen, judged. Runs in a worker thread; touches only its own totals."""
        arm_rows: dict[str, list[dict]] = {"judgments": [], "rubric": []}
        t, rec = totals[reader], case.recording
        try:
            result = run_screen(case.company, rec.as_of, rec.lookback_years, ReplayBackend(rec),
                                models[reader], recorded_plan=rec.plan, session_id=session)
        except Exception as exc:  # one broken reader shouldn't sink the comparison
            t.problems.append(f"{case.id}: run failed: {describe_error(exc, reader)}")
            return None, arm_rows, describe_error(exc, reader)
        failed = [(k, r) for k, r in result.categories.items() if not r.parsed]
        if failed:
            first = failed[0][1].errors[-1] if failed[0][1].errors else "no output"
            t.problems.append(f"{case.id}: {len(failed)} of {len(result.categories)} categories not assessed "
                              f"(first: {first})")
        sink: list[str] = []
        J.FAILURE_SINK.set(sink)
        _score(t, case, result, judge, rubric, arm_rows, out_dir, prices)
        t.problems += [f"{case.id}: judge {f}" for f in sink]
        return result, arm_rows, None

    workers = max(1, min(parallel, len(readers)))
    init_tracing(out_dir / "trace.jsonl")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for case in cases:
                log(f"{case.id}: {len(readers)} readers, {workers} at a time")
                pending = {pool.submit(contextvars.copy_context().run, run_arm, r, case): r for r in readers}
                outcomes = {}
                for future in as_completed(pending):
                    reader = pending[future]
                    result, arm_rows, error = outcomes[reader] = future.result()
                    if error:
                        totals[reader].errors += 1
                        log(f"  {reader} failed: {error}")
                    else:
                        log(f"  {reader} done: {len(result.findings)} findings")
                results: dict[str, ScreenResult] = {}
                for reader in readers:                       # deterministic order in every CSV
                    result, arm_rows, _ = outcomes[reader]
                    for name, data in arm_rows.items():
                        rows[name].extend(data)
                    if result is not None:
                        results[reader] = result

                if pairwise:
                    pairs = list(combinations([r for r in readers if r in results], 2))
                    log(f"{case.id}: comparing {len(pairs)} pairs of reports")
                    verdicts = list(pool.map(
                        lambda ab, company=case.company, res=results: contextvars.copy_context().run(
                            J.judge_pairwise, judge, company, res[ab[0]], res[ab[1]]), pairs))
                    for (a, b), o in zip(pairs, verdicts, strict=True):
                        ta, tb = totals[a], totals[b]
                        if o.winner == "first":
                            ta.wins, tb.losses = ta.wins + 1, tb.losses + 1
                        elif o.winner == "second":
                            tb.wins, ta.losses = tb.wins + 1, ta.losses + 1
                        else:
                            ta.ties, tb.ties = ta.ties + 1, tb.ties + 1
                        rows["pairwise"].append({
                            "case": case.id, "first": a, "second": b,
                            "winner": {"first": a, "second": b}.get(o.winner, "tie"),
                            "consistent_across_order": o.consistent,
                            "verdict_first_as_A": o.verdict_1, "reason_first_as_A": o.reason_1,
                            "verdict_second_as_A": o.verdict_2, "reason_second_as_A": o.reason_2,
                            "human_winner": ""})
    finally:
        shutdown_tracing()

    results_list = list(totals.values())
    for name, data in rows.items():
        _write_csv(out_dir / f"{name}.csv", data)
    rubric_keys = rubric.keys if rubric else []
    from . import traces as T

    call_stats = T.llm_stats(T.llm_calls(T.load_spans(out_dir / "trace.jsonl")))
    _write_csv(out_dir / "model_calls.csv", call_stats.to_dict("records"))
    (out_dir / "summary.md").write_text(
        summary_markdown(results_list, prices, judge.model, len(cases), rubric_keys, rows["pairwise"])
        + model_calls_markdown(call_stats), encoding="utf-8")
    _write_csv(out_dir / "summary.csv", [summary_row(t, prices, rubric_keys) for t in results_list])
    (out_dir / "metrics.json").write_text(json.dumps([metrics(t, prices, rubric_keys) for t in results_list], indent=2))
    (out_dir / "experiment.json").write_text(json.dumps({
        "kind": "readers", "created": datetime.now().isoformat(timespec="seconds"),
        "arms": [{"planner": planner, "reader": r, "label": r} for r in readers],
        "cases": [c.id for c in cases], "judge": judge.model, "rubric": rubric_keys,
        "pairwise": pairwise, "judge_failures": J.JUDGE_FAILURES, "source": "screens",
        "companies": {c.id: c.company.name for c in cases},
        "screens": {c.id: str(c.recording.path) for c in cases},
    }, indent=2))
    return results_list


def compare_screen(trace: Path, readers: list[str], **kwargs) -> list[ArmTotals]:
    """Swap the reader on a single screen."""
    return compare_screens([trace], readers, **kwargs)


def _score(t: ArmTotals, case: Case, result: ScreenResult, judge: ChatModel, rubric: J.Rubric | None,
           rows: dict, out_dir: Path, prices: dict) -> None:
    from .events import build_events

    case_out = out_dir / "cases" / case.id
    case_out.mkdir(parents=True, exist_ok=True)
    name = slug(t.label)
    (case_out / f"{name}.md").write_text(render_markdown(result), encoding="utf-8")
    (case_out / f"{name}.json").write_text(json.dumps(to_json(result), indent=2))

    t.cases += 1
    t.sources += len(result.evidence)
    t.findings += len(result.findings)
    t.events += len(build_events(result)[0])
    t.prompt_tokens += result.prompt_tokens
    t.completion_tokens += result.completion_tokens
    t.wall_time_s += result.wall_time_s
    for run in result.categories.values():
        t.invalid_citations += run.invalid_citations
        if run.attempts or run.errors:  # errors without attempts = the API call failed
            t.categories_called += 1
            t.categories_first_try += run.first_try
            t.categories_failed += not run.parsed

    judgments = J.judge_support(judge, case.company, result)
    for f in result.findings:
        j = judgments.get(f.id)
        if j is not None:
            t.judged += 1
            t.supported_on_target += j.about_target and j.supported
            t.namesake_errors += not j.about_target
            t.unsupported += j.about_target and not j.supported
        rows["judgments"].append({
            "arm": t.label, "case": case.id, "finding_id": f.id, "category": f.category,
            "severity": f.finding.severity, "event_date": f.finding.event_date or "",
            "summary": f.finding.summary, "source_ids": " ".join(f.finding.source_ids),
            "source_urls": " ".join(result.evidence.get(s).url for s in f.finding.source_ids),
            "judge_about_target": "" if j is None else j.about_target,
            "judge_supported": "" if j is None else j.supported,
            "judge_reason": "" if j is None else j.reason,
            "human_about_target": "", "human_supported": ""})

    if rubric:
        scores = J.judge_rubric(judge, case.company, result, rubric)
        quality = rubric.overall({k: v.score for k, v in scores.items()})
        cost = cost_usd(prices, t.reader, result.prompt_tokens, result.completion_tokens)
        row = {"arm": t.label, "case": case.id, "findings": len(result.findings),
               "quality": "" if quality is None else round(quality, 3),
               "reader_cost_usd": "" if cost is None else round(cost, 6),
               "reader_latency_s": round(result.wall_time_s, 2)}
        for key in rubric.keys:
            sc = scores.get(key)
            if sc is not None:
                t.rubric_sum[key] = t.rubric_sum.get(key, 0) + sc.score
                t.rubric_n[key] = t.rubric_n.get(key, 0) + 1
            row[key] = "" if sc is None else sc.score
            row[f"{key}_reason"] = "" if sc is None else sc.reason
        if quality is not None:
            t.quality_sum += quality
            t.quality_n += 1
        rows["rubric"].append(row)


# ---------------------------------------------------------------- summaries

def _ratio(n: float, d: float) -> float | None:
    return n / d if d else None


def metrics(t: ArmTotals, prices: dict, rubric_keys: list[str]) -> dict:
    """Raw numbers per arm for the dashboard (summary.csv holds the same, formatted)."""
    cost = cost_usd(prices, t.reader, t.prompt_tokens, t.completion_tokens)
    price = prices.get(t.reader, {})
    return {
        "arm": t.label, "planner": t.planner, "reader": t.reader, "cases": t.cases,
        "price_input_per_m": price.get("input"), "price_output_per_m": price.get("output"),
        # quality: rubric judge
        "quality": _ratio(t.quality_sum, t.quality_n),
        "rubric": {k: _ratio(t.rubric_sum.get(k, 0), t.rubric_n.get(k, 0)) for k in rubric_keys},
        "pairwise": {"wins": t.wins, "ties": t.ties, "losses": t.losses},
        # quality: the support check on every finding
        "supported_on_target": _ratio(t.supported_on_target, t.judged),
        "namesake_errors": t.namesake_errors,
        # deterministic checks
        "invalid_citations": t.invalid_citations,
        "json_first_try": _ratio(t.categories_first_try, t.categories_called),
        "categories_failed": t.categories_failed,
        # efficiency
        "cost_usd_per_company": None if cost is None or not t.cases else cost / t.cases,
        "latency_s_per_company": _ratio(t.wall_time_s, t.cases),
        "input_tokens": t.prompt_tokens, "output_tokens": t.completion_tokens, "findings": t.findings,
        "events": t.events, "run_errors": t.errors, "problems": t.problems,
    }


def summary_row(t: ArmTotals, prices: dict, rubric_keys: list[str]) -> dict:
    reader_cost = cost_usd(prices, t.reader, t.prompt_tokens, t.completion_tokens)
    row = {
        "arm": t.label, "planner": t.planner, "reader": t.reader, "cases_completed": t.cases,
        "sources_per_company": f"{t.sources / t.cases:.1f}" if t.cases else "n/a",
        "findings": t.findings, "events": t.events,
        "supported_on_target": _pct(t.supported_on_target, t.judged),
        "namesake_errors": t.namesake_errors, "unsupported": t.unsupported,
        "unjudged": t.findings - t.judged, "invalid_citations": t.invalid_citations,
        "json_first_try": _pct(t.categories_first_try, t.categories_called),
        "categories_failed": t.categories_failed,
    }
    if rubric_keys:
        row["quality"] = f"{t.quality_sum / t.quality_n:.2f}" if t.quality_n else "n/a"
    for k in rubric_keys:
        n = t.rubric_n.get(k, 0)
        row[f"rubric_{k}"] = f"{t.rubric_sum[k] / n:.2f}" if n else "n/a"
    row.update({
        "pairwise_w_t_l": f"{t.wins}-{t.ties}-{t.losses}",
        "reader_usd_per_company": _money(reader_cost, t.cases),
        "reader_sec_per_company": f"{t.wall_time_s / t.cases:.1f}" if t.cases else "n/a",
        "input_tokens": t.prompt_tokens, "output_tokens": t.completion_tokens, "run_errors": t.errors,
    })
    return row


def _table(rows: list[dict], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return out


def summary_markdown(results: list[ArmTotals], prices: dict, judge_model: str, n_cases: int,
                     rubric_keys: list[str], pairwise_rows: list[dict]) -> str:
    rows = [summary_row(t, prices, rubric_keys) for t in results]
    lines = [
        "# Reader comparison",
        "",
        f"{n_cases} screens · judge `{judge_model}` · planner `{results[0].planner}` and its recorded "
        "searches are fixed; the reader varies.",
        "",
    ]
    if J.JUDGE_FAILURES:
        lines += [f"**Warning:** {len(J.JUDGE_FAILURES)} judge calls returned no usable verdict; affected "
                  "findings count as unjudged.", ""]
    lines += ["## Findings", ""] + _table(rows, [
        "arm", "findings", "events", "supported_on_target", "namesake_errors", "unsupported",
        "invalid_citations", "json_first_try", "categories_failed", "reader_usd_per_company",
        "reader_sec_per_company"])
    troubled = [t for t in results if t.problems]
    if troubled:
        lines += ["", "## Problems", ""]
        lines += [f"- **{t.label}**: {p}" for t in troubled for p in t.problems[:5]]
    judge_cols = (["arm"] + (["quality"] if rubric_keys else []) + [f"rubric_{k}" for k in rubric_keys]
                  + (["pairwise_w_t_l"] if pairwise_rows else []))
    if len(judge_cols) > 1:
        lines += ["", "## Judge", ""] + _table(rows, judge_cols)
    if pairwise_rows:
        consistent = sum(1 for r in pairwise_rows if r["consistent_across_order"])
        lines += ["", f"Pairwise verdicts that held when the order was swapped: "
                      f"{consistent} of {len(pairwise_rows)}. A win counts only if it held; otherwise it's a tie."]
    lines += [
        "",
        "- events: distinct events after the reader merged duplicates",
        "- supported_on_target: findings about the right company whose cited sources back them",
        "- quality: weighted mean of the rubric scores (1-5), weights in config/rubric.yaml",
        "- rubric_*: judge's 1-5 score per criterion, averaged over the reports it applied to",
        "- pairwise_w_t_l: wins-ties-losses against the other readers, screen by screen",
        "",
        "Every judge verdict is in the CSVs with empty human_* columns: label a sample and run "
        "`kyb agreement judgments.csv`.",
    ]
    return "\n".join(lines) + "\n"


def model_calls_markdown(stats: pd.DataFrame) -> str:
    """Trace analysis section: every model call in the run, by role and model."""
    if stats.empty:
        return ""
    lines = ["", "## Model calls (from the trace)", "",
             "| role | model | calls | errors | JSON repairs | unusable | p50 s | p95 s | prompt tok | output tok "
             "| output tok/s |", "|" + "---|" * 11]
    for r in stats.to_dict("records"):
        def f(v, fmt):
            return "n/a" if v is None or pd.isna(v) else format(v, fmt)
        lines.append(f"| {r['role']} | {r['model']} | {r['calls']} | {r['errors']} | {r['json_repairs']} | "
                     f"{r['unusable_output']} | {f(r['p50_latency_s'], '.2f')} | {f(r['p95_latency_s'], '.2f')} | "
                     f"{f(r['mean_prompt_tokens'], ',.0f')} | {f(r['mean_completion_tokens'], ',.0f')} | "
                     f"{f(r['output_tokens_per_s'], '.0f')} |")
    lines += ["", "JSON repairs: calls that were a second attempt after unusable output. Unusable: calls whose "
              "output still failed the schema. Latency covers successful calls."]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def agreement(judgments_csv: Path) -> dict[str, tuple[int, int]]:
    """Agreement between judge and human labels where a human label is filled (y/n)."""
    def norm(v: str) -> bool | None:
        v = (v or "").strip().lower()
        if v in ("y", "yes", "true", "1"):
            return True
        if v in ("n", "no", "false", "0"):
            return False
        return None

    stats = {"about_target": [0, 0], "supported": [0, 0]}
    with judgments_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key in stats:
                human, judged = norm(row.get(f"human_{key}", "")), norm(row.get(f"judge_{key}", ""))
                if human is None or judged is None:
                    continue
                stats[key][1] += 1
                stats[key][0] += human == judged
    return {k: (v[0], v[1]) for k, v in stats.items()}
