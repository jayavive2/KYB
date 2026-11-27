"""The two evals.

1. Search eval (`search_eval`): did Tavily, driven by a planner, find the known events listed in
   evals/companies.yaml? Judged per recording; no reader involved.
2. Reader eval (`run_experiment`, `compare_screens`): the plan and search results are recorded
   once and held constant; each reader replays them and a judge scores the reports (rubric,
   per-finding checks, optional pairwise). Works on past screens or on the companies file.

Both read their traces for model-call and search statistics (see traces.py).
"""

from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from . import judge as J
from .llm import ChatModel, describe_error
from .plan import CATEGORIES, CREDITS_PER_SEARCH
from .planner import FIXED, MAX_TOTAL, make_planner
from .report import render_markdown, to_json
from .schema import Company
from .screen import ScreenResult, run_screen
from .search import Recording, ReplayBackend, load_recording
from .tracing import init_tracing, shutdown_tracing

ModelFactory = Callable[[str, str], ChatModel]  # (model id, role) -> model

# ---------------------------------------------------------------- dataset and config

@dataclass
class ExpectedEvent:
    id: str
    category: str
    description: str
    date: str | None = None
    reference: str | None = None


@dataclass
class EvalCase:
    id: str
    company: Company
    expected_events: list[ExpectedEvent] = field(default_factory=list)
    notes: str | None = None


@dataclass
class Dataset:
    lookback_years: int
    as_of: date | None
    cases: list[EvalCase]

    def select(self, ids: list[str] | None) -> list[EvalCase]:
        return [c for c in self.cases if not ids or c.id in ids]


def load_dataset(path: Path) -> Dataset:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = [
        EvalCase(
            id=c["id"],
            company=Company.model_validate(c["company"]),
            expected_events=[ExpectedEvent(**e) for e in c.get("expected_events") or []],
            notes=c.get("notes"),
        )
        for c in raw["cases"]
    ]
    as_of = raw.get("as_of")
    return Dataset(int(raw.get("lookback_years", 5)), date.fromisoformat(str(as_of)) if as_of else None, cases)


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


# ---------------------------------------------------------------- recordings

def recording_path(root: Path, planner: str, case_id: str) -> Path:
    return root / slug(planner) / case_id / "trace.jsonl"


def missing_recordings(cases: list[EvalCase], planners: list[str], root: Path) -> list[tuple[str, str]]:
    return [(p, c.id) for p in planners for c in cases if not recording_path(root, p, c.id).exists()]


def estimate_credits(missing: list[tuple[str, str]]) -> int:
    """Upper bound on Tavily credits to record the missing (planner, case) pairs."""
    fixed_searches = sum(len(c.queries) for c in CATEGORIES)
    return sum((fixed_searches if p == FIXED else MAX_TOTAL) * CREDITS_PER_SEARCH for p, _ in missing)


def record_one(planner: str, company: Company, as_of: date, lookback_years: int, path: Path,
               backend_factory, model_factory: ModelFactory, log=print, label: str = "") -> None:
    """Plan and search live (no reader); the trace file at `path` is the recording."""
    path.unlink(missing_ok=True)
    init_tracing(path)
    try:
        result = run_screen(company, as_of, lookback_years, backend_factory(), None,
                            planner=make_planner(planner, model_factory),
                            session_id=f"record-{slug(planner)}-{label or slug(company.name)}")
    finally:
        shutdown_tracing()
    (path.parent / "evidence.md").write_text(render_markdown(result), encoding="utf-8")
    note = f", {len(result.plan.fell_back)} categories from the fixed plan" if result.plan.fell_back else ""
    log(f"recorded {planner} / {label or company.name}: {len(result.searches)} searches, "
        f"{len(result.evidence)} sources{note}")


def record(dataset: Dataset, planner: str, root: Path, backend_factory, model_factory: ModelFactory,
           cases: list[EvalCase] | None = None, force: bool = False, log=print) -> None:
    """Record a planner's searches for each dataset case."""
    as_of = dataset.as_of or date.today()
    for case in cases or dataset.cases:
        path = recording_path(root, planner, case.id)
        if path.exists() and not force:
            log(f"skip {planner} / {case.id}: already recorded")
            continue
        record_one(planner, case.company, as_of, dataset.lookback_years, path, backend_factory, model_factory,
                   log, label=case.id)


def _evidence(rec: Recording, session: str) -> ScreenResult:
    """Rebuild a recording's evidence (replay, no reader): same source IDs as every replay."""
    return run_screen(rec.company, rec.as_of, rec.lookback_years, ReplayBackend(rec), None,
                      recorded_plan=rec.plan, session_id=session)


def _coverage(judge: ChatModel, case: EvalCase, rec: Recording, session: str,
              baseline: ScreenResult | None = None) -> dict[str, list[str]]:
    """Which known events are in this recording's evidence. Cached next to the recording."""
    if not case.expected_events:  # no answer key: nothing to look for
        return {}
    key = hashlib.sha256(json.dumps(
        [rec.responses, [[e.id, e.description] for e in case.expected_events]],
        sort_keys=True).encode()).hexdigest()[:12]
    cache = rec.path.parent / f"coverage.{slug(judge.model)}.{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    baseline = baseline or _evidence(rec, session)
    coverage = J.judge_coverage(judge, case.company, case.expected_events, baseline.evidence)
    cache.write_text(json.dumps(coverage, indent=2))
    return coverage


# ---------------------------------------------------------------- experiment

@dataclass(frozen=True)
class Arm:
    planner: str
    reader: str

    def label(self, kind: str) -> str:
        return {"readers": self.reader, "planners": self.planner}.get(kind, f"{self.planner} -> {self.reader}")


def readers_experiment(planner: str, readers: list[str]) -> list[Arm]:
    return [Arm(planner, r) for r in readers]


@dataclass
class ArmTotals:
    label: str
    planner: str
    reader: str
    cases: int = 0
    # search side (from the planner's recordings)
    searches: int = 0
    sources: int = 0
    credits: float = 0.0
    events_total: int = 0
    events_in_evidence: int = 0
    planner_prompt_tokens: int = 0
    planner_completion_tokens: int = 0
    # reader side
    events_matched: int = 0
    events_matched_in_evidence: int = 0
    findings: int = 0
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


def run_experiment(dataset: Dataset, kind: str, arms: list[Arm], *, model_factory: ModelFactory,
                   judge: ChatModel, prices: dict, recordings_root: Path, out_dir: Path,
                   rubric: J.Rubric | None = None, pairwise: bool = True,
                   case_ids: list[str] | None = None, log=print,
                   locate: Callable[[str, str], Path] | None = None, source: str = "dataset") -> list[ArmTotals]:
    """`locate(planner, case_id)` gives each recording's path (default: under `recordings_root`)."""
    cases = dataset.select(case_ids)
    locate = locate or (lambda planner, case_id: recording_path(recordings_root, planner, case_id))
    missing = [(a, c.id) for a in sorted({a.planner for a in arms}) for c in cases if not locate(a, c.id).exists()]
    if missing:
        raise FileNotFoundError(f"Record these first (planner, case): {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    J.JUDGE_FAILURES.clear()
    session = out_dir.name
    readers = {a.reader: model_factory(a.reader, "reader") for a in arms}
    totals = {a: ArmTotals(a.label(kind), a.planner, a.reader) for a in arms}
    rows: dict[str, list[dict]] = {k: [] for k in ("judgments", "recall", "coverage", "rubric", "pairwise")}

    init_tracing(out_dir / "trace.jsonl")
    try:
        for case in cases:
            recordings = {p: load_recording(locate(p, case.id))
                          for p in {a.planner for a in arms}}
            coverage = {p: _coverage(judge, case, rec, session) for p, rec in recordings.items()}
            for p, cov in coverage.items():
                for e in case.expected_events:
                    rows["coverage"].append({"planner": p, "case": case.id, "event_id": e.id,
                                             "description": e.description,
                                             "source_ids": " ".join(cov.get(e.id, [])),
                                             "in_evidence": bool(cov.get(e.id)), "human_agree": ""})

            results: dict[Arm, ScreenResult] = {}
            for arm in arms:
                t, rec = totals[arm], recordings[arm.planner]
                log(f"{case.id} · {arm.label(kind)}")
                try:
                    result = run_screen(case.company, rec.as_of, rec.lookback_years, ReplayBackend(rec),
                                        readers[arm.reader], recorded_plan=rec.plan, session_id=session)
                except Exception as exc:  # one broken arm shouldn't sink the comparison
                    log(f"  failed: {describe_error(exc, arm.reader)}")
                    t.errors += 1
                    continue
                results[arm] = result
                _score_arm(t, arm, kind, case, rec, coverage[arm.planner], result, judge, rubric, rows,
                           out_dir, prices)

            if pairwise:
                for a, b in combinations([a for a in arms if a in results], 2):
                    log(f"{case.id} · compare {a.label(kind)} vs {b.label(kind)}")
                    o = J.judge_pairwise(judge, case.company, results[a], results[b])
                    ta, tb = totals[a], totals[b]
                    if o.winner == "first":
                        ta.wins, tb.losses = ta.wins + 1, tb.losses + 1
                    elif o.winner == "second":
                        tb.wins, ta.losses = tb.wins + 1, ta.losses + 1
                    else:
                        ta.ties, tb.ties = ta.ties + 1, tb.ties + 1
                    rows["pairwise"].append({
                        "case": case.id, "first": a.label(kind), "second": b.label(kind),
                        "winner": {"first": a.label(kind), "second": b.label(kind)}.get(o.winner, "tie"),
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
        summary_markdown(kind, results_list, prices, judge.model, len(cases), rubric_keys, rows["pairwise"])
        + model_calls_markdown(call_stats), encoding="utf-8")
    _write_csv(out_dir / "summary.csv", [summary_row(t, prices, rubric_keys) for t in results_list])
    (out_dir / "metrics.json").write_text(json.dumps([metrics(t, prices, rubric_keys) for t in results_list], indent=2))
    (out_dir / "experiment.json").write_text(json.dumps({
        "kind": kind, "created": datetime.now().isoformat(timespec="seconds"),
        "arms": [asdict(a) | {"label": a.label(kind)} for a in arms],
        "cases": [c.id for c in cases], "judge": judge.model, "rubric": rubric_keys,
        "pairwise": pairwise, "judge_failures": J.JUDGE_FAILURES, "source": source,
        "companies": {c.id: c.company.name for c in cases},
    }, indent=2))
    return results_list


def _score_arm(t: ArmTotals, arm: Arm, kind: str, case: EvalCase, rec: Recording, coverage: dict,
               result: ScreenResult, judge: ChatModel, rubric: J.Rubric | None, rows: dict,
               out_dir: Path, prices: dict) -> None:
    case_out = out_dir / "cases" / case.id
    case_out.mkdir(parents=True, exist_ok=True)
    name = slug(arm.label(kind))
    (case_out / f"{name}.md").write_text(render_markdown(result), encoding="utf-8")
    (case_out / f"{name}.json").write_text(json.dumps(to_json(result), indent=2))

    t.cases += 1
    t.searches += len(result.searches)
    t.sources += len(result.evidence)
    t.credits += rec.credits
    if rec.plan:
        t.planner_prompt_tokens += rec.plan.prompt_tokens
        t.planner_completion_tokens += rec.plan.completion_tokens
    t.findings += len(result.findings)
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

    events = case.expected_events
    matches = J.judge_recall(judge, events, result)
    t.events_total += len(events)
    for e in events:
        in_evidence = bool(coverage.get(e.id))
        t.events_in_evidence += in_evidence
        t.events_matched += bool(matches.get(e.id))
        t.events_matched_in_evidence += bool(matches.get(e.id)) and in_evidence
        rows["recall"].append({"arm": t.label, "case": case.id, "event_id": e.id, "in_evidence": in_evidence,
                               "matched_finding": matches.get(e.id) or "", "human_agree": ""})

    if rubric:
        scores = J.judge_rubric(judge, case.company, result, rubric)
        quality = rubric.overall({k: v.score for k, v in scores.items()})
        cost = cost_usd(prices, arm.reader, result.prompt_tokens, result.completion_tokens)
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
        # quality: checks against known events and per-finding judgments
        "recall": _ratio(t.events_matched, t.events_total),
        "recall_given_evidence": _ratio(t.events_matched_in_evidence, t.events_in_evidence),
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
        "run_errors": t.errors,
    }


def summary_row(t: ArmTotals, prices: dict, rubric_keys: list[str]) -> dict:
    reader_cost = cost_usd(prices, t.reader, t.prompt_tokens, t.completion_tokens)
    planner_cost = (0.0 if t.planner == FIXED else
                    cost_usd(prices, t.planner, t.planner_prompt_tokens, t.planner_completion_tokens))
    row = {
        "arm": t.label, "planner": t.planner, "reader": t.reader, "cases_completed": t.cases,
        "searches_per_company": f"{t.searches / t.cases:.1f}" if t.cases else "n/a",
        "sources_per_company": f"{t.sources / t.cases:.1f}" if t.cases else "n/a",
        "tavily_credits": f"{t.credits:g}",
        "coverage": _pct(t.events_in_evidence, t.events_total),
        "recall": _pct(t.events_matched, t.events_total),
        "recall_given_evidence": _pct(t.events_matched_in_evidence, t.events_in_evidence),
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
        "planner_usd_per_company": _money(planner_cost, t.cases),
        "reader_usd_per_company": _money(reader_cost, t.cases),
        "reader_sec_per_company": f"{t.wall_time_s / t.cases:.1f}" if t.cases else "n/a",
        "findings": t.findings, "input_tokens": t.prompt_tokens, "output_tokens": t.completion_tokens,
        "run_errors": t.errors,
    })
    return row


def _table(rows: list[dict], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return out


def summary_markdown(kind: str, results: list[ArmTotals], prices: dict, judge_model: str, n_cases: int,
                     rubric_keys: list[str], pairwise_rows: list[dict]) -> str:
    rows = [summary_row(t, prices, rubric_keys) for t in results]
    fixed = {"readers": f"planner `{results[0].planner}` and its recorded searches are fixed; the reader varies",
             "planners": f"reader `{results[0].reader}` is fixed; the planner (and so the searches) varies"}
    lines = [
        f"# {'Reader' if kind == 'readers' else 'Planner' if kind == 'planners' else 'Pipeline'} comparison",
        "",
        f"{n_cases} companies · judge `{judge_model}` · {fixed.get(kind, 'custom arms')}.",
        "",
    ]
    if J.JUDGE_FAILURES:
        lines += [f"**Warning:** {len(J.JUDGE_FAILURES)} judge calls returned no usable verdict; affected "
                  "findings count as unjudged and affected events as not matched.", ""]
    lines += ["## Search", ""] + _table(rows, ["arm", "searches_per_company", "sources_per_company",
                                               "tavily_credits", "coverage", "planner_usd_per_company"])
    lines += ["", "## Findings", ""] + _table(rows, [
        "arm", "recall", "recall_given_evidence", "supported_on_target", "namesake_errors", "unsupported",
        "invalid_citations", "json_first_try", "categories_failed", "reader_usd_per_company",
        "reader_sec_per_company"])
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
        "- coverage: known events present in the searched evidence (the planner's job)",
        "- recall: known events the reader reported",
        "- recall_given_evidence: the same, counting only events that were in the evidence",
        "- supported_on_target: findings about the right company whose cited sources back them",
        "- quality: weighted mean of the rubric scores (1-5), weights in config/rubric.yaml",
        "- rubric_*: judge's 1-5 score per criterion, averaged over the reports it applied to",
        "- pairwise_w_t_l: wins-ties-losses against the other arms, company by company",
        "",
        "Every judge verdict is in the CSVs with empty human_* columns: label a sample and run "
        "`kyb eval agreement judgments.csv`.",
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


# ---------------------------------------------------------------- eval 1: search vs known events

def search_eval(dataset: Dataset, planners: list[str], *, judge: ChatModel, recordings_root: Path, out_dir: Path,
                case_ids: list[str] | None = None, log=print) -> list[dict]:
    """Did search find the known events? For each planner's recording of each company that lists
    known events, the judge marks every event found (and by which sources) or missed. No reader
    runs, so this measures Tavily and the planner alone. Tavily and planner stats come from the
    recordings' traces."""
    from . import traces as T

    cases = [c for c in dataset.select(case_ids) if c.expected_events]
    if not cases:
        raise ValueError("No selected company lists expected_events; nothing to check search against.")
    missing = missing_recordings(cases, planners, recordings_root)
    if missing:
        raise FileNotFoundError(f"Record these first (planner, case): {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    J.JUDGE_FAILURES.clear()
    rows: list[dict] = []
    stats = {p: {"planner": p, "companies": 0, "events": 0, "found": 0, "searches": 0, "sources": 0,
                 "by_category": {}, "tavily": [], "planner_calls": []} for p in planners}
    init_tracing(out_dir / "trace.jsonl")
    try:
        for case in cases:
            for planner in planners:
                path = recording_path(recordings_root, planner, case.id)
                rec = load_recording(path)
                log(f"{case.id} · {planner}")
                baseline = _evidence(rec, out_dir.name)
                coverage = _coverage(judge, case, rec, out_dir.name, baseline)
                spans = T.load_spans(path)
                st = stats[planner]
                st["companies"] += 1
                st["searches"] += len(baseline.searches)
                st["sources"] += len(baseline.evidence)
                st["tavily"].append(spans)
                st["planner_calls"].append(T.llm_calls(spans))
                for e in case.expected_events:
                    sources = [baseline.evidence.get(s) for s in coverage.get(e.id, [])]
                    sources = [s for s in sources if s]
                    found_in = sorted({c for s in sources for c in s.categories})
                    st["events"] += 1
                    st["found"] += bool(sources)
                    cat = st["by_category"].setdefault(e.category, [0, 0])
                    cat[0] += bool(sources)
                    cat[1] += 1
                    rows.append({
                        "planner": planner, "case": case.id, "company": case.company.name, "event_id": e.id,
                        "category": e.category, "date": e.date or "", "description": e.description,
                        "found": bool(sources), "source_ids": " ".join(s.id for s in sources),
                        "source_urls": " ".join(s.url for s in sources),
                        "source_titles": " | ".join(s.title for s in sources),
                        "found_by_queries": " | ".join(dict.fromkeys(q for s in sources for q in s.queries)),
                        "found_in_categories": " ".join(found_in),
                        "found_in_own_category": e.category in found_in,
                        "reference": e.reference or "", "human_agree": ""})
    finally:
        shutdown_tracing()

    metrics = []
    for p, st in stats.items():
        spans = pd.concat(st["tavily"]) if st["tavily"] else pd.DataFrame()
        tav = T.search_stats(spans) if not spans.empty else {}
        calls = pd.concat(st["planner_calls"]) if st["planner_calls"] else pd.DataFrame()
        planner_calls = calls[calls["role"] == "planner"] if not calls.empty else calls
        metrics.append({
            "planner": p, "companies": st["companies"], "events": st["events"], "found": st["found"],
            "coverage": _ratio(st["found"], st["events"]),
            "searches_per_company": _ratio(st["searches"], st["companies"]),
            "sources_per_company": _ratio(st["sources"], st["companies"]),
            "results_per_search": tav.get("results_per_search"), "empty_searches": tav.get("empty_searches"),
            "search_p50_s": tav.get("p50_latency_s"), "search_p95_s": tav.get("p95_latency_s"),
            "tavily_credits": tav.get("credits"),
            "planner_calls": len(planner_calls),
            "planner_p50_s": planner_calls["latency_s"].median() if len(planner_calls) else None,
            "planner_output_tokens": int(planner_calls["completion_tokens"].sum()) if len(planner_calls) else 0,
            "by_category": {k: {"found": v[0], "events": v[1]} for k, v in st["by_category"].items()},
        })

    _write_csv(out_dir / "search_eval.csv", rows)
    (out_dir / "search_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out_dir / "summary.md").write_text(search_summary(metrics, rows, judge.model), encoding="utf-8")
    (out_dir / "experiment.json").write_text(json.dumps({
        "kind": "search", "created": datetime.now().isoformat(timespec="seconds"), "planners": planners,
        "cases": [c.id for c in cases], "companies": {c.id: c.company.name for c in cases},
        "judge": judge.model, "judge_failures": J.JUDGE_FAILURES}, indent=2))
    return metrics


def search_summary(metrics: list[dict], rows: list[dict], judge_model: str) -> str:
    def f(v, fmt):
        return "n/a" if v is None else format(v, fmt)

    lines = ["# Search eval: known events vs Tavily results", "",
             f"{metrics[0]['companies']} companies, {metrics[0]['events']} known events · judge `{judge_model}`. "
             "An event counts as found if any recorded search result describes it. No reader is involved.", ""]
    if J.JUDGE_FAILURES:
        lines += [f"**Warning:** {len(J.JUDGE_FAILURES)} judge calls failed; affected events count as missed.", ""]
    lines += ["| planner | coverage | found | searches / company | results / search | empty searches | "
              "credits | search p50 s | planner p50 s |", "|" + "---|" * 9]
    for m in metrics:
        lines.append(f"| {m['planner']} | {f(m['coverage'], '.0%')} | {m['found']}/{m['events']} | "
                     f"{f(m['searches_per_company'], '.1f')} | {f(m['results_per_search'], '.1f')} | "
                     f"{f(m['empty_searches'], 'g')} | {f(m['tavily_credits'], 'g')} | "
                     f"{f(m['search_p50_s'], '.2f')} | {f(m['planner_p50_s'], '.1f')} |")
    cats = sorted({c for m in metrics for c in m["by_category"]})
    lines += ["", "## Coverage by category (found / known)", "",
              "| category | " + " | ".join(m["planner"] for m in metrics) + " |", "|" + "---|" * (len(metrics) + 1)]
    for c in cats:
        cells = [f"{m['by_category'][c]['found']}/{m['by_category'][c]['events']}" if c in m["by_category"]
                 else "–" for m in metrics]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    missed = [r for r in rows if not r["found"]]
    if missed:
        lines += ["", "## Missed events", ""]
        lines += [f"- **{r['planner']}** · {r['company']} · {r['event_id']}: {r['description']}" for r in missed]
    lines += ["", "Each event's matching sources and the queries that found them are in search_eval.csv; fill "
              "`human_agree` for a sample to check the judge."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- benchmarks from past screens

AS_RECORDED = "as recorded"  # planner label when the chosen screens were made by different planners


@dataclass
class Screens:
    """Past screens as benchmark cases: each trace already holds the company, plan and evidence."""
    cases: list[EvalCase]
    traces: dict[str, Path]          # case id -> the screen's trace
    planners: dict[str, str]         # case id -> planner that made it
    as_of: dict[str, date]
    lookback: dict[str, int]

    @property
    def planner_label(self) -> str:
        names = set(self.planners.values())
        return names.pop() if len(names) == 1 else AS_RECORDED

    def dataset(self) -> Dataset:
        return Dataset(5, None, self.cases)  # per-case dates come from the traces, not from here


def load_screens(traces: list[Path], answer_key: Path | None = None) -> Screens:
    """One case per trace. If `answer_key` (a companies.yaml) lists the same company, its known
    events are attached, which adds recall and coverage; otherwise the rubric alone grades it."""
    known: dict[str, list[ExpectedEvent]] = {}
    if answer_key and answer_key.exists():
        for c in load_dataset(answer_key).cases:
            for name in [c.company.name, *c.company.aliases]:
                known.setdefault(name.casefold(), c.expected_events)
    screens = Screens([], {}, {}, {}, {})
    for trace in traces:
        rec = load_recording(trace)
        cid, n = slug(rec.company.name), 2
        while cid in screens.traces:
            cid, n = f"{slug(rec.company.name)}-{n}", n + 1
        events = known.get(rec.company.name.casefold(), [])
        screens.cases.append(EvalCase(cid, rec.company, list(events)))
        screens.traces[cid], screens.planners[cid] = trace, rec.planner
        screens.as_of[cid], screens.lookback[cid] = rec.as_of, rec.lookback_years
    return screens


def compare_screens(traces: list[Path], readers: list[str], *, model_factory: ModelFactory, judge: ChatModel,
                    prices: dict, out_dir: Path, answer_key: Path | None = None, rubric: J.Rubric | None = None,
                    pairwise: bool = True, log=print) -> list[ArmTotals]:
    """Eval 2 on past screens: each screen's plan and search results held constant, readers swapped.

    Every reader is re-run on the recorded evidence, the screen's own reader included, so all arms
    are produced and judged the same way."""
    screens = load_screens(traces, answer_key)
    return run_experiment(screens.dataset(), "readers", readers_experiment(screens.planner_label, readers),
                          model_factory=model_factory, judge=judge, prices=prices, recordings_root=Path("."),
                          out_dir=out_dir, rubric=rubric, pairwise=pairwise, log=log,
                          locate=lambda planner, cid: screens.traces[cid], source="screens")


def compare_screen(trace: Path, readers: list[str], **kwargs) -> list[ArmTotals]:
    """Swap the reader on a single screen."""
    return compare_screens([trace], readers, **kwargs)
