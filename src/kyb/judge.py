"""LLM judges. One fixed judge model answers three kinds of question:

  support   - is each finding about the target, and backed by its sources? (per finding; also the
              source check on every screen)
  rubric    - 1-5 scores on criteria from config/rubric.yaml           (per report)
  pairwise  - which of two reports would an analyst rather rely on?    (per pair, both orders)

Every verdict is written to CSV so a person can audit a sample.
"""

from __future__ import annotations

import contextvars
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from pydantic import BaseModel, Field

from .events import SourceCheck, build_events
from .evidence import render_sources
from .llm import ChatModel, JsonCallResult, complete_json, describe_error
from .plan import window_start
from .schema import Company, Judgment
from .screen import ScreenResult
from .tracing import get_tracer, set_io

JUDGE_FAILURES: list[str] = []  # reset per experiment; reported in the summary


# Readers run in parallel and each judges its own report, so judge calls are capped here: enough to
# keep the eval fast, few enough that one judge model isn't flooded. KYB_JUDGE_CONCURRENCY overrides.
JUDGE_CONCURRENCY = int(os.getenv("KYB_JUDGE_CONCURRENCY", "8"))
_JUDGE_SLOTS = threading.BoundedSemaphore(JUDGE_CONCURRENCY)

# Where the current reader's judge failures go. Set per reader in the eval; worker threads started
# with contextvars.copy_context() see the same list, so failures land on the right reader.
FAILURE_SINK: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("judge_failure_sink", default=None)


def judge_call(judge: ChatModel, name: str, system: str, user: str, schema) -> JsonCallResult:
    with _JUDGE_SLOTS, get_tracer().start_as_current_span(f"kyb.judge.{name}") as span:  # wait outside the span
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND,
                           OpenInferenceSpanKindValues.EVALUATOR.value)
        try:
            call = complete_json(judge, [{"role": "system", "content": system},
                                         {"role": "user", "content": user}], schema)
        except Exception as exc:  # a judge outage shouldn't discard the candidates' runs
            span.record_exception(exc)
            call = JsonCallResult(value=None, attempts=0, errors=[describe_error(exc, judge.model)])
        if call.value is None:
            failure = f"{name}: {call.errors[-1] if call.errors else 'no output'}"
            JUDGE_FAILURES.append(failure)
            sink = FAILURE_SINK.get()
            if sink is not None:
                sink.append(failure)
        set_io(span, output_value=call.value.model_dump() if call.value else {"errors": call.errors})
        return call


def report_for_judge(result: ScreenResult, excerpt_chars: int = 400) -> str:
    """A report as the judge sees it: each event (duplicates merged, as in the report) with short
    excerpts of its cited sources."""
    failed = [k for k, run in result.categories.items() if not run.parsed]
    note = f"\n\nCategories not assessed (the author's output was unusable): {', '.join(failed)}" if failed else ""
    events, _ = build_events(result)
    if not events:
        return "(no findings)" + note
    blocks = []
    for e in events:
        f = e.lead.finding
        lines = [f"[{e.lead.id}] ({', '.join(e.categories)}; {f.severity}; {f.status}; "
                 f"{f.event_date or 'date n/a'}) {f.summary}"]
        for sid, category in e.citations:
            s = result.evidence.get(sid)
            if s:
                text = s.text(category)
                text = text[:excerpt_chars] + ("…" if len(text) > excerpt_chars else "")
                lines.append(f"    {sid}: {s.title} ({s.published_date or 'date unknown'}) — {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + note


def _context(company: Company, result: ScreenResult) -> str:
    start = window_start(result.as_of, result.lookback_years)
    return (f"Target company:\n{company.identity_block()}\n\n"
            f"Screen date {result.as_of}; events in scope from {start}.")


# ---------------------------------------------------------------- support

class _Judgments(BaseModel):
    judgments: list[Judgment]


SUPPORT_PROMPT = """You audit findings from an adverse-media screen of the target company below.
For each finding, read only its cited sources and decide:
- about_target: is the finding about the target company, and not a different organization
  with a similar name? (Events of a parent or subsidiary count only if the finding names that entity.)
- supported: do the cited sources state what the finding claims, including actor, amounts and
  dates? Paraphrase is fine. Any claim the sources do not contain makes it unsupported.

Target company:
{identity}

Return only JSON:
{{"judgments": [{{"finding_id": "...", "about_target": true, "supported": true, "reason": "short"}}]}}"""


def judge_support(judge: ChatModel, company: Company, result: ScreenResult,
                  batch_size: int = 8, workers: int = 4) -> dict[str, Judgment]:
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    findings = result.findings
    system = SUPPORT_PROMPT.format(identity=company.identity_block())

    def batch(chunk) -> list[Judgment]:
        blocks = []
        for f in chunk:
            cited = [result.evidence.get(sid) for sid in f.finding.source_ids]
            blocks.append(
                f"Finding {f.id} ({f.category}, {f.finding.event_date or 'date n/a'}): {f.finding.summary}\n"
                f"Cited sources:\n{render_sources([c for c in cited if c], f.category)}"
            )
        call = judge_call(judge, "support", system, "\n\n---\n\n".join(blocks), _Judgments)
        known = {f.id for f in chunk}
        return [j for j in call.value.judgments if j.finding_id in known] if call.value else []

    chunks = [findings[i : i + batch_size] for i in range(0, len(findings), batch_size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = [pool.submit(contextvars.copy_context().run, batch, c) for c in chunks]
        return {j.finding_id: j for f in done for j in f.result()}


def check_sources(judge: ChatModel, result: ScreenResult) -> SourceCheck:
    """The support check on a single screen: every finding against the sources it cites.
    Findings it can't confirm are held back from the report, not deleted."""
    before = len(JUDGE_FAILURES)
    with get_tracer().start_as_current_span("kyb.check") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND,
                           OpenInferenceSpanKindValues.EVALUATOR.value)
        check = SourceCheck(judge.model, judge_support(judge, result.company, result))
        check.errors = JUDGE_FAILURES[before:]
        verdicts = [check.confirmed(f.id) for f in result.findings]
        span.set_attribute("kyb.confirmed", verdicts.count(True))
        span.set_attribute("kyb.held_back", verdicts.count(False))
        span.set_attribute("kyb.unchecked", verdicts.count(None))
    return check


# ---------------------------------------------------------------- rubric

@dataclass
class Criterion:
    key: str
    question: str
    weight: float = 1.0
    applies_to_empty: bool = False


@dataclass
class Rubric:
    question: str
    criteria: list[Criterion]
    scale: dict[int, str]

    @property
    def keys(self) -> list[str]:
        return [c.key for c in self.criteria]

    def overall(self, scores: dict[str, int | None]) -> float | None:
        """Weighted mean of the criteria that were scored (weights renormalized)."""
        pairs = [(c.weight, scores[c.key]) for c in self.criteria if scores.get(c.key) is not None]
        total = sum(w for w, _ in pairs)
        return sum(w * v for w, v in pairs) / total if total else None


def load_rubric(path: Path) -> Rubric:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    criteria = [Criterion(c["key"], " ".join(c["question"].split()), float(c.get("weight", 1.0)),
                          bool(c.get("applies_to_empty", False))) for c in raw["criteria"]]
    return Rubric(" ".join(raw.get("question", "").split()), criteria,
                  {int(k): v for k, v in raw.get("scale", {}).items()})


class _Score(BaseModel):
    score: int | None = Field(default=None, ge=1, le=5)  # null = criterion doesn't apply
    reason: str = ""


class _RubricOut(BaseModel):
    scores: dict[str, _Score | None]


RUBRIC_PROMPT = """You grade one KYB adverse-media screening report against a rubric.

Screening question: {question}

{context}

You get the report (each finding with excerpts of its cited sources), then an overview of all the
evidence the report's author was given. Judge citation, relevance, specificity and calibration from
the report and its excerpts; judge scope against the evidence overview.

Score each criterion from 1 to 5, or null if it does not apply:
{criteria}

Scale: {scale}
{empty_note}
Return only JSON: {{"scores": {{{example}}}}}"""


def evidence_overview(result: ScreenResult, chars: int = 300) -> str:
    """Every source the reader saw: ID, categories, date, title and the start of its text."""
    lines = []
    for s in result.evidence.all():
        text = " ".join(s.text().split())
        lines.append(f"{s.id} [{', '.join(s.categories)}] ({s.published_date or 'date unknown'}) "
                     f"{s.title}: {text[:chars]}{'…' if len(text) > chars else ''}")
    return "\n".join(lines) or "(no search results)"


def judge_rubric(judge: ChatModel, company: Company, result: ScreenResult,
                 rubric: Rubric) -> dict[str, _Score]:
    """Scores per criterion; criteria that don't apply (or the judge left null) are omitted."""
    empty = not result.findings
    criteria = [c for c in rubric.criteria if c.applies_to_empty or not empty]
    if not criteria:
        return {}
    scale = "; ".join(f"{k} = {v}" for k, v in sorted(rubric.scale.items(), reverse=True))
    example = ", ".join(f'"{c.key}": {{"score": 4, "reason": "short"}}' for c in criteria)
    system = RUBRIC_PROMPT.format(
        question=rubric.question, context=_context(company, result), scale=scale, example=example,
        criteria="\n".join(f"- {c.key}: {c.question}" for c in criteria),
        empty_note=("\nThis report has no findings. Grade only the criteria listed.\n" if empty else ""))
    user = (f"Report:\n\n{report_for_judge(result)}\n\n=====\n\n"
            f"Evidence the author was given:\n\n{evidence_overview(result)}")
    call = judge_call(judge, "rubric", system, user, _RubricOut)
    if call.value is None:
        return {}
    keys = {c.key for c in criteria}
    return {k: v for k, v in call.value.scores.items() if k in keys and v is not None and v.score is not None}


# ---------------------------------------------------------------- pairwise

class _Verdict(BaseModel):
    winner: str
    reason: str = ""


PAIRWISE_PROMPT = """You compare two KYB adverse-media screening reports about the same company,
produced by different pipelines from web search results. Decide which one a compliance analyst
would rather rely on.

Weigh, in this order:
1. Findings are about the right company and backed by their cited excerpts.
2. Coverage of the significant adverse events.
3. No noise: nothing irrelevant, duplicated or outside the window.
4. Clear, specific summaries with severity and status that fit the sources.

{context}

Return only JSON: {{"winner": "A" | "B" | "tie", "reason": "one or two sentences"}}"""


@dataclass
class PairOutcome:
    winner: str          # "first", "second" or "tie" (in terms of the arms as passed)
    verdict_1: str       # raw verdict with the first arm shown as A
    verdict_2: str       # raw verdict with the second arm shown as A
    reason_1: str
    reason_2: str

    @property
    def consistent(self) -> bool:
        return self.winner != "tie" or self.verdict_1 == self.verdict_2 == "tie"


def judge_pairwise(judge: ChatModel, company: Company, first: ScreenResult,
                   second: ScreenResult) -> PairOutcome:
    """Ask twice with the order swapped; a win only counts if it survives the swap."""
    system = PAIRWISE_PROMPT.format(context=_context(company, first))

    def ask(a: ScreenResult, b: ScreenResult) -> tuple[str, str]:
        user = f"Report A:\n\n{report_for_judge(a)}\n\n=====\n\nReport B:\n\n{report_for_judge(b)}"
        call = judge_call(judge, "pairwise", system, user, _Verdict)
        if call.value is None:
            return "error", "; ".join(call.errors)
        w = call.value.winner.strip().upper()
        return ("tie" if w not in ("A", "B") else w), call.value.reason

    v1, r1 = ask(first, second)
    v2, r2 = ask(second, first)
    if v1 == "A" and v2 == "B":
        winner = "first"
    elif v1 == "B" and v2 == "A":
        winner = "second"
    else:
        winner = "tie"
    return PairOutcome(winner, v1, v2, r1, r2)
