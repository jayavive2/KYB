"""LLM judges. One fixed judge model answers five kinds of question:

  coverage  - is each known event in the search results at all?        (per recording)
  recall    - which finding, if any, reports each known event?         (per report)
  support   - is each finding about the target, and backed by its sources? (per finding)
  rubric    - 1-5 scores on criteria from config/rubric.yaml           (per report)
  pairwise  - which of two reports would an analyst rather rely on?    (per pair, both orders)

Every verdict is written to CSV so a person can audit a sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from pydantic import BaseModel, Field

from .evidence import EvidenceStore, render_sources
from .llm import ChatModel, JsonCallResult, complete_json
from .plan import window_start
from .schema import Company
from .screen import ScreenResult
from .tracing import get_tracer, set_io

JUDGE_FAILURES: list[str] = []  # reset per experiment; reported in the summary


def judge_call(judge: ChatModel, name: str, system: str, user: str, schema) -> JsonCallResult:
    with get_tracer().start_as_current_span(f"kyb.judge.{name}") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND,
                           OpenInferenceSpanKindValues.EVALUATOR.value)
        try:
            call = complete_json(judge, [{"role": "system", "content": system},
                                         {"role": "user", "content": user}], schema)
        except Exception as exc:  # a judge outage shouldn't discard the candidates' runs
            span.record_exception(exc)
            call = JsonCallResult(value=None, attempts=0, errors=[f"{type(exc).__name__}: {exc}"])
        if call.value is None:
            JUDGE_FAILURES.append(f"{name}: {call.errors[-1] if call.errors else 'no output'}")
        set_io(span, output_value=call.value.model_dump() if call.value else {"errors": call.errors})
        return call


def report_for_judge(result: ScreenResult, excerpt_chars: int = 400) -> str:
    """A report as the judge sees it: each finding with short excerpts of its cited sources."""
    failed = [k for k, run in result.categories.items() if not run.parsed]
    note = f"\n\nCategories not assessed (the author's output was unusable): {', '.join(failed)}" if failed else ""
    if not result.findings:
        return "(no findings)" + note
    blocks = []
    for f in result.findings:
        lines = [f"[{f.id}] ({f.category}; {f.finding.severity}; {f.finding.status}; "
                 f"{f.finding.event_date or 'date n/a'}) {f.finding.summary}"]
        for sid in f.finding.source_ids:
            s = result.evidence.get(sid)
            if s:
                text = s.text(f.category)
                text = text[:excerpt_chars] + ("…" if len(text) > excerpt_chars else "")
                lines.append(f"    {sid}: {s.title} ({s.published_date or 'date unknown'}) — {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + note


def _context(company: Company, result: ScreenResult) -> str:
    start = window_start(result.as_of, result.lookback_years)
    return (f"Target company:\n{company.identity_block()}\n\n"
            f"Screen date {result.as_of}; events in scope from {start}.")


# ---------------------------------------------------------------- coverage, recall, support

class _CoverageItem(BaseModel):
    event_id: str
    source_ids: list[str] = []


class _Coverage(BaseModel):
    events: list[_CoverageItem]


class _Match(BaseModel):
    event_id: str
    finding_id: str | None = None


class _Matches(BaseModel):
    matches: list[_Match]


class Judgment(BaseModel):
    finding_id: str
    about_target: bool
    supported: bool
    reason: str = ""


class _Judgments(BaseModel):
    judgments: list[Judgment]


COVERAGE_PROMPT = """You check whether known events appear in a set of search results about a company.
For each event, list the IDs of sources whose text describes that specific event for the target
company. Use only the text shown. If no source describes it, return an empty list.

Target company:
{identity}

Return only JSON: {{"events": [{{"event_id": "...", "source_ids": ["S1"]}}]}}"""

RECALL_PROMPT = """You match known events to the findings of an adverse-media screen.
For each known event, give the ID of the finding that reports the same underlying event
(same company, same action or incident), even if worded differently or less detailed.
If no finding reports it, use null.

Return only JSON: {"matches": [{"event_id": "...", "finding_id": "regulatory-1"}]}"""

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


def _event_lines(events) -> str:
    return "\n".join(f"- {e.id} ({e.category}, {e.date or 'date n/a'}): {e.description}" for e in events)


def judge_coverage(judge: ChatModel, company: Company, events, evidence: EvidenceStore) -> dict[str, list[str]]:
    if not events:
        return {}
    user = (f"Known events:\n{_event_lines(events)}\n\nSearch results:\n\n"
            f"{render_sources(evidence.all(), max_chars=1500)}")
    call = judge_call(judge, "coverage", COVERAGE_PROMPT.format(identity=company.identity_block()),
                      user, _Coverage)
    known = {s.id for s in evidence.all()}
    out = {e.id: [] for e in events}
    if call.value:
        for item in call.value.events:
            if item.event_id in out:
                out[item.event_id] = [s for s in item.source_ids if s in known]
    return out


def judge_recall(judge: ChatModel, events, result: ScreenResult) -> dict[str, str | None]:
    out: dict[str, str | None] = {e.id: None for e in events}
    if not events or not result.findings:
        return out
    fs = "\n".join(f"- {f.id} ({f.category}, {f.finding.event_date or 'date n/a'}): {f.finding.summary}"
                   for f in result.findings)
    call = judge_call(judge, "recall", RECALL_PROMPT,
                      f"Known events:\n{_event_lines(events)}\n\nFindings:\n{fs}", _Matches)
    ids = {f.id for f in result.findings}
    if call.value:
        for m in call.value.matches:
            if m.event_id in out and m.finding_id in ids:
                out[m.event_id] = m.finding_id
    return out


def judge_support(judge: ChatModel, company: Company, result: ScreenResult,
                  batch_size: int = 8) -> dict[str, Judgment]:
    out: dict[str, Judgment] = {}
    findings = result.findings
    system = SUPPORT_PROMPT.format(identity=company.identity_block())
    for i in range(0, len(findings), batch_size):
        blocks = []
        for f in findings[i : i + batch_size]:
            cited = [result.evidence.get(sid) for sid in f.finding.source_ids]
            blocks.append(
                f"Finding {f.id} ({f.category}, {f.finding.event_date or 'date n/a'}): {f.finding.summary}\n"
                f"Cited sources:\n{render_sources([c for c in cited if c], f.category)}"
            )
        call = judge_call(judge, "support", system, "\n\n---\n\n".join(blocks), _Judgments)
        if call.value:
            for j in call.value.judgments:
                out[j.finding_id] = j
    return out


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
