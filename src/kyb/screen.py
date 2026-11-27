"""The screening pipeline: plan -> search -> evidence -> per-category reading.

Two model roles. The planner (the fixed plan, or a reasoning model) chooses the searches.
The reader decides which sources are about the target company, pulls out adverse events,
and cites source IDs. Code builds the search parameters and renders the report, so no model
can widen the date window, invent a URL, or skip a category.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from .evidence import EvidenceStore, render_sources
from .llm import ChatModel, complete_json
from .plan import CATEGORIES, CATEGORY_BY_KEY, plan_version, window_start
from .planner import FixedPlanner, Planner, PlanResult
from .schema import (
    CATEGORY_OUTPUT_EXAMPLE,
    CategoryOutput,
    Company,
    Finding,
    RejectedSource,
)
from .search import SearchBackend, traced_search
from .tracing import get_tracer, set_io, set_json

SYSTEM_PROMPT = """You are a KYB (know-your-business) analyst screening a company for adverse media.
Today's date is {as_of}. Only report events dated between {start} and {as_of}.

Target company:
{identity}

Risk category: {title}
What counts: {description}

You will receive search results, each labeled with a source ID such as [S3].

Rules:
1. Use only the provided sources. Do not add facts from memory.
2. Report only events about the target company. A source about a different organization
   with a similar name is "different_entity". If an event concerns the target's parent or
   subsidiary, report it only if the source ties it to the target, and name that entity in
   the summary.
3. Every finding must cite the source IDs that directly support it. Merge sources about
   the same event into one finding.
4. Severity: high = criminal charges or pleas, sanctions violations, large regulatory
   penalties, insolvency, or a major breach; medium = civil suits, smaller penalties,
   open investigations; low = minor or peripheral matters.
5. Status: alleged (claims, charges, investigations), confirmed (pleas, judgments,
   settlements, penalties), resolved (closed or lifted), unknown.
6. List sources you did not use in rejected_sources with a reason: different_entity,
   not_adverse, outside_window, or not_relevant.
7. If nothing qualifies, return an empty findings list.

Return only a JSON object shaped like this example:
{example}"""


@dataclass
class ValidatedFinding:
    id: str
    category: str
    finding: Finding

    def as_dict(self) -> dict:
        return {"id": self.id, "category": self.category, **self.finding.model_dump()}


@dataclass
class CategoryRun:
    category: str
    source_ids: list[str]
    findings: list[ValidatedFinding] = field(default_factory=list)
    rejected: list[RejectedSource] = field(default_factory=list)
    parsed: bool = False
    first_try: bool = False
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    invalid_citations: int = 0
    dropped_findings: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class SearchLog:
    category: str
    query: str
    include_domains: list[str]
    result_count: int


@dataclass
class ScreenResult:
    company: Company
    as_of: date
    lookback_years: int
    mode: str
    model: str | None
    trace_id: str
    evidence: EvidenceStore
    searches: list[SearchLog]
    categories: dict[str, CategoryRun] = field(default_factory=dict)
    wall_time_s: float = 0.0
    plan: PlanResult | None = None

    @property
    def planner(self) -> str:
        return self.plan.planner if self.plan else "fixed"

    @property
    def findings(self) -> list[ValidatedFinding]:
        return [f for run in self.categories.values() for f in run.findings]

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.categories.values())

    @property
    def completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.categories.values())


def build_messages(company: Company, as_of: date, lookback_years: int, category: str,
                   evidence_text: str) -> list[dict]:
    cat = CATEGORY_BY_KEY[category]
    system = SYSTEM_PROMPT.format(
        as_of=as_of.isoformat(),
        start=window_start(as_of, lookback_years).isoformat(),
        identity=company.identity_block(),
        title=cat.title,
        description=cat.description,
        example=CATEGORY_OUTPUT_EXAMPLE,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Search results for {company.name}:\n\n{evidence_text}"},
    ]


def validate_output(category: str, output: CategoryOutput, allowed_ids: set[str],
                    run: CategoryRun) -> None:
    """Keep only citations to sources the model was shown; drop findings left uncited."""
    for finding in output.findings:
        valid = [sid for sid in dict.fromkeys(finding.source_ids) if sid in allowed_ids]
        run.invalid_citations += len(set(finding.source_ids)) - len(valid)
        if not valid:
            run.dropped_findings += 1
            continue
        kept = finding.model_copy(update={"source_ids": valid})
        run.findings.append(ValidatedFinding(f"{category}-{len(run.findings) + 1}", category, kept))
    run.rejected = [r for r in output.rejected_sources if r.source_id in allowed_ids]


def extract_category(model: ChatModel, company: Company, as_of: date, lookback_years: int,
                     category: str, evidence: EvidenceStore) -> CategoryRun:
    sources = evidence.for_category(category)
    run = CategoryRun(category=category, source_ids=[s.id for s in sources])
    tracer = get_tracer()
    with tracer.start_as_current_span(f"kyb.extract.{category}") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute("kyb.category", category)
        span.set_attribute("kyb.source_count", len(sources))
        if not sources:  # nothing to read; no model call needed
            run.parsed = run.first_try = True
            return run

        messages = build_messages(company, as_of, lookback_years, category,
                                  render_sources(sources, category))
        try:
            call = complete_json(model, messages, CategoryOutput)
        except Exception as exc:  # API failure: this category is "not assessed", the rest continue
            span.record_exception(exc)
            run.errors.append(f"{type(exc).__name__}: {exc}")
            set_json(span, "kyb.errors", run.errors)
            return run
        run.attempts, run.errors = call.attempts, call.errors
        run.prompt_tokens, run.completion_tokens = call.prompt_tokens, call.completion_tokens
        run.latency_s = call.latency_s
        if call.value is not None:
            run.parsed, run.first_try = True, call.first_try
            validate_output(category, call.value, set(run.source_ids), run)

        span.set_attribute("kyb.parsed", run.parsed)
        span.set_attribute("kyb.attempts", run.attempts)
        span.set_attribute("kyb.invalid_citations", run.invalid_citations)
        span.set_attribute("kyb.dropped_findings", run.dropped_findings)
        set_io(span, output_value=[f.as_dict() for f in run.findings])
        if run.errors:
            set_json(span, "kyb.errors", run.errors)
    return run


def _plan_step(company: Company, as_of: date, lookback_years: int, planner: Planner,
               recorded: PlanResult | None) -> PlanResult:
    with get_tracer().start_as_current_span("kyb.plan") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute("kyb.plan_replayed", recorded is not None)
        if recorded is not None:
            plan = recorded
        else:
            plan = planner.plan(company, as_of, lookback_years)
            span.set_attribute("kyb.prompt_tokens", plan.prompt_tokens)
            span.set_attribute("kyb.completion_tokens", plan.completion_tokens)
        span.set_attribute("kyb.planner", plan.planner)
        span.set_attribute("kyb.search_count", len(plan.searches))
        set_json(span, "kyb.plan_fixes", plan.fixes)
        set_json(span, "kyb.plan_fell_back", plan.fell_back)
        if plan.raw is not None:
            set_json(span, "kyb.plan_raw", plan.raw)
        set_io(span, output_value=[{"category": p.category, "params": p.params} for p in plan.searches])
    return plan


def run_screen(company: Company, as_of: date, lookback_years: int, backend: SearchBackend,
               model: ChatModel | None, session_id: str | None = None, max_workers: int = 4,
               planner: Planner | None = None, recorded_plan: PlanResult | None = None) -> ScreenResult:
    """Plan, search, then read. `model=None` stops after searching (recording mode).
    `recorded_plan` replays a plan instead of calling the planner (the default is the fixed plan)."""
    import time

    started = time.perf_counter()
    tracer = get_tracer()
    planner = planner or FixedPlanner()
    planner_name = recorded_plan.planner if recorded_plan else planner.name
    metadata = {"company": company.name, "mode": backend.mode, "planner": planner_name,
                "model": model.model if model else "search-only"}
    with using_attributes(session_id=session_id or "", metadata=metadata), \
            tracer.start_as_current_span("kyb.screen") as root:
        root.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        root.set_attribute("kyb.company", company.model_dump_json())
        root.set_attribute("kyb.as_of", as_of.isoformat())
        root.set_attribute("kyb.lookback_years", lookback_years)
        root.set_attribute("kyb.plan_version", plan_version())
        root.set_attribute("kyb.planner", planner_name)
        root.set_attribute("kyb.mode", backend.mode)
        root.set_attribute("kyb.model", model.model if model else "")
        set_io(root, input_value=company.model_dump(exclude_none=True))

        plan = _plan_step(company, as_of, lookback_years, planner, recorded_plan)
        evidence = EvidenceStore()
        searches: list[SearchLog] = []
        for planned in plan.searches:
            response = traced_search(backend, planned.category, planned.params)
            evidence.add_response(planned.category, response)
            searches.append(SearchLog(planned.category, planned.params["query"],
                                      planned.params.get("include_domains", []),
                                      len(response.get("results") or [])))

        result = ScreenResult(
            company=company, as_of=as_of, lookback_years=lookback_years, mode=backend.mode,
            model=model.model if model else None,
            trace_id=format(root.get_span_context().trace_id, "032x"),
            evidence=evidence, searches=searches, plan=plan,
        )

        if model is not None:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    c.key: pool.submit(contextvars.copy_context().run, extract_category,
                                       model, company, as_of, lookback_years, c.key, evidence)
                    for c in CATEGORIES
                }
                result.categories = {key: f.result() for key, f in futures.items()}

        root.set_attribute("kyb.source_count", len(evidence))
        root.set_attribute("kyb.finding_count", len(result.findings))
        # Custom names on purpose: the LLM child spans already carry llm.token_count.*,
        # and repeating them here would double-count in trace UIs that sum the tree.
        root.set_attribute("kyb.prompt_tokens", result.prompt_tokens)
        root.set_attribute("kyb.completion_tokens", result.completion_tokens)
        set_io(root, output_value=[f.as_dict() for f in result.findings])
    result.wall_time_s = time.perf_counter() - started
    return result
