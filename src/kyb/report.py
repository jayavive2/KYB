"""Render a screen result. Code writes the report, so every citation links to a source
that was actually retrieved, and a category the model failed on is never shown as clean."""

from __future__ import annotations

from email.utils import parsedate_to_datetime

from .events import SEVERITY_RANK, Event, SourceCheck, build_events
from .plan import CATEGORIES, CATEGORY_BY_KEY, window_start
from .screen import ScreenResult, ValidatedFinding


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _day(published: str | None) -> str:
    """Tavily dates look like 'Thu, 10 Oct 2024 00:00:00 GMT'; show 2024-10-10."""
    if not published:
        return "—"
    try:
        return parsedate_to_datetime(published).date().isoformat()
    except (TypeError, ValueError):
        return published


def _cites(result: ScreenResult, source_ids: list[str]) -> str:
    return ", ".join(f"[{sid}]({result.evidence.get(sid).url})" for sid in source_ids)


def _meta(f: ValidatedFinding) -> str:
    return " · ".join(x for x in (f.finding.severity.upper(), f.finding.status, f.finding.event_date) if x)


def _event_line(result: ScreenResult, e: Event, check: SourceCheck | None) -> str:
    line = f"- **[{_meta(e.lead)}]** {e.lead.finding.summary} {_cites(result, e.source_ids)}"
    others = [CATEGORY_BY_KEY[c].title for c in e.categories[1:]]
    if others:
        line += f" _Also: {', '.join(others)}._"
    elif len(e.members) > 1:
        line += f" _({len(e.members)} reports merged.)_"
    if check is not None and check.confirmed(e.lead.id) is None:
        line += " _(source check returned nothing for this one)_"
    return line


def render_markdown(result: ScreenResult, check: SourceCheck | None = None) -> str:
    check = check or result.check
    start = window_start(result.as_of, result.lookback_years)
    lines = [
        f"# KYB screen: {result.company.name}",
        "",
        f"As of {result.as_of} · events since {start} · mode: {result.mode} · planner: {result.planner} · "
        f"reader: {result.model or 'none (search only)'} · trace: `{result.trace_id}`",
        "",
        "Target:",
        result.company.identity_block(),
        "",
    ]

    if result.categories:
        events, held = build_events(result, check)
        lines += ["## Summary", "", "| Category | Events | Highest severity | Sources reviewed |",
                  "|---|---|---|---|"]
        for cat in CATEGORIES:
            run = result.categories[cat.key]
            if not run.parsed:
                lines.append(f"| {cat.title} | not assessed | — | {len(run.source_ids)} |")
                continue
            here = [e for e in events if cat.key in e.categories]
            top = max((e.lead.finding.severity for e in here), key=lambda s: SEVERITY_RANK[s], default="—")
            lines.append(f"| {cat.title} | {len(here)} | {top} | {len(run.source_ids)} |")
        lines += ["", "## Findings", ""]
        if check is not None:
            lines += [f"Every finding was checked against the sources it cites by `{check.model}`. "
                      + (f"{len(held)} the check could not confirm are under **Held back** below."
                         if held else "None were held back."), ""]

        for cat in CATEGORIES:
            run = result.categories[cat.key]
            lines += [f"### {cat.title}", ""]
            if not run.parsed:
                reason = (f"the model's output could not be parsed after {run.attempts} attempts"
                          if run.attempts else f"the model call failed ({run.errors[-1]})")
                lines += [f"**Not assessed:** {reason}. Review the sources manually.", ""]
                continue
            mine = [e for e in events if e.lead.category == cat.key]
            elsewhere = [e for e in events if cat.key in e.categories[1:]]
            if not mine and not elsewhere:
                lines.append("No adverse findings. Searched:")
                for s in result.searches:
                    if s.category == cat.key:
                        scope = " (regulator sites only)" if s.include_domains else ""
                        lines.append(f"- `{s.query}`{scope}: {s.result_count} results")
                lines.append("")
                continue
            lines += [_event_line(result, e, check) for e in mine]
            for e in elsewhere:
                summary = e.lead.finding.summary
                short = summary if len(summary) <= 110 else summary[:107].rsplit(" ", 1)[0] + "…"
                lines.append(f"- Also here: {short} (listed under {CATEGORY_BY_KEY[e.lead.category].title})")
            lines.append("")

        if held:
            lines += ["## Held back", "",
                      "The source check found these unsupported by the sources they cite, or not about the "
                      "target. They are not in the summary. Review them before relying on them.", ""]
            for f in held:
                reason = check.reason(f.id) if check else ""
                lines.append(f"- **[{_meta(f)}]** {f.finding.summary} {_cites(result, f.finding.source_ids)}"
                             + (f" — _check: {reason}_" if reason else ""))
            lines.append("")

        invalid = sum(r.invalid_citations for r in result.categories.values())
        dropped = sum(r.dropped_findings for r in result.categories.values())
        rejected_namesakes = sum(
            1 for r in result.categories.values() for x in r.rejected if x.reason == "different_entity"
        )
        merged = sum(len(g) - 1 for g in result.duplicate_groups)
        if check is None:
            checked = "- Source check: not run"
        else:
            verdicts = [check.confirmed(f.id) for f in result.findings]
            checked = (f"- Source check (`{check.model}`): {verdicts.count(True)} confirmed, "
                       f"{verdicts.count(False)} not confirmed, {verdicts.count(None)} not checked; "
                       f"{len(held)} held back (a confirmed duplicate keeps the rest in the report)")
            if check.errors:
                checked += f" ({len(check.errors)} check calls failed: {check.errors[0]})"
        lines += [
            "## Checks on the model's output",
            "",
            f"- Citations removed because the cited ID was not in the evidence: {invalid}",
            f"- Findings dropped because no valid citation remained: {dropped}",
            f"- Findings merged because they reported the same event as another: {merged}",
            checked,
            f"- Sources the model set aside as a different entity: {rejected_namesakes}",
            f"- Reader tokens: {result.prompt_tokens:,} in, {result.completion_tokens:,} out",
            "",
        ]
    else:
        lines += ["## Searches", ""]
        for s in result.searches:
            scope = " (regulator sites only)" if s.include_domains else ""
            lines.append(f"- {s.category}: `{s.query}`{scope}: {s.result_count} results")
        lines.append("")

    lines += ["## Sources", "", "| ID | Title | Published | Categories |", "|---|---|---|---|"]
    for s in result.evidence.all():
        lines.append(f"| {s.id} | [{_md_escape(s.title)}]({s.url}) | {_day(s.published_date)} "
                     f"| {', '.join(s.categories)} |")
    return "\n".join(lines) + "\n"


def to_json(result: ScreenResult, check: SourceCheck | None = None) -> dict:
    check = check or result.check
    events, held = build_events(result, check) if result.categories else ([], [])

    def finding(f: ValidatedFinding) -> dict:
        out = f.as_dict()
        if check is not None:
            out["confirmed"] = check.confirmed(f.id)
            out["check_reason"] = check.reason(f.id)
        return out

    verdicts = [check.confirmed(f.id) for f in result.findings] if check else []
    return {
        "company": result.company.model_dump(exclude_none=True),
        "as_of": result.as_of.isoformat(),
        "lookback_years": result.lookback_years,
        "mode": result.mode,
        "planner": result.planner,
        "plan_fixes": result.plan.fixes if result.plan else [],
        "model": result.model,
        "searches": [{"category": s.category, "query": s.query, "regulator_sites_only": bool(s.include_domains),
                      "results": s.result_count} for s in result.searches],
        "trace_id": result.trace_id,
        "findings": [finding(f) for f in result.findings],
        "events": [{"id": e.lead.id, "members": [m.id for m in e.members], "categories": e.categories,
                    "severity": e.lead.finding.severity, "status": e.lead.finding.status,
                    "event_date": e.lead.finding.event_date, "summary": e.lead.finding.summary,
                    "source_ids": e.source_ids} for e in events],
        "held_back": [f.id for f in held],
        "duplicate_groups": result.duplicate_groups,
        "merge": None if result.merge is None else {
            "attempts": result.merge.attempts, "errors": result.merge.errors,
            "prompt_tokens": result.merge.prompt_tokens, "completion_tokens": result.merge.completion_tokens},
        "source_check": None if check is None else {
            "model": check.model, "confirmed": verdicts.count(True), "not_confirmed": verdicts.count(False),
            "held_back": len(held),
            "unchecked": verdicts.count(None), "errors": check.errors},
        "categories": {
            key: {
                "parsed": run.parsed, "attempts": run.attempts, "errors": run.errors,
                "invalid_citations": run.invalid_citations,
                "dropped_findings": run.dropped_findings,
                "rejected_sources": [r.model_dump() for r in run.rejected],
                "prompt_tokens": run.prompt_tokens, "completion_tokens": run.completion_tokens,
                "latency_s": round(run.latency_s, 3),
            }
            for key, run in result.categories.items()
        },
        "sources": [
            {"id": s.id, "url": s.url, "title": s.title, "published_date": s.published_date,
             "categories": s.categories}
            for s in result.evidence.all()
        ],
    }
