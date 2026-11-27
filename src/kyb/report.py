"""Render a screen result. Code writes the report, so every citation links to a source
that was actually retrieved, and a category the model failed on is never shown as clean."""

from __future__ import annotations

from .plan import CATEGORIES, window_start
from .screen import ScreenResult

_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(result: ScreenResult) -> str:
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
        lines += ["## Summary", "", "| Category | Findings | Highest severity | Sources reviewed |",
                  "|---|---|---|---|"]
        for cat in CATEGORIES:
            run = result.categories[cat.key]
            if not run.parsed:
                lines.append(f"| {cat.title} | not assessed | — | {len(run.source_ids)} |")
                continue
            top = max((f.finding.severity for f in run.findings),
                      key=lambda s: _SEVERITY_RANK[s], default="—")
            lines.append(f"| {cat.title} | {len(run.findings)} | {top} | {len(run.source_ids)} |")
        lines += ["", "## Findings", ""]

        for cat in CATEGORIES:
            run = result.categories[cat.key]
            lines += [f"### {cat.title}", ""]
            if not run.parsed:
                reason = (f"the model's output could not be parsed after {run.attempts} attempts"
                          if run.attempts else f"the model call failed ({run.errors[-1]})")
                lines += [f"**Not assessed:** {reason}. Review the sources manually.", ""]
                continue
            if not run.findings:
                lines.append("No adverse findings. Searched:")
                for s in result.searches:
                    if s.category == cat.key:
                        scope = " (regulator sites only)" if s.include_domains else ""
                        lines.append(f"- `{s.query}`{scope}: {s.result_count} results")
                lines.append("")
                continue
            for vf in sorted(run.findings, key=lambda f: -_SEVERITY_RANK[f.finding.severity]):
                f = vf.finding
                cites = ", ".join(
                    f"[{sid}]({result.evidence.get(sid).url})" for sid in f.source_ids
                )
                meta = " · ".join(x for x in (f.severity.upper(), f.status, f.event_date) if x)
                lines.append(f"- **[{meta}]** {f.summary} {cites}")
            lines.append("")

        invalid = sum(r.invalid_citations for r in result.categories.values())
        dropped = sum(r.dropped_findings for r in result.categories.values())
        rejected_namesakes = sum(
            1 for r in result.categories.values() for x in r.rejected if x.reason == "different_entity"
        )
        lines += [
            "## Checks on the model's output",
            "",
            f"- Citations removed because the cited ID was not in the evidence: {invalid}",
            f"- Findings dropped because no valid citation remained: {dropped}",
            f"- Sources the model set aside as a different entity: {rejected_namesakes}",
            f"- Tokens: {result.prompt_tokens:,} in, {result.completion_tokens:,} out",
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
        lines.append(f"| {s.id} | [{_md_escape(s.title)}]({s.url}) | {s.published_date or '—'} "
                     f"| {', '.join(s.categories)} |")
    return "\n".join(lines) + "\n"


def to_json(result: ScreenResult) -> dict:
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
        "findings": [f.as_dict() for f in result.findings],
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
