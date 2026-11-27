"""What a report shows, built from the reader's findings by code.

Findings the reader grouped as the same real-world event are shown once, under every category
they belong to. Findings the source check could not confirm are held back, unless a duplicate of
theirs was confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .plan import CATEGORIES
from .schema import Judgment
from .screen import ScreenResult, ValidatedFinding

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}
_CATEGORY_ORDER = {c.key: i for i, c in enumerate(CATEGORIES)}


@dataclass
class SourceCheck:
    """The judge's verdict on each finding: do its cited sources say what it claims?"""

    model: str
    judgments: dict[str, Judgment] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def confirmed(self, finding_id: str) -> bool | None:
        """True or False once checked; None if the check returned nothing for this finding."""
        j = self.judgments.get(finding_id)
        return None if j is None else (j.supported and j.about_target)

    def reason(self, finding_id: str) -> str:
        j = self.judgments.get(finding_id)
        if j is None:
            return ""
        if not j.about_target:
            return "not about the target company" + (f": {j.reason}" if j.reason else "")
        return j.reason


@dataclass
class Event:
    """One real-world event. `members` are the confirmed findings that report it, the lead first;
    `extra_categories` come from duplicates the check didn't confirm (the event still belongs there)."""

    members: list[ValidatedFinding]
    extra_categories: list[str] = field(default_factory=list)

    @property
    def lead(self) -> ValidatedFinding:
        return self.members[0]

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys([m.category for m in self.members] + self.extra_categories))

    @property
    def citations(self) -> list[tuple[str, str]]:
        """(source ID, category it was cited under), each source once."""
        seen: dict[str, str] = {}
        for m in self.members:
            for sid in m.finding.source_ids:
                seen.setdefault(sid, m.category)
        return list(seen.items())

    @property
    def source_ids(self) -> list[str]:
        return [sid for sid, _ in self.citations]


def build_events(result: ScreenResult, check: SourceCheck | None = None
                 ) -> tuple[list[Event], list[ValidatedFinding]]:
    """(events to show, findings held back). A finding is held back when the source check rejected
    it and no duplicate of it was confirmed; a confirmed duplicate carries the event."""
    findings = result.findings
    order = {f.id: i for i, f in enumerate(findings)}
    by_id = {f.id: f for f in findings}

    def ok(f: ValidatedFinding) -> bool:
        return check is None or check.confirmed(f.id) is not False

    def lead_first(f: ValidatedFinding) -> tuple:
        return (-SEVERITY_RANK[f.finding.severity], -len(f.finding.source_ids), order[f.id])

    grouped = {i for g in result.duplicate_groups for i in g}
    groups = [[by_id[i] for i in g if i in by_id] for g in result.duplicate_groups]
    groups += [[f] for f in findings if f.id not in grouped]
    events: list[Event] = []
    held: list[ValidatedFinding] = []
    for group in groups:
        confirmed = sorted((f for f in group if ok(f)), key=lead_first)
        rejected = [f for f in group if not ok(f)]
        if confirmed:
            events.append(Event(confirmed, [f.category for f in rejected]))
        else:
            held.extend(rejected)
    events.sort(key=lambda e: (_CATEGORY_ORDER.get(e.lead.category, 99), lead_first(e.lead)))
    held.sort(key=lambda f: order[f.id])
    return events, held
