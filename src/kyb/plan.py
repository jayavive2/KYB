"""The fixed screening plan.

Every company is screened with the same categories, queries and Tavily parameters.
That is a compliance property (a documented, repeatable method) and an evaluation
property (replay is exact because no model decides what to search).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date

from .schema import Company

# Applied to every search. `exact_match` keeps only results containing the quoted
# company name; `advanced` + `chunks_per_source` returns the most query-relevant
# passages per page instead of whole pages, which is our first layer of compaction.
BASE_PARAMS: dict = {
    "search_depth": "advanced",
    "topic": "general",
    "max_results": 8,
    "chunks_per_source": 3,
    "include_published_date": True,
    "exact_match": True,
    "include_usage": True,
}

CREDITS_PER_SEARCH = 2  # Tavily charges 2 credits for an `advanced` search

# Social platforms: posts repeat or speculate on news, and a reader can't tell a rumor from a
# report. Left out of open-web searches; the news they echo is still found at its source.
EXCLUDED_DOMAINS: tuple[str, ...] = (
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "linkedin.com",
    "reddit.com",
    "youtube.com",
    "pinterest.com",
    "threads.net",
    "quora.com",
)

# Primary sources for enforcement actions. Searched in addition to the open web so a
# regulator's own press release is in the evidence when it exists.
REGULATOR_DOMAINS: tuple[str, ...] = (
    "sec.gov",
    "justice.gov",
    "ftc.gov",
    "consumerfinance.gov",
    "fincen.gov",
    "treasury.gov",
    "federalreserve.gov",
    "occ.gov",
    "fdic.gov",
    "cftc.gov",
    "finra.org",
    "fca.org.uk",
)


@dataclass(frozen=True)
class QuerySpec:
    template: str
    include_domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class Category:
    key: str
    title: str
    description: str
    queries: tuple[QuerySpec, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        "regulatory",
        "Regulatory and enforcement actions",
        "Fines, penalties, consent orders, settlements with regulators, license actions, "
        "and formal regulatory investigations.",
        (
            QuerySpec('"{name}" regulator fine penalty settlement enforcement action'),
            QuerySpec('"{name}" enforcement action', include_domains=REGULATOR_DOMAINS),
        ),
    ),
    Category(
        "litigation",
        "Material litigation",
        "Lawsuits, class actions and court rulings in which the company is a party, "
        "mainly as defendant.",
        (QuerySpec('"{name}" lawsuit class action court ruling'),),
    ),
    Category(
        "financial_crime",
        "Financial crime",
        "Fraud, money laundering, bribery or corruption: criminal charges, pleas, "
        "convictions, deferred prosecution agreements or credible investigations.",
        (QuerySpec('"{name}" fraud money laundering bribery investigation'),),
    ),
    Category(
        "sanctions",
        "Sanctions and export controls",
        "Sanctions designations, sanctions violations, and export-control breaches.",
        (QuerySpec('"{name}" sanctions OFAC export controls violation'),),
    ),
    Category(
        "data_breach",
        "Data breaches and cyber incidents",
        "Data breaches, ransomware or other cyberattacks, and security incidents that "
        "affected customers or operations.",
        (QuerySpec('"{name}" data breach cyberattack security incident'),),
    ),
    Category(
        "financial_distress",
        "Financial distress",
        "Bankruptcy, insolvency, failure, default, going-concern warnings, rescue "
        "financing, and major restructurings or layoffs.",
        (QuerySpec('"{name}" bankruptcy default restructuring layoffs'),),
    ),
    Category(
        "leadership",
        "Leadership and governance",
        "Executive misconduct, executives charged or forced out, board or governance "
        "failures, and whistleblower allegations.",
        (QuerySpec('"{name}" CEO executive resigns charged whistleblower'),),
    ),
)

CATEGORY_BY_KEY = {c.key: c for c in CATEGORIES}


@dataclass(frozen=True)
class PlannedSearch:
    category: str
    params: dict = field(hash=False)


def window_start(as_of: date, lookback_years: int) -> date:
    try:
        return as_of.replace(year=as_of.year - lookback_years)
    except ValueError:  # 29 February
        return as_of.replace(year=as_of.year - lookback_years, day=28)


TOPICS = ("general", "news")


def make_search(category: str, query: str, start: date, *, topic: str = "general",
                exact_match: bool = True, include_domains: tuple[str, ...] = ()) -> PlannedSearch:
    """The one place Tavily parameters are assembled, for the fixed plan and planners alike.
    The date window always comes from the screen, never from a planner."""
    params = dict(BASE_PARAMS)
    params.update(query=query, topic=topic, exact_match=exact_match, start_date=start.isoformat())
    if include_domains:
        params["include_domains"] = list(include_domains)
    else:
        params["exclude_domains"] = list(EXCLUDED_DOMAINS)
    return PlannedSearch(category, params)


def build_plan(company: Company, as_of: date, lookback_years: int) -> list[PlannedSearch]:
    start = window_start(as_of, lookback_years)
    return [
        make_search(c.key, spec.template.format(name=company.name), start,
                    include_domains=spec.include_domains)
        for c in CATEGORIES for spec in c.queries
    ]


def plan_version() -> str:
    """Fingerprint of the plan definition, stored in every trace so replay can refuse a mismatch."""
    payload = {
        "base": BASE_PARAMS,
        "excluded": list(EXCLUDED_DOMAINS),
        "categories": [
            [c.key, [[q.template, list(q.include_domains)] for q in c.queries]] for c in CATEGORIES
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def search_key(params: dict) -> str:
    """Canonical key for a Tavily request; recorded and replayed searches must match on it."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))
