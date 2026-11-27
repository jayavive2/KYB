"""Planners decide which Tavily searches a screen runs.

`FixedPlanner` is the hand-written plan and the baseline. `LLMPlanner` asks a model
(intended: a reasoning model) to write the queries for each risk category. Either way,
code builds the final Tavily parameters, enforces the budget, and keeps the date window,
so planners compete on query choice only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field

from .llm import ChatModel, complete_json
from .plan import (
    CATEGORIES,
    CATEGORY_BY_KEY,
    REGULATOR_DOMAINS,
    TOPICS,
    PlannedSearch,
    build_plan,
    make_search,
    window_start,
)
from .schema import Company

FIXED = "fixed"
MAX_PER_CATEGORY = 3
MAX_TOTAL = 12


@dataclass
class PlanResult:
    planner: str
    searches: list[PlannedSearch]
    fixes: list[str] = field(default_factory=list)      # changes code made to the model's plan
    fell_back: list[str] = field(default_factory=list)  # categories filled from the fixed plan
    raw: dict | None = None                             # the model's plan as returned
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class Planner(Protocol):
    name: str

    def plan(self, company: Company, as_of: date, lookback_years: int) -> PlanResult: ...


class FixedPlanner:
    name = FIXED

    def plan(self, company: Company, as_of: date, lookback_years: int) -> PlanResult:
        return PlanResult(self.name, build_plan(company, as_of, lookback_years))


class _Search(BaseModel):
    category: str
    query: str
    topic: str = "general"
    regulator_sites_only: bool = False
    exact_match: bool = True
    why: str = ""


class _Plan(BaseModel):
    searches: list[_Search] = Field(default_factory=list)


PLANNER_PROMPT = """You plan web searches for a KYB (know-your-business) adverse-media screen.
Today's date is {as_of}. The screen covers events from {start} to {as_of}; the date filter is
applied for you.

Target company:
{identity}

Risk categories. Plan at least one search for every category:
{categories}

Each search goes to Tavily web search. For each, choose:
- query: put the company's name, or an alias, in double quotes, then the words a news article,
  court filing or regulator press release would use. Example: "Acme Corp" consent order penalty
- topic: "general" (default, includes regulator and court sites) or "news" (news coverage)
- regulator_sites_only: true to search only regulator websites (SEC, DOJ, FTC, CFPB, FinCEN,
  Treasury/OFAC, Federal Reserve, OCC, FDIC, CFTC, FINRA, UK FCA)
- exact_match: true to require the quoted name in every result. Use it unless the company is
  commonly referred to by another name, and then quote that alias in another search.

Budget: 1 to {per_cat} searches per category, at most {total} in total. Spend extra searches where
this kind of company most often has records, and to cover aliases.

Return only JSON:
{{"searches": [{{"category": "regulatory", "query": "\\"Acme Corp\\" consent order penalty", "topic": "general", "regulator_sites_only": false, "exact_match": true, "why": "short reason"}}]}}"""  # noqa: E501


class LLMPlanner:
    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.name = model.model

    def plan(self, company: Company, as_of: date, lookback_years: int) -> PlanResult:
        start = window_start(as_of, lookback_years)
        categories = "\n".join(f"- {c.key}: {c.title}. {c.description}" for c in CATEGORIES)
        system = PLANNER_PROMPT.format(as_of=as_of.isoformat(), start=start.isoformat(),
                                       identity=company.identity_block(), categories=categories,
                                       per_cat=MAX_PER_CATEGORY, total=MAX_TOTAL)
        began = time.perf_counter()
        call = complete_json(self.model, [{"role": "system", "content": system},
                                          {"role": "user", "content": f"Plan the searches for {company.name}."}],
                             _Plan)
        result = validate_plan(self.name, call.value, company, as_of, lookback_years)
        if call.value is None:
            result.fixes.append(f"planner output unusable ({'; '.join(call.errors)}); used the fixed plan")
        result.prompt_tokens, result.completion_tokens = call.prompt_tokens, call.completion_tokens
        result.latency_s = time.perf_counter() - began
        return result


def validate_plan(planner: str, plan: _Plan | None, company: Company, as_of: date,
                  lookback_years: int) -> PlanResult:
    """Turn a model's plan into Tavily requests: drop what's invalid, cap the budget, and fill
    any category left without a search from the fixed plan."""
    start = window_start(as_of, lookback_years)
    fixed = build_plan(company, as_of, lookback_years)
    result = PlanResult(planner, [], raw=plan.model_dump() if plan else None)
    if plan is None:
        result.searches, result.fell_back = fixed, [c.key for c in CATEGORIES]
        return result

    candidates: list[PlannedSearch] = []
    seen: set[tuple] = set()
    for s in plan.searches:
        query = " ".join(s.query.split())
        if s.category not in CATEGORY_BY_KEY:
            result.fixes.append(f"dropped search for unknown category {s.category!r}")
            continue
        if not query:
            result.fixes.append(f"dropped empty query in {s.category}")
            continue
        topic = s.topic if s.topic in TOPICS else "general"
        if topic != s.topic:
            result.fixes.append(f"topic {s.topic!r} -> 'general' for {query!r}")
        exact = s.exact_match
        if exact and '"' not in query:  # Tavily's exact_match needs a quoted phrase
            exact = False
            result.fixes.append(f"exact_match off for {query!r}: no quoted phrase")
        key = (s.category, query, topic, exact, s.regulator_sites_only)
        if key in seen:
            result.fixes.append(f"dropped duplicate {query!r}")
            continue
        seen.add(key)
        candidates.append(make_search(s.category, query, start, topic=topic, exact_match=exact,
                                      include_domains=REGULATOR_DOMAINS if s.regulator_sites_only else ()))

    # Budget: every category gets one search (the model's first, or the fixed plan's), and
    # the rest of the budget goes to the model's extra searches in the order it gave them.
    firsts: dict[str, PlannedSearch] = {}
    for c in candidates:
        firsts.setdefault(c.category, c)
    chosen = list(firsts.values())
    per_cat = {k: 1 for k in firsts}
    extra_budget = MAX_TOTAL - len(CATEGORIES)
    for c in candidates:
        if firsts[c.category] is c:
            continue
        if extra_budget <= 0 or per_cat[c.category] >= MAX_PER_CATEGORY:
            result.fixes.append(f"over budget, dropped {c.params['query']!r}")
            continue
        chosen.append(c)
        per_cat[c.category] += 1
        extra_budget -= 1

    for category in CATEGORIES:
        if category.key not in per_cat:
            chosen.append(next(f for f in fixed if f.category == category.key))
            result.fell_back.append(category.key)

    order = {c.key: i for i, c in enumerate(CATEGORIES)}
    result.searches = sorted(chosen, key=lambda p: order[p.category])  # stable within a category
    return result


def make_planner(spec: str, model_factory) -> Planner:
    """'fixed' or a model ID."""
    return FixedPlanner() if spec == FIXED else LLMPlanner(model_factory(spec, "planner"))
