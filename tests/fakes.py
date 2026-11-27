"""Offline stand-ins for Tavily and the chat model, shaped like the real responses."""

from __future__ import annotations

import hashlib
import json
import re

from kyb.llm import ChatResult


class FakeTavily:
    """Deterministic Tavily-shaped responses. Every query returns a page shared by all
    queries (to exercise dedupe), a page specific to the query, and a namesake page."""

    mode = "live"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, params: dict) -> dict:
        self.calls.append(params)
        q = params["query"]
        h = hashlib.sha1(q.encode()).hexdigest()[:8]
        return {
            "query": q,
            "results": [
                {"title": "Acme Corp profile", "url": "https://news.example.com/acme-profile?utm_source=x",
                 "content": f"Acme Corp overview relevant to: {q}", "score": 0.9,
                 "published_date": "2025-03-01"},
                {"title": f"Acme Corp fined by regulator ({h})", "url": f"https://news.example.com/{h}",
                 "content": f"Regulator fined Acme Corp $5 million in 2024. Query: {q}", "score": 0.8,
                 "published_date": "2024-06-10"},
                {"title": "Acme Foods recall lawsuit", "url": "https://other.example.com/acme-foods",
                 "content": "Acme Foods, a grocery chain unrelated to Acme Corp, faces a lawsuit.",
                 "score": 0.5, "published_date": "2025-01-20"},
            ],
            "response_time": 1.2,
            "usage": {"credits": 2},
            "request_id": f"req-{h}",
        }


class FakeModel:
    """Cites the first source it is shown, plus a made-up ID; adds a finding that cites
    only the made-up ID. Optionally returns garbage on the first call."""

    def __init__(self, model: str = "fake/model", bad_first: bool = False, always_bad: bool = False):
        self.model = model
        self.bad_first = bad_first
        self.always_bad = always_bad
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> ChatResult:
        self.calls.append(messages)
        if self.always_bad or (self.bad_first and len(messages) == 2):
            return ChatResult("Sure! Here are the findings: none", 100, 10, 0.01)
        if "for duplicates" in messages[0]["content"]:   # the merge step: group identical summaries
            by_summary: dict[str, list[str]] = {}
            for fid, summary in re.findall(r"^- (\S+) \(.*?\): (.*)$", messages[-1]["content"], re.M):
                by_summary.setdefault(summary, []).append(fid)
            groups = [ids for ids in by_summary.values() if len(ids) > 1] + [["made-up-1", "made-up-2"]]
            return ChatResult(json.dumps({"groups": groups}), 300, 30, 0.02)
        ids = re.findall(r"^\[(S\d+)\]", messages[1]["content"], re.M)
        payload = {
            "findings": [
                {"summary": "Regulator fined Acme Corp $5 million.", "event_date": "2024-06",
                 "severity": "HIGH", "status": "confirmed", "source_ids": [ids[0], "S999"]},
                {"summary": "Something unsupported.", "severity": "low", "source_ids": ["S999"]},
            ],
            "rejected_sources": [{"source_id": ids[-1], "reason": "different_entity"}],
        }
        text = "<think>reasoning here</think>\n```json\n" + json.dumps(payload) + "\n```"
        return ChatResult(text, 1000, 200, 0.05)


class FakeJudge:
    model = "fake/judge"

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> ChatResult:
        self.calls.append(messages)
        system, user = messages[0]["content"], messages[1]["content"]
        if "against a rubric" in system:
            out = {"scores": {"citation": {"score": 4, "reason": "ok"}, "scope": {"score": 3},
                              "relevance": {"score": 5}, "specificity": {"score": 4},
                              "calibration": {"score": None}, "made_up_key": {"score": 1}}}
        elif "compare two KYB" in system:
            # Prefers the report with more findings, so verdicts are consistent across order.
            a, b = user.split("=====")
            na, nb = len(re.findall(r"^\[", a, re.M)), len(re.findall(r"^\[", b, re.M))
            out = {"winner": "A" if na > nb else "B" if nb > na else "tie", "reason": "count"}
        else:
            ids = re.findall(r"^Finding (\S+) \(", user, re.M)
            out = {"judgments": [{"finding_id": f, "about_target": True, "supported": i == 0,
                                  "reason": "test"} for i, f in enumerate(ids)]}
        return ChatResult(json.dumps(out), 500, 50, 0.02)


class FakePlannerModel:
    """A 'thinking' planner: wraps its JSON in reasoning tags and breaks a few rules
    so validation has something to fix."""

    def __init__(self, model: str = "fake/planner", garbage: bool = False):
        self.model = model
        self.garbage = garbage
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> ChatResult:
        self.calls.append(messages)
        if self.garbage:
            return ChatResult("I would search for news.", 300, 40, 0.1)
        q = '"Acme Corp"'
        searches = [
            {"category": "regulatory", "query": f"{q} consent order penalty"},
            {"category": "regulatory", "query": f"{q} enforcement", "regulator_sites_only": True},
            {"category": "litigation", "query": "Acme lawsuit", "exact_match": True},
            {"category": "esg", "query": f"{q} emissions"},
            {"category": "data_breach", "query": f"{q} data breach", "topic": "news"},
            {"category": "data_breach", "query": f"{q} ransomware"},
            {"category": "data_breach", "query": f"{q} ransomware"},
            {"category": "data_breach", "query": f"{q} hack"},
            {"category": "data_breach", "query": f"{q} leak", "topic": "finance"},
        ]
        text = "<think>Acme is a bank, so...</think>" + json.dumps({"searches": searches})
        return ChatResult(text, 800, 1500, 2.0)
