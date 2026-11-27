"""Trace analysis: read a run's OpenTelemetry JSONL and turn it into eval numbers.

Every model call is a `kyb.llm.call` span (whatever the provider), every search a
`tavily.search` span, and the pipeline stages are `kyb.plan`, `kyb.extract.<category>` and
`kyb.judge.<name>`. So the trace alone answers: how many calls each model made, how often they
failed or needed a JSON repair, how long each call took, how many tokens it used, and where a
screen's time went.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .dashboard import short

# Stage colors: categorical slots 1-4 of the reference palette (light, dark). Every row of the
# timeline is also labeled, so color is never the only cue.
STAGE_COLORS = {
    "light": {"plan": "#257de0", "search": "#f47230", "read": "#00b894", "judge": "#f2a600"},
    "dark": {"plan": "#338ced", "search": "#e76624", "read": "#00a888", "judge": "#d28e05"},
}


def load_spans(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        rows.append({
            "name": s["name"], "trace_id": s["context"]["trace_id"], "span_id": s["context"]["span_id"],
            "parent_id": s.get("parent_id"), "start": pd.Timestamp(s["start_time"]),
            "end": pd.Timestamp(s["end_time"]), "status": s.get("status", {}).get("status_code", "UNSET"),
            "attrs": s.get("attributes", {}),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["duration_s"] = (df["end"] - df["start"]).dt.total_seconds()
    names = dict(zip(df["span_id"], df["name"], strict=True))
    df["parent_name"] = df["parent_id"].map(names)
    roots = df[df["name"] == "kyb.screen"].set_index("trace_id")["attrs"]
    df["screen_reader"] = df["trace_id"].map(lambda t: roots[t].get("kyb.model") if t in roots else None)
    df["company"] = df["trace_id"].map(
        lambda t: json.loads(roots[t]["kyb.company"])["name"] if t in roots and "kyb.company" in roots[t] else None)
    return df


def _role(parent: str | None) -> str:
    if not parent:
        return "other"
    if parent == "kyb.plan":
        return "planner"
    if parent.startswith("kyb.extract.") or parent == "kyb.merge":
        return "reader"
    if parent.startswith("kyb.judge."):
        return "judge"
    return "other"


def llm_calls(spans: pd.DataFrame) -> pd.DataFrame:
    """One row per model call."""
    calls = spans[spans["name"] == "kyb.llm.call"].copy()
    if calls.empty:
        return pd.DataFrame(columns=["role", "model", "latency_s", "prompt_tokens", "completion_tokens",
                                     "attempt", "parsed", "error", "trace_id", "start"])
    a = calls["attrs"]
    calls["role"] = calls["parent_name"].map(_role)
    calls["model"] = a.map(lambda x: x.get("kyb.model"))
    calls["latency_s"] = a.map(lambda x: x.get("kyb.latency_s")).fillna(calls["duration_s"])
    calls["prompt_tokens"] = a.map(lambda x: x.get("kyb.prompt_tokens", 0))
    calls["completion_tokens"] = a.map(lambda x: x.get("kyb.completion_tokens", 0))
    calls["attempt"] = a.map(lambda x: x.get("kyb.attempt", 1))
    calls["parsed"] = a.map(lambda x: x.get("kyb.parsed", False))
    calls["error"] = calls["status"] == "ERROR"
    return calls[["role", "model", "latency_s", "prompt_tokens", "completion_tokens", "attempt", "parsed",
                  "error", "trace_id", "start"]]


def llm_stats(calls: pd.DataFrame) -> pd.DataFrame:
    """Per (role, model): call counts, failures, repairs, latency percentiles, tokens, output speed."""
    if calls.empty:
        return pd.DataFrame()
    rows = []
    for (role, model), g in calls.groupby(["role", "model"], sort=False):
        ok = g[~g["error"]]
        secs = ok["latency_s"].sum()
        rows.append({
            "role": role, "model": model, "calls": len(g),
            "errors": int(g["error"].sum()),
            "json_repairs": int((g["attempt"] > 1).sum()),
            "unusable_output": int((~g["parsed"] & ~g["error"]).sum()),
            "p50_latency_s": ok["latency_s"].median() if len(ok) else None,
            "p95_latency_s": ok["latency_s"].quantile(0.95) if len(ok) else None,
            "mean_prompt_tokens": ok["prompt_tokens"].mean() if len(ok) else None,
            "mean_completion_tokens": ok["completion_tokens"].mean() if len(ok) else None,
            "output_tokens_per_s": ok["completion_tokens"].sum() / secs if secs else None,
        })
    order = {"planner": 0, "reader": 1, "judge": 2, "other": 3}
    return pd.DataFrame(rows).sort_values(["role", "model"], key=lambda c: c.map(order) if c.name == "role" else c)


def search_stats(spans: pd.DataFrame, live_only: bool = True) -> dict:
    """Tavily calls in these spans: count, results per search, latency, credits."""
    s = spans[spans["name"] == "tavily.search"]
    if live_only:
        s = s[~s["attrs"].map(lambda x: bool(x.get("kyb.replayed")))]
    if s.empty:
        return {"searches": 0}
    results = s["attrs"].map(lambda x: x.get("tavily.result_count", 0))
    return {
        "searches": len(s),
        "results_per_search": float(results.mean()),
        "empty_searches": int((results == 0).sum()),
        "p50_latency_s": float(s["duration_s"].median()),
        "p95_latency_s": float(s["duration_s"].quantile(0.95)),
        "credits": float(s["attrs"].map(lambda x: x.get("tavily.credits", 0) or 0).sum()),
    }


def stage_of(name: str) -> str | None:
    if name == "kyb.plan":
        return "plan"
    if name == "tavily.search":
        return "search"
    if name.startswith("kyb.extract.") or name == "kyb.merge":
        return "read"
    if name.startswith("kyb.judge.") or name == "kyb.check":
        return "judge"
    return None


def stage_times(spans: pd.DataFrame, trace_id: str) -> dict[str, float]:
    """Wall time per stage in one screen (stages that ran in parallel count once)."""
    t = spans[spans["trace_id"] == trace_id].copy()
    t["stage"] = t["name"].map(stage_of)
    out = {}
    for stage, g in t.dropna(subset=["stage"]).groupby("stage"):
        out[stage] = (g["end"].max() - g["start"].min()).total_seconds()
    return out


def waterfall(spans: pd.DataFrame, trace_id: str, mode: str = "light") -> go.Figure:
    """Timeline of one screen: plan, each search, each category read, and the model calls inside."""
    from .dashboard import PALETTES, _layout

    p, colors = PALETTES[mode], STAGE_COLORS[mode]
    t = spans[spans["trace_id"] == trace_id].sort_values("start")
    t0 = t["start"].min()
    # Each stage span followed by the model calls made inside it.
    calls = {pid: g for pid, g in t[t["name"] == "kyb.llm.call"].groupby("parent_id")}
    ordered = []
    for _, r in t[t["name"] != "kyb.llm.call"].iterrows():
        ordered.append(r)
        if r["span_id"] in calls:
            ordered.extend(c for _, c in calls[r["span_id"]].iterrows())
    rows = []
    for r in ordered:
        stage = stage_of(r["name"])
        if r["name"] == "kyb.llm.call":
            stage = stage_of(r["parent_name"] or "") or "read"
            a = r["attrs"]
            label = f"↳ model call · {short(a.get('kyb.model', ''))}" + (
                f" · attempt {a['kyb.attempt']}" if a.get("kyb.attempt", 1) > 1 else "")
        elif stage == "search":
            label = f"search · {r['attrs'].get('kyb.category', '')}"
        elif r["name"] == "kyb.merge":
            label = "merge duplicates"
        elif r["name"] == "kyb.check":
            label = "source check"
        elif stage == "read":
            label = f"read · {r['name'].removeprefix('kyb.extract.')}"
        elif stage in ("plan", "judge"):
            label = r["name"].removeprefix("kyb.").replace(".", " · ")
        else:
            continue
        rows.append({"label": label, "stage": stage, "start": (r["start"] - t0).total_seconds(),
                     "dur": max(r["duration_s"], 0.002), "error": r["status"] == "ERROR"})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["row"] = range(len(d))
    replayed = bool(t[t["name"] == "tavily.search"]["attrs"].map(lambda a: bool(a.get("kyb.replayed"))).any())
    names = {"search": "search (replayed from the recording)" if replayed else "search"}
    fig = go.Figure()
    for stage in ("plan", "search", "read", "judge"):
        g = d[d["stage"] == stage] if not d.empty else d
        if g.empty:
            continue
        fig.add_trace(go.Bar(
            name=names.get(stage, stage), y=g["row"],
            x=g["dur"], base=g["start"], orientation="h", marker={"color": colors[stage], "cornerradius": 3},
            customdata=g[["label"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>starts %{base:.2f} s · takes %{x:.2f} s<extra></extra>"))
    fig.update_layout(barmode="overlay", bargap=0.35, showlegend=True,
                      legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0,
                              "font": {"color": p["text2"]}})
    fig.update_yaxes(tickvals=list(range(len(d))), ticktext=list(d["label"]) if not d.empty else [],
                     autorange="reversed", showgrid=False, tickfont={"color": p["text"], "size": 11})
    fig.update_xaxes(title={"text": "seconds from start"}, rangemode="tozero")
    fig = _layout(fig, p, "Where the time went", 120 + 20 * len(d))
    fig.update_layout(showlegend=True)
    return fig


def latency_strip(calls: pd.DataFrame, role: str, mode: str = "light") -> go.Figure:
    """Every call's latency for one role, one row per model; the bar marks the median."""
    from .dashboard import PALETTES, _layout

    p = PALETTES[mode]
    c = calls[(calls["role"] == role) & ~calls["error"]].copy()
    c["m"] = c["model"].map(short)
    order = c.groupby("m")["latency_s"].median().sort_values().index.tolist()
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=c["latency_s"], y=c["m"], orientation="h", boxpoints="all", jitter=0.4, pointpos=0,
        fillcolor="rgba(0,0,0,0)", line={"color": p["axis"], "width": 1},
        marker={"color": p["series"], "size": 8, "opacity": 0.7, "line": {"color": p["surface"], "width": 1}},
        hovertemplate="%{y}<br>%{x:.2f} s<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=order, showgrid=False, tickfont={"color": p["text"]})
    fig.update_xaxes(title={"text": "seconds per call"}, rangemode="tozero")
    return _layout(fig, p, f"Latency of every {role} call", 120 + 48 * max(len(order), 1))
