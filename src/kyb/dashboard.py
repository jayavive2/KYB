"""Reader-sweep dashboard: how each reader trades quality against cost and latency when the
planner and its recorded search results are held constant.

Pure functions that turn a results folder into data frames and Plotly figures, so they can be
tested without Streamlit. Colors follow one rule each: identity by direct label (never by hue,
since five models exceed what a categorical palette can keep apart on a scatter), magnitude by a
single blue ramp, and Pareto-frontier membership by blue versus gray.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

PALETTES = {
    "light": {"text": "#191919", "text2": "#666666", "grid": "#e7e7e7", "axis": "#c2c2c2",
              "series": "#257de0", "context": "#b3b3b3", "surface": "#ffffff", "high": "darker",
              "ramp": ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]},
    "dark": {"text": "#efefef", "text2": "#9c9c9c", "grid": "#303030", "axis": "#474747",
             "series": "#338ced", "context": "#5c5c5c", "surface": "#191919", "high": "brighter",
             "ramp": ["#0e3d7e", "#1757ac", "#2370cd", "#338ced", "#6aabf1", "#9dc8f7", "#cce3fd"]},
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def short(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def wrap(name: str, width: int = 16) -> str:
    """Break a long model name at hyphens so point labels stay narrow."""
    lines, cur = [], ""
    for part in name.split("-"):
        piece = part if not cur else f"{cur}-{part}"
        if len(piece) > width and cur:
            lines.append(cur + "-")
            cur = part
        else:
            cur = piece
    return "<br>".join(lines + [cur])


def _nice_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 ticks between lo and hi, for log axes."""
    import math

    ticks, exp = [], math.floor(math.log10(lo))
    while 10 ** exp <= hi * 10:
        for m in (1, 2, 5):
            v = m * 10 ** exp
            if lo / 1.5 <= v <= hi * 1.5:
                ticks.append(v)
        exp += 1
    return ticks


# ---------------------------------------------------------------- data

def load_sweep(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(one row per reader, one row per reader x company, experiment.json)."""
    exp = json.loads((run_dir / "experiment.json").read_text())
    rows = []
    for m in json.loads((run_dir / "metrics.json").read_text()):
        row = {k: v for k, v in m.items() if k not in ("rubric", "pairwise")}
        row.update({f"rubric_{k}": v for k, v in (m.get("rubric") or {}).items()})
        row.update({f"pairwise_{k}": v for k, v in (m.get("pairwise") or {}).items()})
        row["model"] = short(m["reader"])
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values("quality", ascending=False, na_position="last")
    metrics = metrics.reset_index(drop=True)
    rubric_path = run_dir / "rubric.csv"
    per_case = pd.read_csv(rubric_path) if rubric_path.exists() and rubric_path.stat().st_size else pd.DataFrame()
    return metrics, per_case, exp


def pareto_front(df: pd.DataFrame, cost_col: str, quality_col: str = "quality") -> pd.Series:
    """True where no other reader is at least as cheap (or fast) and at least as good, and
    strictly better on one of the two."""
    x, y = df[cost_col], df[quality_col]
    on = []
    for i in df.index:
        if pd.isna(x[i]) or pd.isna(y[i]):
            on.append(False)
            continue
        dominated = any(
            j != i and not pd.isna(x[j]) and not pd.isna(y[j])
            and x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i])
            for j in df.index)
        on.append(not dominated)
    return pd.Series(on, index=df.index)


def composite(df: pd.DataFrame, w_quality: float, w_cost: float, w_latency: float) -> pd.DataFrame:
    """Blend quality, cost and latency into one score in [0, 1].

    quality: rubric quality mapped from 1-5 to 0-1. cost and latency: the best reader's value
    divided by this reader's, so twice the cost scores 0.5. Weights renormalize over the
    measures a reader has (a reader with no price is scored on quality and latency only)."""
    out = df.copy()
    out["quality_score"] = (out["quality"] - 1) / 4
    for col, name in (("cost_usd_per_company", "cost_score"), ("latency_s_per_company", "latency_score")):
        best = out[col].min()
        out[name] = best / out[col] if pd.notna(best) and best > 0 else 1.0
    parts = [("quality_score", w_quality), ("cost_score", w_cost), ("latency_score", w_latency)]

    def blend(row: pd.Series) -> float | None:
        used = [(w, row[c]) for c, w in parts if w > 0 and pd.notna(row[c])]
        total = sum(w for w, _ in used)
        return sum(w * v for w, v in used) / total if total else None

    out["composite"] = out.apply(blend, axis=1)
    return out.sort_values("composite", ascending=False, na_position="last")


def runs_table(df: pd.DataFrame, rubric_keys: list[str]) -> pd.DataFrame:
    """The W&B-style runs table: one row per reader, columns grouped by evaluation layer."""
    cols = {"model": "Reader", "events": "Events"}
    cols |= {"invalid_citations": "L1 invalid citations", "json_first_try": "L1 JSON first try",
             "categories_failed": "L1 categories failed"}
    cols |= {"quality": "L2 quality"} | {f"rubric_{k}": f"L2 {k}" for k in rubric_keys}
    cols |= {"supported_on_target": "L2 supported on target", "namesake_errors": "L2 namesake errors"}
    cols |= {"cost_usd_per_company": "L3 $ / company", "latency_s_per_company": "L3 sec / company",
             "input_tokens": "L3 input tokens", "output_tokens": "L3 output tokens"}
    present = {k: v for k, v in cols.items() if k in df.columns}
    return df[list(present)].rename(columns=present)


# ---------------------------------------------------------------- figures

def _layout(fig: go.Figure, p: dict, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": p["text"]}, "x": 0, "xanchor": "left"},
        height=height, margin={"l": 8, "r": 16, "t": 44, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": p["text2"], "size": 12}, showlegend=False,
        hoverlabel={"font": {"family": FONT}},
    )
    axis = {"gridcolor": p["grid"], "gridwidth": 1, "linecolor": p["axis"], "zeroline": False, "automargin": True,
            "tickfont": {"color": p["text2"]}, "title": {"font": {"color": p["text2"]}}}
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def ranking_bar(scored: pd.DataFrame, p: dict) -> go.Figure:
    d = scored.dropna(subset=["composite"]).iloc[::-1]  # best at the top
    fig = go.Figure(go.Bar(
        x=d["composite"], y=d["model"], orientation="h",
        marker={"color": p["series"], "cornerradius": 4},
        text=[f"{v:.2f}" for v in d["composite"]], textposition="outside",
        textfont={"color": p["text"]}, cliponaxis=False,
        customdata=d[["quality", "cost_usd_per_company", "latency_s_per_company"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>score %{x:.2f}<br>quality %{customdata[0]:.2f} / 5"
                      "<br>$%{customdata[1]:.4f} per company<br>%{customdata[2]:.1f} s per company<extra></extra>",
    ))
    fig.update_layout(bargap=0.6)
    fig.update_xaxes(range=[0, 1.12], showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, tickfont={"color": p["text"]})
    return _layout(fig, p, "Score with your weights (0-1)", 60 + 44 * len(d))


def tradeoff_scatter(df: pd.DataFrame, x_col: str, x_title: str, p: dict, hover_fmt: str,
                     tick_fmt=lambda v: f"{v:g}") -> go.Figure:
    d = df.dropna(subset=[x_col, "quality"]).copy()
    d["front"] = pareto_front(d, x_col)
    fig = go.Figure()
    front = d[d["front"]].sort_values(x_col)
    if len(front) > 1:  # the frontier: no reader beats these on both axes
        fig.add_trace(go.Scatter(x=front[x_col], y=front["quality"], mode="lines", hoverinfo="skip",
                                 line={"color": p["series"], "width": 2}, opacity=0.35))
    leftmost = d[x_col].idxmin() if len(d) else None
    for on, color in ((False, p["context"]), (True, p["series"])):
        g = d[d["front"] == on]
        # Frontier labels above the line, dominated ones below; the leftmost label leans right off the axis.
        pos = ["bottom right" if i == leftmost else "top center" if on else "bottom center" for i in g.index]
        fig.add_trace(go.Scatter(
            x=g[x_col], y=g["quality"], mode="markers+text", text=[wrap(m) for m in g["model"]],
            textposition=pos,
            textfont={"color": p["text"], "size": 11}, cliponaxis=False, customdata=g["model"],
            marker={"size": 13, "color": color, "line": {"color": p["surface"], "width": 2}},
            hovertemplate=f"<b>%{{customdata}}</b><br>quality %{{y:.2f}} / 5<br>{hover_fmt}"
                          f"<br>{'on' if on else 'off'} the frontier<extra></extra>"))
    lo, hi = (d[x_col].min(), d[x_col].max()) if len(d) else (1, 1)
    if lo > 0 and hi / lo > 8:
        import math

        pad = (math.log10(hi) - math.log10(lo)) * 0.12
        ticks = _nice_ticks(lo, hi)
        fig.update_xaxes(type="log", range=[math.log10(lo) - pad, math.log10(hi) + pad],
                         tickvals=ticks, ticktext=[tick_fmt(t) for t in ticks])
    else:
        pad = (hi - lo) * 0.12 or abs(hi) * 0.1 or 1
        fig.update_xaxes(range=[max(0, lo - pad), hi + pad])
    fig.update_xaxes(title={"text": x_title})
    fig.update_yaxes(title={"text": "Quality (rubric, 1-5)"}, range=[0.8, 5.45], dtick=1)
    return _layout(fig, p, f"Quality vs {x_title.split(' (')[0].lower()}", 380)


def parallel_coordinates(df: pd.DataFrame, rubric_keys: list[str], p: dict) -> go.Figure:
    d = df.reset_index(drop=True)
    pos = [len(d) - 1 - i for i in d.index]  # first row (highest quality) at the top of the axis
    dims = [{"label": "reader", "values": pos, "tickvals": pos, "ticktext": list(d["model"])}]
    measures = (("cost_usd_per_company", "cost ($)", "$.3f"), ("latency_s_per_company", "latency (s)", ".1f"))
    for col, label, fmt in measures:
        if d[col].notna().any():
            dims.append({"label": label, "values": d[col].fillna(d[col].max()), "tickformat": fmt})
    for k in rubric_keys:
        col = f"rubric_{k}"
        if col in d and d[col].notna().any():
            dims.append({"label": k, "values": d[col].fillna(1), "range": [1, 5]})
    dims.append({"label": "quality", "values": d["quality"].fillna(1), "range": [1, 5]})
    ramp = p["ramp"][2:]  # keep the lightest steps off the plot so every line stays visible
    fig = go.Figure(go.Parcoords(
        line={"color": d["quality"].fillna(1), "cmin": 1, "cmax": 5, "showscale": False,
              "colorscale": [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]},
        dimensions=dims, labelfont={"color": p["text"], "size": 12},
        tickfont={"color": p["text2"], "size": 11}, rangefont={"color": "rgba(0,0,0,0)", "size": 1},  # ticks say it
    ))
    fig = _layout(fig, p, f"Every reader across every measure ({p['high']} line = higher quality)", 400)
    fig.update_layout(margin={"l": 210, "r": 60, "t": 84, "b": 24})
    return fig


def heatmap(matrix: pd.DataFrame, title: str, p: dict, value_name: str, zmin: float = 1, zmax: float = 5,
            fmt: str = ".2f") -> go.Figure:
    ramp = p["ramp"]
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(dtype=float), x=list(matrix.columns), y=list(matrix.index), zmin=zmin, zmax=zmax,
        colorscale=[[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)], xgap=2, ygap=2,
        colorbar={"title": {"text": value_name, "font": {"color": p["text2"]}}, "thickness": 10,
                  "tickfont": {"color": p["text2"]}, "outlinewidth": 0},
        hovertemplate=f"<b>%{{y}}</b> · %{{x}}<br>{value_name} %{{z:{fmt}}}<extra></extra>", hoverongaps=False,
    ))
    fig.update_xaxes(showgrid=False, side="bottom", tickfont={"color": p["text"]})
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont={"color": p["text"]})
    return _layout(fig, p, title, 90 + 46 * len(matrix))


def rubric_matrix(df: pd.DataFrame, rubric_keys: list[str]) -> pd.DataFrame:
    cols = [f"rubric_{k}" for k in rubric_keys if f"rubric_{k}" in df] + ["quality"]
    m = df.set_index("model")[cols].astype(float)
    return m.rename(columns=lambda c: c.removeprefix("rubric_"))


def company_matrix(per_case: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Reader x company quality, rows in `order` (model short names)."""
    if per_case.empty or "quality" not in per_case:
        return pd.DataFrame()
    m = per_case.pivot_table(index="arm", columns="case", values="quality", aggfunc="mean")
    m.index = [short(i) for i in m.index]
    return m.reindex([o for o in order if o in m.index])
