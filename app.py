"""Streamlit front end for the KYB screener.

    uv run streamlit run app.py

Screen       screen one company with the default planner and reader; see its trace; optionally
             compare other readers on the same search results (reader eval on one screen).
Search eval  eval 1: did search find the known events in evals/companies.yaml? Per planner.
Reader eval  eval 2: search held constant, readers swapped, rubric judge; dashboard and trace analysis.
Results      per-run detail: summaries, two reports side by side, every judge verdict.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from kyb import dashboard as dash
from kyb import evaluate as ev
from kyb import factory
from kyb import traces as T
from kyb.judge import load_rubric
from kyb.planner import FIXED, make_planner
from kyb.report import render_markdown, to_json
from kyb.schema import Company
from kyb.screen import run_screen
from kyb.tracing import init_tracing, shutdown_tracing

load_dotenv()

CONFIG = Path("config/models.toml")
RUBRIC = Path("config/rubric.yaml")
DATASET = Path("evals/companies.yaml")
RECORDINGS = Path("recordings")
RESULTS = Path("results")
RUNS = Path("runs")

st.set_page_config(page_title="KYB screener", layout="wide")
cfg = ev.load_config(CONFIG)
prices = cfg.get("prices", {})
defaults = cfg.get("defaults", {})
planner_options = list(dict.fromkeys([FIXED] + cfg["eval"]["planners"] + [defaults.get("planner", FIXED)]))
reader_options = list(dict.fromkeys(cfg["eval"]["readers"] + [defaults.get("reader", cfg["eval"]["readers"][0])]))


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def with_extra(selected: list[str], extra: str) -> list[str]:
    return selected + [m.strip() for m in extra.split(",") if m.strip() and m.strip() not in selected]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def missing_keys(*names: str) -> list[str]:
    return [n for n in names if not os.getenv(n)]


CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1320px;}
.kyb-hero {padding: 0.4rem 0 0.6rem 0;}
.kyb-eyebrow {text-transform: uppercase; letter-spacing: .09em; font-size: .72rem; font-weight: 650;
              color: #257de0; margin-bottom: .35rem;}
.kyb-hero h1 {margin: 0; padding: 0; font-size: 2.1rem; letter-spacing: -0.01em;}
.kyb-hero p {margin: .35rem 0 0 0; opacity: .72; font-size: 1rem; max-width: 60rem;}
.kyb-stat-label {font-size: .8rem; font-weight: 600; opacity: .68; margin-bottom: .15rem;}
.kyb-stat-value {font-size: 1.75rem; font-weight: 650; line-height: 1.15; letter-spacing: -0.01em;}
.kyb-stat-sub {font-size: .82rem; opacity: .68; margin-top: .2rem; white-space: nowrap; overflow: hidden;
               text-overflow: ellipsis;}
button[data-baseweb="tab"] p {font-size: 1rem; font-weight: 600;}
div[data-testid="stExpander"] details summary p {font-weight: 600;}
</style>
"""


def stat_card(col, label: str, value: str, sub: str = "") -> None:
    with col.container(border=True):
        st.markdown(f'<div class="kyb-stat-label">{label}</div><div class="kyb-stat-value">{value}</div>'
                    f'<div class="kyb-stat-sub" title="{sub}">{sub or "&nbsp;"}</div>', unsafe_allow_html=True)


def demote(markdown: str, levels: int = 2) -> str:
    """Report headings are sized for a standalone page; shrink them inside the app."""
    import re

    return re.sub(r"^(#{1,4}) ", lambda m: "#" * (len(m.group(1)) + levels) + " ", markdown, flags=re.M)


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


@st.cache_data(show_spinner=False)
def cached_spans(path: str, mtime: float) -> pd.DataFrame:
    return T.load_spans(Path(path))


def spans_of(path: Path) -> pd.DataFrame:
    return cached_spans(str(path), path.stat().st_mtime) if path.exists() else pd.DataFrame()


def fmt_stats(stats: pd.DataFrame) -> pd.DataFrame:
    return stats.rename(columns={
        "p50_latency_s": "p50 s", "p95_latency_s": "p95 s", "mean_prompt_tokens": "prompt tokens / call",
        "mean_completion_tokens": "output tokens / call", "output_tokens_per_s": "output tokens / s",
        "json_repairs": "JSON repairs", "unusable_output": "unusable output"}).round(
        {"p50 s": 2, "p95 s": 2, "prompt tokens / call": 0, "output tokens / call": 0, "output tokens / s": 1})


def runs_of(kind: str) -> list[Path]:
    return [r for r in sorted(RESULTS.glob("*"), reverse=True) if (r / "experiment.json").exists()
            and json.loads((r / "experiment.json").read_text())["kind"] == kind]


def reader_runs() -> list[Path]:
    return [r for r in sorted(RESULTS.glob("*"), reverse=True) if (r / "metrics.json").exists()
            and json.loads((r / "experiment.json").read_text())["kind"] == "readers"]


def run_label(run_dir: Path) -> str:
    exp = json.loads((run_dir / "experiment.json").read_text())
    companies = list((exp.get("companies") or {}).values()) or exp["cases"]
    what = companies[0] if len(companies) == 1 else f"{len(companies)} companies"
    return f"{run_dir.name} · {what} · {len(exp['arms'])} readers"


def show_results(run_dir: Path, where: str) -> None:
    """`where` prefixes widget keys: the same run can be on screen in two tabs."""
    k = f"{where}_{run_dir.name}"
    exp = json.loads((run_dir / "experiment.json").read_text())
    arms = [a["label"] for a in exp["arms"]]
    st.markdown((run_dir / "summary.md").read_text())

    summary = read_csv(run_dir / "summary.csv")
    rubric_cols = [c for c in summary.columns if c.startswith("rubric_")]
    if rubric_cols:
        st.subheader("Rubric scores (1-5)")
        chart = summary.set_index("arm")[rubric_cols].apply(pd.to_numeric, errors="coerce")
        st.bar_chart(chart.rename(columns=lambda c: c.removeprefix("rubric_")), stack=False)

    st.subheader("Compare two reports")
    c0, c1, c2 = st.columns([1, 2, 2])
    case = c0.selectbox("Company", exp["cases"], key=f"cmp_case_{k}")
    left = c1.selectbox("Left", arms, index=0, key=f"cmp_left_{k}")
    right = c2.selectbox("Right", arms, index=min(1, len(arms) - 1), key=f"cmp_right_{k}")
    pairwise = read_csv(run_dir / "pairwise.csv")
    if not pairwise.empty and left != right:
        row = pairwise[(pairwise["case"] == case) & pairwise["first"].isin([left, right])
                       & pairwise["second"].isin([left, right])]
        if not row.empty:
            r = row.iloc[0]
            st.info(f"Judge's pick: **{r['winner']}** "
                    f"({'held' if r['consistent_across_order'] else 'did not hold'} when the order was swapped). "
                    f"{r['reason_first_as_A']}")
    rubric = read_csv(run_dir / "rubric.csv")
    for col, arm in ((c1, left), (c2, right)):
        report = run_dir / "cases" / case / f"{ev.slug(arm)}.md"
        if not rubric.empty:
            r = rubric[(rubric["arm"] == arm) & (rubric["case"] == case)]
            if not r.empty and exp["rubric"]:
                row = r.iloc[0]
                col.caption(" · ".join(f"{k}: {row.get(k, '–')}" for k in ["quality"] + exp["rubric"]))
        col.markdown(demote(report.read_text()) if report.exists() else "_This arm failed for this company._")

    st.subheader("Judge verdicts")
    st.caption("Label a sample in judgments.csv (human_* columns), then run `kyb eval agreement`.")
    for name in ("judgments", "recall", "coverage", "rubric", "pairwise"):
        df = read_csv(run_dir / f"{name}.csv")
        if df.empty:
            continue
        with st.expander(f"{name}.csv ({len(df)} rows)"):
            st.dataframe(df)
            st.download_button(f"Download {name}.csv", (run_dir / f"{name}.csv").read_bytes(),
                               file_name=f"{run_dir.name}-{name}.csv", key=f"dl_{name}_{k}")
    st.caption(f"Trace: {run_dir / 'trace.jsonl'}")


def plural(n: int, word: str, many: str = "") -> str:
    return f"{n} {word if n == 1 else many or word + 's'}"


def show_screen(run_dir: Path) -> None:
    data = json.loads((run_dir / "findings.json").read_text())
    findings, cats = data["findings"], data.get("categories") or {}
    counts = f"{plural(len(findings), 'finding')} from {plural(len(data['sources']), 'source')}"
    st.success(f"Screened **{data['company']['name']}** with planner `{data['planner']}` and reader "
               f"`{data['model']}`: {counts}.")
    tiles = st.columns(4)
    flagged = sorted({f["category"] for f in findings})
    stat_card(tiles[0], "Findings", str(len(findings)),
              f"{sum(f['severity'] == 'high' for f in findings)} high severity" if findings else "nothing adverse")
    stat_card(tiles[1], "Categories flagged", f"{len(flagged)} of {len(cats) or 7}", ", ".join(flagged) or "none")
    searches = plural(len(data["searches"]), "search", "searches")
    stat_card(tiles[2], "Sources reviewed", str(len(data["sources"])), searches)
    not_assessed = [c for c, r in cats.items() if not r.get("parsed")]
    invalid = sum(r.get("invalid_citations", 0) for r in cats.values())
    stat_card(tiles[3], "Output checks", "✓ passed" if not (not_assessed or invalid) else "⚠ see report",
              f"{invalid} invalid citations · {len(not_assessed)} not assessed")
    if data.get("plan_fixes"):
        st.warning("Plan fixes: " + "; ".join(data["plan_fixes"]))
    with st.expander(f"Plan: {len(data['searches'])} searches"):
        st.dataframe(pd.DataFrame(data["searches"]), hide_index=True)
    with st.expander("Report", expanded=True):
        st.markdown(demote((run_dir / "report.md").read_text()))
    spans = spans_of(run_dir / "trace.jsonl")
    if not spans.empty:
        trace_id = spans.loc[spans["name"] == "kyb.screen", "trace_id"].iloc[0]
        with st.expander("Trace: where the time went and every model call"):
            stages = T.stage_times(spans, trace_id)
            cols = st.columns(len(stages) or 1)
            for col, (stage, secs) in zip(cols, sorted(stages.items(), key=lambda kv: "psr".index(kv[0][0])),
                                          strict=False):
                stat_card(col, f"{stage.capitalize()} time", f"{secs:.1f} s")
            st.plotly_chart(T.waterfall(spans, trace_id, theme_mode()), theme=None, key="screen_waterfall")
            st.dataframe(fmt_stats(T.llm_stats(T.llm_calls(spans))), hide_index=True)
            search = T.search_stats(spans)
            if search.get("searches"):
                st.caption(f"Tavily: {search['searches']} searches, {search['results_per_search']:.1f} results "
                           f"each, {search['empty_searches']} empty, p50 {search['p50_latency_s']:.2f} s, "
                           f"{search['credits']:g} credits.")
    st.caption(f"Trace (replayable): {run_dir / 'trace.jsonl'}")


def show_search(run_dir: Path, where: str) -> None:
    """Eval 1 results: which known events search found, per planner."""
    k = f"{where}_{run_dir.name}"
    exp = json.loads((run_dir / "experiment.json").read_text())
    metrics = json.loads((run_dir / "search_metrics.json").read_text())
    rows = read_csv(run_dir / "search_eval.csv")
    p = dash.PALETTES[theme_mode()]
    st.caption(f"{len(exp['cases'])} companies, {metrics[0]['events']} known events · judge `{exp['judge']}`. "
               "An event counts as found if any recorded search result describes it. No reader is involved.")
    if exp.get("judge_failures"):
        st.warning(f"{len(exp['judge_failures'])} judge calls failed; affected events count as missed.")
    tiles = st.columns(min(len(metrics), 4))
    for tile, m in zip(tiles, metrics, strict=False):
        stat_card(tile, f"Coverage · {dash.short(m['planner'])}", f"{(m['coverage'] or 0):.0%}",
                  f"{m['found']} of {m['events']} events · {m['searches_per_company']:.1f} searches per company")

    rows["event"] = rows["company"] + " · " + rows["event_id"]
    found = rows.pivot_table(index="event", columns="planner", values="found", aggfunc="max")
    st.subheader("Found or missed, event by event")
    labels = found.map(lambda v: "✓ found" if v else "✗ missed")
    st.dataframe(labels.style.map(lambda v: "color: #0a8a1e; font-weight: 600" if v.startswith("✓")
                                  else "color: #dc3c3d; font-weight: 600"), key=f"found_{k}")

    cats = sorted({c for m in metrics for c in m["by_category"]})
    by_cat = pd.DataFrame({dash.short(m["planner"]): [
        m["by_category"][c]["found"] / m["by_category"][c]["events"] if c in m["by_category"] else None
        for c in cats] for m in metrics}, index=cats).T
    left, right = st.columns(2)
    left.plotly_chart(dash.heatmap(by_cat, "Coverage by category", p, "coverage", zmin=0, zmax=1, fmt=".0%"),
                      theme=None, key=f"cats_{k}")
    table = pd.DataFrame([{
        "planner": m["planner"], "searches / company": m["searches_per_company"],
        "results / search": m["results_per_search"], "empty searches": m["empty_searches"],
        "search p50 s": m["search_p50_s"], "search p95 s": m["search_p95_s"], "credits": m["tavily_credits"],
        "planner calls": m["planner_calls"], "planner p50 s": m["planner_p50_s"],
        "planner output tokens": m["planner_output_tokens"]} for m in metrics]).round(2)
    right.markdown("**From the recordings' traces**")
    right.dataframe(table, hide_index=True, key=f"stats_{k}")

    with st.expander("Evidence for each event: matching sources and the queries that found them"):
        st.dataframe(rows[["planner", "company", "event_id", "found", "source_titles", "source_urls",
                           "found_by_queries", "found_in_own_category", "description"]], hide_index=True)
    missed = rows[~rows["found"].astype(bool)]
    if not missed.empty:
        st.markdown("**Missed:** " + "; ".join(f"{dash.short(r.planner)} · {r.event}" for r in missed.itertuples()))
    st.caption(f"Results folder: {run_dir} · search_eval.csv has a human_agree column for checking the judge.")


def show_dashboard(run: Path, where: str) -> None:
    """The sweep view for one reader comparison. `where` keeps widget keys unique per tab."""
    k = f"{where}_{run.name}"
    df, per_case, exp = dash.load_sweep(run)
    keys = exp.get("rubric", [])
    if df["quality"].isna().all():
        if not keys:
            st.warning("This comparison ran without rubric scores, so it has no quality measure.")
        else:
            failures = exp.get("judge_failures", [])
            st.error(f"The judge returned no usable rubric scores ({len(failures)} failed calls), so there is no "
                     f"quality measure. Check that `{exp['judge']}` is available and returns JSON. "
                     + (f"First error: {failures[0]}" if failures else ""))
        return
    p = dash.PALETTES[theme_mode()]
    companies = list((exp.get("companies") or {}).values()) or exp["cases"]
    what = companies[0] if len(companies) == 1 else f"{len(companies)} companies"
    planner = df["planner"].iloc[0]
    held = (f"Each screen's own plan and search results for {what} are" if planner == ev.AS_RECORDED
            else f"Planner `{planner}` and its search results for {what} are")
    st.caption(f"{held} held constant; each reader replays them. Judge `{exp['judge']}`. "
               "Layers: L1 deterministic checks · L2 judge · L3 cost and latency.")

    with st.container(border=True):
        st.markdown("**How much each measure matters to you**")
        c1, c2, c3 = st.columns(3)
        w_q = c1.slider("Quality", 0.0, 1.0, 0.6, 0.05, key=f"w_quality_{k}")
        w_c = c2.slider("Cost", 0.0, 1.0, 0.25, 0.05, key=f"w_cost_{k}")
        w_l = c3.slider("Latency", 0.0, 1.0, 0.15, 0.05, key=f"w_latency_{k}")
    scored = dash.composite(df, w_q, w_c, w_l)

    tiles = st.columns(4)
    best = scored.iloc[0]
    stat_card(tiles[0], "Best with your weights", f"{best['composite']:.2f}", best["model"])
    top = df.loc[df["quality"].idxmax()]
    stat_card(tiles[1], "Highest quality", f"{top['quality']:.2f} / 5", top["model"])
    if df["cost_usd_per_company"].notna().any():
        cheap = df.loc[df["cost_usd_per_company"].idxmin()]
        stat_card(tiles[2], "Lowest cost per company", f"${cheap['cost_usd_per_company']:.4f}", cheap["model"])
    fast = df.loc[df["latency_s_per_company"].idxmin()]
    stat_card(tiles[3], "Fastest per company", f"{fast['latency_s_per_company']:.1f} s", fast["model"])

    left, right = st.columns(2)
    with left.container(border=True):
        st.plotly_chart(dash.ranking_bar(scored, p), theme=None, key=f"rank_{k}")
        st.caption("Quality 1-5 maps to 0-1. Cost and latency score as best reader ÷ this reader, so twice the "
                   "cost scores 0.5.")
    with right.container(border=True):
        st.plotly_chart(dash.heatmap(dash.rubric_matrix(df, keys), "Rubric scores by reader", p, "score"),
                        theme=None, key=f"rubric_{k}")

    left, right = st.columns(2)
    with left.container(border=True):
        if df["cost_usd_per_company"].notna().any():
            st.plotly_chart(dash.tradeoff_scatter(df, "cost_usd_per_company", "Reader cost ($ per company)", p,
                                                  "$%{x:.4f} per company", tick_fmt=lambda v: f"${v:g}"),
                            theme=None, key=f"cost_{k}")
        else:
            st.info("No prices in config/models.toml for these readers, so there is no cost view.")
    with right.container(border=True):
        st.plotly_chart(dash.tradeoff_scatter(df, "latency_s_per_company", "Reader latency (sec per company)", p,
                                              "%{x:.1f} s per company"), theme=None, key=f"latency_{k}")
    st.caption("Blue points are on the Pareto frontier: no other reader is both cheaper (or faster) and better. "
               "Gray points are beaten on both.")

    with st.container(border=True):
        st.plotly_chart(dash.parallel_coordinates(df, keys, p), theme=None, key=f"parallel_{k}")
    by_company = dash.company_matrix(per_case, list(df["model"]))
    with st.container(border=True):
        if by_company.shape[1] > 1:
            st.plotly_chart(dash.heatmap(by_company, "Quality by company", p, "quality"),
                            theme=None, key=f"company_{k}")
        elif not per_case.empty and any(c.endswith("_reason") for c in per_case.columns):
            st.markdown("**Why the judge scored them this way**")
            cols = [c for c in per_case.columns if c.endswith("_reason")]
            reasons = per_case.assign(model=per_case["arm"].map(dash.short)).set_index("model")[cols]
            st.dataframe(reasons.rename(columns=lambda c: c.removesuffix("_reason")))

    st.subheader("Trace analysis")
    spans = spans_of(run / "trace.jsonl")
    if spans.empty:
        st.info("No trace file for this run.")
    else:
        calls = T.llm_calls(spans)
        with st.container(border=True):
            st.markdown("**Every model call in this run**")
            st.dataframe(fmt_stats(T.llm_stats(calls)), hide_index=True, key=f"calls_{k}")
            st.caption("From the run's OpenTelemetry trace. JSON repairs are second attempts after unusable "
                       "output; unusable output failed the schema even then.")
        if (calls["role"] == "reader").any():
            with st.container(border=True):
                st.plotly_chart(T.latency_strip(calls, "reader", theme_mode()), theme=None, key=f"strip_{k}")
        roots = spans[spans["name"] == "kyb.screen"]
        roots = roots[roots["attrs"].map(lambda a: bool(a.get("kyb.model")))]
        if not roots.empty:
            with st.container(border=True):
                st.markdown("**Timeline of one replay**")
                a, b = st.columns(2)
                pick_reader = a.selectbox("Reader", sorted(roots["screen_reader"].unique()),
                                          format_func=dash.short, key=f"wf_reader_{k}")
                mine = roots[roots["screen_reader"] == pick_reader]
                pick_company = b.selectbox("Company", sorted(mine["company"].unique()), key=f"wf_company_{k}")
                trace_id = mine.loc[mine["company"] == pick_company, "trace_id"].iloc[0]
                st.plotly_chart(T.waterfall(spans, trace_id, theme_mode()), theme=None, key=f"wf_{k}")

    st.subheader("Runs")
    table = dash.runs_table(df, keys)
    pct = [c for c in table.columns if any(w in c for w in ("first try", "supported", "recall"))]
    fmt = {c: st.column_config.NumberColumn(format="%.0f%%") for c in pct}
    fmt |= {c: st.column_config.NumberColumn(format="%.2f") for c in table.columns if c.startswith("L2 ")
            and c not in pct and "errors" not in c}
    fmt |= {"L3 $ / company": st.column_config.NumberColumn(format="$%.4f"),
            "L3 sec / company": st.column_config.NumberColumn(format="%.1f")}
    table[pct] = table[pct] * 100
    st.dataframe(table, hide_index=True, column_config=fmt, key=f"table_{k}")
    st.caption(f"Results folder: {run} · reports and every judge verdict are in the Results tab.")


# ---------------------------------------------------------------- layout

st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="kyb-hero"><div class="kyb-eyebrow">Tavily search · Nebius Token Factory</div>'
            '<h1>KYB screener</h1><p>Screen a company for adverse media, then evaluate both halves: did search '
            'find what it should, and which model reads it best.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Setup**")
    for key in ("TAVILY_API_KEY", "NEBIUS_API_KEY"):
        if os.getenv(key):
            st.badge(key, icon=":material/check_circle:", color="green")
        else:
            st.badge(f"{key} missing", icon=":material/error:", color="red")
    if not (os.getenv("TAVILY_API_KEY") and os.getenv("NEBIUS_API_KEY")):
        st.caption("Add the missing keys to `.env` (or as Codespaces secrets, then rebuild the codespace) "
                   "and restart the app.")
    judge_id = st.text_input("Judge model", cfg["eval"]["judge"], key="judge")
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        st.caption(f"Traces also go to {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")

tab_screen, tab_search, tab_readers, tab_results = st.tabs(["Screen", "Search eval", "Reader eval", "Results"])
with tab_screen:
    with st.form("screen"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Company name (searched as an exact phrase)", key="screen_name")
        country = c2.text_input("Country")
        website = c1.text_input("Website")
        industry = c2.text_input("Industry")
        aliases = c1.text_input("Aliases (comma-separated)")
        notes = c2.text_input("Notes that tell it apart from similarly named companies")
        s_planner = c1.selectbox("Planner", planner_options, key="screen_planner",
                                 index=planner_options.index(defaults.get("planner", FIXED)))
        s_reader = c2.selectbox("Reader", reader_options, key="screen_reader",
                                index=reader_options.index(defaults.get("reader", reader_options[0])))
        lookback = c1.number_input("Lookback (years)", 1, 20, 5)
        submitted = st.form_submit_button("Screen", type="primary", key="screen_submit")
    if submitted and missing_keys("TAVILY_API_KEY", "NEBIUS_API_KEY"):
        st.error("Set TAVILY_API_KEY and NEBIUS_API_KEY in .env to screen.")
    elif submitted and name:
        run_dir = RUNS / f"{stamp()}-{ev.slug(name)}"
        company = Company(name=name, country=country or None, website=website or None,
                          industry=industry or None, notes=notes or None,
                          aliases=[a.strip() for a in aliases.split(",") if a.strip()])
        with st.spinner("Planning, searching and reading"):
            init_tracing(run_dir / "trace.jsonl")
            try:
                result = run_screen(company, date.today(), int(lookback), factory.make_search_backend(),
                                    factory.make_model(s_reader, "reader"),
                                    planner=make_planner(s_planner, factory.make_model), session_id=run_dir.name)
            finally:
                shutdown_tracing()
        (run_dir / "report.md").write_text(render_markdown(result))
        (run_dir / "findings.json").write_text(json.dumps(to_json(result), indent=2))
        st.session_state["screen_dir"] = str(run_dir)
        st.session_state.pop("compare_dir", None)

    if st.session_state.get("screen_dir"):
        screen_dir = Path(st.session_state["screen_dir"])
        show_screen(screen_dir)
        st.divider()
        st.subheader("Compare other readers on this screen")
        st.caption("Replays this screen's plan and search results through each reader (no new searches) and "
                   "grades every report with the rubric judge. No answer key needed.")
        used = json.loads((screen_dir / "findings.json").read_text())["model"]
        options = list(dict.fromkeys(reader_options + [used]))
        chosen = st.multiselect("Readers", options, default=options, key="compare_readers")
        chosen = with_extra(chosen, st.text_input("More reader IDs (comma-separated)", key="compare_extra"))
        cmp_pairwise = st.checkbox(f"Also compare reports two at a time ({len(chosen) * (len(chosen) - 1)} more "
                                   "judge calls)", False, key="compare_pairwise")
        if judge_id in chosen:
            st.warning("The judge is also a reader; it will grade its own output.")
        blocked = missing_keys("NEBIUS_API_KEY")
        if blocked:
            st.error("Set NEBIUS_API_KEY in .env to compare.")
        if st.button("Compare", type="primary", key="compare_run", disabled=len(chosen) < 2 or bool(blocked)):
            out_dir = RESULTS / f"{stamp()}-compare-{screen_dir.name.split('-', 2)[-1]}"
            with st.status("Replaying and judging", expanded=True) as status:
                ev.compare_screen(screen_dir / "trace.jsonl", chosen, model_factory=factory.make_model,
                                  judge=factory.make_model(judge_id, "judge"), prices=prices, out_dir=out_dir,
                                  rubric=load_rubric(RUBRIC), pairwise=cmp_pairwise, log=status.write)
                status.update(label=f"Done: {out_dir}", state="complete", expanded=False)
            st.session_state["compare_dir"] = str(out_dir)
        if st.session_state.get("compare_dir"):
            show_dashboard(Path(st.session_state["compare_dir"]), "screen")

with tab_search:
    st.caption("Eval 1. Checks whether search finds the known events listed in evals/companies.yaml, for each "
               "planner. Only companies that list known events are used. No reader runs, so it isolates search.")
    if not DATASET.exists():
        st.info(f"Add {DATASET} with known events per company to run the search eval.")
    else:
        sdata = ev.load_dataset(DATASET)
        with_events = [c.id for c in sdata.cases if c.expected_events]
        s_planners = st.multiselect("Planners", planner_options, default=[FIXED] + cfg["eval"]["planners"],
                                    key="search_planners")
        s_planners = with_extra(s_planners, st.text_input("More planner IDs (comma-separated)",
                                                          key="search_extra"))
        s_cases = st.multiselect("Companies with known events", with_events, default=with_events,
                                 key="search_cases")
        s_missing = ev.missing_recordings(sdata.select(s_cases), s_planners, RECORDINGS)
        s_allow = True
        if s_missing:
            st.warning(f"{len(s_missing)} planner recordings are missing. Recording them runs live searches: "
                       f"up to {ev.estimate_credits(s_missing)} Tavily credits.")
            s_allow = st.checkbox("Record them now", key="search_allow_record")
        s_blocked = missing_keys("NEBIUS_API_KEY", *(("TAVILY_API_KEY",) if s_missing else ()))
        if s_blocked:
            st.error(f"Set {', '.join(s_blocked)} in .env to run this.")
        if st.button("Run search eval", type="primary", key="search_run",
                     disabled=not s_planners or not s_cases or not s_allow or bool(s_blocked)):
            out_dir = RESULTS / f"{stamp()}-search"
            with st.status("Recording and judging", expanded=True) as status:
                for planner in sorted({q for q, _ in s_missing}):
                    ev.record(sdata, planner, RECORDINGS, factory.make_search_backend, factory.make_model,
                              cases=sdata.select([c for q, c in s_missing if q == planner]), log=status.write)
                ev.search_eval(sdata, s_planners, judge=factory.make_model(judge_id, "judge"),
                               recordings_root=RECORDINGS, out_dir=out_dir, case_ids=s_cases, log=status.write)
                status.update(label=f"Done: {out_dir}", state="complete", expanded=False)
    search_runs = runs_of("search")
    if search_runs:
        st.divider()
        show_search(st.selectbox("Search eval run", search_runs, format_func=lambda r: r.name,
                                 key="search_pick"), "search")

with tab_readers:
    st.caption("Eval 2. The plan and search results are recorded once and held constant; each reader replays "
               "them and the judge scores every report with the rubric. Use past screens, or the companies file.")
    with st.expander("Run a reader eval", expanded=not reader_runs()):
        screens_available = sorted((r for r in RUNS.glob("*") if (r / "findings.json").exists()), reverse=True)
        sources = ["Past screens"] + (["evals/companies.yaml"] if DATASET.exists() else [])
        source = st.radio("Companies from", sources, horizontal=True, key="bench_source")
        readers = st.multiselect("Readers to compare", reader_options, default=cfg["eval"]["readers"],
                                 key="readers")
        readers = with_extra(readers, st.text_input("More reader IDs (comma-separated)", key="extra_readers"))
        if judge_id in readers:
            st.warning("The judge is also a reader; it will grade its own output.")
        use_pairwise = st.checkbox("Also compare reports two at a time (both orders)", False, key="use_pairwise")
        missing = []
        if source == "Past screens":
            def screen_label(r: Path) -> str:
                f = json.loads((r / "findings.json").read_text())
                return f"{f['company']['name']} · {f['as_of']} · planner {f['planner']}"

            picked = st.multiselect("Screens", screens_available, format_func=screen_label, key="bench_screens",
                                    default=screens_available[:10])
            if not screens_available:
                st.info("No past screens yet. Screen a few companies in the Screen tab first.")
            ready = bool(picked)
        else:
            data = ev.load_dataset(DATASET)
            sweep_planner = cfg["eval"].get("sweep_planner", FIXED)
            start_at = planner_options.index(sweep_planner) if sweep_planner in planner_options else 0
            planner = st.selectbox("Fixed planner", planner_options, key="fixed_planner", index=start_at)
            case_ids = st.multiselect("Companies", [c.id for c in data.cases], default=[c.id for c in data.cases],
                                      key="cases")
            missing = ev.missing_recordings(data.select(case_ids), [planner], RECORDINGS)
            ready = bool(case_ids)
        allow_record = True
        if missing:
            st.warning(f"{len(missing)} planner recordings are missing. Recording them runs live searches: "
                       f"up to {ev.estimate_credits(missing)} Tavily credits.")
            allow_record = st.checkbox("Record them now", key="allow_record")
        blocked = missing_keys("NEBIUS_API_KEY", *(("TAVILY_API_KEY",) if missing else ()))
        if blocked:
            st.error(f"Set {', '.join(blocked)} in .env to run this.")
        if st.button("Run reader eval", type="primary", key="run",
                     disabled=not ready or not readers or not allow_record or bool(blocked)):
            run_dir = RESULTS / f"{stamp()}-readers"
            common = dict(model_factory=factory.make_model, judge=factory.make_model(judge_id, "judge"),
                          prices=prices, out_dir=run_dir, rubric=load_rubric(RUBRIC), pairwise=use_pairwise)
            with st.status("Replaying and judging", expanded=True) as status:
                if source == "Past screens":
                    ev.compare_screens([r / "trace.jsonl" for r in picked], readers, answer_key=DATASET,
                                       log=status.write, **common)
                else:
                    if missing:
                        ev.record(data, planner, RECORDINGS, factory.make_search_backend, factory.make_model,
                                  cases=data.select([c for _, c in missing]), log=status.write)
                    ev.run_experiment(data, "readers", ev.readers_experiment(planner, readers),
                                      recordings_root=RECORDINGS, case_ids=case_ids, log=status.write, **common)
                status.update(label=f"Done: {run_dir}", state="complete", expanded=False)
    runs = reader_runs()
    if runs:
        show_dashboard(st.selectbox("Reader eval run", runs, format_func=run_label, key="dash_run"), "dash")

with tab_results:
    all_runs = sorted((p for p in RESULTS.glob("*") if (p / "experiment.json").exists()), reverse=True)
    if not all_runs:
        st.write("No results yet.")
    else:
        chosen_run = st.selectbox("Run", all_runs, format_func=lambda p: p.name, key="result_run")
        if json.loads((chosen_run / "experiment.json").read_text())["kind"] == "search":
            st.markdown((chosen_run / "summary.md").read_text())
        else:
            show_results(chosen_run, "results")
