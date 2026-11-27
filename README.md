# KYB screening on Tavily + Nebius, with a two-part eval harness

A vendor-risk / KYB (know-your-business) screener. Give it a company, and it runs four stages:

1. **Planner**: a reasoning ("thinking") model on Nebius Token Factory, or the hand-written fixed
   plan, chooses the Tavily searches.
2. **Tavily search** runs them in parallel, skipping social media.
3. **Reader**: a second model reads the results per risk category and returns findings that cite
   source IDs, then groups findings that report the same event so each event appears once.
4. **Source check**: a judge model reads each finding's cited sources and confirms the finding or
   holds it back. Code renders the report, so every citation links to a source that was actually
   retrieved, and every finding in it was confirmed against those sources.

Models are called through [LiteLLM](https://docs.litellm.ai/docs/providers), so switching provider is
a config edit: Nebius Token Factory models are `nebius/<model ID>`, and `openai/...`,
`anthropic/...`, `gemini/...` work the same way with their own key.

Every run is traced with OpenTelemetry, and **the trace doubles as a recording**: it stores the plan
and the full Tavily responses. That makes two separate evals possible, one per half of the pipeline:

- **Eval 1, search:** did Tavily, driven by the planner, find the known events listed in
  `evals/companies.yaml`? Each event is marked found or missed, with the sources and queries that
  found it. No reader is involved.
- **Eval 2, reader:** with the search results held constant, which reader model writes the best
  report, at what cost and latency? A judge model grades every report on a rubric. This needs no
  answer key and runs on any past screen.

Both read their traces for **trace analysis**: model calls, errors, JSON repairs, latency per call,
tokens, Tavily latency and credits, and a timeline of where each screen's time went.

See [TECHNICAL_STATEMENT.md](TECHNICAL_STATEMENT.md) for the reasoning and results.

## How it works

```
company ─► PLANNER  reasoning model writing queries per risk category (or the fixed plan)
             │       code builds the Tavily params: exact_match, advanced depth + chunks_per_source,
             │       the date window, regulator-only domains; budget ≤3 per category, ≤12 total
             ▼
           TAVILY SEARCH  in parallel, social media excluded
             │       ─► evidence store: deduped by URL, stable IDs S1…Sn in plan order
             │                    ▲
             │                    └── eval 1: are the known events in here?
             ▼
           READER   one call per category ─► JSON findings citing source IDs
             │       (schema-validated, one repair retry, unknown IDs removed)
             │       then one call grouping duplicates; code validates the groups
             │                    ▲
             │                    └── eval 2: same evidence, other readers, rubric judge
             ▼
           SOURCE CHECK  judge reads each finding's cited sources: confirmed, or held back
             ▼
           report rendered by code: one entry per event, held-back findings listed separately,
           a category the reader failed on is "not assessed", never "clean"

trace.jsonl (OpenTelemetry, OpenInference conventions; also to Phoenix, Langfuse or any OTLP backend)
  kyb.plan · tavily.search (full responses) · kyb.extract.* · kyb.merge · kyb.check · kyb.llm.call · kyb.judge.*
```

## Run it in GitHub Codespaces

Nothing to install locally. Pushing to GitHub, adding two secrets and opening a codespace gets you the app in a browser tab.

1. Push this folder to a new GitHub repository (private is fine).
2. Add your keys as Codespaces secrets named `TAVILY_API_KEY` and `NEBIUS_API_KEY`, either at
   github.com/settings/codespaces (scoped to that repository) or in the repository's
   **Settings → Secrets and variables → Codespaces**. If you create the codespace with
   **Code → Codespaces → ⋯ → New with options**, the page also asks for any of the two that are missing.
3. **Code → Codespaces → Create codespace on main.** The first build installs uv and the
   dependencies (about two minutes). The app then starts and opens in a preview tab on port 8501;
   to open it in a full browser tab, use the globe icon on port 8501 in the **Ports** panel.

The sidebar shows a green badge for each key the app can see. If you add a secret after the codespace
exists, rebuild it (command palette: **Codespaces: Rebuild Container**) or put the keys in `.env`.
In the codespace terminal, `uv run kyb ...` runs the same pipeline from the command line, and
`uv run pytest` runs the offline tests. For traces in Phoenix, run `uvx arize-phoenix serve` in a
second terminal, then `export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006` before starting
the app. Port 6006 is already forwarded.

## Quickstart (local)

Requires [uv](https://docs.astral.sh/uv/), a [Tavily](https://app.tavily.com) key and a
[Nebius Token Factory](https://tokenfactory.nebius.com) key.

```bash
uv sync
cp .env.example .env            # add TAVILY_API_KEY and NEBIUS_API_KEY
uv run kyb models               # models your key can use; edit config/models.toml to match
uv run streamlit run app.py
```

To use another provider for any role, put its LiteLLM model string in `config/models.toml` (for
example `reader = "openai/gpt-4.1-mini"` or a reader list mixing `nebius/...` and
`anthropic/...` models) and add that provider's key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) to
`.env`. The sidebar shows which provider keys are present. Costs come from `config/models.toml` and
fall back to LiteLLM's price list for models not priced there.

The app has four tabs:

| Tab | What it does |
|---|---|
| Screen | Screen one company with the default planner and reader. Shows the report, the plan, and the trace: where the time went and every model call. Then, optionally, compare other readers on the same search results. |
| Search eval | Eval 1 for each planner against the known events in `evals/companies.yaml`. |
| Reader eval | Eval 2 on past screens or the companies file: the dashboard and the trace analysis. |
| Results | Per-run detail: summaries, two reports side by side, every judge verdict. |

The same from the command line:

```bash
uv run kyb screen "TD Bank" --country "United States" --website td.com
uv run kyb compare runs/<timestamp>-TD_Bank/trace.jsonl        # eval 2 on that screen
uv run kyb compare runs/*/trace.jsonl                           # eval 2 across past screens
uv run kyb eval search --record-missing                         # eval 1
uv run kyb eval readers --record-missing                        # eval 2 on the companies file
uv run kyb eval agreement results/<run>/judgments.csv           # after you hand-label a sample
```

To browse traces in a UI, start [Arize Phoenix](https://arize.com/docs/phoenix) and point the
standard OTLP variable at it:

```bash
uvx arize-phoenix serve                              # UI on http://localhost:6006
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006
```

[Langfuse](https://langfuse.com/integrations/native/opentelemetry) works the same way: set
`OTEL_EXPORTER_OTLP_ENDPOINT` to its `/api/public/otel` endpoint and `OTEL_EXPORTER_OTLP_HEADERS` to
its Basic Auth header.

## Eval 1: search against known events

`evals/companies.yaml` is the answer key: companies with public, referenced adverse events (a
guilty plea, a sanctions settlement, a breach, a bankruptcy) plus two namesake traps. The search
eval records each planner's searches once per company, then the judge reads the recorded results
and marks each known event found (and by which sources) or missed.

It reports, per planner:

- **coverage:** known events found, overall and by risk category
- **every event:** found or missed, the matching sources and the queries that surfaced them, and
  whether it was found by its own category's search or by another one
- **from the traces:** searches per company, results per search, empty searches, Tavily latency
  (median and slowest 5%), credits, and the planner's own call latency and output tokens

Only companies that list known events are used, and no reader runs, so this isolates search. It's
also the place to compare planners, for example the fixed plan against a reasoning model.

## Eval 2: readers on fixed search results

A screen's plan and search results are replayed, unchanged, through each reader. Every reader,
the original included, is re-run on the same evidence, so all reports are produced and judged the
same way. It runs on one screen, on any set of past screens, or on the companies file.

Each reader is evaluated in three layers:

| Layer | What | How |
|---|---|---|
| L1 deterministic | invalid citations, JSON first try, categories failed | code, no model |
| L2 judge | rubric quality and per-criterion scores; supported on target, namesake errors; recall when known events exist | fixed judge model |
| L3 efficiency | cost and latency per company, tokens | provider usage, prices in `config/models.toml` |

The rubric (`config/rubric.yaml`) has five weighted criteria. The judge sees the screening question,
the report with excerpts of each finding's cited sources, and an overview of all the evidence the
reader was given; that last part is what makes scope gradable.

| Criterion | Weight | Question, in short |
|---|---|---|
| citation | 0.25 | Is every claim backed by the cited excerpts? |
| scope | 0.25 | Does the report capture every significant adverse event in the evidence? |
| relevance | 0.20 | Is everything reported about the target, in the window, in the right category, without noise? |
| specificity | 0.15 | Who acted, what happened, amounts and dates where the sources give them? |
| calibration | 0.15 | Do severity and status match the sources? |

A report with no findings is graded on scope only: it scores well only if the evidence held nothing
material. Quality is the weighted mean of the criteria that apply.

The **dashboard** shows how the readers trade quality, cost and latency: weight sliders with a
ranking by the blended score, parallel coordinates across every measure, quality against cost and
against latency with the Pareto frontier (readers nobody beats on both), rubric and per-company
heatmaps, and a runs table grouped by layer. By default the sweep runs five readers across price
tiers, all on Nebius: `GLM-5.3`, `MiniMax-M3`, `nemotron-3-super-120b-a12b`, `gpt-oss-120b` and
`Qwen3-30B-A3B-Instruct-2507`.

## Trace analysis

Every model call is a `kyb.llm.call` span whatever the provider, alongside the OpenInference LiteLLM
span, and every search a `tavily.search` span. From the trace alone, each run reports:

- **per model and role (planner, reader, judge):** calls, errors, JSON repairs, unusable output,
  latency per call (median and slowest 5%), prompt and output tokens per call, output tokens per second
- **per reader:** the latency of every call, as a strip chart
- **per screen:** a timeline of plan, each search, each category read and the model calls inside it
- **for search:** results per search, empty searches, Tavily latency and credits

These appear in the app, in `model_calls.csv`, and in each run's `summary.md`.

## The judge

The judge is a fixed model kept out of the reader list. Every verdict lands in a CSV with an empty
human column (`judgments.csv`, `recall.csv`, `coverage.csv`, `rubric.csv`, `search_eval.csv`).
Label a sample, and `kyb eval agreement` reports how often you agree with the judge.

## Results

_Filled in from `results/<run>/summary.md` after running the evals._

## Design decisions and their costs

- **Two evals, one per half.** Search quality and reading quality fail for different reasons and
  are fixed by different people. Measuring them separately says which half to work on.
- **Record once, replay everywhere.** A screen or planner is recorded once, so reader comparisons
  see identical evidence and spend no Tavily credits. The recording also freezes the screen date,
  which both the search window and the reader's "today" depend on.
- **Planners compete on queries only.** Code builds every Tavily request, keeps the date window,
  enforces the budget, and fills any category a planner skips from the fixed plan.
- **The reader reads; code writes.** The reader returns findings that cite source IDs; code
  validates them and renders the report. No model can produce a URL that wasn't retrieved.
- **Check, then show.** A valid citation ID only proves the source exists. On every screen a judge
  model reads each finding's cited sources; findings it can't confirm move to a "Held back" section
  instead of disappearing, so a reviewer sees both. If the check itself fails, nothing is hidden
  and the report says the findings weren't checked. Costs one judge call per eight findings.
- **Duplicates are merged, not avoided.** Reading per category keeps each prompt focused, but one
  event (a guilty plea with regulatory penalties) lands in several categories. One extra reader call
  groups them; code accepts only groups of known IDs and lists the event under each category.
- **The answer key is optional.** The rubric judges any screen against its own evidence; known
  events add a check against what actually happened.
- **LiteLLM's `completion` directly, no agent framework.** Any provider by config, exact requests
  and provider-reported token counts, plus a provider-independent span per model call for the trace
  analysis. JSON mode is requested and dropped per model if rejected; the output is validated either way.

## Limitations

- **The rubric grades against the evidence, not the truth.** If search missed something, a
  well-cited report can still score well. Eval 1 is what catches that, for the companies with known events.
- **Known companies favor reasoning planners.** A planner may already know a famous company's
  scandals and aim its queries at them. Include lesser-known companies before drawing conclusions.
- A planner's recording is one sample: rerunning it, even at temperature 0, may produce a different plan.
- The system covers adverse media. Ownership, formal sanctions-list matching, registry and
  licensing checks need structured sources, not web search.
- Eight companies and thirteen known events give a directional comparison, not a statistically
  strong one. An LLM judge has its own errors; the agreement check measures them.

## Layout

```
app.py                 Streamlit app
src/kyb/plan.py        fixed plan, categories, Tavily parameter builder
src/kyb/planner.py     fixed and model planners, plan validation and budget
src/kyb/search.py      live Tavily backend, recordings, replay backend, traced search span
src/kyb/evidence.py    evidence store and model-facing rendering
src/kyb/screen.py      pipeline: plan, parallel search, read, merge; citation validation
src/kyb/llm.py         LiteLLM model, JSON parsing and repair, per-call span, readable errors
src/kyb/judge.py       coverage, recall, support (and the per-screen source check), rubric, pairwise
src/kyb/events.py      findings into events: duplicates merged, unconfirmed findings held back
src/kyb/evaluate.py    eval 1 (search_eval), eval 2 (run_experiment, compare_screens), summaries
src/kyb/traces.py      trace analysis: model calls, search stats, stage timeline
src/kyb/dashboard.py   sweep data, Pareto frontier, blended score, Plotly figures
src/kyb/report.py      markdown / JSON report
src/kyb/tracing.py     OpenTelemetry setup (file + optional OTLP)
src/kyb/factory.py     builds live models and the search backend
src/kyb/cli.py         `kyb` command
evals/companies.yaml   answer key: known events per company, with references
config/models.toml     default planner and reader, planners, readers, judge, prices
config/rubric.yaml     rubric criteria and weights
tests/                 offline tests: fake Tavily, fake models, LiteLLM against a mock endpoint, AppTest
.devcontainer/         GitHub Codespaces: Python 3.12, uv, key secrets, app on port 8501
.streamlit/config.toml app theme (light and dark), shared with the chart palette
```

`uv run pytest` runs everything without API keys.
