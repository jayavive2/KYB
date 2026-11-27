<!-- DRAFT: rewrite in your own words and fill in the numbers from results/<run>/summary.md. -->

# Technical statement

## The customer and the problem

The starter is a general research agent: the model decides what to search, writes an answer,
and is asked to "include source URLs". For a customer that has to act on the answer, three things
are missing:

1. **Nothing makes the output auditable.** URLs come from the model's own text, so nothing
   guarantees they were retrieved or that they say what the answer claims. The model is also never
   told today's date, so any date filter it passes to Tavily comes from its training data.
2. **Nothing makes it repeatable.** The model chooses queries on every run, and the web changes
   daily, so two runs differ for reasons nobody can see.
3. **There is no way to choose models.** Token Factory offers many open models at very different
   prices. Comparing them on live search mixes model differences with search differences.

I picked a customer for whom all three are requirements: a risk or compliance team screening
vendors and counterparties for adverse media (KYB). That is an established Tavily use case. BMO's
AML team, for example, works where regulators dictate the method as well as the answer.

## What I built

- **A three-stage pipeline.** A reasoning model plans the searches (or a hand-written fixed plan);
  Tavily runs them; a reader model extracts findings per risk category and cites source IDs. Code
  builds every Tavily request, enforces a search budget, keeps the date window, validates citations,
  and renders the report.
- **OpenTelemetry tracing with OpenInference conventions**, viewable in Phoenix, Langfuse or any
  OTLP backend. The plan and every full Tavily response are on the spans, so each trace is also a
  recording that can be replayed.
- **A reader eval on fixed search results.** Any screen, or set of screens, is replayed through
  several readers. A judge grades each report on a weighted rubric (citation, scope, relevance,
  specificity, calibration) against the evidence the reader had, so no answer key is needed. A
  dashboard shows how the readers trade quality against cost and latency.
- **Trace analysis** for screens and evals: calls, errors, JSON repairs and latency per call for every model,
  Tavily latency and credits, and a timeline of where each screen's time went.
- **A Streamlit app** that puts the default screen first and the evals one click away.

## Why this approach

Recording a screen once and replaying it keeps every reader comparison exact and free of extra
search cost. The eval needs no answer key, so it works on any screen a team runs, but it can only
judge a report against the evidence it was given.

## Results

_From `results/<run>/summary.md`:_

**Reader eval** (planner fixed: _planner_)

| Reader | Quality (1-5) | Citation | Scope | Relevance | $ / company | Sec / company | JSON repairs |
|---|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … | … |

- Judge agreement on a hand-labeled sample of _N_ findings: _X%_ (about target), _Y%_ (supported).
- Takeaway: _which planner and reader you'd recommend for this workflow, and at what cost difference._

## Value

- **For the customer:** a screen they can defend: a documented method, every claim linked to a
  retrieved source, and a trace of every step. Plus a repeatable way to pick, and re-check, the
  cheapest planner and reader that meet their bar.
- **For Tavily and Nebius:** the harness sells search and inference together. Recordings let a
  customer test new Token Factory models against their own past cases in minutes, with no extra
  search spend, which shortens a POC and makes model upgrades low-risk.

## Next steps

- Run replays in CI so a prompt or model change has to pass the recorded cases.
- Sample production traces into the eval set; every real screen becomes a regression case.
- Repeat each model planner a few times to measure plan variance, and add lesser-known companies
  so planners can't lean on what they already know.
- Scheduled re-screening that diffs against the previous trace and reports only what changed.
