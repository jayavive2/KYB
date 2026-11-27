# Reader comparison

1 companies · judge `nebius/deepseek-ai/DeepSeek-V4-Pro-0813` · planner `nebius/moonshotai/Kimi-K3.1-Pro` and its recorded searches are fixed; the reader varies.

**Warning:** 1 judge calls returned no usable verdict; affected findings count as unjudged and affected events as not matched.

## Search

| arm | searches_per_company | sources_per_company | tavily_credits | coverage | planner_usd_per_company |
|---|---|---|---|---|---|
| nebius/zai-org/GLM-5.3 | 12.0 | 58.0 | 24 | 100% | 0.0389 |
| nebius/MiniMaxAI/MiniMax-M3 | 12.0 | 58.0 | 24 | 100% | 0.0389 |
| nebius/nvidia/nemotron-3-super-120b-a12b | 12.0 | 58.0 | 24 | 100% | 0.0389 |
| nebius/openai/gpt-oss-120b | 12.0 | 58.0 | 24 | 100% | 0.0389 |
| nebius/Qwen/Qwen3-30B-A3B-Instruct-2507 | 12.0 | 58.0 | 24 | 100% | 0.0389 |

## Findings

| arm | recall | recall_given_evidence | supported_on_target | namesake_errors | unsupported | invalid_citations | json_first_try | categories_failed | reader_usd_per_company | reader_sec_per_company |
|---|---|---|---|---|---|---|---|---|---|---|
| nebius/zai-org/GLM-5.3 | 0% | 0% | n/a | 0 | 0 | 0 | 14% | 6 | 0.5373 | 182.8 |
| nebius/MiniMaxAI/MiniMax-M3 | 100% | 100% | 91% | 0 | 3 | 0 | 100% | 0 | 0.0180 | 9.0 |
| nebius/nvidia/nemotron-3-super-120b-a12b | 100% | 100% | 94% | 1 | 1 | 0 | 100% | 0 | 0.0369 | 40.9 |
| nebius/openai/gpt-oss-120b | 100% | 100% | 77% | 3 | 4 | 0 | 100% | 0 | 0.0127 | 12.2 |
| nebius/Qwen/Qwen3-30B-A3B-Instruct-2507 | 100% | 100% | 86% | 1 | 2 | 0 | 100% | 0 | 0.0054 | 14.7 |

## Judge

| arm | quality | rubric_citation | rubric_scope | rubric_relevance | rubric_specificity | rubric_calibration |
|---|---|---|---|---|---|---|
| nebius/zai-org/GLM-5.3 | 1.00 | n/a | 1.00 | n/a | n/a | n/a |
| nebius/MiniMaxAI/MiniMax-M3 | n/a | n/a | n/a | n/a | n/a | n/a |
| nebius/nvidia/nemotron-3-super-120b-a12b | 3.30 | 3.00 | 3.00 | 3.00 | 4.00 | 4.00 |
| nebius/openai/gpt-oss-120b | 3.25 | 2.00 | 3.00 | 4.00 | 3.00 | 5.00 |
| nebius/Qwen/Qwen3-30B-A3B-Instruct-2507 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 |

- coverage: known events present in the searched evidence (the planner's job)
- recall: known events the reader reported
- recall_given_evidence: the same, counting only events that were in the evidence
- supported_on_target: findings about the right company whose cited sources back them
- quality: weighted mean of the rubric scores (1-5), weights in config/rubric.yaml
- rubric_*: judge's 1-5 score per criterion, averaged over the reports it applied to
- pairwise_w_t_l: wins-ties-losses against the other arms, company by company

Every judge verdict is in the CSVs with empty human_* columns: label a sample and run `kyb eval agreement judgments.csv`.

## Model calls (from the trace)

| role | model | calls | errors | JSON repairs | unusable | p50 s | p95 s | prompt tok | output tok | output tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| reader | nebius/MiniMaxAI/MiniMax-M3 | 8 | 0 | 0 | 0 | 4.00 | 6.22 | 5,135 | 589 | 161 |
| reader | nebius/Qwen/Qwen3-30B-A3B-Instruct-2507 | 8 | 0 | 0 | 0 | 6.70 | 9.17 | 5,284 | 497 | 87 |
| reader | nebius/nvidia/nemotron-3-super-120b-a12b | 8 | 0 | 0 | 0 | 13.62 | 25.60 | 5,405 | 3,328 | 248 |
| reader | nebius/openai/gpt-oss-120b | 8 | 0 | 0 | 0 | 4.08 | 7.80 | 4,924 | 1,420 | 337 |
| reader | nebius/zai-org/GLM-5.3 | 13 | 0 | 6 | 12 | 48.95 | 58.05 | 5,629 | 7,602 | 169 |
| judge | nebius/deepseek-ai/DeepSeek-V4-Pro-0813 | 34 | 0 | 7 | 8 | 53.28 | 121.22 | 7,789 | 5,230 | 94 |

JSON repairs: calls that were a second attempt after unusable output. Unusable: calls whose output still failed the schema. Latency covers successful calls.
