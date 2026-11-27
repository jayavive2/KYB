# Reader comparison

1 screens · judge `nebius/deepseek-ai/DeepSeek-V4-Pro-0813` · planner `nebius/moonshotai/Kimi-K3.1-Pro` and its recorded searches are fixed; the reader varies.

## Findings

| arm | findings | events | supported_on_target | namesake_errors | unsupported | invalid_citations | json_first_try | categories_failed | reader_usd_per_company | reader_sec_per_company |
|---|---|---|---|---|---|---|---|---|---|---|
| nebius/moonshotai/Kimi-K3.1-Pro | 26 | 23 | 96% | 1 | 0 | 0 | 100% | 0 | 0.5264 | 120.7 |
| nebius/meta-models/muse-spark-open | 24 | 22 | 92% | 0 | 2 | 0 | 100% | 0 | 0.1835 | 72.0 |
| nebius/zai-org/GLM-5.3 | 31 | 26 | 100% | 0 | 0 | 0 | 86% | 0 | 0.6238 | 405.5 |
| nebius/nvidia/Nemotron-3-Ultra-550b-a55b | 34 | 26 | 94% | 0 | 2 | 0 | 100% | 0 | 0.1308 | 37.4 |
| nebius/MiniMaxAI/MiniMax-M3 | 31 | 18 | 97% | 0 | 1 | 0 | 100% | 0 | 0.0170 | 8.1 |
| nebius/openai/gpt-oss-120b | 31 | 28 | 94% | 0 | 2 | 0 | 86% | 0 | 0.0146 | 16.3 |

## Judge

| arm | quality | rubric_citation | rubric_scope | rubric_relevance | rubric_specificity | rubric_calibration |
|---|---|---|---|---|---|---|
| nebius/moonshotai/Kimi-K3.1-Pro | 3.60 | 2.00 | 5.00 | 4.00 | 4.00 | 3.00 |
| nebius/meta-models/muse-spark-open | 3.60 | 2.00 | 5.00 | 4.00 | 4.00 | 3.00 |
| nebius/zai-org/GLM-5.3 | 3.75 | 2.00 | 5.00 | 4.00 | 5.00 | 3.00 |
| nebius/nvidia/Nemotron-3-Ultra-550b-a55b | 3.50 | 2.00 | 4.00 | 4.00 | 5.00 | 3.00 |
| nebius/MiniMaxAI/MiniMax-M3 | 3.15 | 2.00 | 4.00 | 3.00 | 4.00 | 3.00 |
| nebius/openai/gpt-oss-120b | 3.15 | 2.00 | 4.00 | 3.00 | 4.00 | 3.00 |

- events: distinct events after the reader merged duplicates
- supported_on_target: findings about the right company whose cited sources back them
- quality: weighted mean of the rubric scores (1-5), weights in config/rubric.yaml
- rubric_*: judge's 1-5 score per criterion, averaged over the reports it applied to
- pairwise_w_t_l: wins-ties-losses against the other readers, screen by screen

Every judge verdict is in the CSVs with empty human_* columns: label a sample and run `kyb agreement judgments.csv`.

## Model calls (from the trace)

| role | model | calls | errors | JSON repairs | unusable | p50 s | p95 s | prompt tok | output tok | output tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| reader | nebius/MiniMaxAI/MiniMax-M3 | 8 | 0 | 0 | 0 | 3.35 | 4.96 | 4,773 | 574 | 187 |
| reader | nebius/meta-models/muse-spark-open | 8 | 0 | 0 | 0 | 21.08 | 31.10 | 4,793 | 3,986 | 192 |
| reader | nebius/moonshotai/Kimi-K3.1-Pro | 8 | 0 | 0 | 0 | 43.84 | 93.06 | 4,804 | 5,619 | 122 |
| reader | nebius/nvidia/Nemotron-3-Ultra-550b-a55b | 8 | 0 | 0 | 0 | 11.98 | 29.60 | 5,076 | 3,760 | 260 |
| reader | nebius/openai/gpt-oss-120b | 9 | 0 | 1 | 1 | 5.12 | 7.81 | 4,972 | 1,461 | 302 |
| reader | nebius/zai-org/GLM-5.3 | 9 | 0 | 1 | 1 | 87.87 | 189.76 | 4,676 | 14,266 | 151 |
| judge | nebius/deepseek-ai/DeepSeek-V4-Pro-0813 | 30 | 0 | 0 | 0 | 51.32 | 178.80 | 8,736 | 4,995 | 66 |

JSON repairs: calls that were a second attempt after unusable output. Unusable: calls whose output still failed the schema. Latency covers successful calls.
