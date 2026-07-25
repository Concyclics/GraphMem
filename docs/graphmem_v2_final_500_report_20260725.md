# GraphMem V2 LongMemEval Final 500 Report

Date: 2026-07-25

## Result

- Variant: hierarchical_state_graph_v2
- Backbone and judge: deepseek-v4-flash
- Embedding: Qwen3-Embedding-0.6B, local OpenAI-compatible service on port 8001
- Judge: Mem0 LongMemEval prompt at commit bd063eea04de4f8a19927beea155afa094a01905
- Overall accuracy: **462/500 = 92.4%**
- Build budget pass: **500/500**
- Answer budget pass: **500/500**
- Build P50/P95/max: **253,946 / 271,194 / 284,693**
- Answer P50/P95/max: **7,181 / 8,154 / 8,863**
- Build, answer, and judge reasoning tokens: **0**

## Accuracy by question type

| Type | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| single-session-user | 68 | 70 | 97.14% |
| multi-session | 123 | 133 | 92.48% |
| single-session-preference | 26 | 30 | 86.67% |
| temporal-reasoning | 119 | 133 | 89.47% |
| knowledge-update | 74 | 78 | 94.87% |
| single-session-assistant | 52 | 56 | 92.86% |

The overall 90% target is met. The stricter stretch targets of 90% for
temporal reasoning and 95% for single-session-assistant are not met.

## DeepSeek token accounting

Only build and answer calls are included in the two hard budgets.

| Stage | Cache-miss input | Cache-hit input | Output | Total |
| --- | ---: | ---: | ---: | ---: |
| Build | 63,056,006 | 18,749,824 | 45,446,463 | 127,252,293 |
| Answer | 890,843 | 2,693,376 | 13,096 | 3,597,315 |

Judge usage is recorded separately and excluded from both budgets:

- cache-miss input: 30,788
- cache-hit input: 758,912
- output: 70,353
- total: 860,053
- reasoning: 0

## Index and retrieval quality

- Gold answer-session recall: 98.30% average
- Source-leaf expansion recall: 93.87% average
- Post-pack gold-term support: 84.00% average
- Fact semantic retrieval: 100.0% any-session recall, 96.2% all-session recall
- Leaf BM25: 99.6% any-session recall, 95.4% all-session recall
- Leaf semantic: 98.6% any-session recall, 93.2% all-session recall
- Source ID errors: 0
- Routing pointer errors: 0
- Edge endpoint errors: 0
- State-chain errors: 0
- Routing cards over 180 provider tokens: 0

The remaining 38 errors were classified as:

- answer reasoning or format: 30
- retrieval ranking or graph: 3
- context rendering: 4
- index extraction: 1

The dominant remaining limitation is therefore answer/operator coverage, not
raw retrieval availability.

## Graph usage audit

Graph expansion was retained in 484/500 questions. It expanded 9,111 nodes,
of which 1,982 survived final packing. All 8 gold-rescue candidates were
retained. Graph traversal introduced a previously unavailable operator source
fact in 37 questions.

Relations with concrete post-pack/operator usage included:

- before: 98 post-pack nodes, 46 operator sources
- after: 42 post-pack nodes, 10 operator sources
- same_measure: 586 post-pack nodes, 205 operator sources
- same_collection: 92 post-pack nodes, 36 operator sources
- operand_of: 146 post-pack nodes, 65 operator sources
- same_predicate: 475 post-pack nodes
- contains: 431 post-pack nodes, 259 operator sources

This confirms that the graph is used for typed lateral and temporal expansion,
not only for routing-card-to-leaf descent.

## Residual risks

- Session extraction parse-error rate is 9.73%; lossless L0 fallback keeps the
  final fact-source error count at zero, but it still increases reliance on
  deterministic L0 operators.
- Temporal scope diagnostics reported 15 warnings. Temporal accuracy is
  89.47%, narrowly below the 90% stretch target.
- Post-pack gold-term support is 84.00%, below the 95% retrieval-sufficiency
  stretch target, even though answer-session recall is 98.30%.
- supersedes, contradicts, and supports are traversed, but they contribute
  fewer final operator sources than measure, collection, and operand edges.

## Artifacts

- Final run: runs/iterations/v4_iter2_20260725/full500_final_20260725
- Pipeline diagnostics:
  runs/iterations/v4_iter2_20260725/full500_final_20260725/analysis/pipeline_diagnostics.json
- Retrieval-channel audit:
  runs/iterations/v4_iter2_20260725/full500_final_20260725/analysis/retrieval_channel_audit.json
- Mem0 judgments:
  runs/iterations/v4_iter2_20260725/full500_final_20260725/mem0_judge/auto_eval.jsonl
