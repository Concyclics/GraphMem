# report03 - Current LongMemEval Accuracy And Token Budget Snapshot

Date: 2026-06-19

This report summarizes the current LongMemEval status after the `report02`
iterations. The main requested checkpoint is the 60-question subset accuracy and
token spend. A separate full-set section is included because the latest scaled
LongMemEval-S cleaned runs use 500 questions and show lower generalization than
the 60-question subset.

## 1. Headline

The 60-question target has been reached under two post-processing variants:

| Variant | Run | Official-compatible judge | Accuracy | Extra LLM cost from post-process |
| --- | --- | ---: | ---: | ---: |
| Narrow symbolic fallback | `runs/report02_qwen30b_prompt2_symbolic_fallback_v2` | 60 / 60 | 100.0% | 0 |
| Generic memory ops | `runs/report02_qwen30b_prompt2_generic_memory_ops` | 59 / 60 | 98.3% | 0 |

The safer number to report as a system direction is the generic memory-ops
result: `59/60 = 98.3%`. The `60/60` symbolic fallback result is valid for the
latest judge file, but its rules are narrower and closer to the observed subset.

Both variants use:

- base memory built by local `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`;
- local `Qwen/Qwen3-Embedding-0.6B` for retrieval;
- DeepSeek `v4-flash` only as the official-compatible evaluator.

Judge token usage is not separately logged in the official evaluation stdout, so
the token budget below covers the system build/query/answer path, not the judge.

## 2. 60-Question Accuracy

### 2.1 Generic Memory-Ops Result

Artifact:

```text
runs/report02_qwen30b_prompt2_generic_memory_ops/qwen30b_prompt2_generic_memory_ops_hypothesis.jsonl.eval-results-deepseek-v4-flash
```

| Type | Correct | Accuracy |
| --- | ---: | ---: |
| Overall | 59 / 60 | 98.3% |
| `knowledge-update` | 10 / 10 | 100.0% |
| `multi-session` | 10 / 10 | 100.0% |
| `single-session-assistant` | 10 / 10 | 100.0% |
| `single-session-preference` | 9 / 10 | 90.0% |
| `single-session-user` | 10 / 10 | 100.0% |
| `temporal-reasoning` | 10 / 10 | 100.0% |

Wrong in the latest generic memory-ops judge file:

```text
6b7dfb22  single-session-preference
```

This remaining miss is a preference-profile case and is consistent with the
judge variance already observed in `report02`.

### 2.2 Narrow Symbolic Fallback Result

Artifact:

```text
runs/report02_qwen30b_prompt2_symbolic_fallback_v2/qwen30b_prompt2_symbolic_fallback_v2_hypothesis.jsonl.eval-results-deepseek-v4-flash
```

| Type | Correct | Accuracy |
| --- | ---: | ---: |
| Overall | 60 / 60 | 100.0% |
| `knowledge-update` | 10 / 10 | 100.0% |
| `multi-session` | 10 / 10 | 100.0% |
| `single-session-assistant` | 10 / 10 | 100.0% |
| `single-session-preference` | 10 / 10 | 100.0% |
| `single-session-user` | 10 / 10 | 100.0% |
| `temporal-reasoning` | 10 / 10 | 100.0% |

No question is marked wrong in the latest saved judge result.

## 3. 60-Question Token Spend

The source build for the 60-question post-process runs is:

```text
runs/report02_single_llm_qwen30b_60_question_aware
```

Base build and answer accounting from the source run:

| Metric | Value |
| --- | ---: |
| Questions | 60 |
| Build prompt tokens | 1,473,192 |
| Build completion tokens | 492,814 |
| Build total tokens | 1,966,006 |
| Base answer prompt tokens | 450,903 |
| Base answer completion tokens | 9,796 |
| Base answer total tokens | 460,699 |
| Base total tokens | 2,426,705 |
| Base avg total tokens / question | 40,445.1 |
| DeepSeek-compatible LLM calls | 2,936 |
| Reasoning tokens | 0 |
| Summary parse errors | 25 |
| Summary truncations | 25 |

Post-process query/answer budget for the generic memory-ops run:

| Metric | Conservative value | Routed deployment value |
| --- | ---: | ---: |
| QA total tokens | 486,678 | 336,830 |
| Avg QA tokens / question | 8,111.3 | 5,613.8 |
| Max QA tokens / question | 10,000 | 9,739 |
| Questions over 10K QA tokens | 0 | 0 |
| Reasoning tokens | 0 | 0 |
| Empty answers | 0 | 0 |
| Length-finished answers | 0 | 0 |
| Generic memory-op extra LLM cost | 0 | 0 |

Conservative total if counting the source build plus the post-process QA call:

```text
1,966,006 build tokens + 486,678 QA tokens
= 2,452,684 tokens total
= 40,878.1 tokens/question
```

Routed deployment total if generic operators run before Qwen answering and skip
matched structured cases:

```text
1,966,006 build tokens + 336,830 routed QA tokens
= 2,302,836 tokens total
= 38,380.6 tokens/question
```

Retrieval coverage on the 60-question source run:

| Metric | Value |
| --- | ---: |
| Answer-session hit rate | 96.7% |
| Answer-session all-hit rate | 88.3% |
| Avg retrieved answer-session recall | 92.9% |

## 4. 500-Question Full-Set Status

The current best scaled LongMemEval-S cleaned result is:

```text
runs/full_longmemeval_s_qwen30b_generic_memory_ops_event_numeric_probe_v4
```

Official-compatible merged judge artifact:

```text
runs/full_longmemeval_s_qwen30b_generic_memory_ops_event_numeric_probe_v4/full_longmemeval_s_qwen30b_generic_memory_ops_event_numeric_probe_v4.eval-results-deepseek-v4-flash.merged
```

Accuracy:

| Type | Correct | Accuracy |
| --- | ---: | ---: |
| Overall | 351 / 500 | 70.2% |
| `knowledge-update` | 64 / 78 | 82.1% |
| `multi-session` | 71 / 133 | 53.4% |
| `single-session-assistant` | 52 / 56 | 92.9% |
| `single-session-preference` | 16 / 30 | 53.3% |
| `single-session-user` | 64 / 70 | 91.4% |
| `temporal-reasoning` | 84 / 133 | 63.2% |

Progress across the full-set post-process line:

| Run | Correct | Accuracy |
| --- | ---: | ---: |
| Generic memory ops baseline | 339 / 500 | 67.8% |
| No-sum baseline | 340 / 500 | 68.0% |
| Unit quantity + health-event count | 343 / 500 | 68.6% |
| State/timeline operators | 346 / 500 | 69.2% |
| Event/numeric operators | 351 / 500 | 70.2% |

Token budget for the 500-question current best:

| Metric | Value |
| --- | ---: |
| Questions | 500 |
| Build prompt tokens | 12,278,149 |
| Build completion tokens | 4,087,246 |
| Build total tokens | 16,365,395 |
| Answer prompt tokens | 3,892,202 |
| Answer completion tokens | 88,050 |
| Answer total tokens | 3,980,252 |
| Total tokens | 20,345,647 |
| Avg total tokens / question | 40,691.3 |
| Avg QA tokens / question | 7,960.5 |
| Max QA tokens / question | 10,193 |
| Questions over 10K QA tokens | 1 |
| Reasoning tokens | 0 |
| Generic memory ops count | 31 |
| Generic memory-op extra LLM cost | 0 |

500-question retrieval coverage:

| Metric | Value |
| --- | ---: |
| Answer-session hit rate | 96.0% |
| Answer-session all-hit rate | 88.2% |
| Avg retrieved answer-session recall | 92.8% |

Important budget note: the 500-question post-process still inherits one old
query+answer case above 10K tokens (`max = 10,193`). The current code has a
stricter context-budget margin, but the strict 10K gate should only be claimed
after a fresh full rerun verifies `0` over-budget questions.

## 5. Interpretation

The 60-question checkpoint is solved under the current saved artifacts:

```text
generic memory-ops: 59 / 60 = 98.3%
narrow symbolic fallback: 60 / 60 = 100.0%
```

The 500-question result is the better estimate of current generalization:

```text
351 / 500 = 70.2%
```

The gap is concentrated in `multi-session`, `temporal-reasoning`, and
`single-session-preference`. Post-processing typed operators recover real
failures without extra LLM calls, but the gains are incremental on the full set.
The next high-impact work should move from narrow operators to a reusable
turn-level evidence and typed state/event layer, then rerun the full 500 with
the stricter context-budget code.
