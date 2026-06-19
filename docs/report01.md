# report01 - Direct GraphMem V2 Demo Report

## 1. Report Scope

This report records the first successful real 10-question run of the current
token-efficient GraphMem demo on the `multi-session` subset of LongMemEval-S.
It focuses on:

1. the method that currently works end to end,
2. the experiment setup and compared variants,
3. DeepSeek token cost at aggregate, stage, and question level,
4. local embedding and LLMLingua2 auxiliary cost,
5. retrieval coverage and manually judged answer accuracy.

The reported run artifacts are under:

- run summary: `GraphMem/runs/token_demo_v2_deepseek10/summary.md`
- primary GraphMem stats:
  `GraphMem/runs/token_demo_v2_deepseek10/direct_session_k16_compact_graphmem/`
- primary manual evaluation:
  `GraphMem/runs/token_demo_v2_deepseek10/direct_session_k16_compact_graphmem/manual_eval.md`

The main result is:

```text
direct_session_k16_compact_graphmem
10-question average DeepSeek tokens = 29,766.2 / question
manual strict accuracy = 8 / 10
manual relaxed accuracy = 8 / 10
```

The token target of `< 300K DeepSeek tokens / question` is already satisfied
with large margin. The next bottleneck is accuracy on multi-session coverage and
answer-time reasoning, not DeepSeek token budget.

## 2. Current Working Method

### 2.1 Design Goal

The successful path is a direct-session GraphMem build intended to avoid the
large leaf-summary cost of the older K-way tree.

The method keeps the following separation:

```text
raw leaf evidence = answer-time source of truth
compressed or summarized text = build and retrieval aid
```

Raw user-assistant evidence is never replaced by compressed text or summary
text.

### 2.2 Memory Construction

For each question:

1. Load only that question's `haystack_sessions`.
2. Convert session turns into raw leaf evidence nodes.
3. Keep full raw leaf text and session metadata.
4. Build a user-focused text view for summary build and leaf retrieval.
5. Summarize sessions with the direct-session K16 builder.
6. Embed leaves and session summaries with the local embedding service.
7. Build root graph edges from local embedding similarity plus temporal
   adjacency.

The direct-session builder avoids the previous small leaf-group summary layer
for normal sessions:

```text
raw leaves of one session -> one compact session summary
```

Only unusually long sessions use the fallback path:

```text
raw leaves -> raw-group summaries -> session merge summary
```

In this run:

- `480` sessions produced `483` summary nodes.
- `479` summary calls were direct session summaries.
- `3` calls were long-session raw-group summaries.
- `1` call was a session merge summary.

### 2.3 Compact Summary Schema

The V2 summary schema is intentionally short:

```json
{
  "m": ["short memory fact or update"],
  "k": ["keyword"]
}
```

The build prompt asks DeepSeek to keep short user memory facts, including:

- preferences,
- events and visits,
- purchases and costs,
- counts and dates,
- updates and negations.

Assistant filler and generic advice are not supposed to dominate the summary.

### 2.4 Compression Policy

The primary GraphMem variant compresses only build inputs with LLMLingua2:

```text
LLMLingua2 input = user-focused child text for summary build
LLMLingua2 output = prompt material for DeepSeek summary extraction
```

Compression does not change answer evidence. Final QA context still uses raw
leaf evidence.

The LLMLingua2 model used by the run is:

```text
microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
```

### 2.5 Retrieval And Answering

The retrieval path is root + leaf hybrid:

1. Retrieve relevant session roots.
2. Expand root candidates through graph neighbors.
3. Retrieve leaves under root candidates.
4. Add global leaf fallback from all raw leaves.
5. Select a bounded cross-session raw evidence set.
6. Add only a small set of matching session summary hints.
7. Answer from raw evidence with DeepSeek.

The root graph itself does not add DeepSeek build tokens. Graph edges in this
demo are built from local embeddings and temporal adjacency only.

## 3. Experiment Setup

### 3.1 Data

Input data:

```text
GraphMem/data/longmemeval_s_subset_10_per_type.json
```

Selection:

```text
question_type = multi-session
max_questions = 10
```

Run scale:

| Item | Count |
| --- | ---: |
| Questions | 10 |
| Haystack sessions | 480 |
| Raw leaf nodes | 2,525 |

### 3.2 Models And Services

| Component | Setting |
| --- | --- |
| Answer and summary LLM | `deepseek-v4-pro` |
| Build summary thinking | disabled |
| Final QA thinking | enabled |
| Final QA reasoning effort | high |
| Embedding endpoint | local OpenAI-compatible service on `127.0.0.1:8002` |
| Embedding model | `Qwen/Qwen3-Embedding-0.6B` |
| Compressor | LLMLingua2 |
| Compressor model | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` |

### 3.3 Reported Variant Matrix

| Variant | Direct session | Compact schema | Build compression | Root graph | Role |
| --- | --- | --- | --- | --- | --- |
| `direct_session_k16_compact_no_compress` | yes | yes | no | no | cost comparator |
| `direct_session_k16_compact_graphmem` | yes | yes | yes | yes | primary method |

Reported run configuration:

| Parameter | Value |
| --- | ---: |
| `fanout_k` | 16 |
| `max_group_rough_tokens` | 6000 |
| `root_candidate_k` | 8 |
| `global_leaf_top_k` | 16 |
| `leaf_top_k` for this run | 10 |
| `per_session_leaf_k` | 2 |
| `qa_summary_top_k` | 4 |
| `graph_neighbor_k` | 2 |
| `question_workers` | 2 |
| `summary_workers` | 0 |
| `max_inflight_deepseek` | 0 |

The current code may change defaults after this run. The table above is the
configuration represented by the reported artifacts.

## 4. Main Cost Results

### 4.1 DeepSeek Aggregate Cost

| Variant | DeepSeek calls | Build input | Build output | Answer input | Answer output | Reasoning | Total DeepSeek tokens | Avg / question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_session_k16_compact_no_compress` | 493 | 236,778 | 83,005 | 53,717 | 4,887 | 4,109 | 378,387 | 37,838.7 |
| `direct_session_k16_compact_graphmem` | 493 | 147,239 | 91,155 | 54,566 | 4,702 | 4,181 | 297,662 | 29,766.2 |

`reasoning` is logged as a provider-returned completion detail for answer calls.
It is reported explicitly but is not added a second time when computing
`Total DeepSeek tokens`.

Primary method budget check:

```text
target average budget     < 300,000 DeepSeek tokens / question
observed average budget   = 29,766.2 DeepSeek tokens / question
budget result             = pass
```

Compression comparison:

| Comparison | Reduction |
| --- | ---: |
| Total DeepSeek tokens | 80,725 |
| Total DeepSeek token reduction | 21.3% |
| Build input tokens | 89,539 |
| Build input token reduction | 37.8% |

The main compression gain is on summary-build prompt input. Answer prompt cost
stays close between the two variants because both variants answer from raw leaf
evidence.

### 4.2 Primary Variant DeepSeek Cost By Stage

Primary variant: `direct_session_k16_compact_graphmem`.

| Stage | Calls | Prompt tokens | Completion tokens | Reasoning tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `build_summary_session_direct` | 479 | 146,270 | 90,521 | 0 | 236,791 |
| `build_summary_raw_group` | 3 | 658 | 424 | 0 | 1,082 |
| `build_summary_session_merge` | 1 | 311 | 210 | 0 | 521 |
| `answer_qa` | 10 | 54,566 | 4,702 | 4,181 | 59,268 |

Build cost remains the dominant DeepSeek expense:

```text
build summary tokens = 238,394
answer QA tokens     = 59,268
```

### 4.3 Primary Variant Prompt Cache Breakdown

| Stage | Prompt tokens | Cache hit tokens | Cache miss tokens |
| --- | ---: | ---: | ---: |
| `build_summary_session_direct` | 146,270 | 18,816 | 127,454 |
| `build_summary_raw_group` | 658 | 384 | 274 |
| `build_summary_session_merge` | 311 | 0 | 311 |
| `answer_qa` | 54,566 | 0 | 54,566 |
| Total | 201,805 | 19,200 | 182,605 |

### 4.4 Primary Variant Per-Question DeepSeek Cost

| Question ID | Total tokens | Build input | Build output | Answer input | Answer output | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aae3761f` | 28,295 | 13,608 | 8,715 | 5,454 | 518 | 48 |
| `gpt4_f2262a51` | 28,582 | 13,573 | 8,759 | 5,560 | 690 | 46 |
| `d682f1a2` | 29,649 | 15,521 | 9,067 | 4,717 | 344 | 48 |
| `81507db6` | 30,492 | 15,353 | 9,325 | 5,088 | 726 | 51 |
| `e5ba910e_abs` | 29,866 | 14,437 | 9,206 | 6,000 | 223 | 49 |
| `60bf93ed_abs` | 29,628 | 15,001 | 9,315 | 4,891 | 421 | 52 |
| `gpt4_d84a3211` | 30,995 | 14,839 | 9,638 | 6,035 | 483 | 49 |
| `1a8a66a6` | 29,755 | 14,160 | 9,374 | 5,801 | 420 | 52 |
| `73d42213` | 29,770 | 15,371 | 8,629 | 5,318 | 452 | 50 |
| `88432d0a_abs` | 30,630 | 15,376 | 9,127 | 5,702 | 425 | 48 |

Observed per-question range:

```text
min = 28,295 DeepSeek tokens
max = 30,995 DeepSeek tokens
```

The direct-session V2 path is therefore stable on this 10-question subset at
roughly `28K-31K` DeepSeek tokens per question.

## 5. Local Auxiliary Cost

Local embedding and compression usage is recorded separately from DeepSeek API
tokens.

### 5.1 Local Embedding

Primary variant embedding stats:

| Item | Value |
| --- | ---: |
| Embedding API calls | 64 |
| Embedded items | 3,018 |
| Local embedding prompt tokens | 234,594 |
| Local embedding total tokens | 234,594 |
| Recorded embedding latency sum | 4.64 s |

The local embedding token count is not included in the DeepSeek total.

### 5.2 LLMLingua2 Compression

Primary variant compression stats:

| Item | Value |
| --- | ---: |
| Compression calls | 483 |
| Origin rough tokens | 170,513 |
| Compressed rough tokens | 81,052 |
| Total compression chunks | 690 |
| Max chunks in one compression call | 9 |
| Recorded compression latency sum | 17,556.18 s |

The compression latency number is a sum over concurrent compression records, not
the end-to-end run wall time. It still shows that the current compressed path
trades local compute time for lower DeepSeek build input cost.

## 6. Retrieval And Graph Results

### 6.1 Primary Graph Size

| Item | Value |
| --- | ---: |
| Leaf nodes | 2,525 |
| Summary nodes | 483 |
| Graph edges | 1,248 |
| Build calls per session | 1.00625 |

The graph edge count adds no DeepSeek edge-refinement token in this experiment.

### 6.2 Answer-Session Coverage

Primary variant retrieval metrics:

| Metric | Value |
| --- | ---: |
| Retrieved answer-session any-hit rate | 1.0 |
| Retrieved answer-session all-hit rate | 0.8 |
| Average answer-session recall | 0.9 |

Per-question answer-session recall:

| Question ID | Recall |
| --- | ---: |
| `aae3761f` | 1.0 |
| `gpt4_f2262a51` | 1.0 |
| `d682f1a2` | 1.0 |
| `81507db6` | 1.0 |
| `e5ba910e_abs` | 1.0 |
| `60bf93ed_abs` | 0.5 |
| `gpt4_d84a3211` | 1.0 |
| `1a8a66a6` | 0.5 |
| `73d42213` | 1.0 |
| `88432d0a_abs` | 1.0 |

### 6.3 Retrieval Top-K Policy

The retrieval report below describes the successful 10-question artifact in
`token_demo_v2_deepseek10`, not a later tuned rerun.

For the reported primary run, hybrid retrieval used:

| Retrieval step | Reported run value | Meaning |
| --- | ---: | --- |
| Root candidate retrieval | `root_candidate_k = 8` | Rank session-root summaries by query embedding and take the first 8 root candidates. |
| Graph expansion | `graph_neighbor_k = 2` | Expand each selected root candidate through up to 2 root-graph neighbors. |
| Global raw-leaf fallback | `global_leaf_top_k = 16` | Add the top 16 raw-leaf candidates from the full per-question leaf set. |
| Final raw evidence | `leaf_top_k = 10` | Send at most 10 selected raw leaf evidence chunks into QA context. |
| Per-session raw evidence cap | `per_session_leaf_k = 2` | Prefer cross-session coverage before filling the final leaf budget. |
| Final summary hints | `qa_summary_top_k = 4` | Send at most 4 session-root summary hints into QA context. |

The graph-expanded root candidate pool is not injected wholesale into the QA
prompt. It is used to collect candidate raw leaves. The final QA prompt receives
only:

```text
up to 4 session-root summary hints
+ up to 10 raw leaf evidence chunks
```

For this reported run every question hit both caps:

```text
4 summary hints / question
10 raw leaf evidence chunks / question
```

The `direct_session` method does not put legacy K-way leaf-group summaries into
the answer context. The selected summary IDs in this run are all direct session
summary nodes with IDs containing:

```text
summary:build_summary_session_direct
```

Therefore the answer context mix is:

| Context type | Count across 10 questions | Stored text type |
| --- | ---: | --- |
| Raw leaf evidence | 100 | Original raw user-assistant leaf chunks |
| Session-root summary hints | 40 | Compact DeepSeek summary nodes |
| Legacy leaf-group/internal K-way summaries | 0 | Not used by this direct-session run |

### 6.4 Retrieval Context Volume Per Question

`Raw evidence chars` and `summary chars` below are measured from the exact
`context_text` sent to answer QA, split at the `Raw evidence:` section marker.
They measure serialized context size, not DeepSeek tokenizer counts. The
DeepSeek answer prompt token count is reported separately.

| Question ID | Summary hints | Raw leaves | Retrieved sessions | Graph edges used | Summary chars | Raw evidence chars | Context chars | Answer prompt tokens | Answer-session recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aae3761f` | 4 | 10 | 6 | 16 | 3,234 | 18,744 | 21,995 | 5,454 | 1.0 |
| `gpt4_f2262a51` | 4 | 10 | 6 | 14 | 3,967 | 20,556 | 24,540 | 5,560 | 1.0 |
| `d682f1a2` | 4 | 10 | 6 | 16 | 3,521 | 16,319 | 19,857 | 4,717 | 1.0 |
| `81507db6` | 4 | 10 | 7 | 16 | 3,472 | 21,018 | 24,507 | 5,088 | 1.0 |
| `e5ba910e_abs` | 4 | 10 | 7 | 14 | 3,528 | 21,848 | 25,393 | 6,000 | 1.0 |
| `60bf93ed_abs` | 4 | 10 | 8 | 15 | 3,180 | 18,028 | 21,225 | 4,891 | 0.5 |
| `gpt4_d84a3211` | 4 | 10 | 6 | 14 | 3,249 | 23,898 | 27,164 | 6,035 | 1.0 |
| `1a8a66a6` | 4 | 10 | 8 | 15 | 3,741 | 23,796 | 27,554 | 5,801 | 0.5 |
| `73d42213` | 4 | 10 | 8 | 16 | 3,278 | 19,624 | 22,919 | 5,318 | 1.0 |
| `88432d0a_abs` | 4 | 10 | 6 | 14 | 3,966 | 21,291 | 25,274 | 5,702 | 1.0 |

Across the 10 answer contexts:

| Serialized context type | Characters |
| --- | ---: |
| Summary section | 35,136 |
| Raw evidence section | 205,122 |

The raw evidence text is therefore the dominant answer-context payload. Summary
hints are a smaller localization aid.

### 6.5 Full Retrieval Records

The complete retrieval record for every reported question is already stored in:

```text
GraphMem/runs/token_demo_v2_deepseek10/direct_session_k16_compact_graphmem/retrieval_results.jsonl
```

Each JSONL row contains:

| Field | Meaning |
| --- | --- |
| `question_id` | Question key |
| `summary_node_ids` | Exact summary nodes inserted into the context |
| `leaf_node_ids` | Exact raw leaf evidence nodes inserted into the context |
| `retrieved_session_ids` | Distinct sessions represented by selected raw leaves |
| `edge_count` | Number of graph edges used while expanding root candidates |
| `context_text` | Full serialized answer context, including summary hints and full raw evidence text |
| `answer_session_hit` | Whether any gold answer session was retrieved |
| `answer_session_all_hit` | Whether all gold answer sessions were retrieved |
| `answer_session_recall` | Gold answer-session recall for this question |

The node payloads used by those IDs are stored in:

```text
GraphMem/runs/token_demo_v2_deepseek10/direct_session_k16_compact_graphmem/nodes.jsonl
```

This means the run preserves both levels of audit detail:

1. a compact per-question retrieval breakdown in this report,
2. the full retrieved text and exact node IDs in the JSONL artifacts.

## 7. Answer Accuracy

### 7.1 Evaluation Policy

Manual evaluation is applied to the primary optimized variant:

```text
direct_session_k16_compact_graphmem
```

Strict policy:

- numeric questions must state the gold numeric result,
- insufficient-information gold answers must explicitly say information is
  insufficient,
- conflicting or wrong numeric answers are incorrect.

Relaxed policy:

- semantic match to the gold answer is allowed,
- minor extra explanation is allowed if it does not change the final answer.

No LLM judge is used. Judge token cost is therefore zero.

### 7.2 Primary Accuracy

| Metric | Result |
| --- | ---: |
| Strict correct | 8 / 10 |
| Strict accuracy | 80% |
| Relaxed correct | 8 / 10 |
| Relaxed accuracy | 80% |

Correct cases:

| Question | Result |
| --- | --- |
| road trip total driving hours | correct |
| number of doctors | correct |
| number of delivery services | correct |
| graduation ceremonies attended | correct |
| headphones plus missing iPad cost | correct insufficient-information answer |
| missing iPad case arrival | correct insufficient-information answer |
| bike expense total | correct |
| missing egg tart baking count | correct insufficient-information answer |

Incorrect cases:

| Question | Error type | Reason |
| --- | --- | --- |
| magazine subscription count | `partial_multi_session_recall` | The context misses answer sessions carrying the Architectural Digest subscription and predicts `1` instead of gold `2`. |
| clinic arrival time | `answer_reasoning_error` | Retrieved evidence contains a `7 AM` departure and a two-hour trip, but the answer does not infer the gold `9:00 AM` arrival. |

## 8. Runtime And Quality Signals

### 8.1 Wall Time

| Variant | Sum of question wall times | Actual reported variant elapsed time |
| --- | ---: | ---: |
| `direct_session_k16_compact_no_compress` | 239.89 s | 123.82 s |
| `direct_session_k16_compact_graphmem` | 1354.63 s | 696.38 s |

The compressed GraphMem path reduces DeepSeek build input tokens but costs much
more local wall time in this run because LLMLingua2 compression is expensive.

### 8.2 Summary Parse And Truncation Signals

| Variant | Summary parse errors | Summary truncations |
| --- | ---: | ---: |
| `direct_session_k16_compact_no_compress` | 16 | 10 |
| `direct_session_k16_compact_graphmem` | 20 | 20 |

The compact schema is usable, but summary output caps still truncate a minority
of real summary calls. This should be tracked in later runs when adjusting
summary output budgets.

## 9. Current Conclusion

The first successful method is:

```text
direct-session K16 compact GraphMem
+ user-focused build view
+ LLMLingua2 build-input compression
+ local embedding root graph
+ hybrid root/global-leaf retrieval
+ raw evidence QA
```

It succeeds on the main token-efficiency objective:

```text
29.8K DeepSeek tokens / question on the 10-question multi-session subset
```

It also shows the expected cost tradeoff:

- LLMLingua2 compression reduces DeepSeek build prompt tokens substantially.
- The current compressed path is slower than no-compress because the local
  compressor is expensive.
- Graph edge construction adds retrieval structure without adding DeepSeek edge
  refinement tokens.

The current accuracy result is useful but not yet the target:

```text
current strict accuracy = 8 / 10
target strict accuracy  = at least 9 / 10
```

The next accuracy work should prioritize:

1. stronger multi-session leaf coverage for count/current-state questions,
2. answer instructions and evidence presentation for direct arithmetic and time
   inference,
3. a stable rerun of the tuned retrieval/QA path before spending more build
   tokens.
