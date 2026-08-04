# GraphMem V4.1 dual-backbone evaluation report

Date: 2026-08-04 (Asia/Singapore)

## Evaluation contract

This report separates three quantities that must not be conflated:

1. official judge accuracy;
2. retrieval coverage and lexical token-F1 diagnostics;
3. memory-backbone token consumption.

The memory embedding model is `Qwen3-Embedding-0.6B` for every run. The two
memory backbones are `gpt-5.4-mini` and `Qwen3-32B-FP8`, both with internal
thinking disabled. Official answer judgment uses `gpt-5.4-mini`; embedding and
judge tokens are excluded from memory build and query budgets.

No Qwen score is called an accuracy until all answers have been evaluated by
the same pinned official prompt. Partial runs and token-F1 are reported only as
diagnostics.

## Method

### 1. Hypergraph coarsening followed by evidence refinement

GraphMem first compresses a long memory into routing cards and reliable typed
relations. BM25/FTS, dense similarity and exact structured indexes connect a
question to a small graph region. Lossless turns are retained behind the coarse
nodes and expanded only after routing. A bounded number of LLM extraction calls
correct semantic roles that lexical construction cannot recover. This avoids
placing a full transcript or several duplicate representations of the same fact
in every query context.

For Qwen experiments, selective session correction divides the memory timeline
into bins and chooses the most information-dense session in each bin. Remaining
sessions keep lossless turns and deterministic cards. Selection is independent
of the question, benchmark, topic and gold annotations. This reduces build cost
without deleting source evidence.

### 2. Graph-harness QueryIR for constrained retrieval

A deterministic QueryIR describes target entities, relation, owner, temporal or
collection scope and the roles required to answer. Coarse routing is followed by
fine retrieval from FTS, dense, exact/inverted lookup, source projection and
typed graph edges. An evidence certificate records missing roles and permits at
most two depth-two expansions along relations such as `dialogue_pair`,
`same_event`, `state_transition`, `collection_member` and
`temporal_endpoint`. Wide participant or global temporal hubs are forbidden.

This harness addresses a common failure of both pure vector retrieval and an
unconstrained LLM search: a topically similar passage is not sufficient when a
question requires a particular owner, lifecycle state, time endpoint or complete
set of members.

### 3. Lightweight planner under a shared query budget

Easy questions use deterministic retrieval and spend almost all available
tokens on evidence and the final answer. A short planner is invoked only when
the evidence certificate remains incomplete. It receives the question, QueryIR,
candidate entity/event aliases and missing roles, never the full transcript.
Its proposals must resolve to indexed source evidence and cannot directly become
an answer.

The planner therefore occupies an intermediate Pareto point: zero planner cost
is efficient but misses hard aliases and event identities, while a large agentic
search displaces decisive evidence from the roughly 10K shared query budget.
GraphMem caps the planner and deducts its tokens from the evidence-pack allowance.

## Verified frozen GPT-5.4-mini result

The last fully judged, reproducible frozen accuracy reference is V3.7. It is not
a clean dual-backbone build comparison: LongMemEval reuses the persisted V2
evidence ledger, and the LoCoMo component is the immutable V3.4 peer-dialogue
route. The table is therefore the strongest verified GPT answer result, while the
unified V4.1 rerun remains incomplete:

| Benchmark | Correct | Total | Accuracy |
|---|---:|---:|---:|
| LongMemEval | 445 | 500 | 89.00% |
| LoCoMo Category 1-4 | 1,328 | 1,540 | 86.23% |

Relative to this frozen reference, reaching 93% requires 20 additional correct
LongMemEval answers and 105 additional correct LoCoMo answers; reaching 95%
requires 30 and 135, respectively. The requested 93--95% target has therefore
not yet been demonstrated by a complete, uniformly built and officially judged
run.

LongMemEval by type:

| Type | Correct | Total | Accuracy |
|---|---:|---:|---:|
| single-session-user | 66 | 70 | 94.29% |
| multi-session | 118 | 133 | 88.72% |
| single-session-preference | 28 | 30 | 93.33% |
| temporal-reasoning | 111 | 133 | 83.46% |
| knowledge-update | 68 | 78 | 87.18% |
| single-session-assistant | 54 | 56 | 96.43% |

LoCoMo by category:

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| 1 | 235 | 282 | 83.33% |
| 2 | 259 | 321 | 80.69% |
| 3 | 55 | 96 | 57.29% |
| 4 | 779 | 841 | 92.63% |

Frozen answer-stage token accounting:

| Benchmark | Uncached input | Cached input | Output | Total | Max/question | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| LongMemEval | 2,713,852 | 790,016 | 11,695 | 3,515,563 | 8,725 | 0 |
| LoCoMo | 6,430,303 | 2,077,952 | 255,655 | 8,763,910 | 5,834 | 0 |


## Post-frozen generic query repairs

Two query-only repairs were implemented after the frozen score and are not
credited to any accuracy table until a complete replay is judged:

- Commit 1c4dec2 requires a collection planner with missing member/scope roles
  to produce bounded morphological aliases and common child-type retrieval
  terms. These terms are never evidence and every hit must still pass source,
  owner, relation, scope and lifecycle validation.
- Commit 691bca4 adds a lossless dense quota for completed user assertions
  inside already routed sessions. It admits at most two turns per session and
  eight overall, and rejects questions, plans, negations and assistant
  recommendations before normal evidence packing.

The synthetic regression uses an unseen musical-instrument domain rather than a
benchmark question. The complete repository suite passes 871 tests, including
the static gold/benchmark-branch scan.

## Qwen3-32B-FP8 status and measured costs

LoCoMo memory answering is complete for all 1,540 Category 1-4 questions.
Official GPT judgment is pending; local token-F1 is 0.314539 and is not an
accuracy estimate.

Retrieval diagnostics by category:

| Category | Questions | Any gold session | All gold sessions | Mean session recall | Token-F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 282 | 265 (93.97%) | 170 (60.28%) | 78.84% | 0.234260 |
| 2 | 321 | 310 (96.57%) | 305 (95.02%) | 95.90% | 0.282277 |
| 3 | 96 | 77 (80.21%) | 58 (60.42%) | 71.58% | 0.165046 |
| 4 | 841 | 818 (97.27%) | 818 (97.27%) | 97.27% | 0.370837 |

These figures isolate the remaining Qwen retrieval weakness in multi-evidence
Category 1 and reasoning-heavy Category 3; they are not answer-accuracy claims.

The replay-deduplicated LoCoMo build bill covers exactly 10 conversations:

| Stage | Calls | Uncached input | Cached input | Output | Total |
|---|---:|---:|---:|---:|---:|
| Session extraction | 272 | 409,360 | 0 | 332,427 | 741,787 |
| Consolidation | 10 | 99,923 | 0 | 18,006 | 117,929 |
| Build total | 282 | 509,283 | 0 | 350,433 | 859,716 |

Qwen/vLLM does not expose provider cache-hit token breakdown, so all prompt
tokens are conservatively classified as uncached. Build mean is 85,971.6 tokens
per conversation; P50 is 91,254, P95/max is 99,913; reasoning tokens are zero.

| Query stage | Calls | Uncached input | Cached input | Output | Total |
|---|---:|---:|---:|---:|---:|
| Planner | 1,499 | 1,631,529 | 0 | 173,930 | 1,805,459 |
| Final answer | 1,540 | 15,657,842 | 0 | 37,738 | 15,695,580 |
| Query total | 3,039 | 17,289,371 | 0 | 211,668 | 17,501,039 |

Query mean is 11,364.31 tokens, P50 is 11,335, P95 is 12,909 and max is
15,787. There are 1,473 questions above 10K, 229 above 12K and 70 above 13K.
This run therefore measures the current quality/cost point but does not satisfy
the target mean query budget. Retrieval reaches at least one gold session on
95.45% of questions, all gold sessions on 87.73%, with mean session recall
92.005%.

LongMemEval cap-6 and cap-2 throughput probes were stopped when their measured
end-to-end rates could not guarantee completion by the evaluation deadline. The
final fallback run uses cap-1 in a third isolated result root. Cap-1 keeps every
lossless turn and deterministic routing card but performs LLM semantic correction
for one question-independent, timeline-stratified information-dense session per
question. The selected call is nested inside cap-2/cap-6, so compatible
provider-call checkpoints are reused without reusing either larger-cap final
index. Final 500-question accuracy and token tables must be inserted only after
strict merge, replay-aware token audit and official judgment.

The 09:00 Asia/Singapore evaluation deadline passed while this isolated cap-1
batch was still draining the local Qwen queue. The batch was allowed to finish
instead of being interrupted a fourth time, because another strategy change
would make the resulting build and token accounting non-comparable.

## Reproduction and immutable artifacts

The Qwen runner is `scripts/run_v419_qwen32b_unified_full.sh`. It supports
`RUN_JUDGE=0` for isolated local-memory generation, `V36_LLM_SESSION_CAP` for
question-independent selective correction, and independent question/build
concurrency controls. Token aggregation excludes persisted call replays using
the provider call ID and reports raw and unique record counts.

The final deadline fallback was launched without exposing credentials:

    RUN_ROOT=/home/chenhan/graphmem_v419_qwen32b_cap1_full_20260804 \
    RUN_JUDGE=0 BENCHMARKS=lme V36_LLM_SESSION_CAP=1 \
    LME_PARALLEL_SHARDS=25 LME_INFLIGHT_PER_SHARD=16 \
    LME_QUESTION_WORKERS=20 \
    bash scripts/run_v419_qwen32b_unified_full.sh

The runner loads QWEN_API_KEY and EMBEDDING_API_KEY from the remote private
environment; neither value is written to logs or this report.

Completed Qwen LoCoMo artifacts:

- answers: /home/chenhan/graphmem_v419_qwen32b_unified_full_20260803/locomo/merged/hierarchical_hybrid_graph_v4_1_query/answers.jsonl,
  SHA-256 e2a44dfe468c8605bfc92bffcb630391894f71c9d83159cf4eeed90e8f92f55d;
- retrieval: the same variant directory, retrieval_results.jsonl,
  SHA-256 56ec42fc728005876d09875ba3efec1b03d098d7b65a43455843cb4c0ddf7c3f;
- replay-deduplicated token report:
  /home/chenhan/graphmem_v419_qwen32b_unified_full_20260803/locomo/report_no_judge/summary.json,
  SHA-256 37bf6e4045b1e778d588a3c67b6a4a2310af60f0881a49c14edd6e72f64526e4.

The final cap-1 LongMemEval paths/hashes and all external judge outputs will be
added after strict merge and authorized judgment.
