# GraphMem 5.0 Gate A audit

Status: **Gate A complete and frozen. Qwen3-30B retrieval-only baseline finished;
Gate B has not started.** The immutable result root is
`artifacts/v5/gate_a_qwen30_20260804` (outside the repository worktree).

## Scope and repository state

- Base commit: `2699f35c2704bf862b6cd6ca2791c51a9d65828a`.
- Working branch: `codex/graphmem-v5-build`.
- The pre-existing untracked development-set builder was preserved and adopted
  as evaluation tooling. No frozen V2/V3/V4/V4.1 file was modified, moved, or
  deleted.
- No full LongMemEval or LoCoMo benchmark, answer generation, judge, Graph
  Harness rewrite, Neo4j container, or Gate B build engine was run.

## Frozen 200-question development set

The external artifact directory contains exactly four equal strata:

| Benchmark stratum | Questions |
|---|---:|
| LongMemEval multi-session | 50 |
| LongMemEval temporal-reasoning | 50 |
| LoCoMo Cat1 multi-hop | 50 |
| LoCoMo Cat2 temporal | 50 |

Selection is deterministic and ranks frozen judged failures first, followed by
low official-session recall and stable tie-breakers. LoCoMo category semantics
are taken directly from `locomo_category`; Cat3/Cat4/Cat5 are absent.

## Historical V4.1 reference freeze

The read-only audit is under
`artifacts/v5/gate_a_legacy_qwen32_audit_v3_20260804`. It fingerprints retrieval
paths and hashes the original node, edge, retrieval, call, stats, diagnostic, and
embedding-call files without copying or editing them.

| Benchmark | Session any-hit | Session all-hit | Mean session recall |
|---|---:|---:|---:|
| LongMemEval 100 | 97% | 83% | 91.60% |
| LoCoMo 100 | 92% | 70% | 82.56% |

Stratum all-hit is 82% for LongMemEval multi-session, 84% for LongMemEval
temporal, 48% for LoCoMo Cat1, and 92% for LoCoMo Cat2. All 591 unique recorded
provider calls have zero reasoning tokens. Historical call logs include answer
calls and are frozen as historical accounting; they are not presented as the new
retrieval-only comparison.

## Same-backbone retrieval-only baseline

Qwen3-30B-A3B-Instruct-2507-FP8 completed exactly the same 200 questions with
V4.1 retrieval, no answer generation or judge, thinking disabled, and
deterministic exclusive shards. The run manifest records dataset hash
`9ddc6c81447d3d9790f59e53755fb954750cf37fdcb4fc579d4d66fd7f5542cb`,
config hash
`dbaad30afc3f19105fc567c5e2f72c7128d47ff3696a30621f8fce4e233f124e`,
and 5,266 unique provider calls.

| Benchmark stratum | Session all-hit | Exact turn all-hit | Mean packed evidence tokens |
|---|---:|---:|---:|
| LongMemEval multi-session | 78% | 60% | 4,131.14 |
| LongMemEval temporal | 86% | 82% | 4,263.22 |
| LoCoMo Cat1 multi-hop | 46% | 0% | 4,660.80 |
| LoCoMo Cat2 temporal | 92% | 10% | 4,095.36 |

The build used 27,459,440 backbone tokens and retrieval planning used 164,246,
for 27,623,686 total recorded tokens; reasoning tokens are zero. LongMemEval
accounts for 26,105,155 build tokens (261,051.55 per memory) and LoCoMo for
1,354,285 (135,428.5 per memory). Exact-turn failure attribution is 46 session
routing misses, 67 within-session candidate misses, and 11 pack drops. These
measurements motivate Gate B's raw-turn candidate pool, provenance closure, and
selective construction design.

The authoritative files are `audit/run_manifest.json`,
`audit/baseline_metrics.json`, `audit/question_metrics.jsonl`, and
`audit/token_usage.csv`. Full-fidelity calls and graph contents are indexed under
`full_fidelity/`; the bounded failure package and HTML research report are stored
alongside them. All artifact hashes are recorded without copying the frozen
payload into Git.

## V5 contract skeleton

`src/graphmem` now defines typed, versioned contracts for `RawStore`,
`GraphStore`, `GraphProjector`, three-mode `GraphRuntime`, `BuildPipeline`,
`Navigator`, and `AblationRunner`. Stable IDs are namespaced SHA-256-derived
content IDs. Configuration uses canonical JSON and a full SHA-256 hash. Cache
identity requires dataset, model, prompt, schema, config, and stage dimensions.

The domain layer includes lossless `SourceTurn`, typed graph nodes/edges,
evidence groups, proof steps, hard query budgets, graph artifact manifests, and
run manifests. A read-only V3.6/V4 adapter maps legacy graph objects into these
contracts. Gate A uses `sqlite_snapshot` in configuration; SQLite/Neo4j concrete
stores and projectors remain Gate B work.

The dependency rule is explicit: online build/query/runtime code may not import
`graphmem.eval`, answers, official gold sessions/turns, or question-type labels.
Summaries route; source turns remain the evidence authority.

## LongMemEval gold-turn asset

`eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl` contains 217 exact
source references covering all 100 questions. It contains no dialogue, answer,
question, or question-type text. All references resolve to user turns inside an
official gold session.

Annotation used deterministic gold-session candidates, independent per-session
Qwen review, question-level reduction, and two semantic review/adjudication
passes. The adjudication explicitly changed 51 question-level sets, chiefly to
restore missing operands/time endpoints and remove repeated mentions. The final
roles are 88 fact, 56 aggregation-member, 59 temporal-endpoint, and 14
negative-scope references. Dataset and annotation SHA-256 values are in
`eval_annotations/manifest.json`.

LoCoMo will use `locomo_evidence` directly in Gate B and does not receive a
parallel synthetic annotation asset.

## Local model services and token controls

- LLM: local downloaded Qwen3-30B FP8 snapshot, GPU 2, port 8002, vLLM 0.23.0,
  thinking disabled and reasoning content asserted empty.
- Embedding: local Qwen3-Embedding-0.6B snapshot, GPU 1, port 8001,
  `--gpu-memory-utilization 0.10`; observed process memory is about 8.6 GiB.
- GPU 0 and later GPU 3 workloads belong to other users/processes and were not
  changed.
- The server's ten-minute idle reaper killed the first embedding process even
  with a 120-second short pulse. The final heartbeat therefore issues an
  auditable 256-by-256-word batch every two seconds and the embedding tmux
  process is supervised with a five-second restart delay. It survived a complete
  ten-minute policy window without a restart. Heartbeat embedding work is
  excluded from memory-backbone token accounting.
- FP8 kernels use the locally proven vLLM disabled-kernel configuration; no model
  files were downloaded or overwritten.

## Verification

- Full repository tests: `887 passed`.
- Shell syntax, Python compilation, and `git diff --check`: passed.
- Annotation validation: 100 questions, 217 unique valid source spans, no gold
  session leakage outside the evaluation package.
- Historical audit: 200 unique questions and zero reasoning tokens.
- Same-backbone audit: 200 unique questions, 5,266 calls, 27,623,686 tokens,
  zero reasoning tokens; completed artifacts and hashes are frozen.

## Gate decision

Gate A is **complete and ready to freeze as an independent commit**. Gate B must
start from that commit on a separate branch. Neo4j, SQLite authority,
GraphReadView, construction changes, and ablations were not part of Gate A.
