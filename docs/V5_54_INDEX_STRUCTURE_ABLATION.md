# V5.54 Index-Structure Ablation

This experiment replaces the historical V5.20 monotonic mechanism chain with
two separately attributable evaluations.

## Query-side factorial

The full 2,040-question corpus is evaluated at 32 and 64 evidence turns. Every
arm uses the same V5.21 Safe-Witness authority graph, source-turn FAISS indexes,
QueryIR implementation, 12K evidence-token ceiling, Qwen3-30B answer model,
V5.54 label-free readout, and GPT-5.6-luna judge.

| Arm | Hierarchical routing | Relation traversal |
|---|---:|---:|
| `seed_only` | off | off |
| `hierarchy_only` | on | off |
| `flat_graph` | off | on |
| `full` | on | on |

The runner records the effective navigator values rather than the raw CLI
flags. `audit_v5_54_index_ablation.py` rejects an arm if a disabled hierarchy
has non-zero hierarchy-route timings or disabled traversal has relation-signal
visits. It also freezes the aggregate authority graph checksum and verifies all
question IDs, prompt policies, and budgets. Its final gate independently
recomputes API Token sums and nearest-rank p50/p95/p99/max statistics, verifies
prepared/answer/usage prompt hashes, and checks every paired Luna verdict
against the SHA-256 of the exact prediction bytes.

Accuracy differences use paired McNemar tests and a Memory-cluster bootstrap.
The interaction is

```text
A(full) - A(hierarchy_only) - A(flat_graph) + A(seed_only).
```

## Construction-side Pareto

The controlled scaling benchmark compares exact All-pairs counts, flat sparse
candidate generation, and the current Coarsen--Reconnect--Refine path. Token
values in this microbenchmark are a fixed per-candidate relation-decision
envelope, not API usage. The real-corpus panel independently reads the frozen
510-Memory build ledger and reconstructs the per-Memory All-pairs bound from
the authority database.

Semantic extraction Token and relation-induction Token must remain separate.
The Safe-Witness rebuild reused frozen semantic extraction and made no new
generative relation calls, so its zero generated Token is reported as such and
the cached extraction ledger is not presented as relation-edge cost.

## Reproduction

```bash
V554_INDEX_PHASE=prepare bash scripts/run_v5_54_index_structure_ablation.sh
V554_INDEX_PHASE=answer bash scripts/run_v5_54_index_structure_ablation.sh
V554_INDEX_PHASE=judge bash scripts/run_v5_54_index_structure_ablation.sh
V554_INDEX_PHASE=summarize bash scripts/run_v5_54_index_structure_ablation.sh
```

Every phase is resumable. Local vLLM failures are retried after the managed
service restarts; judge deltas carry a prior verdict only when prediction bytes
are identical. A single resumable judge slice can be selected with
`V554_INDEX_ONLY_BUDGET`, `V554_INDEX_ONLY_ARM`, and
`V554_INDEX_ONLY_BENCHMARK`; this is useful for keeping aggregate remote
concurrency below the service saturation point.
