# V5.20 graph-structure and evidence-budget evaluation

## Frozen implementation

- Core implementation: `ba9421f`.
- Reproducible experiment harness: `08bb40e`.
- Streaming navigation/answer pipeline: `d375d4b`.
- Build/answer model: Qwen3-30B-A3B-Instruct-2507-FP8.
- Judge: `gpt-5.6-luna`, temperature 0, seed 0, reasoning disabled.
- Candidate Answer injection is disabled. Algebraic drafts remain in the audit
  trace and never enter the answer prompt.
- Answer completion is capped at 2,000 tokens. Evidence uses the frozen Qwen
  tokenizer and at most 12,000 tokens.

## Why the relation-signal deletion table is no longer a main ablation

Deleting scene/entity/temporal/lexical signals one at a time does not isolate
the graph mechanism: signals can substitute for each other, and a materialized
edge can exist without being traversed or used by the answer model. The main
ablation therefore follows the complete causal path from candidate retrieval to
the final answer on one immutable multi-attribute graph.

## Hard200 graph-mechanism ladder

The set contains 50 LongMemEval multi-session, 50 LongMemEval temporal, 50
LoCoMo category-1 multi-hop, and 50 LoCoMo category-2 temporal questions. Every
arm uses the same source graph, FAISS indexes, QueryIR, 64-turn budget, prompt
contract, answer model, and Luna judge.

1. `seed_only`: QueryIR plus dense/lexical seeds and the flat fact reservoir;
   relation traversal and hierarchical seed routing are disabled.
2. `flat_graph`: enables relation traversal while hierarchical seed routing
   remains disabled.
3. `hierarchical`: additionally enables coarse-to-fine seed routing and
   bounded top-down expansion.
4. `topology_layout`: keeps the same hierarchical retrieval and groups evidence
   by QueryIR operand/graph branch in root-to-leaf order. Unbound candidates are
   placed in small relevance-anchored auxiliary windows.

The report includes the four original strata, structural/temporal aggregates,
and the overall result. Each adjacent transition and the end-to-end
`seed_only -> topology_layout` transition report paired gains/losses, exact
McNemar p-value, and a 10,000-sample paired-bootstrap 95% confidence interval.

Retrieval diagnostics include final/candidate recall, precision, F1 and
all-hit; candidate AP/NDCG; graph-reachable recall/precision; visited
nodes/edges; latency; and the number of evidence turns rendered as `CHAIN`,
`GRAPH`, or `AUX`. This distinguishes an absent path, an unused path, a noisy
path, and evidence that was retrieved but not converted into a correct answer.

## Full 32/64-turn budget curve

Both arms run all 500 LongMemEval and 1,540 LoCoMo category 1--4 questions on
the same 510-memory graph. Only `max_evidence_turns` changes. Both use
topological evidence layout, the 12K evidence ceiling, the 2K completion cap,
and 256 total local-API concurrency split across simultaneously running jobs.

The comparison to Mem0 top-50/top-200 is made by measured API Answer Token and
accuracy, not by treating a GraphMem source turn and a Mem0 extracted item as
the same unit. GraphMem build cost is shared by its 32/64-turn rows; each Mem0
cutoff uses its own archived per-question answer usage.

## Artifact roots

- Hard200: `artifacts/report/v5_20/graph_structure_ablation_dev200`
- Full budget curve: `artifacts/report/v5_20/full_budget_benchmark`

Both launchers are resumable and wait for the managed local services to restart
after connection failures. Summary JSON is rendered into the report only after
all expected question IDs and Luna verdicts are present.
