# GraphMem V3 cross-benchmark validation (2026-07-26)

## Outcome

The current V3 implementation preserves the frozen LongMemEval control result
and the LoCoMo control result while satisfying both per-item token gates.
These control subsets establish absence of an obvious regression; they do not
establish 90% full-benchmark accuracy.

| Evaluation | Judge | Accuracy | Build max | Answer max | Budget failures |
|---|---|---:|---:|---:|---:|
| LongMemEval frozen control | Mem0 `bd063ee` | 10/12 (83.33%) | 242,138/question | 8,574/question | 0 |
| LoCoMo conversation-0 control | memory-benchmarks | 20/20 (100%) | 65,801/conversation | 7,660/question | 0 |

All build and answer calls used `deepseek-v4-flash` with thinking disabled.
The summed API reasoning-token count was zero. Judge and embedding usage are
excluded from both budgets.

## Token accounting

LongMemEval 12 questions:

- build cache-miss input: 1,290,026
- build cache-hit input: 420,736
- build output: 1,019,957
- answer cache-miss input: 48,315
- answer cache-hit input: 43,648
- answer output: 498

LoCoMo 20 questions sharing one conversation:

- build cache-miss input: 1,275
- build cache-hit input: 31,232
- build output: 33,294
- answer cache-miss input: 140,031
- answer cache-hit input: 9,984
- answer output: 258

## Architecture validated

- L0 keeps lossless, role-neutral turns.
- L1 stores source-grounded claims, events, event frames, and typed operands.
- L2/L3 episodes, overlapping themes, state chains, and provenance-bearing
  hyperedges participate directly in retrieval.
- Dense, BM25, and exact channels seed all levels and are fused with RRF.
- Query-conditioned relation posteriors drive bounded, bidirectional
  node-to-hyperedge-to-node expansion to depth two.
- Routed sessions are refined with local turn retrieval, adjacency closure,
  catalog projections, source provenance, and evidence-budget packing.
- Deterministic operators act on local typed evidence for state, temporal,
  count/list, duration, recurrence, ordering, and relative-entity binding.
- The final answer is a single DeepSeek call.

## Generic recommendation regression fix

A LongMemEval recommendation control exposed a cross-domain routing error:
the correct coarse session was present, but a distractor session matched a
request modifier and was promoted as a user constraint. The fix is generic:

1. Select recommendation scopes by coverage of target content terms.
2. Do not let request modifiers such as suggest/complement/current/setup define
   the subject scope.
3. Accept constraints only from compatible speakers/subjects.
4. Prefer owned/current state, preference, and explicit need relations.
5. Keep graph-expanded evidence available, but do not convert a low-scope
   distractor into an authoritative user constraint.

No benchmark name, category, question ID, or topic term is present in the V3
core implementation. Source inspection also confirms that gold answer and gold
session IDs are not read by V3 planning, retrieval, packing, or answer prompt.
Gold session IDs are consulted only after answer generation for offline recall
metrics.

## Tests and artifacts

- Automated tests: 362 passed.
- LongMemEval run:
  `runs/v3_20260726/lme_control12_v3_bl`
- LoCoMo run:
  `runs/v3_20260726/locomo_control20_v3_bl`
- Focused recommendation regression:
  `runs/v3_20260726/lme_v3_bl_recommendation1`

Previously frozen unseen evaluations remain the proper estimate of current
generalization and are materially below 90%. Therefore the architecture has a
credible path to better accuracy, but a 90% claim requires a parameter-frozen
expanded blind run or full evaluation.
