# GraphMem V3 unseen evaluation report (2026-07-25)

## Executive result

GraphMem V3 now has a benchmark-neutral hypergraph/coarse-to-fine implementation,
strict DeepSeek token accounting, compact evidence packing, and local temporal,
state, and frequency operators.  The implementation satisfies both token gates,
but the unseen results do **not** support a claim of 90% cross-benchmark accuracy.

| Evaluation | Judged | Accuracy | Build max | Answer max | Budget failures |
|---|---:|---:|---:|---:|---:|
| LongMemEval frozen development control | 12 | 91.67% | 230,699 | 9,504 | 0 |
| LongMemEval unseen, seed 20260724 | 50 | 60.00% | 250,233 | 9,799 | 0 |
| LoCoMo frozen development control | 20 | 95.00% | 56,198 | 8,796 | 0 |
| LoCoMo unseen, Memory Benchmarks categories 1–4 | 151/200 | 85.43% | 85,074/conversation | 7,319 | 0 |

The LoCoMo judge intentionally excludes 49 category-5 questions, following the
`mem0ai/memory-benchmarks` prompt implementation.  Judge and embedding tokens are
excluded from the build and answer budgets.  Build, answer, and judge API
reasoning tokens are zero.

## Frozen V3 design

- Lossless role-neutral turns, atomic claims/events, overlapping episodes/themes,
  hyperedges, and state chains.
- Dense, BM25, and exact retrieval over all levels with RRF fusion.
- Claim-aware session scope posterior.  It is a continuous routing weight, never
  a hard session filter, and is disabled when no grounded claim match exists.
- Query-conditioned relation weights and bidirectional hyperedge traversal with
  fanout normalization and a per-edge expansion cap.
- Compact coarse evidence: episode/theme text remains visible, while long pointer
  lists and repeated summaries are reduced.  Provenance remains in the ledger.
- Generic local operators for duration, relative-time scope, newest state,
  recurring-frequency counts, planned events, and scalar chronology.
- One DeepSeek answer call, no answer repair or note-extraction calls.

The V3 core contains no benchmark name, category ID, question ID, or topic rule
table.  A static test enforces this property.

## LongMemEval unseen diagnosis

Accuracy by type:

| Type | Questions | Accuracy | Gold-session recall |
|---|---:|---:|---:|
| knowledge-update | 9 | 77.78% | 100.00% |
| multi-session | 15 | 40.00% | 94.44% |
| single-session-assistant | 6 | 66.67% | 100.00% |
| single-session-preference | 2 | 50.00% | 100.00% |
| single-session-user | 4 | 100.00% | 100.00% |
| temporal-reasoning | 14 | 57.14% | 89.29% |

Overall gold-session recall is 95.33%, and every question retrieves at least one
gold session.  Therefore the principal failure is not coarse session routing.
It is the transition from a relevant session to a complete, correctly scoped
operand set:

1. Multi-session collection closure is incomplete or mixes action identity,
   recurrence, and entity identity.
2. Temporal questions retrieve the right sessions but bind dates, companions,
   or event attributes to the wrong event instance.
3. State chains are present but are not yet the authoritative source for every
   current-state answer; some values still depend on answer-model selection.
4. Fallback `said` claims preserve losslessness but add long, weakly structured
   assistant content that competes with atomic autobiographical claims.
5. The current closure certificate only describes the retrieved local
   hypergraph.  It cannot prove that all operands across memory were enumerated.

The generic operator iteration fixed representative unseen failures (duration,
relative-time event scope, weekly recurrence, event ordering, and latest state),
but the uniform 50-question score rose only from 58% to 60%.  Continuing to add
question-pattern prompts would be benchmark overfitting and is not recommended.

LongMemEval token percentiles:

| Metric | P50 | P95 | Max |
|---|---:|---:|---:|
| build cache-miss input | 3,054 | 3,581 | 3,905 |
| build cache-hit input | 136,576 | 140,032 | 141,440 |
| build output | 77,570.5 | 87,973 | 108,110 |
| build total | 215,399 | 230,098 | 250,233 |
| answer cache-miss input | 7,846 | 8,951 | 9,541 |
| answer cache-hit input | 256 | 256 | 384 |
| answer output | 6 | 76 | 198 |
| answer total | 8,161.5 | 9,215 | 9,799 |

## LoCoMo unseen diagnosis

| Category | Judged | Accuracy | Gold-session recall |
|---|---:|---:|---:|
| 1 | 24 | 87.50% | 74.90% |
| 2 | 36 | 77.78% | 94.40% |
| 3 | 6 | 66.67% | 83.30% |
| 4 | 85 | 89.41% | 100.00% |

The ten conversation builds range from 48,594 to 85,074 DeepSeek tokens.  Answer
tokens have P50 6,336, P95 6,932, and max 7,319.  LoCoMo is materially stronger
because one shared conversation graph is smaller and named participants reduce
scope ambiguity.  Categories requiring multi-hop or temporal composition remain
the weakest.

## Required architecture work before claiming 90%

1. Replace question-time local closure with a persisted operand catalog:
   canonical event/entity instances, occurrence identity, recurrence schedules,
   and explicit negative/unknown attributes.
2. Make state chains authoritative and deterministic.  Query a typed
   `(subject, predicate, context)` chain first; use semantic retrieval only to
   recover missing aliases or provenance.
3. Add event-frame unification across sessions.  Dates, participants, location,
   quantity, modality, and source turns must attach to one event ID before graph
   traversal.
4. Introduce retrieval-time completeness tests independent of gold:
   competing-scope margin, unresolved contradiction count, open collection
   boundary, and missing event slots.  If closure is open, spend remaining
   evidence budget on targeted graph expansion instead of answering immediately.
5. Persist ready-to-query dense matrices, BM25 statistics, inverted indexes, and
   incidence adjacency.  Current cached runs still repeatedly scan and sort all
   nodes, causing avoidable 6–12 minute batch latency.
6. Evaluate on another unseen benchmark before tuning either LongMemEval or
   LoCoMo again.  Parameters should be selected from retrieval completeness and
   provenance metrics, not answer-judge errors.

## Reproducible artifacts

- LongMemEval manifest:
  `runs/v3_20260725/splits/lme_unseen50_seed20260724.manifest.json`
- LongMemEval run:
  `runs/v3_20260725/lme_unseen50_v3_compact_ops_p`
- LoCoMo manifest:
  `runs/v3_20260725/splits/locomo_unseen200_seed20260724.manifest.json`
- LoCoMo run:
  `runs/v3_20260725/locomo_unseen200_v3_compact_ops_p`
- LoCoMo token report:
  `runs/v3_20260725/locomo_unseen200_v3_compact_ops_p/token_analysis/locomo_token_report.md`

All 280 automated tests pass.
