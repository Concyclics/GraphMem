# GraphMem V3 frozen blind evaluation — 2026-07-25

## Outcome

- Development run G: 103/113 (91.15%). One subsequently verified event-binding
  regression raises the expected frozen development result to 104/113 (92.04%).
- Frozen blind set: 40/50 (80.0%). The 90% gate was not met, so the 500-question
  run was not started.
- Build and answer budgets passed for every blind question; reasoning tokens were
  zero.

## Blind token accounting

| stage | cache-miss input | cache-hit input | output | total | per-question max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Build | 7,091,795 | 1,147,776 | 4,597,162 | 12,836,733 | 274,432 |
| Answer | 360,591 | 256 | 1,021 | 361,868 | 8,565 |

Judge and embedding usage are excluded from both budget rows.

## Retrieval and graph findings

- Fact semantic retrieval: 100% any-gold-session recall, 96% all-gold-session
  recall.
- Fact BM25: 98% any, 92% all.
- Leaf BM25: 100% any, 98% all.
- Entity retrieval added no unique fact-level gold sessions on this split.
- Typed graph expansion retained non-seed nodes on 43/50 questions, but produced
  zero unique gold-session rescues because dense/BM25 already found the relevant
  sessions.
- 109 expanded nodes survived final selection (13.29% of expanded nodes).
  Retained relationship paths included `semantic_neighbor`, `same_predicate`,
  `before`, `after`, `supersedes`, `supports`, and `same_entity`.

The graph is therefore traversed after seed selection and affects the final
evidence set, but its marginal session-recall contribution on this blind split is
small. The main remaining gap is operand discovery and deterministic aggregation,
not first-stage session retrieval.

## Failure decomposition

- 6 answer reasoning/format failures.
- 3 retrieval-ranking/graph failures according to the conservative heuristic
  classifier; all three nevertheless retrieved the annotated sessions, indicating
  missing operand facts or operators rather than total session miss.
- 1 context-rendering failure.

Five of ten failures are multi-session questions. Missing generic operations
include per-unit division, cross-category sums/counts, grouped delta comparison,
frequency argmax, and evidence-backed current-state acceptance. This explains why
high session recall did not translate into 90% answer accuracy.

## Decision

Do not run the full 500-question evaluation from this frozen configuration.
The next iteration should be designed from aggregate failure families, not from
blind question IDs, and evaluated on a newly sealed split.
