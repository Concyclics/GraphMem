# GraphMem V3.6 retrieval/answer repair gate — 2026-07-30

## Decision

The repair candidate did not pass the pre-commit gate on LongMemEval. It was
therefore not evaluated on the previous-correct control set and was not
committed or pushed.

The gate is based on net fixes required to reach 85% before regression:

| Benchmark | Previous correct | Previous wrong | Wrong fixes required | Best measured wrong fixes | Gate |
|---|---:|---:|---:|---:|---|
| LongMemEval | 328/500 | 172 | 97 | 66/172 (38.37%) | fail |
| LoCoMo Cat. 1–4 | 1224/1540 | 316 | 85 | 142/316 (44.94%) | pass on wrong set only |

The optimistic no-regression projections are 78.8% for LongMemEval and 88.7%
for LoCoMo. These are not final scores because the previous-correct controls
were intentionally not run after the joint gate failed.

## Implemented candidate

- Globally ranks lossless turns with exact, BM25, dense and fine-node source
  projection.
- Uses routing-card rank as a bounded boost instead of a mandatory per-card
  quota.
- Adds local dialogue adjacency only around selected source anchors.
- Fuses proposition-focused ranking with routed-region coverage ranking.
- Packs source turns before compact RoleFrame/EvidenceGroup navigation records.
- Removes routing-card text and unverified operator calculations from the
  final answer context.
- Adds a source-binding diagnostic for owner/entity, relation and comparison
  endpoint co-occurrence.
- Uses one compact class-level answer prompt and does not force abstention from
  an incomplete extraction certificate.
- Splits compound acquisition members and recovers a maintained parent asset
  from a bounded two-sentence lossless window.
- Adds conversation-shared persisted-index replay and answer-only replay tools.

No gold answer, answer session, benchmark topic, question ID or item-specific
branch is used by production retrieval or answer code.

## Ablation results on previous wrongs

| Experiment | LongMemEval fixes |
|---|---:|
| Source-first retrieval + compact answer prompt | 57/172 (33.14%) |
| Same retrieval + restored long class prompt | 44/172 (25.58%) |
| Dual-rank source fusion + compact answer prompt | 66/172 (38.37%) |
| Broad fallback-ledger injection, remaining collection errors only | 4/60 (6.67%) |

The broad fallback-ledger change was reverted. More candidates are not a
substitute for relation-bound evidence closure.

LongMemEval dual-rank fixes by original type:

| Type | Fixed / previous wrong |
|---|---:|
| knowledge-update | 9/24 |
| multi-session | 23/58 |
| single-session-assistant | 3/5 |
| single-session-preference | 4/13 |
| single-session-user | 7/11 |
| temporal-reasoning | 20/61 |

LoCoMo source-first fixes by category:

| Category | Fixed / previous wrong |
|---|---:|
| 1 | 35/83 |
| 2 | 38/82 |
| 3 | 25/45 |
| 4 | 44/106 |

## Token results

Only backbone answer calls are included below. Embedding and judge calls are
excluded; all reasoning-token counts are zero.

| Run | Questions | Avg total | P95 total | Max total | Cache miss input | Cache hit input | Output | >12,100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LME source-first | 172 | 6,815.8 | 10,499 | 12,148 | 1,119,745 | 50,688 | 1,881 | 1 |
| LME dual-rank | 172 | 7,102.5 | 10,733 | 12,048 | 1,219,691 | 0 | 1,947 | 0 |
| LoCoMo source-first | 316 | 4,052.3 | 4,574 | 5,908 | 1,269,158 | 7,168 | 4,186 | 0 |

No build calls were made; all experiments reused the immutable V3.6 indexes.

## Root cause after repair

LongMemEval is not blocked by answer verbosity or abstention. In the best run,
only one remaining wrong answer was an abstention. The unresolved errors are
mainly incomplete relation-bound operand closure:

- a routed session is present, but the exact member-bearing turn is displaced
  by same-topic turns;
- a component operation must be bound to its parent asset;
- a single source contains multiple coordinated members;
- implicit ownership/acquisition or device use is not represented as an
  independent frame;
- temporal questions retrieve both topics but not both independently dated
  endpoints;
- preference answers require transfer of supported attributes rather than an
  exact target lookup.

The evidence pack can contain answer tokens while still missing the semantic
binding needed to answer. Conversely, injecting every lossless fallback causes
large accuracy loss. The next architecture change must therefore rank
relation-bound source spans, not simply increase source count.

## Required next implementation

1. Build a question-time, source-span candidate table from routed sessions.
   Each row must bind owner, operation/relation, target entity, lifecycle and
   time in the same sentence or bounded adjacent window.
2. Run dense/BM25 only to propose spans; use the candidate table to close the
   requested role set and exclude same-topic non-members.
3. For collection questions, split coordinated noun phrases and map component
   operations to a source-supported parent asset before counting.
4. For temporal questions, require one independently dated span per named
   endpoint before packing.
5. For preference/recommendation questions, represent supported positive and
   negative attributes separately from the requested future target.
6. Re-run the 172/316 wrong gates. Only after both meet their required fix
   counts should the 328/1224 previous-correct controls be run.
7. Commit and push only if net projected accuracy after the full controls is at
   least 85% on both benchmarks with no material type regression.

## Verification

- `PYTHONPATH=. .venv/bin/pytest -q`: 695 passed.
- Static benchmark/topic/ID branch checks are included in the passing suite.
- Working tree is intentionally uncommitted because the accuracy gate failed.
