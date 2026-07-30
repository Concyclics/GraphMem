# GraphMem V3.6 locked validation report (2026-07-30)

## Locked implementation

- Production variant: `hierarchical_role_graph_v3_6`
- Retrieval revision: `graphmem_v36_generic_evidence_closure_20260730_locked`
- Final response path: one GPT-5.4-mini call; deterministic operators only emit provenance-bound evidence ledgers.
- No gold answer, gold session ID, benchmark ID, question ID, or per-topic answer table is read by retrieval or answering.
- Persistent V3.6 RoleFrame/RouterCard/EvidenceGroup indexes were reused; this run is an answer-only replay over immutable indexes.

## Fixed 80-question weak-type validation

Dataset: `longmemeval_weaktypes80_dev_seed20260730.json` (40 previously weak questions plus 20 unseen multi-session and 20 unseen temporal-reasoning questions).

| Slice | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Overall | 67 | 80 | 83.75% |
| Multi-session | 36 | 40 | 90.00% |
| Temporal reasoning | 31 | 40 | 77.50% |
| Previous weak slice | 32 | 40 | 80.00% |
| Newly sampled slice | 35 | 40 | 87.50% |

The result uses the pinned Mem0 LongMemEval judge prompt at commit `bd063eea04de4f8a19927beea155afa094a01905` with GPT-5.4-mini and no API reasoning mode.

## General changes validated in this iteration

- Provenance-bound count/sum/difference ledgers now cover repeated events, age arithmetic, subset percentages, labeled scalar differences, paired metrics, repeated durations, dated event counts, same-unit state changes, maintenance parent assets, category acquisition members, and operation-target pairs.
- Temporal binders put an explicit `answer_value` in the compact decisive ledger while still requiring the final LLM answer call.
- Category acquisition requires both structured category membership and local source co-occurrence with the acquisition action; broad card context cannot become a member.
- Provider abbreviations such as `Dr.` no longer split a dated visit away from its date.
- Maintenance of a component can be normalized to its parent asset, while different assets remain distinct.

## Remaining validation errors

Multi-session (4): `d682f1a2`, `gpt4_59c863d7`, `88432d0a_abs`, `gpt4_194be4b3`.

Temporal reasoning (9): `2c63a862`, `gpt4_93159ced`, `b29f3365`, `gpt4_2f56ae70`, `gpt4_f420262d`, `gpt4_2655b836`, `gpt4_7f6b06db`, `f0853d11`, `eac54add`.

The dominant residuals are long-span endpoint binding, incomplete collection closure, and abstention boundary errors. No further development-set tuning is allowed after this lock; the next measurements are full benchmark evaluations.

## Verification

- Repository test suite: 688 passed.
- Focused V3.6 tests: 89 passed.
- Secrets scan: no API-key pattern found outside ignored local configuration/results.
