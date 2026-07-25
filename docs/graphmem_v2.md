# GraphMem V2: Hierarchical State Graph

`hierarchical_state_graph_v2` is an additive benchmark variant. Legacy variants and their JSONL formats remain readable.

## Index contract

- L0 `LeafNode`: lossless user/assistant turns, expanded only for final evidence.
- L1 `AtomicFactNode`: source-grounded state/event/preference/quantity/assistant facts.
- L2 `RoutingCardNode`: compact session routing evidence (at most 180 rough tokens).
- L3 `StateChain`: directed current/history state over `(subject_key, predicate_key, context_key)`.
- Typed edges preserve direction, confidence, provenance and `graphmem_v2` schema markers.

Every fact cites an existing L0 source. Consolidation may normalize aliases and propose relations but cannot add facts or reference unknown IDs. Temporal edges exist only inside an entity/event/state chain. Semantic links are reciprocal kNN links above an adaptive floor.

## Retrieval and answering

Query planning is deterministic and never receives `question_type`, gold answer, or gold session IDs. Dense, BM25 and entity/predicate/status/date signals are fused with RRF. Typed best-first expansion is capped at depth 2. Routing cards, atomic facts, a deterministic evidence ledger, and short L0 excerpts are packed under the configured evidence budget. The answer stage performs one `deepseek-v4-flash` call with thinking disabled and `max_tokens=512`.

Gold support-session IDs are read only after answer generation for offline recall metrics.

## Token accounting

The build and answer phases separately report cache-miss input, cache-hit input, output and total tokens. Per-question hard gates default to 300,000 build tokens and 10,000 answer tokens. Missing provider cache breakdown is inferred as cache miss and marked. Embedding and judge calls are excluded. Judge calls are stored in separate files.

## Reproducible run

1. Put the DeepSeek key in the ignored, mode-0600 `.env` file.
2. Keep `Qwen3-Embedding-0.6B` running at `http://127.0.0.1:8001/v1`.
3. Run `scripts/run_v2_longmemeval.sh [dataset] [output_dir]`.

The runner health-checks the model name and 1024-dimensional embedding response. It never starts or stops the embedding service. Defaults are question concurrency 2, build LLM concurrency 32, and judge concurrency 16. Runs are resumable.

The pinned Mem0 judge source is `src/graphmem_demo/mem0_longmemeval_prompts.py`, from commit `bd063eea04de4f8a19927beea155afa094a01905`, SHA-256 `ba8cf60d26f1390ecbef0f07b3e950556fe3bc5a37ba4b5343f28217f18c144f`.


## Stage-level diagnostics

Every real run writes enough information to separate index, retrieval, packing, answering, and budget failures:

- `index_diagnostics.jsonl`: per-session parse/fallback/length results and question consolidation acceptance.
- `nodes.jsonl`, `state_chains.jsonl`, `edges.jsonl`: L0-L3 structure, source pointers, directed validity/update order, relation confidence, and generator provenance.
- `retrieval_results.jsonl`: deterministic query type, RRF channel ranks, typed expansion, adjacent/source expansion, pre-pack and post-pack IDs, dropped evidence, packed provider-token estimate, and the evidence ledger.
- `question_stats.jsonl`: cache-miss input, cache-hit input, output, total, reasoning, and hard-budget pass/fail for build and answer.
- `scripts/analyze_v2_pipeline.py`: structural QA, term/session/source recall proxies, packer retention, phase attribution, and P50/P95/max token reports.

L0 now has a bounded direct rescue channel for cases where an extraction call omitted a crucial fact. The normal path remains L2 routing to L1 facts followed by provenance expansion. User-backed L0 excerpts are packed user-first; assistant text is retained only when a selected assistant fact needs it. This keeps lossless recovery without filling the answer prompt with long recommendation lists.

The ledger includes query-aware current state, category-aware distinct counts, exact-entity checks, explicit event-time extraction, contextual preference constraints, and deterministic arithmetic. `observed_at` is explicitly labeled as the recording date and is never silently substituted for an unknown `event_time`. Operator outputs that survive packing become answer constraints, while the answer stage still makes exactly one DeepSeek call.

## Frozen experimental evidence (2026-07-24)

The exact pinned Mem0 judge re-scored the legacy 500-answer baseline at **77.0% (385/500)**. The fixed seed `20260724` split contains 48 development errors, 24 legacy-correct controls, and 67 blind errors.

The frozen development set reached **72/72** answer accuracy, including all 48 tuned errors and all 24 controls. The corrected retrieval-sufficiency judge reached **66/72 (91.67%)**, below the 95% target and with repeated scores ranging from 90.28% to 93.06%.

The untouched blind-error run reached **26/67 (38.81%)**. Its errors were attributed to answer reasoning/format (25), retrieval ranking or graph expansion (9), index extraction (3), context rendering (3), and evidence packing (1). Average gold-session recall was 85.97%, source-leaf expansion recall 52.52%, index term support 92.15%, and post-pack term support 68.10%. Typed expansion added exactly 44 candidates per question on average, showing cap saturation and weak discrimination.

Structural integrity remained strong: no source, routing-pointer, edge-endpoint, or state-chain integrity failures were found. Two temporal-scope warnings remained. Session extraction had a 10.02% parse-error rate and 7.79% length-finish rate.

All measured questions passed both hard token gates. Blind build P50/P95/max were **253,192 / 273,507 / 283,685**; blind answer P50/P95/max were **7,471 / 8,836 / 9,293**. Reasoning tokens were zero, and judge/embedding usage was excluded.

The development-inclusive optimistic projection is **459/500 (91.8%)**, but it includes the 48 tuned development errors and assumes no regression on 385 legacy-correct questions. Applying only the blind recovery rate to all 115 legacy errors gives an estimated **85.9%**, also assuming no regression. Neither is a measured full-500 result. Because the blind gate failed, the planned one-time final 500 run was not executed and the 90% target is not demonstrated.
