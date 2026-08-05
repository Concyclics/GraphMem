# GraphMem V4.1 query-time enhancement

`hierarchical_hybrid_graph_v4_1_query` reuses the immutable V4 memory graph and
build cache. It adds only a disposable query sidecar, deterministic QueryIR
augmentation, typed gap expansion, an optional short planner, evidence
certificates, and source-bound answer constraints.

## Run

```bash
.venv/bin/python scripts/run_token_demo.py \
  --data /path/to/questions.json \
  --variants hierarchical_hybrid_graph_v4_1_query \
  --tree-mode hierarchical_hybrid_graph_v4_1_query \
  --summary-schema graphmem_v4_1_query \
  --memory-cache-dir /path/to/existing/v4/cache \
  --output-dir /path/to/v41/output \
  --deepseek-model deepseek-v4-flash \
  --deepseek-base-url https://api.deepseek.com \
  --llm-api-key-env DEEPSEEK_API_KEY \
  --llm-request-profile deepseek \
  --embedding-base-url http://127.0.0.1:8001/v1 \
  --embedding-model Qwen3-Embedding-0.6B \
  --reasoning-effort none
```

The original V4 cache key and graph files are not changed. The query sidecar is
written separately as `retrieval_v41.sqlite`.

## Paired retrieval ablation

```bash
.venv/bin/python scripts/paired_v41_retrieval_ablation.py \
  --data /path/to/questions.json \
  --memory-cache-dir /path/to/existing/v4/cache \
  --output /path/to/paired_v41.jsonl
```

The paired script reuses the same query vectors for V4 and V4.1 and reports
source additions/deletions, evidence completeness, and context-token deltas.

## Query-token accounting

Planner and final-answer calls are included. Embedding and judge calls are
excluded. Per-question traces contain cache-hit input, cache-miss input, output,
reasoning and total tokens, plus flags for 10K, 12K and 13K.

Normal questions target 10K total. Complex collection, temporal, state and
multi-hop questions may use the 12K P95 window. Every query retains the 13K hard
limit. The evidence pack reserves prompt space for certificates and the final
answer instruction.

## Safety and fallback

The sidecar never creates graph nodes or changes embeddings. Every planner
proposal is revalidated against source provenance. Invalid planner JSON falls
back to deterministic retrieval. Sidecar construction failure automatically
falls back to the V4 navigator and records `v41_sidecar_fallback`.

## Validation snapshot

The repository passes 724 local tests. The final dialogue-pair completion pass
runs before optional multi-channel anchors: when a selected, query-relevant
prompt turn has a direct next-turn reply, the reply receives budget priority as
part of the same evidence unit. This is a structural rule and contains no
benchmark ID, topic, brand, or answer-specific branch.

With `deepseek-v4-flash` and thinking disabled, the fixed 10-question
LongMemEval regression set scored 10/10 under the pinned Mem0 judge. Mean query
tokens were 9,702.4, P95/max were 10,560, two questions exceeded 10K, none
exceeded 12K, and reasoning tokens were zero.

On the fixed 20-question LoCoMo control set, the memory-benchmarks judge scored
15/20 (75%), up from 12/20 before dialogue-pair budget priority. Category scores
were 2/2, 1/2, 1/3, and 11/13 for Categories 1 through 4. Mean query tokens were
7,907.6, P95 was 9,307, max was 9,366, no question exceeded 10K, and reasoning
tokens were zero.

These are development/regression subsets, not replacements for the required
500-question LongMemEval and 1,540-question LoCoMo formal evaluations.
