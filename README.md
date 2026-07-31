# GraphMem

GraphMem builds a hierarchical state graph from multi-session dialogue and answers long-horizon questions on benchmarks such as [LongMemEval](https://github.com/LongMemEval/LongMemEval) and [LoCoMo](https://github.com/snap-research/locomo).

This repository’s **default and maintained path is GraphMem V2** (`hierarchical_state_graph_v2`): lossless leaves, atomic facts, routing cards, directed state chains, typed retrieval, an evidence ledger, and hard per-question token budgets.

Design reference: [`docs/graphmem_v2.md`](docs/graphmem_v2.md).

## Architecture

```text
Sessions ──► Build
              L0 LeafNode          lossless user/assistant turns
              L1 AtomicFactNode    source-grounded facts
              L2 RoutingCardNode   compact session routing (≤ ~180 tokens)
              L3 StateChain        current/history state over
                                   (subject, predicate, context)
                    │
Question ──► Retrieve
              deterministic query plan (no gold leakage)
              dense + BM25 + entity/predicate RRF
              typed best-first expansion (depth ≤ 2)
                    │
             Pack   routing cards + facts + evidence ledger + short L0 excerpts
                    │
             Answer one DeepSeek call (thinking off, max_tokens=512)
```

Every L1 fact cites an existing L0 source. Consolidation may normalize aliases and propose relations but cannot invent facts or unknown IDs. Gold support-session IDs are read only after answering, for offline recall metrics.

Build and answer phases separately report cache-miss / cache-hit input, output, and total tokens. Default hard gates: **300,000** build tokens and **10,000** answer tokens per question.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set DEEPSEEK_API_KEY; chmod 600 .env recommended
```

Required local embedding service (runner health-checks it; does not start/stop it):

| Setting | Value |
| --- | --- |
| Model | `Qwen3-Embedding-0.6B` |
| Endpoint | `http://127.0.0.1:8001/v1` |
| Dimension | 1024 |

## Data

Datasets are not bundled. Place converted JSON under `data/`, for example:

- `data/longmemeval_s_cleaned.json` — LongMemEval-S (cleaned)
- `data/locomo10_graphmem.json` — LoCoMo via `scripts/convert_locomo10.py`

## Quick start

LongMemEval:

```bash
scripts/run_v2_longmemeval.sh /path/to/longmemeval_s_cleaned.json /path/to/output
```

LoCoMo:

```bash
scripts/run_v2_locomo.sh
```

Custom OpenAI-compatible API (optional):

```bash
scripts/run_v2_custom_api.sh
scripts/run_v2_locomo_custom_api.sh
```

Runs are resumable. Judge scoring uses the pinned Mem0 LongMemEval prompts in `src/graphmem_demo/mem0_longmemeval_prompts.py`.

Structural QA, recall proxies, packer retention, and token percentiles:

```bash
python scripts/analyze_v2_pipeline.py --help
```

Service check before a run:

```bash
python scripts/check_v2_services.py
```

## Tests

```bash
python -m pytest test/ -q
```

Tests use synthetic fixtures under `test/fixtures/` and do not require benchmark data.

## Project layout

```
src/graphmem_demo/
  hierarchical_v2.py   V2 index / retrieval / packing
  pipeline.py          shared runner; V2 variant = hierarchical_state_graph_v2
scripts/
  run_v2_*.sh          LongMemEval / LoCoMo entrypoints
  check_v2_services.py
  analyze_v2_pipeline.py
  evaluate_mem0_judge.py
docs/
  graphmem_v2.md
  graphmem_v2_final_500_report_20260725.md
  locomo_v2_full_report_20260725.md
test/
```

## Results

LongMemEval-S, 500 questions (2026-07-25): **462/500 (92.4%)** under the pinned Mem0 judge; all questions passed the 300K/10K token gates.

Details: [`docs/graphmem_v2_final_500_report_20260725.md`](docs/graphmem_v2_final_500_report_20260725.md), LoCoMo: [`docs/locomo_v2_full_report_20260725.md`](docs/locomo_v2_full_report_20260725.md).

---

## Appendix: Legacy V1

Earlier GraphMem variants (`direct_session_*`, `summary_tree_*`, etc.) remain in the codebase for compatibility and ablation. They are **not** the recommended path.

| | V1 (legacy) | V2 (this README) |
| --- | --- | --- |
| Variant | `direct_session_*`, `summary_tree_*`, … | `hierarchical_state_graph_v2` |
| Index | Session root + leaf summaries | L0 → L1 → L2 → L3 state graph |
| Retrieval | Hybrid / PPR-style | RRF + typed depth-2 + evidence ledger |
| Budgets | Soft / script-level | Hard 300K build / 10K answer |

Do not use V1 scripts (`run_subset_eval.sh`, `smoke_longmemeval.sh`, …) for new experiments. Historical notes: [`docs/2026-07-08_graphmem_architecture_overview.md`](docs/2026-07-08_graphmem_architecture_overview.md).
