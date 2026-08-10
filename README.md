# GraphMem

GraphMem is a graph-structured long-term memory system for conversational QA benchmarks such as [LongMemEval](https://github.com/LongMemEval/LongMemEval) and LoCoMo.

## Current V5.11 query plane

The measured V5.11 retrieval path is now available through one runtime schema
instead of benchmark-only hardcoded options. It includes H11 QueryIR,
hierarchical routing, bounded process workers, rendezvous memory affinity,
per-tenant admission, versioned compiled views, frequency-aware cache admission,
CPU pinning, deadlines, worker restart, and cache/RSS/PSS telemetry.

```bash
python scripts/serve_v5_11.py \
  --db /path/to/graphmem.sqlite \
  --runtime-config configs/v5/runtime_v5_11_balanced.json \
  --compiled-cache-dir /trusted/local/compiled_views \
  --cpu-ids 0-7
```

Alternative frozen profiles are provided for low-tail-latency, low-memory and
the exact 8-core report setup. See
[`docs/V5_11_RUNTIME_DEPLOYMENT.md`](docs/V5_11_RUNTIME_DEPLOYMENT.md) for the
endpoint, full parameter table, sidecar lifecycle and benchmark reproduction.

The opt-in V5.9 report path (recursive coarsening, parent-gated relation
construction, physical hierarchical QueryIR routing, post-pack certificates and
affected-path snapshot publication) is documented in
[`docs/V5_9_REPORT_IMPLEMENTATION.md`](docs/V5_9_REPORT_IMPLEMENTATION.md). It
does not change the frozen V5.8 B0--B5 profiles.
The measured report numbers and their interpretation boundaries are recorded in
[`docs/V5_9_REPORT_RESULTS.md`](docs/V5_9_REPORT_RESULTS.md).

It builds a hierarchical memory graph from dialogue sessions, retrieves evidence with hybrid semantic / structured / graph signals, and answers questions with timeline-aware context assembly.

## Architecture

See [`docs/2026-07-08_graphmem_architecture_overview.md`](docs/2026-07-08_graphmem_architecture_overview.md) for the current design overview.

## Historical V4.1 development snapshot

The latest cross-backbone implementation and experiment history are maintained on
`codex/graphmem-v419-qwen32b`. It includes the V4.1 online Graph Harness,
resumable GPT/Qwen runners, token accounting, shard recovery, and the unified
dual-backbone report. See:

- [Repository handoff and artifact index](docs/repository_handoff_20260804.md)
- [Graph Harness Query IR presentation notes](docs/graph_harness_query_ir_speaker_notes_20260804.md)
- [GraphMem vs Mem0 comprehensive report](docs/graphmem_mem0_comprehensive_report_20260804.md)
- [V4.1 query design](docs/V4_1_QUERY.md)

```bash
git clone https://github.com/Concyclics/GraphMem.git
cd GraphMem
git switch codex/graphmem-v419-qwen32b
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GraphMem V4.0

The `hierarchical_hybrid_graph_v4_0` variant uses one physical RoleFrame graph with topology-aware state/collection/temporal and peer-dialogue capability projections. See [the V4.0 design and reproducibility guide](docs/graphmem_v4_0.md).

```bash
DATA=data/longmemeval_s_cleaned.json bash scripts/run_v4_benchmark.sh
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

The SGAO OpenAI-compatible endpoint supports both verified remote model options:

```bash
# Default model from .env
bash scripts/run_locomo_custom_api.sh

# One-run override; the API key remains in the ignored .env file
MODEL=gpt-5.6-luna bash scripts/run_locomo_custom_api.sh
```

Set `SGAO_MODEL` in `.env` to `gpt-5.4-mini` or `gpt-5.6-luna` to change the
repository-wide default. The endpoint and key are read from `SGAO_BASE_URL` and
`SGAO_API_KEY`; `.env` is ignored by Git and should remain mode `0600`.

## Data

Benchmark datasets are not bundled in this repo. Place converted JSON files under `data/`, for example:

- `data/longmemeval_s_cleaned.json` — LongMemEval-S (cleaned)
- `data/locomo10_graphmem.json` — LoCoMo converted to GraphMem format

Use `scripts/build_eval_subset.py` to create a fixed evaluation subset once you have the source benchmark file.

## Quick start

Run the main demo CLI:

```bash
python scripts/run_token_demo.py \
  --data data/longmemeval_s_subset50_balanced.json \
  --output-dir runs/my_run \
  --variants direct_session_k16_compact_graphmem \
  --max-questions 10
```

Evaluate generated answers:

```bash
python scripts/evaluate_answers.py \
  --answers runs/my_run/direct_session_k16_compact_graphmem/answers.jsonl \
  --data data/longmemeval_s_subset50_balanced.json \
  --output-jsonl runs/my_run/direct_session_k16_compact_graphmem/auto_eval.jsonl \
  --output-md runs/my_run/direct_session_k16_compact_graphmem/auto_eval.md
```

## Local vLLM workflow

For fully local runs with embedding + LLM services:

```bash
scripts/smoke_longmemeval.sh up      # start services
scripts/smoke_longmemeval.sh run     # smoke test (3 questions)
scripts/smoke_longmemeval.sh stop    # stop services
```

For a fixed 50-question subset with judging and stage audit:

```bash
scripts/run_subset_eval.sh
```

## Tests

```bash
python -m pytest test/ -q
```

Unit tests use synthetic fixtures under `test/fixtures/` and do not require benchmark data.

## Project layout

```
src/graphmem/        Core library (build, retrieve, answer and serving pipeline)
scripts/             CLI entrypoints and evaluation utilities
test/                Unit tests and fixtures
docs/                Design notes and experiment reports
```

## GraphMem V2

The additive `hierarchical_state_graph_v2` variant implements lossless leaves, atomic facts, compact routing cards, directed state chains, typed depth-2 retrieval, an evidence ledger, and separate 300K/10K DeepSeek token gates. See [docs/graphmem_v2.md](docs/graphmem_v2.md).

```bash
# Put DEEPSEEK_API_KEY in the ignored mode-0600 .env first.
scripts/run_v2_longmemeval.sh /path/to/longmemeval_s_cleaned.json /path/to/output
```
