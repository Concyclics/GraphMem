# Reproduce the November 2026 GraphMem research pipeline

This snapshot contains the accepted LoCoMo and LongMemEval retrieval and answer
policies, both the Qwen stage and the later Astra stage. It adds a research CLI;
the existing `Memory` SDK and October product reproduction remain unchanged.

The pipeline is **conversation graph → 117 source/graph features → frozen CPU
ranker → token-budgeted original turns → question/source-based reader branches**.
The final Astra policy performs one answer call. Qwen can make format repairs or
source rereads, and every executed call is charged.

## 1. Install and configure

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -c config/research-constraints.txt '.[research,service,dev]'
graphmem-research verify-assets
cp config/research-endpoints.example.json runs-endpoints.json
```

Edit `runs-endpoints.json` for your own endpoints. Credentials are read only from
the named environment variables; keep this local config out of Git if it contains
private endpoint addresses. The Astra example intentionally has no live provider
address. Set the base URL to your authorized Responses-compatible provider and
export `GRAPHMEM_ASTRA_API_KEY` in your terminal or secret manager.

Models used in the recorded run:

| Role | Model / revision | Settings |
|---|---|---|
| Qwen answer + judge | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, `5a5a776300a41aaa681dd7ff0106608ef2bc90db` | Answer: temperature 0, seed 20261030. Judge: temperature 0, seed 0. |
| Graph-context/raw-view embeddings | `Qwen/Qwen3-Embedding-0.6B`, `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418` | BF16, last-token pooling, left padding, 2048-token truncation, stored FP16 vectors. |
| Base dense features, LoCoMo | `BAAI/bge-m3` | Must match the graph's stored base embeddings. |
| Base dense features, LongMemEval | `Qwen/Qwen3-Embedding-0.6B` | Must match the graph's stored base embeddings. |
| Astra answer | `gpt-6-astra` | Responses API, reasoning high, output cap 4096, strict answer JSON, store false. |

The recorded Qwen server used vLLM 0.23.0 / Torch 2.11.0 / CUDA 13.0 on H100.
For an already downloaded model, the corresponding launch command is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$QWEN_MODEL" --served-model-name graphmem-eval-qwen30b \
  --host 127.0.0.1 --port 8002 --gpu-memory-utilization .85 \
  --max-model-len 49152 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --enforce-eager --linear-backend triton --moe-backend triton
```

Use separate GPU/server environments for model serving and local embedding.
The research package does not start, stop, or discover other people's services.

## 2. Data and graph inputs

Supply the original [LoCoMo `locomo10.json`](https://github.com/snap-research/locomo)
and [LongMemEval cleaned S split](https://github.com/xiaowu0162/LongMemEval).
Raw data and graph databases are not distributed in Git. The recorded input
hashes and graph hashes are in [PROTOCOL.json](PROTOCOL.json).

There are two explicit reproduction modes:

- **Frozen graph:** supply the graph SQLite snapshot used by the study. This
  isolates retrieval/reader reproduction from stochastic graph construction.
- **Fresh graph:** build from original data using the existing repository index
  implementation. The resulting graph and scores are a new run, not a claim of
  byte-identical historical construction. The historical graph-builder model
  settings are not fully captured by the newer product build profile.

Fresh build, if you do not already have the frozen graph:

```bash
# TOKENIZER is the Qwen30B tokenizer.json; model weights stay outside the repo.
graphmem-research build --benchmark locomo --data "$LOCOMO_JSON" \
  --tokenizer "$TOKENIZER" --build-config config/research-build-locomo.json \
  --output runs/build-locomo

graphmem-research build --benchmark longmemeval --data "$LME_JSON" \
  --tokenizer "$TOKENIZER" --build-config config/research-build-longmemeval.json \
  --output runs/build-longmemeval
```

Each build writes `memory.sqlite`, the resolved build configuration and complete
chat/embedding usage ledgers. Set `LOCOMO_GRAPH` and `LME_GRAPH` to these databases,
or to your frozen snapshots. Prepare source-only inference inputs:

```bash
graphmem-research prepare --benchmark locomo --data "$LOCOMO_JSON" \
  --graph-db "$LOCOMO_GRAPH" --tokenizer "$TOKENIZER" --output runs/locomo

graphmem-research prepare --benchmark longmemeval --data "$LME_JSON" \
  --graph-db "$LME_GRAPH" --tokenizer "$TOKENIZER" --output runs/longmemeval
```

`prepare` takes a consistent SQLite copy. Questions and original turns are written
to `input/`; answers, categories and gold evidence are isolated in `evaluation/`.
Normal retrieval and answering never open `evaluation/`.

## 3. Embeddings and retrieval

Install `.[research-encoder]` in your separate local GPU encoder environment and
provide a downloaded Qwen embedding model via `EMBED_MODEL`.

```bash
for benchmark in locomo longmemeval; do
  graphmem-research embed-base --root "runs/$benchmark" --config runs-endpoints.json
  # Run this line in the GPU encoder environment; the other stages use CPU.
  graphmem-research embed-views --root "runs/$benchmark" --model-path "$EMBED_MODEL"
  graphmem-research features --root "runs/$benchmark"
  graphmem-research rank --root "runs/$benchmark"
done
```

`embed-base` uses the benchmark-specific base embedding endpoint. `embed-views`
builds raw-turn and graph-context views plus the second query embedding. It logs
both full token lengths and actually attended lengths; the recorded LME encoder
truncated 7,359 **document-view** inputs at 2048 tokens. Queries were not truncated.
These document-view encoding costs are indexing costs, separate from LLM graph
construction and the two query embedding costs.

The shipped 11 XGBoost models are frozen assets, about 4.7 MB total. LoCoMo uses
10 conversation-held-out models; each question uses the model that excluded its
own conversation. LongMemEval uses the model fitted to all LoCoMo training labels,
with no LME fitting. No training runs during reproduction. Hyperparameters and
reader branches were development-selected; these scores are not untouched-test
estimates. The final reader optimization added no new reranker or training.

The batch budget allocator uses scores and text lengths only: 8,050 conservative
input tokens on average for LoCoMo, 8,700 for LME; up to 32,000 per query and at
least eight ranked turns before the cap. These are packing targets, **not** the
actual provider query-token totals. Run the complete cohort to reproduce batch
allocation; a smaller query batch changes the threshold.

To reuse your original caches, populate the same files under `input/`,
`qwen_embeddings/`, `features/`, and `features_qwen/`. Also supply
`base_query_usage.jsonl` (`qid`, `base_query_tokens`) and
`enhanced_query_usage.jsonl` (`qid`, `enhanced_query_tokens`) from their original
ledgers. A cache without query usage cannot produce a valid cost comparison.
Never substitute zeros for unknown embedding costs.

## 4. Answers, judge and reports

```bash
for benchmark in locomo longmemeval; do
  for phase in qwen astra; do
    graphmem-research answer --root "runs/$benchmark" \
      --output "runs/$benchmark/$phase" --phase "$phase" \
      --config runs-endpoints.json --workers 4
    graphmem-research judge --root "runs/$benchmark" \
      --output "runs/$benchmark/$phase" --config runs-endpoints.json --workers 4
    graphmem-research report --root "runs/$benchmark" \
      --output "runs/$benchmark/$phase"
  done
done
```

For a paid endpoint smoke check, add `--limit 2` to `answer` and use a separate
output directory. Such a run is intentionally incomplete and `report` exits
nonzero. Do not mix smoke outputs with the complete run.

Commands resume recorded requests only when their identities and hashes match.
API transport failures remain **unknown**, block automatic reissue, and invalidate
a complete cost claim. Malformed outputs can invoke only the frozen Qwen format
repair rules. There is no best-of sampling or answer-correctness routing.

Each answer's `call_keys` identifies every charged request. `summary.json` reports
full-cohort accuracy, question-type counts, mean source-ID recall and all-query
model tokens. Judge calls use a separate ledger. Astra reasoning tokens are
already included in output tokens; cached input tokens are still counted.

Matched full-gold-session Astra diagnostic:

```bash
graphmem-research answer --root runs/locomo --output runs/locomo/astra-oracle \
  --phase astra --oracle --config runs-endpoints.json
# Then run judge/report against the same output directory.
```

`--oracle` is explicitly evaluation-only: it replaces the evidence with complete
gold-bearing sessions while retaining the normal GraphMem reader branch. The
recorded oracle comparison and transport-conditioned denominators are explained
in [RESULTS.md](RESULTS.md). Source-ID recall, semantic fact coverage and answer
accuracy are distinct metrics; the CLI does not relabel one as another.

## 5. Validation and code map

```bash
pytest -q test/test_research_reproduction.py
pytest -q
```

The legacy development-set calibration test needs the separately supplied
200-question development split. Set `GRAPHMEM_DEVSET_DIR` to its directory to run
that test; an ordinary public checkout skips it when the data is absent.

The migration was checked against all saved accepted requests and responses,
without a new paid full-benchmark generation. Details are in
[VALIDATION.json](VALIDATION.json). Fresh model calls can differ with backend,
precision, scheduling and provider revisions, even at temperature zero.

| Code | Purpose |
|---|---|
| `src/graphmem/build/` | Existing within-session aggregation and cross-session graph construction |
| `research/features.py`, `enhanced_features.py` | Original source/graph feature extraction |
| `research/retrieval.py`, `assets/` | Frozen rankers, context completion and token allocation |
| `research/astra.py` | Date/topic/source expansion and the final single-call Astra reader |
| `research/qwen.py`, `contexts.py`, `readers/` | Accepted Qwen policies, repairs and fallbacks |
| `research/transport.py`, `judging.py`, `cli.py` | Request identity, native token accounting, frozen grading, reports |

The October product results remain in [the original reproduction guide](../REPRODUCTION.md).
The November research profiles do not silently replace the product service's defaults.
