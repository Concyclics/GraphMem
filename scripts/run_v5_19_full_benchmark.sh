#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ABLATION_ROOT="${V519_ABLATION_ROOT:-${WORKSPACE}/artifacts/report/v5_19/attribute_ablation_dev200}"
ROOT="${V519_FULL_ROOT:-${WORKSPACE}/artifacts/report/v5_19/full_benchmark}"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_17_budget230.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
REPORT_REPO="${WORKSPACE}/GraphMem_report"
MEM0_RESULTS="${WORKSPACE}/artifacts/report/v5_19/mem0_qwen30_cutoffs.json"
MEM0_ARCHIVE="${V519_MEM0_ARCHIVE:-/shared/s3/GraphMem_eval}"
QWEN_ANSWER_WORKERS="${V519_QWEN_ANSWER_WORKERS:-256}"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
GPT_BASE_URL="${SGAO_BASE_URL:-https://sub2api.sgao.me/v1/}"

for path in "${PYTHON_BIN}" "${LME}" "${LOCOMO}" "${GOLD}" "${CONFIG}" "${GRAPHMEM_TOKENIZER_PATH}" \
            "${ABLATION_ROOT}/full/graph/graphmem.sqlite" \
            "${ABLATION_ROOT}/full/graph/relation_embeddings.sqlite"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
mkdir -p "${ROOT}/graph"

wait_local_models() {
  local delay=5
  while true; do
    if curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
        && curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; then
      return 0
    fi
    echo "local vLLM unavailable; waiting ${delay}s for the managed service restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

while true; do
  wait_local_models
  if "${PYTHON_BIN}" "${REPO}/scripts/run_v5_6_full_build.py" \
    --seed-db "${ABLATION_ROOT}/full/graph/graphmem.sqlite" \
    --seed-report "${ABLATION_ROOT}/full/build_report.json" \
    --seed-relation-embedding-db "${ABLATION_ROOT}/full/graph/relation_embeddings.sqlite" \
    --target-db "${ROOT}/graph/graphmem.sqlite" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
    --config "${CONFIG}" --profile b5 --embedding \
    --relation-embedding-db "${ROOT}/graph/relation_embeddings.sqlite" \
    --memory-workers 16 --max-concurrency 256 \
    --require-zero-retries --require-complete-diagnostics \
    --enabled-relation-signals \
      scene_similar,shared_entity,state_compatible,collection_related,temporal_near,lexical_rare \
    --report "${ROOT}/build_report.json"; then
    break
  fi
  echo "full build interrupted or failed; waiting for services and resuming"
  sleep 5
done

"${PYTHON_BIN}" "${REPO}/scripts/precompile_dense_indexes.py" \
  --db "${ROOT}/graph/graphmem.sqlite" --config "${CONFIG}" \
  --output "${ROOT}/dense_indexes" --backend faiss_flat --workers 16

while true; do
  wait_local_models
  if "${PYTHON_BIN}" "${REPO}/scripts/run_v5_6_answer.py" \
    --source-db "${ROOT}/graph/graphmem.sqlite" \
    --output-root "${ROOT}" --run-root "${ROOT}/qwen30" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
    --config "${CONFIG}" --profile h11 \
    --max-evidence-turns 64 --max-evidence-tokens 12000 \
    --span-pack-window 96 --obligation-aware-packing \
    --native-seed-fusion --queryir-soft-fallback --source-time-normalization \
    --graph-hop-decay 0.3 --expansion-beam 2 --rare-lexical-relations \
    --embedding --dense-sidecar-dir "${ROOT}/dense_indexes" \
    --dense-backend faiss_flat --query-embedding-cache "${ROOT}/query_embeddings.sqlite" \
    --answer-workers "${QWEN_ANSWER_WORKERS}" --checkpoint-every 25 \
    --label v519_full_accuracy64_qwen30 \
    --resume; then
    break
  fi
  echo "Qwen full answer interrupted; waiting for vLLM and resuming checkpoint"
  sleep 5
done

judge_root() {
  local answer_root="$1"
  "${PYTHON_BIN}" "${REPO}/scripts/evaluate_mem0_judge.py" \
    --answers "${answer_root}/answers_longmemeval.jsonl" \
    --output-dir "${answer_root}/judge_lme" --model gpt-5.6-luna \
    --api-key-env SGAO_API_KEY --request-profile openai --workers 32 --resume
  "${PYTHON_BIN}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
    --data "${LOCOMO}" --answers "${answer_root}/answers_locomo.jsonl" \
    --output-dir "${answer_root}/judge_locomo" \
    --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
    --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
    --request-profile openai --workers 32 --resume
}

judge_root "${ROOT}/qwen30"

if [[ -d "${MEM0_ARCHIVE}" ]]; then
  "${PYTHON_BIN}" "${REPO}/scripts/import_v5_19_mem0_baseline.py" \
    --archive "${MEM0_ARCHIVE}" --output "${MEM0_RESULTS}"
else
  echo "Mem0 archive unavailable at ${MEM0_ARCHIVE}; preserving pending rows" >&2
fi

"${PYTHON_BIN}" "${REPO}/scripts/render_v5_19_benchmark_manifest.py" \
  --build-report "${ROOT}/build_report.json" --qwen-root "${ROOT}/qwen30" \
  --gpt-root "${ROOT}/gpt54" --mem0 "${MEM0_RESULTS}" \
  --output "${ROOT}/benchmark_manifest.json"
"${PYTHON_BIN}" "${REPO}/scripts/render_v5_19_report_assets.py" \
  --ablation "${ABLATION_ROOT}/ablation_summary.json" \
  --benchmark "${ROOT}/benchmark_manifest.json" --report "${REPORT_REPO}"
"${PYTHON_BIN}" "${REPO}/scripts/audit_v5_19_experiment.py" full \
  --root "${ROOT}" --expected-lme 500 --expected-locomo 1540 --qwen-only

# GPT-5.4-mini is intentionally not replayed over a Qwen-built graph.  Its
# end-to-end result must use a separately rebuilt GPT graph so construction
# and answering use the same model.
