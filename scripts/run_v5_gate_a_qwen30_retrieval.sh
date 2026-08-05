#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPHMEM_ARTIFACTS="${GRAPHMEM_ARTIFACTS:-/ssd3/chenhan/Spark_MemGraph_Dev/artifacts}"
DEVSET_ROOT="${DEVSET_ROOT:-${GRAPHMEM_ARTIFACTS}/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804}"
RUN_ROOT="${RUN_ROOT:-${GRAPHMEM_ARTIFACTS}/v5/gate_a_qwen30_20260804}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenhan/miniconda3/envs/agent/bin/python}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8002/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
LME_SHARDS="${LME_SHARDS:-8}"
LOCOMO_SHARDS="${LOCOMO_SHARDS:-10}"

export QWEN_API_KEY="${QWEN_API_KEY:-local-vllm}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-local-vllm}"

curl -fsS --max-time 10 "${LLM_BASE_URL}/models" >/dev/null
curl -fsS --max-time 10 "${EMBEDDING_BASE_URL}/models" >/dev/null

common=(
  --question-type all
  --variants hierarchical_hybrid_graph_v4_1_query
  --tree-mode hierarchical_hybrid_graph_v4_1_query
  --summary-schema graphmem_v4_1_query
  --llm-model "${LLM_MODEL}"
  --llm-base-url "${LLM_BASE_URL}"
  --llm-api-key-env QWEN_API_KEY
  --llm-request-profile qwen
  --embedding-base-url "${EMBEDDING_BASE_URL}"
  --embedding-model "${EMBEDDING_MODEL}"
  --reasoning-effort none
  --retrieval-only
  --qa-context-token-budget 10000
  --v41-query-target-tokens 10000
  --v41-query-hard-limit-tokens 13000
  --v36-llm-session-cap 0
  --summary-workers 16
  --question-workers 4
  --max-inflight-deepseek 16
  --resume
)

mkdir -p "${RUN_ROOT}"/{lme,locomo,memory_cache_lme,memory_cache_locomo,shard_data_lme,shard_data_locomo}
"${PYTHON_BIN}" "${REPO}/scripts/shard_v5_devset.py" \
  --data "${DEVSET_ROOT}/longmemeval_hard_multisession50_temporal50.json" \
  --output-dir "${RUN_ROOT}/shard_data_lme" --shards "${LME_SHARDS}"
"${PYTHON_BIN}" "${REPO}/scripts/shard_v5_devset.py" \
  --data "${DEVSET_ROOT}/locomo_hard_cat1_multihop50_cat2_temporal50.json" \
  --output-dir "${RUN_ROOT}/shard_data_locomo" --shards "${LOCOMO_SHARDS}" \
  --group-field locomo_sample_id

run_phase() {
  local benchmark="$1" shards="$2" cache="$3" shard_data="$4" index output log
  local pids=()
  mkdir -p "${RUN_ROOT}/${benchmark}/shards"
  local ordinal
  for ordinal in $(seq 0 $((shards - 1))); do
    printf -v index '%02d' "${ordinal}"
    output="${RUN_ROOT}/${benchmark}/shards/shard_${index}"
    log="${output}/run.log"
    mkdir -p "${output}"
    (
      for attempt in 1 2 3; do
        if "${PYTHON_BIN}" "${REPO}/scripts/run_token_demo.py" \
          --data "${shard_data}/shard_${index}.json" \
          --output-dir "${output}" --memory-cache-dir "${cache}" \
          --max-questions 100 "${common[@]}" >>"${log}" 2>&1; then
          exit 0
        fi
        sleep $((attempt * 5))
      done
      exit 1
    ) &
    pids+=("$!")
  done
  local pid failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || return 1
  "${PYTHON_BIN}" "${REPO}/scripts/merge_locomo_shards.py" \
    --shard-root "${RUN_ROOT}/${benchmark}/shards" \
    --output-dir "${RUN_ROOT}/${benchmark}/merged/hierarchical_hybrid_graph_v4_1_query" \
    --variant hierarchical_hybrid_graph_v4_1_query --expected-questions 100
}

run_phase lme "${LME_SHARDS}" "${RUN_ROOT}/memory_cache_lme" "${RUN_ROOT}/shard_data_lme"
run_phase locomo "${LOCOMO_SHARDS}" "${RUN_ROOT}/memory_cache_locomo" "${RUN_ROOT}/shard_data_locomo"

"${PYTHON_BIN}" "${REPO}/scripts/v5_audit_baseline.py" \
  --lme-data "${DEVSET_ROOT}/longmemeval_hard_multisession50_temporal50.json" \
  --locomo-data "${DEVSET_ROOT}/locomo_hard_cat1_multihop50_cat2_temporal50.json" \
  --lme-run "${RUN_ROOT}/lme/merged/hierarchical_hybrid_graph_v4_1_query" \
  --locomo-run "${RUN_ROOT}/locomo/merged/hierarchical_hybrid_graph_v4_1_query" \
  --output-dir "${RUN_ROOT}/audit"
