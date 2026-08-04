#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/mnt/ssd1/graphmem_v419_gpt54_latest_full_20260804}"
LME_SHARD_ROOT="${LME_SHARD_ROOT:-/mnt/ssd1/graphmem_v4_0_deepseek_20260730/query_shards}"
MEMORY_CACHE_DIR="${MEMORY_CACHE_DIR:-/mnt/ssd1/graphmem_v419_gpt54_unified_full_20260803/memory_cache_lme}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-25}"
QUESTION_WORKERS="${QUESTION_WORKERS:-20}"
INFLIGHT_PER_SHARD="${INFLIGHT_PER_SHARD:-20}"
JUDGE_WORKERS="${JUDGE_WORKERS:-256}"
LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-600}"
SHARD_RETRIES="${SHARD_RETRIES:-8}"
export OMP_NUM_THREADS="${GRAPHMEM_CPU_THREADS:-1}"
export MKL_NUM_THREADS="${GRAPHMEM_CPU_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${GRAPHMEM_CPU_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${GRAPHMEM_CPU_THREADS:-1}"
export TOKENIZERS_PARALLELISM="false"
PYTHON_BIN="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "${REPO}/../.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO}/../.venv/bin/python"
fi
[[ -x "${PYTHON_BIN}" ]] || {
  echo "Python virtual environment not found: ${PYTHON_BIN}" >&2
  exit 2
}

set -a
source "${REPO}/.env"
set +a

: "${SGAO_API_KEY:?SGAO_API_KEY is required}"
: "${EMBEDDING_API_KEY:?EMBEDDING_API_KEY is required}"

VARIANT=hierarchical_hybrid_graph_v4_1_query
mkdir -p "${RUN_ROOT}/lme/shards" "${RUN_ROOT}/lme/judge" \
  "${RUN_ROOT}/lme/report"

curl -fsS --max-time 10 -H "Authorization: Bearer ${EMBEDDING_API_KEY}" \
  http://127.0.0.1:8001/v1/models >/dev/null

COMMON_ARGS=(
  --question-type all
  --variants "${VARIANT}"
  --tree-mode "${VARIANT}"
  --summary-schema graphmem_v4_1_query
  --llm-model gpt-5.4-mini
  --llm-base-url https://sub2api.sgao.me/v1/
  --llm-api-key-env SGAO_API_KEY
  --llm-request-profile openai
  --llm-timeout-sec "${LLM_TIMEOUT_SEC}"
  --embedding-base-url http://127.0.0.1:8001/v1
  --embedding-model Qwen3-Embedding-0.6B
  --reasoning-effort none
  --qa-context-token-budget 10000
  --qa-max-tokens 512
  --v41-query-target-tokens 10000
  --v41-query-hard-limit-tokens 14000
  --record-query-budget-overflow
  --summary-workers 32
  --resume
)

run_shard() {
  local index="$1" shard_out attempt
  shard_out="${RUN_ROOT}/lme/shards/shard_${index}"
  mkdir -p "${shard_out}"
  for attempt in $(seq 1 "${SHARD_RETRIES}"); do
    printf 'attempt=%s shard=%s\n' "${attempt}" "${index}" >>"${shard_out}/run.log"
    if "${PYTHON_BIN}" "${REPO}/scripts/run_token_demo.py" \
      --data "${LME_SHARD_ROOT}/lme_${index}.json" \
      --output-dir "${shard_out}" \
      --memory-cache-dir "${MEMORY_CACHE_DIR}" \
      --max-questions 20 \
      --question-workers "${QUESTION_WORKERS}" \
      --max-inflight-deepseek "${INFLIGHT_PER_SHARD}" \
      "${COMMON_ARGS[@]}" >>"${shard_out}/run.log" 2>&1; then
      return 0
    fi
    sleep $((attempt * 5))
  done
  return 1
}

pids=()
labels=()
wait_batch() {
  local position failed=0
  for position in "${!pids[@]}"; do
    if ! wait "${pids[$position]}"; then
      printf '%s failed\n' "${labels[$position]}" >&2
      failed=1
    fi
  done
  pids=()
  labels=()
  [[ "${failed}" == 0 ]]
}

for index in $(seq -w 0 24); do
  run_shard "${index}" &
  pids+=("$!")
  labels+=("lme_${index}")
  if [[ ${#pids[@]} -ge ${PARALLEL_SHARDS} ]]; then
    wait_batch
  fi
done
[[ ${#pids[@]} -eq 0 ]] || wait_batch

MERGED_DIR="${RUN_ROOT}/lme/merged/${VARIANT}"
"${PYTHON_BIN}" "${REPO}/scripts/merge_locomo_shards.py" \
  --shard-root "${RUN_ROOT}/lme/shards" \
  --output-dir "${MERGED_DIR}" --variant "${VARIANT}" \
  --expected-questions 500

"${PYTHON_BIN}" "${REPO}/scripts/evaluate_mem0_judge.py" \
  --answers "${MERGED_DIR}/answers.jsonl" \
  --output-dir "${RUN_ROOT}/lme/judge" \
  --model gpt-5.4-mini --base-url https://sub2api.sgao.me/v1/ \
  --api-key-env SGAO_API_KEY --request-profile openai \
  --workers "${JUDGE_WORKERS}" --resume

"${PYTHON_BIN}" "${REPO}/scripts/summarize_unified_benchmark.py" \
  --run-root "${RUN_ROOT}/lme" --benchmark longmemeval \
  --judge "${RUN_ROOT}/lme/judge" --variant "${VARIANT}" \
  --expected 500 --output-dir "${RUN_ROOT}/lme/report"

echo "GPT-5.4-mini V4.1 LongMemEval answer, judge and report complete: ${RUN_ROOT}"
