#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/mnt/ssd1/graphmem_v4_1_full_20260730}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/ssd1/graphmem_v4_0_deepseek_20260730}"
LME_SHARDS="${LME_SHARDS:-25}"
LOCOMO_SHARDS="${LOCOMO_SHARDS:-10}"
QUESTION_WORKERS="${QUESTION_WORKERS:-20}"
MAX_INFLIGHT="${MAX_INFLIGHT:-20}"
MAX_PARALLEL_SHARDS="${MAX_PARALLEL_SHARDS:-5}"
SHARD_RETRIES="${SHARD_RETRIES:-5}"

set -a
source "${REPO}/.env"
set +a

COMMON_ARGS=(
  --question-type all
  --variants hierarchical_hybrid_graph_v4_1_query
  --tree-mode hierarchical_hybrid_graph_v4_1_query
  --summary-schema graphmem_v4_1_query
  --deepseek-model deepseek-v4-flash
  --deepseek-base-url https://api.deepseek.com
  --llm-api-key-env DEEPSEEK_API_KEY
  --llm-request-profile deepseek
  --embedding-base-url http://127.0.0.1:8001/v1
  --embedding-model Qwen3-Embedding-0.6B
  --reasoning-effort none
  --qa-context-token-budget 10000
  --qa-max-tokens 512
  --v41-query-hard-limit-tokens 15000
  --question-workers "${QUESTION_WORKERS}"
  --summary-workers 32
  --max-inflight-deepseek "${MAX_INFLIGHT}"
  --resume
)

mkdir -p \
  "${RUN_ROOT}/lme/shards" \
  "${RUN_ROOT}/locomo/shards" \
  "${RUN_ROOT}/locomo/shard_data"

"${REPO}/.venv/bin/python" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); json.dump([x for x in p if x.get("question_type") in {"category_1","category_2","category_3","category_4"}],open(sys.argv[2],"w"),ensure_ascii=False,separators=(",",":"))' \
  "${REPO}/data/locomo10_graphmem.json" \
  "${RUN_ROOT}/locomo/locomo_category_1_4.json"

"${REPO}/.venv/bin/python" "${REPO}/scripts/shard_locomo_graphmem.py" \
  --data "${RUN_ROOT}/locomo/locomo_category_1_4.json" \
  --output-dir "${RUN_ROOT}/locomo/shard_data" \
  --shards "${LOCOMO_SHARDS}"

pids=()
labels=()

wait_batch() {
  local failed=0
  local position
  for position in "${!pids[@]}"; do
    if ! wait "${pids[$position]}"; then
      echo "${labels[$position]} failed after ${SHARD_RETRIES} attempts" >&2
      failed=1
    fi
  done
  pids=()
  labels=()
  [[ "${failed}" == "0" ]]
}

run_lme_shard() {
  local index="$1"
  local shard_out="${RUN_ROOT}/lme/shards/shard_${index}"
  local attempt
  mkdir -p "${shard_out}"
  for attempt in $(seq 1 "${SHARD_RETRIES}"); do
    echo "attempt=${attempt}" >>"${shard_out}/run.log"
    if "${REPO}/.venv/bin/python" "${REPO}/scripts/run_token_demo.py" \
      --data "${SOURCE_ROOT}/query_shards/lme_${index}.json" \
      --output-dir "${shard_out}" \
      --memory-cache-dir "${SOURCE_ROOT}/memory_cache_lme" \
      --max-questions 20 "${COMMON_ARGS[@]}" >>"${shard_out}/run.log" 2>&1; then
      return 0
    fi
    sleep $((attempt * 5))
  done
  return 1
}

run_locomo_shard() {
  local index="$1"
  local shard_out="${RUN_ROOT}/locomo/shards/shard_${index}"
  local attempt
  mkdir -p "${shard_out}"
  for attempt in $(seq 1 "${SHARD_RETRIES}"); do
    echo "attempt=${attempt}" >>"${shard_out}/run.log"
    if "${REPO}/.venv/bin/python" "${REPO}/scripts/run_token_demo.py" \
      --data "${RUN_ROOT}/locomo/shard_data/shard_${index}.json" \
      --output-dir "${shard_out}" \
      --memory-cache-dir "${SOURCE_ROOT}/memory_cache_locomo" \
      --max-questions 1540 "${COMMON_ARGS[@]}" >>"${shard_out}/run.log" 2>&1; then
      return 0
    fi
    sleep $((attempt * 5))
  done
  return 1
}

for index in $(seq -w 0 $((LME_SHARDS - 1))); do
  run_lme_shard "${index}" &
  pids+=("$!")
  labels+=("lme_${index}")
  if [[ "${#pids[@]}" -ge "${MAX_PARALLEL_SHARDS}" ]]; then
    wait_batch
  fi
done
[[ "${#pids[@]}" -eq 0 ]] || wait_batch

for index in $(seq 0 $((LOCOMO_SHARDS - 1))); do
  run_locomo_shard "${index}" &
  pids+=("$!")
  labels+=("locomo_${index}")
  if [[ "${#pids[@]}" -ge "${MAX_PARALLEL_SHARDS}" ]]; then
    wait_batch
  fi
done
[[ "${#pids[@]}" -eq 0 ]] || wait_batch

"${REPO}/.venv/bin/python" "${REPO}/scripts/merge_locomo_shards.py" \
  --shard-root "${RUN_ROOT}/lme/shards" \
  --output-dir "${RUN_ROOT}/lme/merged/hierarchical_hybrid_graph_v4_1_query" \
  --variant hierarchical_hybrid_graph_v4_1_query \
  --expected-questions 500

"${REPO}/.venv/bin/python" "${REPO}/scripts/merge_locomo_shards.py" \
  --shard-root "${RUN_ROOT}/locomo/shards" \
  --output-dir "${RUN_ROOT}/locomo/merged/hierarchical_hybrid_graph_v4_1_query" \
  --variant hierarchical_hybrid_graph_v4_1_query \
  --expected-questions 1540

echo "Frozen V4.1 full answer runs completed under ${RUN_ROOT}"
