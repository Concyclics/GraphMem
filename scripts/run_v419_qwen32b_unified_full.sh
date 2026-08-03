#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/chenhan/graphmem_v419_qwen32b_unified_full_20260803}"
DATA_ROOT="${DATA_ROOT:-/home/chenhan/GraphMem-Qwen/benchmark_data}"
LME_SHARD_ROOT="${LME_SHARD_ROOT:-${DATA_ROOT}/lme_query_shards}"
LME_PARALLEL_SHARDS="${LME_PARALLEL_SHARDS:-8}"
LME_INFLIGHT_PER_SHARD="${LME_INFLIGHT_PER_SHARD:-16}"
LOCOMO_PARALLEL_SHARDS="${LOCOMO_PARALLEL_SHARDS:-10}"
LOCOMO_INFLIGHT_PER_SHARD="${LOCOMO_INFLIGHT_PER_SHARD:-8}"
QUERY_HARD_LIMIT="${QUERY_HARD_LIMIT:-14000}"
SHARD_RETRIES="${SHARD_RETRIES:-6}"

set -a
source "${REPO}/.env"
set +a

: "${SGAO_API_KEY:?SGAO_API_KEY is required}"
: "${QWEN_API_KEY:?QWEN_API_KEY is required}"
: "${EMBEDDING_API_KEY:?EMBEDDING_API_KEY is required}"

mkdir -p "${RUN_ROOT}"/{lme,locomo}/{shards,judge} \
  "${RUN_ROOT}"/{memory_cache_lme,memory_cache_locomo}

curl -fsS --max-time 10 -H "Authorization: Bearer ${EMBEDDING_API_KEY}" \
  http://127.0.0.1:8001/v1/models >/dev/null

"${REPO}/.venv/bin/python" - "${REPO}/data/locomo10_graphmem.json" \
  "${RUN_ROOT}/locomo/locomo_category_1_4.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
selected=[row for row in rows if row.get("question_type") in {
    "category_1","category_2","category_3","category_4",
}]
json.dump(selected,open(sys.argv[2],"w"),ensure_ascii=False,separators=(",",":"))
assert len(selected)==1540,len(selected)
PY

"${REPO}/.venv/bin/python" "${REPO}/scripts/shard_locomo_graphmem.py" \
  --data "${RUN_ROOT}/locomo/locomo_category_1_4.json" \
  --output-dir "${RUN_ROOT}/locomo/shard_data" --shards 10

COMMON_ARGS=(
  --question-type all
  --variants hierarchical_hybrid_graph_v4_1_query
  --tree-mode hierarchical_hybrid_graph_v4_1_query
  --summary-schema graphmem_v4_1_query
  --llm-model Qwen3-32B-FP8
  --llm-base-url http://127.0.0.1:8002/v1
  --llm-api-key-env QWEN_API_KEY
  --llm-request-profile qwen
  --embedding-base-url http://127.0.0.1:8001/v1
  --embedding-model Qwen3-Embedding-0.6B
  --reasoning-effort none
  --qa-context-token-budget 10000
  --qa-max-tokens 512
  --v41-query-target-tokens 10000
  --v41-query-hard-limit-tokens "${QUERY_HARD_LIMIT}"
  --record-query-budget-overflow
  --summary-workers 32
  --resume
)

run_with_retries() {
  local label="$1"
  local log="$2"
  shift 2
  local attempt
  for attempt in $(seq 1 "${SHARD_RETRIES}"); do
    printf 'attempt=%s label=%s\n' "${attempt}" "${label}" >>"${log}"
    if "$@" >>"${log}" 2>&1; then
      return 0
    fi
    sleep $((attempt * 5))
  done
  return 1
}

run_lme() {
  local pids=() labels=() index shard_out
  for index in $(seq -w 0 24); do
    shard_out="${RUN_ROOT}/lme/shards/shard_${index}"
    mkdir -p "${shard_out}"
    run_with_retries "lme_${index}" "${shard_out}/run.log" \
      "${REPO}/.venv/bin/python" "${REPO}/scripts/run_token_demo.py" \
      --data "${LME_SHARD_ROOT}/lme_${index}.json" \
      --output-dir "${shard_out}" \
      --memory-cache-dir "${RUN_ROOT}/memory_cache_lme" \
      --max-questions 20 --question-workers 20 \
      --max-inflight-deepseek "${LME_INFLIGHT_PER_SHARD}" \
      "${COMMON_ARGS[@]}" &
    pids+=("$!") labels+=("lme_${index}")
    if [[ ${#pids[@]} -ge ${LME_PARALLEL_SHARDS} ]]; then
      local i
      for i in "${!pids[@]}"; do wait "${pids[$i]}" || return 1; done
      pids=() labels=()
    fi
  done
  local i
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || return 1; done
  "${REPO}/.venv/bin/python" "${REPO}/scripts/merge_locomo_shards.py" \
    --shard-root "${RUN_ROOT}/lme/shards" \
    --output-dir "${RUN_ROOT}/lme/merged/hierarchical_hybrid_graph_v4_1_query" \
    --variant hierarchical_hybrid_graph_v4_1_query --expected-questions 500
}

run_locomo() {
  local pids=() index shard_out
  for index in $(seq 0 9); do
    shard_out="${RUN_ROOT}/locomo/shards/shard_${index}"
    mkdir -p "${shard_out}"
    run_with_retries "locomo_${index}" "${shard_out}/run.log" \
      "${REPO}/.venv/bin/python" "${REPO}/scripts/run_token_demo.py" \
      --data "${RUN_ROOT}/locomo/shard_data/shard_${index}.json" \
      --output-dir "${shard_out}" \
      --memory-cache-dir "${RUN_ROOT}/memory_cache_locomo" \
      --max-questions 1540 --question-workers 64 \
      --max-inflight-deepseek "${LOCOMO_INFLIGHT_PER_SHARD}" \
      "${COMMON_ARGS[@]}" &
    pids+=("$!")
  done
  local i
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || return 1; done
  "${REPO}/.venv/bin/python" "${REPO}/scripts/merge_locomo_shards.py" \
    --shard-root "${RUN_ROOT}/locomo/shards" \
    --output-dir "${RUN_ROOT}/locomo/merged/hierarchical_hybrid_graph_v4_1_query" \
    --variant hierarchical_hybrid_graph_v4_1_query --expected-questions 1540
}

run_lme >"${RUN_ROOT}/lme/full.log" 2>&1 & lme_pid=$!
run_locomo >"${RUN_ROOT}/locomo/full.log" 2>&1 & locomo_pid=$!
wait "${lme_pid}"
wait "${locomo_pid}"

"${REPO}/.venv/bin/python" "${REPO}/scripts/evaluate_mem0_judge.py" \
  --answers "${RUN_ROOT}/lme/merged/hierarchical_hybrid_graph_v4_1_query/answers.jsonl" \
  --output-dir "${RUN_ROOT}/lme/judge" --model gpt-5.4-mini \
  --base-url https://sub2api.sgao.me/v1/ --api-key-env SGAO_API_KEY \
  --request-profile openai --workers 128 --resume & lme_judge_pid=$!

"${REPO}/.venv/bin/python" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
  --data "${RUN_ROOT}/locomo/locomo_category_1_4.json" \
  --answers "${RUN_ROOT}/locomo/merged/hierarchical_hybrid_graph_v4_1_query/answers.jsonl" \
  --output-dir "${RUN_ROOT}/locomo/judge" --model gpt-5.4-mini \
  --base-url https://sub2api.sgao.me/v1/ --api-key-env SGAO_API_KEY \
  --request-profile openai --workers 128 --resume & locomo_judge_pid=$!

wait "${lme_judge_pid}"
wait "${locomo_judge_pid}"
echo "Unified Qwen3-32B-FP8 memory + GPT-5.4-mini judge full run complete: ${RUN_ROOT}"
