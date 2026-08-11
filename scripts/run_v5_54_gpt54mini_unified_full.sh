#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V554_GPT54_ROOT:-${WORKSPACE}/artifacts/report/v5_54/gpt54mini_unified_full}"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_17_budget230.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
MB="${WORKSPACE}/third_party/memory-benchmarks"
BUILD_WORKERS="${GPT54_BUILD_MEMORY_WORKERS:-2}"
BUILD_CONCURRENCY="${GPT54_BUILD_CONCURRENCY:-4}"
ANSWER_WORKERS="${GPT54_ANSWER_WORKERS_PER_ARM:-32}"
JUDGE_WORKERS="${GPT54_JUDGE_WORKERS_PER_BENCHMARK:-16}"

set -a
# shellcheck disable=SC1091
source "${REPO}/.env"
set +a
: "${SGAO_API_KEY:?SGAO_API_KEY is required}"
SGAO_BASE_URL="${SGAO_BASE_URL:-https://sub2api.sgao.me/v1/}"
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
mkdir -p "${ROOT}" "${ROOT}/graph"

event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "${ROOT}/orchestrator.log"
}

run_build() {
  "${PY}" "${REPO}/scripts/run_v5_6_full_build.py" \
    --target-db "${ROOT}/graph/graphmem.sqlite" \
    --relation-embedding-db "${ROOT}/graph/relation_embeddings.sqlite" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
    --config "${CONFIG}" --profile b5 \
    --llm-model gpt-5.4-mini --llm-base-url "${SGAO_BASE_URL}" \
    --llm-api-key-env SGAO_API_KEY --llm-request-profile openai \
    --llm-request-timeout-seconds 60 \
    --enabled-relation-signals scene_similar,shared_entity,state_compatible \
    --embedding --memory-workers "${BUILD_WORKERS}" \
    --max-concurrency "${BUILD_CONCURRENCY}" \
    --require-zero-retries --require-complete-diagnostics \
    --report "${ROOT}/build_report.json"
}

until run_build >>"${ROOT}/build.log" 2>&1; do
  event "build pass incomplete; preserving caches and resuming after backoff"
  sleep 30
done
event "510/510 GPT-5.4-mini memories built"

answer_arm() {
  local turns="$1"
  local runtime="${REPO}/configs/v5/runtime_v5_54_gpt54mini_accuracy${turns}.json"
  local arm="${ROOT}/turn${turns}"
  mkdir -p "${arm}/answer"
  until "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${ROOT}/graph/graphmem.sqlite" \
      --output-root "${arm}" --run-root "${arm}/answer" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
      --config "${CONFIG}" --runtime-config "${runtime}" \
      --answer-policy v5_54 --embedding \
      --answer-model gpt-5.4-mini --answer-base-url "${SGAO_BASE_URL}" \
      --answer-api-key-env SGAO_API_KEY --answer-request-profile openai \
      --max-output-tokens 2000 --answer-workers "${ANSWER_WORKERS}" \
      --navigate-workers 32 --checkpoint-every 50 \
      --label "v554_gpt54mini_turn${turns}" --resume \
      >>"${arm}/answer.log" 2>&1; do
    event "turn${turns} answer interrupted; resuming after backoff"
    sleep 30
  done
  event "turn${turns} answers complete"
}

answer_arm 32 & answer32_pid=$!
answer_arm 64 & answer64_pid=$!
wait "${answer32_pid}"
wait "${answer64_pid}"

judge_arm() {
  local turns="$1"
  local answer="${ROOT}/turn${turns}/answer"
  until "${PY}" "${REPO}/scripts/evaluate_mem0_judge.py" \
      --answers "${answer}/answers_longmemeval.jsonl" \
      --output-dir "${answer}/judge_lme" --model gpt-5.6-luna \
      --base-url "${SGAO_BASE_URL}" --api-key-env SGAO_API_KEY \
      --request-profile openai --workers "${JUDGE_WORKERS}" --resume; do
    event "turn${turns} LME judge interrupted; resuming"
    sleep 30
  done
  until "${PY}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
      --data "${LOCOMO}" --answers "${answer}/answers_locomo.jsonl" \
      --output-dir "${answer}/judge_locomo" \
      --memory-benchmarks-repo "${MB}" --model gpt-5.6-luna \
      --base-url "${SGAO_BASE_URL}" --api-key-env SGAO_API_KEY \
      --request-profile openai --workers "${JUDGE_WORKERS}" --resume; do
    event "turn${turns} LoCoMo judge interrupted; resuming"
    sleep 30
  done
  event "turn${turns} judges complete"
}

judge_arm 32 & judge32_pid=$!
judge_arm 64 & judge64_pid=$!
wait "${judge32_pid}"
wait "${judge64_pid}"

"${PY}" "${REPO}/scripts/summarize_v5_54_gpt54mini_benchmark.py" \
  --root "${ROOT}" --output "${ROOT}/summary.json"
event "unified GPT-5.4-mini benchmark complete"
