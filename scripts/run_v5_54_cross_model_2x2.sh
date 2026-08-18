#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V554_CROSS_ROOT:-${WORKSPACE}/artifacts/report/v5_54/cross_model_2x2}"
Q30_NATIVE="${WORKSPACE}/artifacts/report/v5_54/index_structure_ablation"
G54_NATIVE="${WORKSPACE}/artifacts/report/v5_54/gpt54mini_unified_full"
Q30_GRAPH="${WORKSPACE}/artifacts/report/v5_21/full_minimal_repair/m2_safe_witness/graph/graphmem.sqlite"
G54_GRAPH="${G54_NATIVE}/graph/graphmem.sqlite"
CONFIG="${REPO}/configs/v5/v5_17_budget230.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
MB="${WORKSPACE}/third_party/memory-benchmarks"
Q30_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
Q30_WORKERS="${V554_CROSS_Q30_WORKERS_PER_ARM:-128}"
G54_WORKERS="${V554_CROSS_G54_WORKERS_PER_ARM:-32}"
JUDGE_WORKERS="${V554_CROSS_JUDGE_WORKERS_PER_JOB:-8}"

set -a
# shellcheck disable=SC1091
source "${REPO}/.env"
set +a
: "${SGAO_API_KEY:?SGAO_API_KEY is required}"
SGAO_BASE_URL="${SGAO_BASE_URL:-https://sub2api.sgao.me/v1}"
export GRAPHMEM_LOCAL_API_KEY="${GRAPHMEM_LOCAL_API_KEY:-EMPTY}"
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json}"
export PYTHONHASHSEED=0
mkdir -p "${ROOT}"

event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "${ROOT}/orchestrator.log"
}

wait_q30() {
  until curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; do
    event "Qwen3-30B endpoint unavailable; waiting for managed restart"
    sleep 10
  done
}

answer_complete() {
  local output="$1"
  [[ -s "${output}/run_manifest.json" ]] || return 1
  "${PY}" - "${output}/run_manifest.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
audit = payload.get("prompt_identity_audit", {})
raise SystemExit(0 if (
    int(payload.get("completed_questions", 0)) == 2040
    and bool(audit.get("question_ids_match"))
    and int(audit.get("prompt_hash_mismatches", -1)) == 0
) else 1)
PY
}

answer_cross() {
  local graph_owner="$1" budget="$2"
  local prepared metadata graph model base key_env profile workers output
  if [[ "${graph_owner}" == "q30_graph" ]]; then
    prepared="${Q30_NATIVE}/turn${budget}/full/prepare/prepared_answers.jsonl"
    metadata="${Q30_NATIVE}/turn${budget}/full/answer/answers.jsonl"
    graph="${Q30_GRAPH}"
    model="gpt-5.4-mini"; base="${SGAO_BASE_URL}"
    key_env="SGAO_API_KEY"; profile="openai"; workers="${G54_WORKERS}"
    output="${ROOT}/q30_graph_gpt54/turn${budget}/answer"
  else
    prepared="${G54_NATIVE}/turn${budget}/answer/prepared_answers.jsonl"
    metadata="${G54_NATIVE}/turn${budget}/answer/answers.jsonl"
    graph="${G54_GRAPH}"
    model="${Q30_MODEL}"; base="http://127.0.0.1:8002/v1"
    key_env="GRAPHMEM_LOCAL_API_KEY"; profile="qwen"; workers="${Q30_WORKERS}"
    output="${ROOT}/gpt54_graph_q30/turn${budget}/answer"
  fi
  mkdir -p "${output}"
  if answer_complete "${output}"; then
    event "${graph_owner}/turn${budget} cross answer already complete"
    return 0
  fi
  while true; do
    [[ "${profile}" != "qwen" ]] || wait_q30
    if "${PY}" "${REPO}/scripts/replay_v5_prepared_answers.py" \
        --prepared "${prepared}" --metadata-answers "${metadata}" \
        --source-db "${graph}" --config "${CONFIG}" --output-root "${output}" \
        --answer-model "${model}" --answer-base-url "${base}" \
        --answer-api-key-env "${key_env}" --answer-request-profile "${profile}" \
        --packing-model "${Q30_MODEL}" --max-output-tokens 2000 \
        --workers "${workers}" --checkpoint-every 64 --resume \
        >>"${output}/run.log" 2>&1; then
      break
    fi
    event "${graph_owner}/turn${budget} answer interrupted; preserving cache and resuming"
    sleep 15
  done
  event "${graph_owner}/turn${budget} cross answer complete"
}

judge_cross() {
  local cross="$1" budget="$2" answer
  answer="${ROOT}/${cross}/turn${budget}/answer"
  until "${PY}" "${REPO}/scripts/evaluate_mem0_judge.py" \
      --answers "${answer}/answers_longmemeval.jsonl" \
      --output-dir "${answer}/judge_lme" --model gpt-5.6-luna \
      --base-url "${SGAO_BASE_URL}" --api-key-env SGAO_API_KEY \
      --request-profile openai --workers "${JUDGE_WORKERS}" --resume; do
    event "${cross}/turn${budget} LME judge interrupted; resuming"
    sleep 15
  done
  until "${PY}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
      --data "${LOCOMO}" --answers "${answer}/answers_locomo.jsonl" \
      --output-dir "${answer}/judge_locomo" --memory-benchmarks-repo "${MB}" \
      --model gpt-5.6-luna --base-url "${SGAO_BASE_URL}" \
      --api-key-env SGAO_API_KEY --request-profile openai \
      --workers "${JUDGE_WORKERS}" --resume; do
    event "${cross}/turn${budget} LoCoMo judge interrupted; resuming"
    sleep 15
  done
  event "${cross}/turn${budget} judges complete"
}

answer_cross q30_graph 32 & a1=$!
answer_cross q30_graph 64 & a2=$!
answer_cross gpt54_graph 32 & a3=$!
answer_cross gpt54_graph 64 & a4=$!
wait "${a1}"; wait "${a2}"; wait "${a3}"; wait "${a4}"

judge_cross q30_graph_gpt54 32 & j1=$!
judge_cross q30_graph_gpt54 64 & j2=$!
judge_cross gpt54_graph_q30 32 & j3=$!
judge_cross gpt54_graph_q30 64 & j4=$!
wait "${j1}"; wait "${j2}"; wait "${j3}"; wait "${j4}"

"${PY}" "${REPO}/scripts/summarize_v5_54_cross_model_2x2.py" \
  --root "${ROOT}" --output "${ROOT}/summary.json"
event "cross-model 2x2 benchmark complete"
