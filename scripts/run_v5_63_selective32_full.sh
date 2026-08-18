#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V563_32_ROOT:-${WORKSPACE}/artifacts/report/v5_63/selective32_v1}"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_57_lossless_atomic.json"
RUNTIME="${REPO}/configs/v5/runtime_v5_63_selective32.json"
GRAPH_ROOT="${WORKSPACE}/artifacts/report/v5_57/full"
GRAPH_DB="${GRAPH_ROOT}/graph/graphmem.sqlite"
RELATION_DB="${GRAPH_ROOT}/graph/relation_embeddings.sqlite"
BASELINE="${WORKSPACE}/artifacts/report/v5_54/index_structure_ablation/turn32/full/answer"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
ANSWER_WORKERS="${V563_32_ANSWER_WORKERS:-256}"
JUDGE_WORKERS="${V563_32_JUDGE_WORKERS:-32}"
CHECKPOINT_EVERY="${V563_32_CHECKPOINT_EVERY:-256}"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
mkdir -p "${ROOT}/answer"

event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "${ROOT}/orchestrator.log"
}

wait_local_models() {
  local delay=5
  until curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
      && curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; do
    event "local model service unavailable; waiting ${delay}s for managed restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

while true; do
  wait_local_models
  if "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${GRAPH_DB}" --output-root "${ROOT}" \
      --run-root "${ROOT}/answer" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
      --config "${CONFIG}" --runtime-config "${RUNTIME}" \
      --answer-policy v5_63 --embedding \
      --embedding-request-model Qwen3-Embedding-0.6B \
      --relation-embedding-db "${RELATION_DB}" \
      --answer-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
      --answer-base-url http://127.0.0.1:8002/v1 \
      --answer-request-profile qwen --max-output-tokens 2000 \
      --answer-workers "${ANSWER_WORKERS}" --navigate-workers 32 \
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --label v563_selective32_qwen30 --resume \
      >>"${ROOT}/answer.log" 2>&1; then
    break
  fi
  event "32-turn answer pass interrupted; preserving checkpoints and resuming"
  sleep 15
done
event "32-turn answers complete"

judge_delta() {
  local benchmark="$1"
  local judge_root="${ROOT}/answer/judge_${benchmark}"
  local delta="${judge_root}/delta_answers.jsonl"
  local delta_manifest="${judge_root}/delta_manifest.json"
  local delta_judge="${judge_root}/delta"
  local baseline_judge="${BASELINE}/judge_${benchmark}/paired_verdicts.jsonl"
  mkdir -p "${judge_root}"
  "${PY}" "${REPO}/scripts/paired_judge_delta.py" prepare \
    --baseline-answers "${BASELINE}/answers.jsonl" \
    --candidate-answers "${ROOT}/answer/answers.jsonl" \
    --benchmark "${benchmark}" --output "${delta}" \
    --manifest "${delta_manifest}"
  if [[ -s "${delta}" ]]; then
    while true; do
      if [[ "${benchmark}" == "longmemeval" ]]; then
        "${PY}" "${REPO}/scripts/evaluate_mem0_judge.py" \
          --answers "${delta}" --output-dir "${delta_judge}" \
          --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
          --request-profile openai --workers "${JUDGE_WORKERS}" --resume && break
      else
        "${PY}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
          --data "${LOCOMO}" --answers "${delta}" \
          --output-dir "${delta_judge}" \
          --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
          --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
          --request-profile openai --workers "${JUDGE_WORKERS}" --resume && break
      fi
      event "${benchmark} judge unavailable; waiting for service recovery"
      sleep 15
    done
  else
    mkdir -p "${delta_judge}"
    : >"${delta_judge}/auto_eval.jsonl"
  fi
  "${PY}" "${REPO}/scripts/paired_judge_delta.py" merge \
    --baseline-answers "${BASELINE}/answers.jsonl" \
    --candidate-answers "${ROOT}/answer/answers.jsonl" \
    --baseline-judge "${baseline_judge}" \
    --delta-judge "${delta_judge}/auto_eval.jsonl" \
    --benchmark "${benchmark}" \
    --output "${judge_root}/paired_verdicts.jsonl" \
    --manifest "${judge_root}/paired_manifest.json"
}

judge_delta longmemeval & lme_pid=$!
judge_delta locomo & locomo_pid=$!
wait "${lme_pid}"
wait "${locomo_pid}"
event "32-turn paired judges complete"
