#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V520_FULL_BUDGET_ROOT:-${WORKSPACE}/artifacts/report/v5_20/full_budget_benchmark}"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_17_budget230.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
GRAPH_ROOT="${WORKSPACE}/artifacts/report/v5_19/full_benchmark"
GRAPH_DB="${GRAPH_ROOT}/graph/graphmem.sqlite"
DENSE="${GRAPH_ROOT}/dense_indexes"
QUERY_CACHE="${GRAPH_ROOT}/query_embeddings.sqlite"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
MEM0="${WORKSPACE}/artifacts/report/v5_19/mem0_qwen30_cutoffs.json"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0

for path in "${PYTHON_BIN}" "${LME}" "${LOCOMO}" "${GOLD}" "${CONFIG}" \
            "${GRAPHMEM_TOKENIZER_PATH}" "${GRAPH_DB}" "${DENSE}" \
            "${QUERY_CACHE}" "${MEMORY_BENCHMARKS}" "${MEM0}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
mkdir -p "${ROOT}"

wait_local_models() {
  local delay=5
  while true; do
    if curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
        && curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; then
      return 0
    fi
    echo "local vLLM unavailable; waiting ${delay}s for the managed restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

answer_arm() {
  local turns="$1"
  local arm_root="${ROOT}/turn${turns}"
  mkdir -p "${arm_root}"
  while true; do
    wait_local_models
    if "${PYTHON_BIN}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${GRAPH_DB}" --output-root "${arm_root}" \
      --run-root "${arm_root}/answer" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
      --config "${CONFIG}" --profile h11 \
      --max-evidence-turns "${turns}" --max-evidence-tokens 12000 \
      --max-answer-tokens 12000 --max-output-tokens 2000 \
      --span-pack-window 96 --obligation-aware-packing \
      --native-seed-fusion --queryir-soft-fallback \
      --source-time-normalization --precision-grounded-prompt \
      --graph-hop-decay 0.3 --expansion-beam 2 \
      --rare-lexical-relations --query-gated-rare-lexical \
      --embedding --dense-sidecar-dir "${DENSE}" --dense-backend faiss_flat \
      --query-embedding-cache "${QUERY_CACHE}" \
      --evidence-order topological \
      --answer-workers "${V520_FULL_ANSWER_WORKERS:-192}" \
      --checkpoint-every 100 --label "v520_full_topology${turns}_qwen30" \
      --resume; then
      break
    fi
    echo "turn${turns} full answer interrupted; waiting and resuming"
    sleep 5
  done
}

judge_arm() {
  local turns="$1"
  local answer_root="${ROOT}/turn${turns}/answer"
  judge_lme() {
    while true; do
      if "${PYTHON_BIN}" "${REPO}/scripts/evaluate_mem0_judge.py" \
        --answers "${answer_root}/answers_longmemeval.jsonl" \
        --output-dir "${answer_root}/judge_lme" --model gpt-5.6-luna \
        --api-key-env SGAO_API_KEY --request-profile openai \
        --workers 32 --resume; then
        break
      fi
      echo "turn${turns} LME judge unavailable; waiting and resuming"
      sleep 15
    done
  }
  judge_locomo() {
    while true; do
      if "${PYTHON_BIN}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
        --data "${LOCOMO}" --answers "${answer_root}/answers_locomo.jsonl" \
        --output-dir "${answer_root}/judge_locomo" \
        --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
        --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
        --request-profile openai --workers 32 --resume; then
        break
      fi
      echo "turn${turns} LoCoMo judge unavailable; waiting and resuming"
      sleep 15
    done
  }
  judge_lme &
  local lme_pid=$!
  judge_locomo &
  local locomo_pid=$!
  wait "${lme_pid}"
  wait "${locomo_pid}"
}

# Judge turn32 remotely while the local Qwen service answers turn64.
answer_arm 32
judge_arm 32 &
judge32_pid=$!
answer_arm 64
judge_arm 64
wait "${judge32_pid}"

"${PYTHON_BIN}" "${REPO}/scripts/summarize_v5_20_budget_benchmark.py" \
  --root "${ROOT}" --mem0 "${MEM0}" --output "${ROOT}/summary.json"
