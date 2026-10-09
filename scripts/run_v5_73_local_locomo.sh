#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
SOURCE_ROOT="${V573_SOURCE_ROOT:-${WORKSPACE}/artifacts/report/v5_72_local_locomo_best_of_8}"
ROOT="${V573_ROOT:-${WORKSPACE}/artifacts/report/v5_73_local_locomo_split8}"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_57_lossless_atomic.json"
RUNTIME="${REPO}/configs/v5/runtime_v5_73_accuracy64.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
GRAPH_DB="${SOURCE_ROOT}/graph/graphmem.sqlite"
RELATION_DB="${SOURCE_ROOT}/graph/relation_embeddings.sqlite"
DENSE_DIR="${SOURCE_ROOT}/dense_indexes"
QUERY_CACHE="${SOURCE_ROOT}/query_embedding_cache.sqlite"
LLM_URL="http://127.0.0.1:8002/v1"
EMBED_URL="http://127.0.0.1:8003/v1"
LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
LUNA_MODEL="${SGAO_MODEL_GPT56_LUNA:-gpt-5.6-luna}"
mkdir -p "${ROOT}"

event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "${ROOT}/orchestrator.log"
}

wait_local_models() {
  local delay=5
  until curl -fsS --max-time 5 "${LLM_URL}/models" >/dev/null \
      && curl -fsS --max-time 5 "${EMBED_URL}/models" >/dev/null; do
    event "local model service unavailable; waiting ${delay}s for managed restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

PREPARE="${ROOT}/prepare"
until wait_local_models && "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
    --source-db "${GRAPH_DB}" --relation-embedding-db "${RELATION_DB}" \
    --output-root "${ROOT}" --run-root "${PREPARE}" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
    --lme-type __locomo_only__ \
    --locomo-category 1 --locomo-category 2 \
    --locomo-category 3 --locomo-category 4 \
    --config "${CONFIG}" --runtime-config "${RUNTIME}" \
    --answer-policy v5_73 --embedding \
    --embedding-model BAAI/bge-m3 --embedding-base-url "${EMBED_URL}" \
    --embedding-request-model bge-m3 --dense-sidecar-dir "${DENSE_DIR}" \
    --compiled-cache-dir "${ROOT}/compiled_memory_views" \
    --query-embedding-cache "${QUERY_CACHE}" \
    --prepare-only --navigate-workers 32 --checkpoint-every 50 --resume \
    >>"${ROOT}/prepare.log" 2>&1; do
  event "V5.73 preparation interrupted; preserving checkpoints and waiting"
  sleep 20
done
event "1,540 V5.73 64-turn prompts prepared from the frozen local graph"

"${PY}" "${REPO}/scripts/audit_v5_73_retrieval_gate.py" \
  --baseline "${SOURCE_ROOT}/prepare/retrieval.jsonl" \
  --candidate "${PREPARE}/retrieval.jsonl" \
  --output "${ROOT}/retrieval_gate.json" --expected 1540 \
  | tee "${ROOT}/retrieval_gate.txt"
event "paired retrieval gate passed before answer generation"
if [[ "${V573_PREPARE_ONLY:-0}" == "1" ]]; then
  event "preparation-only mode complete"
  exit 0
fi

ENSEMBLE="${ROOT}/answer_split8"
until wait_local_models && "${PY}" \
    "${REPO}/scripts/replay_v5_prepared_ensemble.py" \
    --prepared "${PREPARE}/prepared_answers.jsonl" \
    --locomo-data "${LOCOMO}" --output-root "${ENSEMBLE}" \
    --model "${LLM_MODEL}" --base-url "${LLM_URL}" \
    --n-per-family 4 --temperature 0.7 --top-p 0.8 --top-k 20 \
    --seed 7300 --max-output-tokens 2000 --verifier-max-tokens 256 \
    --workers 48 --max-retries 0 --resume \
    >>"${ROOT}/answer_split8.log" 2>&1; do
  event "split-prompt answer/verifier pass interrupted; resuming after restart"
  sleep 20
done
event "two n=4 answer families and local evidence selection complete"

JUDGE="${ROOT}/judge_luna_medium"
until "${PY}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
    --data "${LOCOMO}" --answers "${ENSEMBLE}/answers.jsonl" \
    --output-dir "${JUDGE}" --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
    --model "${LUNA_MODEL}" --base-url "${SGAO_BASE_URL}" \
    --api-key-env SGAO_API_KEY --request-profile openai \
    --reasoning-effort medium --max-tokens 2048 --workers 64 --resume \
    >>"${ROOT}/judge.log" 2>&1; do
  event "Luna-medium judge interrupted; preserving verdicts and resuming"
  sleep 20
done
event "1,540 selected answers judged with Luna-medium"

"${PY}" "${REPO}/scripts/summarize_v5_73_locomo.py" \
  --judge "${JUDGE}/auto_eval.jsonl" \
  --candidates "${ENSEMBLE}/candidates.jsonl" \
  --selections "${ENSEMBLE}/selections.jsonl" \
  --retrieval "${PREPARE}/retrieval.jsonl" \
  --output "${ROOT}/summary.json" | tee "${ROOT}/summary.txt"
event "V5.73 full LoCoMo validation complete"
