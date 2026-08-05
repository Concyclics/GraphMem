#!/usr/bin/env bash
# Standard GraphMem evaluation workflow for a fixed subset.
#
# Pipeline:
#   1. start local vLLM services (embed + answer/summary LLM)
#   2. run build + retrieve + answer via run_token_demo.py
#   3. judge answers with evaluate_answers.py
#   4. run stage audit (retrieval vs reasoning bottleneck split)
#
# Usage:
#   scripts/run_subset_eval.sh
#   RUN_TAG=leaf_k18 scripts/run_subset_eval.sh
#   KEEP_UP=1 scripts/run_subset_eval.sh
#   SKIP_RUN=1 RUN_DIR=runs/subset50_baseline/... scripts/run_subset_eval.sh
#
# Defaults target the fixed 50-question balanced subset under data/.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
SMOKE="${SCRIPT_DIR}/smoke_longmemeval.sh"

RUN_ENV="${RUN_ENV:-graphmem}"
DATA="${DATA:-${REPO}/data/longmemeval_s_subset50_balanced.json}"
SUBSET_NAME="${SUBSET_NAME:-longmemeval_s_subset50_balanced}"
RUN_TAG="${RUN_TAG:-baseline}"
OUTDIR="${OUTDIR:-${REPO}/runs/subset50_${RUN_TAG}}"
VARIANT="${VARIANT:-direct_session_k16_compact_graphmem}"
VARIANT_DIR="${VARIANT_DIR:-${OUTDIR}/${VARIANT}}"
RUN_DIR="${RUN_DIR:-${VARIANT_DIR}}"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
LLM_PORT="${LLM_PORT:-8001}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_PORT="${EMBED_PORT:-8002}"
EMBED_TRUNCATE_TOKENS="${EMBED_TRUNCATE_TOKENS:-16384}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"
JUDGE_WORKERS="${JUDGE_WORKERS:-6}"

QWORKERS="${QWORKERS:-2}"
LEAF_TOP_K="${LEAF_TOP_K:-14}"
GLOBAL_LEAF_TOP_K="${GLOBAL_LEAF_TOP_K:-24}"
PER_SESSION_LEAF_K="${PER_SESSION_LEAF_K:-2}"
QA_CONTEXT_TOKEN_BUDGET="${QA_CONTEXT_TOKEN_BUDGET:-10000}"
QA_MAX_TOKENS="${QA_MAX_TOKENS:-1024}"
RUN_HF_HOME="${RUN_HF_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}}"
LLMLINGUA_DEVICE="${LLMLINGUA_DEVICE:-cpu}"

SKIP_RUN="${SKIP_RUN:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
KEEP_UP="${KEEP_UP:-0}"

CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
_activate() {
  set +u
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "$1"
  set -u
}

_require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[fatal] missing file: $1" >&2
    exit 1
  fi
}

_run_demo() {
  _activate "${RUN_ENV}"
  cd "${REPO}"
  export HF_HOME="${RUN_HF_HOME}"
  export HF_HUB_OFFLINE="0"
  mkdir -p "${RUN_HF_HOME}"
  export SGAO_API_KEY="dummy"
  export SGAO_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
  export SGAO_MODEL="${LLM_MODEL}"
  export EMBEDDING_TRUNCATE_TOKENS="${EMBED_TRUNCATE_TOKENS}"
  export EMBEDDING_BATCH_SIZE="${EMBED_BATCH_SIZE}"
  python scripts/run_token_demo.py \
    --data "${DATA}" \
    --question-type all \
    --max-questions 100000 \
    --output-dir "${OUTDIR}" \
    --variants "${VARIANT}" \
    --embedding-base-url "http://127.0.0.1:${EMBED_PORT}/v1" \
    --embedding-model "${EMBED_MODEL}" \
    --question-workers "${QWORKERS}" \
    --leaf-top-k "${LEAF_TOP_K}" \
    --global-leaf-top-k "${GLOBAL_LEAF_TOP_K}" \
    --per-session-leaf-k "${PER_SESSION_LEAF_K}" \
    --qa-context-token-budget "${QA_CONTEXT_TOKEN_BUDGET}" \
    --qa-max-tokens "${QA_MAX_TOKENS}" \
    --llmlingua-device-map "${LLMLINGUA_DEVICE}" \
    --resume
}

_run_judge() {
  _require_file "${RUN_DIR}/answers.jsonl"
  if [[ -z "${SGAO_API_KEY:-}" || "${SGAO_API_KEY}" == "dummy" ]]; then
    echo "[warn] SGAO_API_KEY is not set for judge; export a real key before judging." >&2
  fi
  _activate "${RUN_ENV}"
  cd "${REPO}"
  export SGAO_MODEL="${JUDGE_MODEL}"
  python scripts/evaluate_answers.py \
    --answers "${RUN_DIR}/answers.jsonl" \
    --data "${DATA}" \
    --output-jsonl "${RUN_DIR}/auto_eval.jsonl" \
    --output-md "${RUN_DIR}/auto_eval.md" \
    --model "${JUDGE_MODEL}" \
    --workers "${JUDGE_WORKERS}"
}

_run_audit() {
  _require_file "${RUN_DIR}/answers.jsonl"
  _activate "${RUN_ENV}"
  cd "${REPO}"
  python scripts/analyze_stage_audit.py \
    --run-dir "${RUN_DIR}" \
    --data "${DATA}" \
    --output-dir "${OUTDIR}/analysis"
}

echo "=== GraphMem subset eval ==="
echo "subset: ${SUBSET_NAME}"
echo "data:   ${DATA}"
echo "run:    ${RUN_DIR}"
echo "tag:    ${RUN_TAG}"
echo

_require_file "${DATA}"

if [[ "${SKIP_RUN}" != "1" ]]; then
  echo "[1/4] start services"
  bash "${SMOKE}" up || { echo "[fatal] services not ready" >&2; exit 1; }
  echo "[2/4] run token demo"
  _run_demo
  status=$?
  if [[ "${KEEP_UP}" != "1" ]]; then
    bash "${SMOKE}" stop
  fi
  if [[ ${status} -ne 0 ]]; then
    exit ${status}
  fi
else
  echo "[1/4] skip run (SKIP_RUN=1)"
  echo "[2/4] skip run"
fi

if [[ "${SKIP_JUDGE}" != "1" ]]; then
  echo "[3/4] judge answers"
  _run_judge
else
  echo "[3/4] skip judge"
fi

if [[ "${SKIP_AUDIT}" != "1" ]]; then
  echo "[4/4] stage audit"
  _run_audit
else
  echo "[4/4] skip audit"
fi

echo
echo "done."
echo "answers:    ${RUN_DIR}/answers.jsonl"
echo "judge:      ${RUN_DIR}/auto_eval.md"
echo "audit:      ${OUTDIR}/analysis/stage_summary.json"
echo "summary:    ${OUTDIR}/summary.md"
