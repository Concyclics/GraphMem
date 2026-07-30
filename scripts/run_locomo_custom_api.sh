#!/usr/bin/env bash
set -euo pipefail

# Run (loads repo-root .env automatically):
#   bash scripts/run_locomo_custom_api.sh
#
# Optional overrides:
#   MODEL=gpt-5.4-mini RUN_TAG=myrun MAX_QUESTIONS=1986 bash scripts/run_locomo_custom_api.sh
#   API_KEY_FILE=~/.secrets/sub2api.key bash scripts/run_locomo_custom_api.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Auto-load .env from repo root if present.
if [[ -f "${REPO}/.env" ]]; then
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      export "${key}=${value}"
    fi
  done < "${REPO}/.env"
fi

# --- Required: provide API key via env or file ---
API_KEY="${API_KEY:-${SGAO_API_KEY:-}}"
API_KEY_FILE="${API_KEY_FILE:-}"

# --- API/model config ---
BASE_URL="${BASE_URL:-${SGAO_BASE_URL:-https://sub2api.sgao.me/v1/}}"
MODEL="${MODEL:-${SGAO_MODEL:-gpt-5.4-mini}}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"

# --- Experiment config (same as notes-reader Locomo run) ---
# Use LOCOMO_DATA to override; ignore generic DATA from .env (often LongMemEval path).
DATA="${LOCOMO_DATA:-${REPO}/data/locomo10_graphmem.json}"
VARIANT="${VARIANT:-direct_session_k16_compact_graphmem}"
RUN_TAG="${RUN_TAG:-locomo10_subset50_custom_api_${MODEL//\//_}}"
OUTDIR="${OUTDIR:-${REPO}/runs/${RUN_TAG}}"
RUN_DIR="${OUTDIR}/${VARIANT}"
MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
QUESTION_TYPE="${QUESTION_TYPE:-all}"
QUESTION_WORKERS="${QUESTION_WORKERS:-1}"
JUDGE_WORKERS="${JUDGE_WORKERS:-6}"
ENABLE_ITERATIVE_LEAF_DENOISE="${ENABLE_ITERATIVE_LEAF_DENOISE:-1}"
ITERATIVE_DENOISE_MAX_ROUNDS="${ITERATIVE_DENOISE_MAX_ROUNDS:-3}"
ITERATIVE_DENOISE_MAX_KICK_PER_ROUND="${ITERATIVE_DENOISE_MAX_KICK_PER_ROUND:-5}"
ITERATIVE_DENOISE_MIN_RELEVANCE_RATIO="${ITERATIVE_DENOISE_MIN_RELEVANCE_RATIO:-0.35}"
ITERATIVE_DENOISE_PROTECT_TOP_K="${ITERATIVE_DENOISE_PROTECT_TOP_K:-3}"
RESUME="${RESUME:-1}"

if [[ -z "${API_KEY}" && -n "${API_KEY_FILE}" ]]; then
  if [[ ! -f "${API_KEY_FILE}" ]]; then
    echo "[fatal] API_KEY_FILE not found: ${API_KEY_FILE}" >&2
    exit 1
  fi
  API_KEY="$(tr -d '\r\n' < "${API_KEY_FILE}")"
fi

if [[ -z "${API_KEY}" ]]; then
  echo "[fatal] Missing API key. Set SGAO_API_KEY in .env, or pass API_KEY / API_KEY_FILE." >&2
  echo "example: cp .env.example .env && fill in SGAO_API_KEY" >&2
  exit 1
fi

if [[ ! -f "${DATA}" ]]; then
  echo "[fatal] data file not found: ${DATA}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"

echo "=== [1/3] API smoke test ==="
curl -sS "${BASE_URL%/}/chat/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "model": "${MODEL}",
  "reasoning_effort": "${REASONING_EFFORT}",
  "messages": [{"role": "user", "content": "Reply OK."}],
  "max_tokens": 8
}
EOF
)" >/tmp/locomo_custom_api_smoke.json
echo "Smoke response saved: /tmp/locomo_custom_api_smoke.json"

echo "=== [2/3] Run GraphMem on LoCoMo (${MAX_QUESTIONS} questions) ==="
cd "${REPO}"
export SGAO_API_KEY="${API_KEY}"
export SGAO_BASE_URL="${BASE_URL}"
export SGAO_MODEL="${MODEL}"

python scripts/run_token_demo.py \
  --data "${DATA}" \
  --question-type "${QUESTION_TYPE}" \
  --max-questions "${MAX_QUESTIONS}" \
  --question-workers "${QUESTION_WORKERS}" \
  --output-dir "${OUTDIR}" \
  --variants "${VARIANT}" \
  --llm-base-url "${BASE_URL}" \
  --llm-model "${MODEL}" \
  --enable-graph-first-retrieval \
  --enable-fusion-retrieval \
  --enable-answer-note-extraction \
  $( [[ "${ENABLE_ITERATIVE_LEAF_DENOISE}" == "1" ]] && echo "--enable-iterative-leaf-denoise" ) \
  --iterative-leaf-denoise-max-rounds "${ITERATIVE_DENOISE_MAX_ROUNDS}" \
  --iterative-leaf-denoise-max-kick-per-round "${ITERATIVE_DENOISE_MAX_KICK_PER_ROUND}" \
  --iterative-leaf-denoise-min-relevance-ratio "${ITERATIVE_DENOISE_MIN_RELEVANCE_RATIO}" \
  --iterative-leaf-denoise-protect-top-k "${ITERATIVE_DENOISE_PROTECT_TOP_K}" \
  $( [[ "${RESUME}" == "1" ]] && echo "--resume" )

echo "=== [3/3] Evaluate answers ==="
python scripts/evaluate_answers.py \
  --answers "${RUN_DIR}/answers.jsonl" \
  --data "${DATA}" \
  --output-jsonl "${RUN_DIR}/auto_eval.jsonl" \
  --output-md "${RUN_DIR}/auto_eval.md" \
  --model "${MODEL}" \
  --workers "${JUDGE_WORKERS}"

echo
echo "Done."
echo "Run dir:  ${RUN_DIR}"
echo "Summary:  ${OUTDIR}/summary.md"
echo "Eval md:  ${RUN_DIR}/auto_eval.md"
