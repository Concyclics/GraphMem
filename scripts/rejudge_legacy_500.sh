#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_DIR="${LEGACY_DIR:-/mnt/ssd1/pzx/share/graphmem/full_s_500_gpt/direct_session_k16_compact_graphmem}"
OUT="${OUT:-/mnt/ssd1/pzx/share/graphmem/mem0_rejudge_legacy_500}"
"${REPO}/.venv/bin/python" "${REPO}/scripts/check_v2_services.py"
"${REPO}/.venv/bin/python" "${REPO}/scripts/evaluate_mem0_judge.py" \
  --answers "${LEGACY_DIR}/answers.jsonl" --metadata-jsonl "${LEGACY_DIR}/auto_eval.jsonl" \
  --output-dir "${OUT}" --model gpt-5.4-mini --base-url https://sub2api.sgao.me/v1/ --workers 16 --resume
"${REPO}/.venv/bin/python" "${REPO}/scripts/build_v2_tuning_split.py" \
  --answers "${LEGACY_DIR}/answers.jsonl" --judgments "${OUT}/auto_eval.jsonl" --output-dir "${OUT}/tuning_split"
