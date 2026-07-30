#!/usr/bin/env bash
# LongMemEval smoke-test runner for GraphMem with local vLLM (qwen3-30b + embedding).
#
# Usage:
#   scripts/smoke_longmemeval.sh embed        # start embedding service (GPU0, :8002)
#   scripts/smoke_longmemeval.sh llm          # start main LLM service  (GPU0-3 TP4, :8001)
#   scripts/smoke_longmemeval.sh up           # start embed + llm, wait until both ready
#   scripts/smoke_longmemeval.sh check        # curl both /v1/models
#   scripts/smoke_longmemeval.sh thinking     # test DeepSeek-style "thinking" param compat
#   scripts/smoke_longmemeval.sh run          # run the token demo (build + retrieve + answer)
#   scripts/smoke_longmemeval.sh status       # show pids / ports / GPU memory
#   scripts/smoke_longmemeval.sh stop         # stop both services
#   scripts/smoke_longmemeval.sh logs embed   # tail a service log (embed|llm)
#
# When something fails, just tweak the CONFIG block below and re-run the relevant step.
set -uo pipefail

# ============================ CONFIG (edit here) ============================
VLLM_ENV="graphmem"           # conda env that has vLLM
RUN_ENV="graphmem"            # conda env that runs scripts/run_token_demo.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

DATA="${DATA:-${REPO}/data/longmemeval_s_cleaned.json}"

# LLMLingua compression model cache (used only inside cmd_run).
RUN_HF_HOME="${RUN_HF_HOME:-${HF_HOME}}"

# ---- embedding service ----
EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"
EMBED_PORT="8002"
EMBED_GPUS="0"
EMBED_UTIL="0.15"            # 0.08 时 KV cache 仅 0.28GiB < 8192 所需的 0.88GiB，会启动失败
EMBED_MAXLEN="16384"

# ---- main LLM service (answer / summary; sits in the "DeepSeek" slot) ----
LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
LLM_PORT="8001"
LLM_GPUS="0,1,2,3"
LLM_TP="4"
LLM_UTIL="0.8" # lower to 0.75 if GPU0 OOMs
LLM_MAXLEN="32768"            # lower to 16384 if GPU0 OOMs
LLM_MAX_NUM_SEQS="1"         # 默认 256，sampler 预热会 OOM；冒烟测试用 1 即可，跑通后可调大

# ---- token demo run ----
MAX_Q="3"                     # bump to 10 once the 3-question run passes
QTYPE="multi-session"
OUTDIR="${REPO}/runs/smoke3"
VARIANTS="direct_session_k16_compact_no_compress direct_session_k16_compact_graphmem"
QWORKERS="1"
# ---- optional: LLM-assisted root edges (default off, easy remove) ----
ENABLE_LLM_ROOT_EDGES="${ENABLE_LLM_ROOT_EDGES:-0}"
LLM_ROOT_EDGE_MAX_TOKENS="${LLM_ROOT_EDGE_MAX_TOKENS:-256}"
LLM_ROOT_EDGE_NEIGHBORS_PER_REL="${LLM_ROOT_EDGE_NEIGHBORS_PER_REL:-2}"
LLM_ROOT_EDGE_MIN_SHARED="${LLM_ROOT_EDGE_MIN_SHARED:-1}"
LLM_ROOT_EDGE_ANCHOR_LIMIT="${LLM_ROOT_EDGE_ANCHOR_LIMIT:-8}"
# ===========================================================================

LOGDIR="${REPO}/runs/services"
mkdir -p "${LOGDIR}"

CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"

_activate() {
  # conda 的 activate/deactivate 钩子（如 gcc 的）不兼容 `set -u`，
  # 会因 _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED 等未定义变量导致子 shell 直接退出，
  # 所以在切换环境前后临时关闭 nounset。
  set +u
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "$1"
  set -u
}

_pidfile() { echo "${LOGDIR}/$1.pid"; }
_logfile() { echo "${LOGDIR}/$1.log"; }

_is_running() {
  local f; f="$(_pidfile "$1")"
  [[ -s "${f}" ]] && kill -0 "$(cat "${f}")" 2>/dev/null
}

_wait_ready() {
  local name="$1" port="$2" deadline="${3:-600}" start
  start="$(date +%s)"
  echo "[wait] ${name} on :${port} (timeout ${deadline}s) ..."
  while true; do
    if curl -s -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[ready] ${name} on :${port}"
      return 0
    fi
    if ! _is_running "${name}"; then
      echo "[fail] ${name} exited early. Last log lines:" >&2
      tail -n 25 "$(_logfile "${name}")" >&2
      return 1
    fi
    if (( "$(date +%s)" - start > deadline )); then
      echo "[timeout] ${name} not ready within ${deadline}s; see $(_logfile "${name}")" >&2
      return 1
    fi
    sleep 5
  done
}

start_embed() {
  if _is_running embed; then echo "embed already running (pid $(cat "$(_pidfile embed)"))"; return 0; fi
  echo "[start] embedding ${EMBED_MODEL} on GPU ${EMBED_GPUS} :${EMBED_PORT}"
  ( _activate "${VLLM_ENV}"
    CUDA_VISIBLE_DEVICES="${EMBED_GPUS}" setsid vllm serve "${EMBED_MODEL}" \
      --served-model-name "${EMBED_MODEL}" \
      --runner pooling \
      --port "${EMBED_PORT}" --host 127.0.0.1 \
      --max-model-len "${EMBED_MAXLEN}" \
      --gpu-memory-utilization "${EMBED_UTIL}" \
      --trust-remote-code \
      >"$(_logfile embed)" 2>&1 < /dev/null &
    echo $! > "$(_pidfile embed)"
  )
  echo "  log: $(_logfile embed)"
}

start_llm() {
  if _is_running llm; then echo "llm already running (pid $(cat "$(_pidfile llm)"))"; return 0; fi
  echo "[start] LLM ${LLM_MODEL} on GPU ${LLM_GPUS} TP=${LLM_TP} :${LLM_PORT}"
  ( _activate "${VLLM_ENV}"
    CUDA_VISIBLE_DEVICES="${LLM_GPUS}" setsid vllm serve "${LLM_MODEL}" \
      --served-model-name "${LLM_MODEL}" \
      --port "${LLM_PORT}" --host 127.0.0.1 \
      --tensor-parallel-size "${LLM_TP}" \
      --max-model-len "${LLM_MAXLEN}" \
      --gpu-memory-utilization "${LLM_UTIL}" \
      --max-num-seqs "${LLM_MAX_NUM_SEQS}" \
      --trust-remote-code \
      >"$(_logfile llm)" 2>&1 < /dev/null &
    echo $! > "$(_pidfile llm)"
  )
  echo "  log: $(_logfile llm)"
}

cmd_up() {
  start_embed
  _wait_ready embed "${EMBED_PORT}" || return 1
  start_llm
  _wait_ready llm "${LLM_PORT}" 900 || return 1
  cmd_check
}

cmd_check() {
  echo "=== embedding :${EMBED_PORT} ==="; curl -s -m 5 "http://127.0.0.1:${EMBED_PORT}/v1/models" | head -c 300; echo
  echo "=== llm :${LLM_PORT} ==="; curl -s -m 5 "http://127.0.0.1:${LLM_PORT}/v1/models" | head -c 300; echo
}

cmd_thinking() {
  echo "=== testing DeepSeek-style 'thinking' param against ${LLM_MODEL} ==="
  curl -s -m 30 "http://127.0.0.1:${LLM_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${LLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"thinking\":{\"type\":\"disabled\"}}" \
    | head -c 600
  echo
  echo "--> If you see an error mentioning 'thinking'/'extra fields', clients.py needs a patch."
}

cmd_run() {
  echo "[run] ${MAX_Q} ${QTYPE} questions -> ${OUTDIR}"
  local llm_root_edge_args=()
  if [[ "${ENABLE_LLM_ROOT_EDGES}" == "1" ]]; then
    llm_root_edge_args=(
      --enable-llm-root-edges
      --llm-root-edge-max-tokens "${LLM_ROOT_EDGE_MAX_TOKENS}"
      --llm-root-edge-neighbors-per-relation "${LLM_ROOT_EDGE_NEIGHBORS_PER_REL}"
      --llm-root-edge-min-shared "${LLM_ROOT_EDGE_MIN_SHARED}"
      --llm-root-edge-anchor-limit "${LLM_ROOT_EDGE_ANCHOR_LIMIT}"
    )
  fi
  ( _activate "${RUN_ENV}"
    cd "${REPO}"
    # demo 需要联网下载 LLMLingua 压缩模型，缓存写到有权限的自有目录。
    export HF_HOME="${RUN_HF_HOME}"
    export HF_HUB_OFFLINE="0"
    mkdir -p "${RUN_HF_HOME}"
    export SGAO_API_KEY="dummy"
    export SGAO_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
    export SGAO_MODEL="${LLM_MODEL}"
    # 超长输入让 vLLM 截断到 embedding 的上下文上限，避免 400。
    export EMBEDDING_TRUNCATE_TOKENS="${EMBED_MAXLEN}"
    # 批小一点，降低 embed 在与 llm 抢 GPU 时的单请求超时概率。
    export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
    python scripts/run_token_demo.py \
      --data "${DATA}" \
      --question-type "${QTYPE}" \
      --max-questions "${MAX_Q}" \
      --output-dir "${OUTDIR}" \
      --variants ${VARIANTS} \
      --embedding-base-url "http://127.0.0.1:${EMBED_PORT}/v1" \
      --embedding-model "${EMBED_MODEL}" \
      --question-workers "${QWORKERS}" \
      "${llm_root_edge_args[@]}"
  )
}

cmd_status() {
  for n in embed llm; do
    if _is_running "${n}"; then echo "${n}: running pid=$(cat "$(_pidfile "${n}")")"; else echo "${n}: not running"; fi
  done
  echo "--- ports ---"
  curl -s -m 3 "http://127.0.0.1:${EMBED_PORT}/v1/models" >/dev/null 2>&1 && echo "embed :${EMBED_PORT} ready" || echo "embed :${EMBED_PORT} down"
  curl -s -m 3 "http://127.0.0.1:${LLM_PORT}/v1/models"  >/dev/null 2>&1 && echo "llm   :${LLM_PORT} ready" || echo "llm   :${LLM_PORT} down"
  echo "--- gpu ---"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
}

stop_one() {
  local n="$1" f; f="$(_pidfile "${n}")"
  if ! _is_running "${n}"; then echo "${n}: not running"; rm -f "${f}"; return 0; fi
  local pid; pid="$(cat "${f}")"
  echo "stopping ${n} (pid ${pid}) ..."
  # vLLM spawns child workers; kill the whole process group.
  kill -- -"${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "${pid}" 2>/dev/null || { rm -f "${f}"; echo "${n} stopped"; return 0; }; sleep 1; done
  kill -9 -- -"${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
  rm -f "${f}"; echo "${n} force-killed"
}

cmd_stop() { stop_one llm; stop_one embed; }

cmd_logs() { tail -n 60 -f "$(_logfile "${1:-embed}")"; }

case "${1:-}" in
  embed)    start_embed ;;
  llm)      start_llm ;;
  up)       cmd_up ;;
  check)    cmd_check ;;
  thinking) cmd_thinking ;;
  run)      cmd_run ;;
  status)   cmd_status ;;
  stop)     cmd_stop ;;
  logs)     cmd_logs "${2:-embed}" ;;
  *) sed -n '2,18p' "$0" ;;
esac
