#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${STATE_DIR:-/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5/gate_a_services_20260804}"
VLLM_BIN="${VLLM_BIN:-/home/chenhan/miniconda3/envs/vllm/bin/vllm}"
HF_CACHE_ROOT="${HF_HOME:-/home/chenhan/.cache/huggingface}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-${HF_CACHE_ROOT}/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-${HF_CACHE_ROOT}/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8/snapshots/5a5a776300a41aaa681dd7ff0106608ef2bc90db}"
EMBED_SERVED_NAME="Qwen/Qwen3-Embedding-0.6B"
LLM_SERVED_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
EMBED_GPU="${EMBED_GPU:-1}"
LLM_GPU="${LLM_GPU:-auto}"
EMBED_PORT="${EMBED_PORT:-8001}"
LLM_PORT="${LLM_PORT:-8002}"
PORT_SEARCH_LIMIT="${PORT_SEARCH_LIMIT:-100}"
LLM_IDLE_MAX_MEMORY_MIB="${LLM_IDLE_MAX_MEMORY_MIB:-512}"
LLM_IDLE_MAX_UTILIZATION="${LLM_IDLE_MAX_UTILIZATION:-5}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-65536}"
LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS:-384}"

mkdir -p "${STATE_DIR}"

ready() {
  curl -fsS --max-time 3 "http://127.0.0.1:$1/v1/models" >/dev/null
}

validate_port_settings() {
  local name value
  for name in EMBED_PORT LLM_PORT; do
    value="${!name}"
    [[ "${value}" =~ ^[0-9]+$ ]] && (( value >= 1 && value <= 65535 )) || {
      echo "${name} must be an integer between 1 and 65535; got: ${value}" >&2
      return 1
    }
  done
  [[ "${PORT_SEARCH_LIMIT}" =~ ^[0-9]+$ ]] || {
    echo "PORT_SEARCH_LIMIT must be a non-negative integer; got: ${PORT_SEARCH_LIMIT}" >&2
    return 1
  }
}

port_is_available() {
  /usr/bin/env python3 - "$1" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

pick_available_port() {
  local requested="$1" label="$2" candidate offset
  for ((offset = 0; offset <= PORT_SEARCH_LIMIT; offset++)); do
    candidate=$((requested + offset))
    (( candidate <= 65535 )) || break
    if port_is_available "${candidate}"; then
      if (( candidate != requested )); then
        echo "${label} port ${requested} is occupied; using ${candidate}" >&2
      fi
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "no free ${label} port found from ${requested} through $(( requested + PORT_SEARCH_LIMIT > 65535 ? 65535 : requested + PORT_SEARCH_LIMIT ))" >&2
  return 1
}

read_service_port() {
  local state_file="$1" fallback="$2" value
  if [[ -r "${state_file}" ]]; then
    value="$(<"${state_file}")"
    if [[ "${value}" =~ ^[0-9]+$ ]] && (( value >= 1 && value <= 65535 )); then
      printf '%s\n' "${value}"
      return 0
    fi
  fi
  printf '%s\n' "${fallback}"
}

write_service_port() {
  printf '%s\n' "$2" >"$1"
}

pick_idle_llm_gpu() {
  local gpu_snapshot busy_snapshot
  local index uuid memory_used memory_total utilization
  local best_gpu="" best_memory="" best_utilization=""
  local -A busy_uuids=()

  [[ "${LLM_IDLE_MAX_MEMORY_MIB}" =~ ^[0-9]+$ ]] || {
    echo "LLM_IDLE_MAX_MEMORY_MIB must be a non-negative integer" >&2
    return 1
  }
  [[ "${LLM_IDLE_MAX_UTILIZATION}" =~ ^[0-9]+$ ]] || {
    echo "LLM_IDLE_MAX_UTILIZATION must be a non-negative integer" >&2
    return 1
  }
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is required for automatic LLM GPU selection" >&2
    return 1
  }
  gpu_snapshot="$(nvidia-smi \
    --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits)" || {
    echo "failed to query GPU state with nvidia-smi" >&2
    return 1
  }
  busy_snapshot="$(nvidia-smi --query-compute-apps=gpu_uuid \
    --format=csv,noheader,nounits)" || {
    echo "failed to query active GPU compute processes" >&2
    return 1
  }

  while IFS= read -r uuid; do
    uuid="${uuid//[[:space:]]/}"
    [[ -n "${uuid}" ]] && busy_uuids["${uuid}"]=1
  done <<<"${busy_snapshot}"

  while IFS=',' read -r index uuid memory_used memory_total utilization; do
    index="${index//[[:space:]]/}"
    uuid="${uuid//[[:space:]]/}"
    memory_used="${memory_used//[[:space:]]/}"
    memory_total="${memory_total//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    [[ "${index}" =~ ^[0-9]+$ && "${memory_used}" =~ ^[0-9]+$ && \
       "${utilization}" =~ ^[0-9]+$ ]] || continue
    [[ -z "${busy_uuids["${uuid}"]:-}" ]] || continue
    (( memory_used <= LLM_IDLE_MAX_MEMORY_MIB )) || continue
    (( utilization <= LLM_IDLE_MAX_UTILIZATION )) || continue
    if [[ -z "${best_gpu}" ]] || (( memory_used < best_memory )) || \
       (( memory_used == best_memory && utilization < best_utilization )) || \
       (( memory_used == best_memory && utilization == best_utilization && index < best_gpu )); then
      best_gpu="${index}"
      best_memory="${memory_used}"
      best_utilization="${utilization}"
    fi
  done <<<"${gpu_snapshot}"

  if [[ -z "${best_gpu}" ]]; then
    echo "no idle GPU found for the LLM (requires no compute process, memory.used <= ${LLM_IDLE_MAX_MEMORY_MIB} MiB, utilization <= ${LLM_IDLE_MAX_UTILIZATION}%)" >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits >&2 || true
    return 1
  fi
  printf '%s\n' "${best_gpu}"
}

start_one() {
  local session="$1" gpu="$2" logfile="$3"
  shift 3
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "${session} already exists" >&2
    return 1
  fi
  tmux new-session -d -s "${session}" \
    "cd '${REPO}' && exec env CUDA_VISIBLE_DEVICES='${gpu}' HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel '${VLLM_BIN}' serve $* >'${logfile}' 2>&1"
}

run_embedding_supervisor() {
  while true; do
    printf '%s embedding supervisor starting vLLM\n' "$(date -Is)" >>"${STATE_DIR}/embedding.log"
    set +e
    env CUDA_VISIBLE_DEVICES="${EMBED_GPU}" HF_HUB_OFFLINE=1 \
      VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel \
      "${VLLM_BIN}" serve "${EMBED_MODEL_PATH}" \
      --served-model-name "${EMBED_SERVED_NAME}" --runner pooling \
      --host 127.0.0.1 --port "${EMBED_PORT}" --max-model-len 32768 \
      --gpu-memory-utilization 0.10 --trust-remote-code \
      >>"${STATE_DIR}/embedding.log" 2>&1
    local exit_code=$?
    set -e
    printf '%s embedding vLLM exited code=%s; restarting in 5s\n' \
      "$(date -Is)" "${exit_code}" >>"${STATE_DIR}/embedding.log"
    sleep 5
  done
}

run_llm_supervisor() {
  while true; do
    printf '%s llm supervisor starting vLLM on physical GPU %s\n' \
      "$(date -Is)" "${LLM_GPU}" >>"${STATE_DIR}/llm.log"
    set +e
    env CUDA_VISIBLE_DEVICES="${LLM_GPU}" HF_HUB_OFFLINE=1 \
      VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
      VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel \
      "${VLLM_BIN}" serve "${LLM_MODEL_PATH}" \
      --served-model-name "${LLM_SERVED_NAME}" --host 127.0.0.1 --port "${LLM_PORT}" \
      --tensor-parallel-size 1 --max-model-len "${LLM_MAX_MODEL_LEN}" \
      --gpu-memory-utilization 0.88 --max-num-seqs "${LLM_MAX_NUM_SEQS}" \
      --max-num-batched-tokens 65536 --trust-remote-code \
      >>"${STATE_DIR}/llm.log" 2>&1
    local exit_code=$?
    set -e
    printf '%s llm vLLM exited code=%s; restarting in 5s\n' \
      "$(date -Is)" "${exit_code}" >>"${STATE_DIR}/llm.log"
    sleep 5
  done
}

wait_one() {
  local session="$1" port="$2" logfile="$3" attempt
  for attempt in $(seq 1 120); do
    if ready "${port}"; then
      echo "${session} ready on ${port}"
      return 0
    fi
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      tail -n 80 "${logfile}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "${session} did not become ready; see ${logfile}" >&2
  return 1
}

start_services() {
  validate_port_settings
  [[ -x "${VLLM_BIN}" ]] || { echo "missing vllm: ${VLLM_BIN}" >&2; return 1; }
  [[ -d "${EMBED_MODEL_PATH}" ]] || { echo "missing embedding snapshot" >&2; return 1; }
  [[ -d "${LLM_MODEL_PATH}" ]] || { echo "missing LLM snapshot" >&2; return 1; }
  start_embedding
  start_heartbeat
  start_llm
  start_llm_heartbeat
}

start_embedding() {
  validate_port_settings
  if tmux has-session -t graphmem-v5-embed 2>/dev/null; then
    EMBED_PORT="$(read_service_port "${STATE_DIR}/embedding.port" "${EMBED_PORT}")"
    if ready "${EMBED_PORT}"; then
      echo "graphmem-v5-embed already ready on ${EMBED_PORT}"
      return 0
    fi
    wait_one graphmem-v5-embed "${EMBED_PORT}" "${STATE_DIR}/embedding.log"
    return
  fi
  EMBED_PORT="$(pick_available_port "${EMBED_PORT}" embedding)" || return 1
  write_service_port "${STATE_DIR}/embedding.port" "${EMBED_PORT}"
  tmux new-session -d -s graphmem-v5-embed \
    "cd '${REPO}' && exec env HF_HOME='${HF_CACHE_ROOT}' EMBED_GPU='${EMBED_GPU}' EMBED_PORT='${EMBED_PORT}' STATE_DIR='${STATE_DIR}' VLLM_BIN='${VLLM_BIN}' EMBED_MODEL_PATH='${EMBED_MODEL_PATH}' bash scripts/v5_gate_a_model_services.sh supervise-embed"
  wait_one graphmem-v5-embed "${EMBED_PORT}" "${STATE_DIR}/embedding.log"
}

start_heartbeat() {
  EMBED_PORT="$(read_service_port "${STATE_DIR}/embedding.port" "${EMBED_PORT}")"
  ready "${EMBED_PORT}" || { echo "embedding service is not ready on ${EMBED_PORT}" >&2; return 1; }
  if tmux has-session -t graphmem-v5-embed-heartbeat 2>/dev/null; then
    echo "graphmem-v5-embed-heartbeat already running"
    return 0
  fi
  tmux new-session -d -s graphmem-v5-embed-heartbeat \
    "cd '${REPO}' && exec /home/chenhan/miniconda3/envs/agent/bin/python scripts/v5_embedding_heartbeat.py --url 'http://127.0.0.1:${EMBED_PORT}/v1/embeddings' --batch-size 512 --batch-jitter 64 --text-words 256 --interval-sec 0.2 --jitter-sec 0.6 --log '${STATE_DIR}/embedding_heartbeat.jsonl'"
}

start_llm() {
  local selected_gpu
  validate_port_settings
  [[ -d "${LLM_MODEL_PATH}" ]] || { echo "missing LLM snapshot: ${LLM_MODEL_PATH}" >&2; return 1; }
  if tmux has-session -t graphmem-v5-llm 2>/dev/null; then
    LLM_PORT="$(read_service_port "${STATE_DIR}/llm.port" "${LLM_PORT}")"
    if ready "${LLM_PORT}"; then
      echo "graphmem-v5-llm already ready on ${LLM_PORT}"
      return 0
    fi
    wait_one graphmem-v5-llm "${LLM_PORT}" "${STATE_DIR}/llm.log"
    return
  fi
  LLM_PORT="$(pick_available_port "${LLM_PORT}" LLM)" || return 1
  if [[ "${LLM_GPU}" == "auto" ]]; then
    selected_gpu="$(pick_idle_llm_gpu)" || return 1
  else
    selected_gpu="${LLM_GPU}"
  fi
  [[ "${selected_gpu}" =~ ^[0-9]+$ ]] || {
    echo "LLM_GPU must be 'auto' or one physical GPU index; got: ${selected_gpu}" >&2
    return 1
  }
  echo "Selected physical GPU ${selected_gpu} and port ${LLM_PORT} for the single-GPU LLM"
  write_service_port "${STATE_DIR}/llm.port" "${LLM_PORT}"
  tmux new-session -d -s graphmem-v5-llm \
    "cd '${REPO}' && exec env HF_HOME='${HF_CACHE_ROOT}' LLM_GPU='${selected_gpu}' LLM_PORT='${LLM_PORT}' STATE_DIR='${STATE_DIR}' VLLM_BIN='${VLLM_BIN}' LLM_MODEL_PATH='${LLM_MODEL_PATH}' LLM_MAX_MODEL_LEN='${LLM_MAX_MODEL_LEN}' LLM_MAX_NUM_SEQS='${LLM_MAX_NUM_SEQS}' bash scripts/v5_gate_a_model_services.sh supervise-llm"
  wait_one graphmem-v5-llm "${LLM_PORT}" "${STATE_DIR}/llm.log"
}

start_llm_heartbeat() {
  LLM_PORT="$(read_service_port "${STATE_DIR}/llm.port" "${LLM_PORT}")"
  ready "${LLM_PORT}" || { echo "LLM service is not ready on ${LLM_PORT}" >&2; return 1; }
  if tmux has-session -t graphmem-v5-llm-heartbeat 2>/dev/null; then
    echo "graphmem-v5-llm-heartbeat already running"
    return 0
  fi
  tmux new-session -d -s graphmem-v5-llm-heartbeat \
    "cd '${REPO}' && exec /home/chenhan/miniconda3/envs/agent/bin/python scripts/v5_llm_heartbeat.py --base-url 'http://127.0.0.1:${LLM_PORT}/v1' --log '${STATE_DIR}/llm_heartbeat.jsonl'"
}

stop_services() {
  local session
  for session in graphmem-v5-embed-heartbeat graphmem-v5-llm-heartbeat graphmem-v5-embed graphmem-v5-llm; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      tmux kill-session -t "${session}"
    fi
  done
}

status_services() {
  local embed_port llm_port
  embed_port="$(read_service_port "${STATE_DIR}/embedding.port" "${EMBED_PORT}")"
  llm_port="$(read_service_port "${STATE_DIR}/llm.port" "${LLM_PORT}")"
  tmux list-sessions 2>/dev/null || true
  echo "embedding endpoint: http://127.0.0.1:${embed_port}/v1"
  curl -fsS --max-time 3 "http://127.0.0.1:${embed_port}/v1/models" || true
  echo "LLM endpoint: http://127.0.0.1:${llm_port}/v1"
  curl -fsS --max-time 3 "http://127.0.0.1:${llm_port}/v1/models" || true
}

case "${1:-}" in
  start) start_services ;;
  start-embed) start_embedding ;;
  start-llm) start_llm ;;
  select-llm-gpu) pick_idle_llm_gpu ;;
  start-llm-heartbeat) start_llm_heartbeat ;;
  start-heartbeat) start_heartbeat ;;
  stop) stop_services ;;
  status) status_services ;;
  supervise-embed) run_embedding_supervisor ;;
  supervise-llm) run_llm_supervisor ;;
  *) echo "usage: $0 {start|start-embed|start-llm|select-llm-gpu|start-heartbeat|start-llm-heartbeat|stop|status}" >&2; exit 2 ;;
esac
