#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${GRAPHMEM_VLLM_STATE_DIR:-$ROOT_DIR/runs/vllm_services}"
VLLM_ENV="${GRAPHMEM_VLLM_ENV:-vllm}"
CONDA_BASE="${GRAPHMEM_CONDA_BASE:-$(conda info --base)}"

EMBED_MODEL="${GRAPHMEM_EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
SUMMARY_MODEL="${GRAPHMEM_SUMMARY_MODEL:-Qwen/Qwen3.5-2B}"
EMBED_PORT="${GRAPHMEM_EMBED_PORT:-8002}"
SUMMARY_PORT="${GRAPHMEM_SUMMARY_PORT:-8003}"
HOST="${GRAPHMEM_VLLM_HOST:-127.0.0.1}"

mkdir -p "$STATE_DIR"

usage() {
  cat <<EOF
Usage: $0 {start|stop|status}

Environment overrides:
  GRAPHMEM_VLLM_ENV         Conda env for vLLM, default: vllm
  GRAPHMEM_EMBED_MODEL      Embedding model, default: Qwen/Qwen3-Embedding-0.6B
  GRAPHMEM_SUMMARY_MODEL    Summary model, default: Qwen/Qwen3.5-2B
  GRAPHMEM_EMBED_PORT       Embedding API port, default: 8002
  GRAPHMEM_SUMMARY_PORT     Summary API port, default: 8003
  GRAPHMEM_VLLM_HOST        API host, default: 127.0.0.1
EOF
}

pid_file() {
  local name="$1"
  echo "$STATE_DIR/$name.pid"
}

log_file() {
  local name="$1"
  echo "$STATE_DIR/$name.log"
}

is_running() {
  local name="$1"
  local file
  file="$(pid_file "$name")"
  [[ -s "$file" ]] && kill -0 "$(cat "$file")" 2>/dev/null
}

port_ready() {
  local port="$1"
  python - "$HOST" "$port" <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

wait_ready() {
  local name="$1"
  local port="$2"
  local deadline="${3:-240}"
  local start
  start="$(date +%s)"
  while true; do
    if port_ready "$port"; then
      echo "$name ready on $HOST:$port"
      return 0
    fi
    if ! is_running "$name"; then
      echo "$name exited before becoming ready; see $(log_file "$name")" >&2
      return 1
    fi
    if (( "$(date +%s)" - start > deadline )); then
      echo "$name did not become ready within ${deadline}s; see $(log_file "$name")" >&2
      return 1
    fi
    sleep 3
  done
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 < 2000 && $3 < 20) print $1}'
}

pick_gpus() {
  mapfile -t gpus < <(free_gpus)
  if (( "${#gpus[@]}" < 2 )); then
    echo "Need at least 2 mostly-free GPUs for embedding + summarizer; found ${#gpus[@]}." >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2
    return 1
  fi
  echo "${gpus[0]} ${gpus[1]}"
}

start_service() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  shift 3
  if is_running "$name"; then
    echo "$name already running with pid $(cat "$(pid_file "$name")")"
    return 0
  fi
  echo "Starting $name on GPU $gpu, port $port"
  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" setsid bash -lc \
      "source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$VLLM_ENV' && exec python -m vllm.entrypoints.openai.api_server --host '$HOST' --port '$port' $(printf '%q ' "$@")" \
      >"$(log_file "$name")" 2>&1 </dev/null &
    echo $! >"$(pid_file "$name")"
  )
  wait_ready "$name" "$port"
}

start_all() {
  read -r embed_gpu summary_gpu < <(pick_gpus)
  start_service embedding "$embed_gpu" "$EMBED_PORT" \
    --model "$EMBED_MODEL" \
    --served-model-name "$EMBED_MODEL" \
    --runner pooling \
    --trust-remote-code
  start_service summarizer "$summary_gpu" "$SUMMARY_PORT" \
    --model "$SUMMARY_MODEL" \
    --served-model-name "$SUMMARY_MODEL" \
    --trust-remote-code \
    --max-model-len 32768
}

stop_service() {
  local name="$1"
  local file
  file="$(pid_file "$name")"
  if ! is_running "$name"; then
    rm -f "$file"
    echo "$name not running"
    return 0
  fi
  local pid
  pid="$(cat "$file")"
  echo "Stopping $name pid $pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$file"
      return 0
    fi
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$file"
}

stop_all() {
  stop_service embedding
  stop_service summarizer
}

status_one() {
  local name="$1"
  local port="$2"
  if is_running "$name"; then
    printf "%s running pid=%s " "$name" "$(cat "$(pid_file "$name")")"
    if port_ready "$port"; then
      echo "ready"
    else
      echo "not-ready"
    fi
  else
    echo "$name not running"
  fi
}

status_all() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  status_one embedding "$EMBED_PORT"
  status_one summarizer "$SUMMARY_PORT"
}

case "${1:-}" in
  start) start_all ;;
  stop) stop_all ;;
  status) status_all ;;
  *) usage; exit 2 ;;
esac
