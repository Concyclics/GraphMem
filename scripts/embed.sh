#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=3
#"${CUDA_VISIBLE_DEVICES:-1}"
HF_HOME="${HF_HOME:-/ssd1/chenhan/huggingface}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}"
MODEL_PATH="${MODEL_PATH:-}"
ALLOW_REMOTE_MODEL="${ALLOW_REMOTE_MODEL:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.1}"
PORT="${PORT:-8003}"
HOST="${HOST:-0.0.0.0}"
HF_OVERRIDES="${HF_OVERRIDES:-{\"is_matryoshka\": true}}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
VLLM_BIN="${VLLM_BIN:-/home/chenhan/miniconda3/envs/vllm/bin/vllm}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_CACHE_DIR="$HF_HOME/hub/models--${MODEL_ID//\//--}"
  if [[ -f "$MODEL_CACHE_DIR/refs/main" ]]; then
    SNAPSHOT_REV="$(<"$MODEL_CACHE_DIR/refs/main")"
    SNAPSHOT_PATH="$MODEL_CACHE_DIR/snapshots/$SNAPSHOT_REV"
    if [[ -d "$SNAPSHOT_PATH" ]]; then
      MODEL_PATH="$SNAPSHOT_PATH"
    fi
  fi
fi

if [[ -n "$MODEL_PATH" ]]; then
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "MODEL_PATH does not exist or is not a directory: $MODEL_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "MODEL_PATH does not look like a valid model snapshot: $MODEL_PATH" >&2
    exit 1
  fi
  MODEL_TARGET="$MODEL_PATH"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  echo "Using local model snapshot: $MODEL_TARGET" >&2
elif [[ "$ALLOW_REMOTE_MODEL" == "1" ]]; then
  MODEL_TARGET="$MODEL_ID"
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  fi
  echo "Using remote model id: $MODEL_TARGET" >&2
else
  echo "No local model snapshot found for $MODEL_ID" >&2
  echo "Checked HF cache under: $HF_HOME/hub/models--${MODEL_ID//\//--}" >&2
  echo "Set MODEL_PATH=/path/to/local/model or ALLOW_REMOTE_MODEL=1 to permit remote loading." >&2
  exit 1
fi

cmd=(
  "$VLLM_BIN"
  serve
  "$MODEL_TARGET"
  --gpu_memory_utilization
  "$GPU_MEMORY_UTILIZATION"
  --port
  "$PORT"
  --host
  "$HOST"
  --hf_overrides
  "$HF_OVERRIDES"
  --served-model-name
  "$SERVED_MODEL_NAME"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${VLLM_EXTRA_ARGS})
  cmd+=("${extra_args[@]}")
fi

exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
