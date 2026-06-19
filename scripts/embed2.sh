vllm serve Qwen/Qwen3-Embedding-0.6B \
  --host 0.0.0.0 \
  --port 8003 \
  --trust-remote-code \
  --served-model-name Qwen/Qwen3-Embedding-0.6B \
  --hf-overrides '{"is_matryoshka": true}' \
  --gpu-memory-utilization 0.1
