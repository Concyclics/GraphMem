export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
/home/chenhan/miniconda3/envs/vllm/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --port 8004 \
  --gpu_memory_utilization 0.80 \
  --tensor-parallel-size 1 \
  --max_model_len 262144 \
  --host 0.0.0.0
