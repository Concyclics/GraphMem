export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

CUDA_VISIBLE_DEVICES=2,3 vllm serve google/gemma-4-12B-it --port 8001 --gpu_memory_utilization 0.9 --tensor-parallel-size 2 --max_model_len 262144 --host 0.0.0.0
