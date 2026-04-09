rm -rf /root/.cache/vllm/torch_compile_cache/*

# VLLM_TORCH_PROFILER_DIR=./vllm_profile \
# VLLM_LOG_LEVEL=debug \
# VLLM_USE_V1=1 \
# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
# VLLM_TORCH_PROFILER_DIR=./vllm_profile \
# VLLM_TORCH_PROFILER_WITH_STACK=1 \
# export LD_LIBRARY_PATH=/usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/lib:$LD_LIBRARY_PATH

VLLM_ENC_ENABLE=1 \
ASCEND_RT_VISIBLE_DEVICES=2,3 vllm serve /mnt/Qwen3-8B-W8A8 --served-model-name "Qwen3-8B-W8A8" \
    --max-model-len 4096 \
    --quantization ascend \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-num-seqs 8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --async-scheduling
