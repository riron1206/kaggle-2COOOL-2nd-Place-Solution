#!/bin/bash

module load cuda/12.4

source .venv/bin/activate

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_FLASHINFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen/Qwen3-VL-235B-A22B-Thinking \
  --served-model-name /data/models/Qwen/Qwen3-VL-235B-A22B-Thinking \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --host 0.0.0.0 \
  --port 22002 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager \
  --allowed-local-media-path / \
  --quantization fp8 \
  --distributed-executor-backend mp
