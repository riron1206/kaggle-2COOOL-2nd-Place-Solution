#!/bin/bash

module load cuda/12.4

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "/data/models/Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 22002 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.70 \
  --trust-remote-code \
  --enforce-eager \
  --allowed-local-media-path /
