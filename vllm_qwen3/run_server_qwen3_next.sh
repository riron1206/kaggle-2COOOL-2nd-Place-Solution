#!/bin/bash

module purge
module load gcc/64/4.1.7a1 slurm/gc1/23.02.7 gc1/cuda/12.4 gc1/cudnn/9.4.0 gc1/nccl/2.21.5 gc1/hpcx/2.18.1

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
