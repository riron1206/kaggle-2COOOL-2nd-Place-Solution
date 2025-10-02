#!/bin/bash

module purge
module load gcc/64/4.1.7a1 slurm/gc1/23.02.7 gc1/cuda/12.4 gc1/cudnn/9.4.0 gc1/nccl/2.21.5 gc1/hpcx/2.18.1

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/models/zai-org/GLM-4.5V \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 4 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --allowed-local-media-path / \
    --enforce-eager \
    --media-io-kwargs '{"video": {"num_frames": -1}}'
