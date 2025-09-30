#!/bin/bash

module load cuda/12.4

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/models/zai-org/GLM-4.5V \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 4 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --allowed-local-media-path / \
    --media-io-kwargs '{"video": {"num_frames": -1}}'
