
#!/bin/bash

module load cuda/12.8.1

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 vllm serve /data/models/openai/gpt-oss-120b \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2
