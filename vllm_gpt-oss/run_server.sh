
#!/bin/bash

module load cuda/12.8.1 cudnn/9.10.2.21_cuda12 nccl/2.26.2-1-cuda-12.8.1 gdrcopy/2.4.1-cuda-12.8.1 hpcx-debug/v2.18.1-cuda12 hpcx-mt/v2.18.1-cuda12 hpcx-prof/v2.18.1-cuda12 hpcx/v2.18.1-cuda12

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 vllm serve /data/models/openai/gpt-oss-120b \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2
