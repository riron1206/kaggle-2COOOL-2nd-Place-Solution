#!/bin/bash

module load cuda/12.8.1 cudnn/9.10.2.21_cuda12 nccl/2.26.2-1-cuda-12.8.1 gdrcopy/2.4.1-cuda-12.8.1 hpcx-debug/v2.18.1-cuda12 hpcx-mt/v2.18.1-cuda12 hpcx-prof/v2.18.1-cuda12 hpcx/v2.18.1-cuda12 awscli-v2/2.18.12

uv venv -p python3.12
source .venv/bin/activate

uv pip sync requirements.txt

uv pip install --pre \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio -U

uv pip install --pre --no-deps \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  "vllm==0.10.1+gptoss"

uv pip install --pre \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  triton-kernels

# https://github.com/vllm-project/vllm/issues/22290
uv pip uninstall triton
uv pip install "triton==3.4.0"
