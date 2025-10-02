#!/bin/bash

module purge
module load gcc/64/4.1.7a1 slurm/gc1/23.02.7 gc1/cuda/12.4 gc1/cudnn/9.4.0 gc1/nccl/2.21.5 gc1/hpcx/2.18.1

uv venv -p python3.10
source .venv/bin/activate

uv pip sync requirements.txt
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
