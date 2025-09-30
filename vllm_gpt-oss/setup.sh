#!/bin/bash

module load cuda/12.8.1

uv venv -p python3.12
source .venv/bin/activate

uv pip install "transformers==4.56.0"
uv pip install "quantizers==1.2.2"
uv pip install tabulate

# https://huggingface.co/openai/gpt-oss-120b
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# https://github.com/vllm-project/vllm/issues/22290
uv pip uninstall triton
uv pip install "triton==3.4.0"
