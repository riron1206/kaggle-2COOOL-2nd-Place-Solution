#!/bin/bash

module load cuda/12.4

uv venv -p python3.10
source .venv/bin/activate

# https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct
uv pip install git+https://github.com/huggingface/transformers
# https://github.com/QwenLM/Qwen3-VL
uv pip install accelerate
uv pip install qwen-vl-utils==0.0.14
uv pip install 'vllm>=0.10.2'
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
uv pip install jupyterlab papermill ipywidgets ipynbname ipyplot black isort jupyterlab_code_formatter
uv pip install pandas
