#!/bin/bash

module load gc1/cuda/12.4

uv venv -p python3.10
source .venv/bin/activate

uv pip install -r requirements.txt

uv pip install -U pyarrow
uv pip install -U datasets
uv pip install -U gdown
uv pip install -U "qwen-vl-utils[decord]" decord av soundfile librosa
uv pip install open-clip-torch ultralytics
uv pip install opencv-contrib-python
uv pip install "transformers==4.56.0"
uv pip install "quantizers==1.2.2"
uv pip install python-dotenv
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
uv pip install transformers-v4.55.0-GLM-4.5V-preview
uv pip install pillow
