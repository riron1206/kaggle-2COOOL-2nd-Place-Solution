#!/bin/bash

mkdir -p logs

python src/gpt-oss-120b_infer_vllm_sc_from_frames_csv.py 2>&1 | tee logs/run_gpt-oss-120b_infer_vllm_sc_from_frames_csv.log
