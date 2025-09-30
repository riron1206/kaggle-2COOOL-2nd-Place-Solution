#!/bin/bash

cd /home/user_00006_821839/workspace/k/2coool/work3
source .venv/bin/activate

python src/gpt-oss-120b_infer_vllm_sc_from_frames_csv.py 2>&1 | tee sbatch_outputs/run_gpt-oss-120b_infer_vllm_sc_from_frames_csv.log
