#!/bin/bash

mkdir -p ./results/merge_submit_2csv

python ../src/merge_submit_2csv.py \
  --sample ./results/run_Qwen3VL_multi_image_select_frames_from_csv_infer_vllm_v2/exp004_n_sample4.csv \
  --full   ./results/run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v2/exp005_n_sample4.csv \
  --out    ./results/merge_submit_2csv/exp004_and_exp005_n_sample4.csv

python ../src/merge_submit_2csv.py \
  --sample ./results/run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v4/exp005_n_sample4_12_11_6.csv \
  --full   ./results/run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v2/exp004_and_exp005_n_sample4.csv \
  --out    ./results/merge_submit_2csv/20250927_ens2.csv
