#!/bin/bash

mkdir -p ./results/merge_submit_2csv

python ../src/merge_submit_2csv.py \
  --sample ./results/run_glm45v_multi_image_select_frames_from_csv_infer_vllm_v2/exp001_n_sample.csv \
  --full   ./results/run_glm45v_multi_image_select_frames_from_gptoss_csv_infer_vllm_v2/exp002_n_sample4_step6.csv \
  --out    ./results/merge_submit_2csv/exp001_and_exp002_n_sample4_step_merged.csv
