#!/bin/bash

INPUT_BASE="../001_video2frames/mp4_vstack_png"
OUT_DIR="./results/glm45v_image_frames_infer_perception_vllm_server"
mkdir -p "$OUT_DIR"

LOG_FILE="logs/glm45v_image_frames_infer_perception_vllm_server.log"
mkdir -p "$(dirname "$LOG_FILE")"
rm -rf "$LOG_FILE"

for subdir in $(ls -d "$INPUT_BASE"/*/ | sort); do
  [[ -d "$subdir" ]] || continue
  subdir_path="${subdir%/}"
  echo "\n[RUN] dir=$subdir_path" >> "$LOG_FILE"

  python ./src/glm45v_image_frames_infer_perception_vllm_server.py \
    --dir "$subdir_path" \
    --out_dir "$OUT_DIR" \
    --skip_if_exists \
     --frame_stride 10 \
    2>&1 | tee -a "$LOG_FILE"

done
