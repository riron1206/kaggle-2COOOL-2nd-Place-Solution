#!/bin/bash

API_BASE="http://localhost:22002/v1"
MODEL="/data/models/Qwen/Qwen3-VL-235B-A22B-Thinking"

GPTOSS_CSV="../../003_frame_detection/results/gpt-oss-120b_infer_vllm_sc_from_frames_csv.csv"

OUTPUT_DIR="./results/run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v3"
mkdir -p "$OUTPUT_DIR"

PRE_FRAMES=6
POST_FRAMES=5
STEP_FRAMES=10

TEMPERATURE=0.6
TOP_P=0.95
TOP_K=-1
REPETITION_PENALTY=1.0

run_select_and_infer() {
  local CSV="$1"
  local IMAGES_DIR="$2"
  local PRE="${3:-$PRE_FRAMES}"
  local POST="${4:-$POST_FRAMES}"
  local TAG="$(basename "$IMAGES_DIR")"
  local SKIP_EXIST_ARG="${5:-$SKIP_IF_EXISTS}"

  local OUT_DIR=$OUTPUT_DIR/"$TAG"
  mkdir -p "$OUT_DIR"
  local FRAMES_FILE="$OUT_DIR/frames.txt"
  local OUTPUT_JSON="$OUT_DIR/ans.json"
  local OUTPUT_JSON_TMP="$OUT_DIR/tmp.json"

  if [ "${SKIP_EXIST_ARG:-0}" != "0" ] && [ -f "$OUTPUT_JSON" ]; then
    echo "[Info] Skip (exists): $OUTPUT_JSON"
    echo "====================================================="
    echo ""
    return 0
  fi

  echo "[Info] CSV        : $CSV"
  echo "[Info] IMAGES_DIR : $IMAGES_DIR"
  echo "[Info] pre/post   : $PRE / $POST"

  python ../src/select_frames_from_gpt-oss_csv.py \
    --csv "$CSV" \
    --images_dir "$IMAGES_DIR" \
    --pre_frames "$PRE" \
    --post_frames "$POST" \
    --frame_step "$STEP_FRAMES" \
    --out "$FRAMES_FILE"

  local NUM
  NUM=$(wc -l < "$FRAMES_FILE" | tr -d '[:space:]')
  echo "[Info] Selected frame count: $NUM"
  if [ "$NUM" -eq 0 ]; then
    echo "[Error] Extracted result is empty" >&2
    return 1
  fi

  local -a IMAGES
  mapfile -t IMAGES < "$FRAMES_FILE"

  python src/glm45v_multi_image_infer_vllm_sc.py \
    --images "${IMAGES[@]}" \
    --api_base "$API_BASE" \
    --model "$MODEL" \
    --output_json "$OUTPUT_JSON" \
    --tmp_json "$OUTPUT_JSON_TMP" \
    --json_mode \
    --sc_at all --n_samples 4 \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --repetition_penalty "$REPETITION_PENALTY"

  echo "[Info] Saved JSON: $(readlink -f "$OUTPUT_JSON")"
  echo "====================================================="
  echo ""
}


IMAGES_ROOT="../../001_video2frames/gdrive_png/videos"
SKIP_IF_EXISTS=1

for VID_ID in $(seq 0 660); do
  IMG_DIR="$IMAGES_ROOT/$VID_ID"
  if [ -f "$GPTOSS_CSV" ] && [ -d "$IMG_DIR" ]; then
    run_select_and_infer "$GPTOSS_CSV" "$IMG_DIR" "$PRE_FRAMES" "$POST_FRAMES" "$SKIP_IF_EXISTS"
  else
    echo "[Warn] スキップ: CSV または 画像DIR が無い: id=$VID_ID (CSV=$GPTOSS_CSV, IMG_DIR=$IMG_DIR)" >&2
  fi
done

# subのcsv作成
python ../src/aggregate_ans_jsons_to_csv.py \
  --root "$OUTPUT_DIR/**/" \
  --output_csv $OUTPUT_DIR/exp005_n_sample4_6_5_10.csv
