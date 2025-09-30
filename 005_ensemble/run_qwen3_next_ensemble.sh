#!/bin/bash

JSON_DIR=./output/jsons
START_IDX=0
END_IDX=660

all_jsons_exist() {
  local dir="$1"
  local s="$2"
  local e="$3"
  for i in $(seq "$s" "$e"); do
    if [ ! -f "${dir}/${i}.json" ]; then
      return 1
    fi
  done
  return 0
}

if all_jsons_exist "$JSON_DIR" "$START_IDX" "$END_IDX"; then
  echo "[exp1_infer.sh] All JSONs already exist. Running exp1_agg.sh ..."
  bash ./exp1_agg.sh
fi

python ./src/qwen3_next_infer_vllm_submit_ensemble_caption.py \
  --base-url http://localhost:22002/v1 \
  --model /data/models/Qwen/Qwen3-Next-80B-A3B-Instruct \
  --input-csvs "./sub_csv/*.csv" \
  --base-sub 20250927_ens2_clip20.csv \
  --json-dir ./output/jsons \
  --skip-existing-json

if all_jsons_exist "$JSON_DIR" "$START_IDX" "$END_IDX"; then
  echo "All JSONs exist after run..."
  python ./src/qwen3_next_infer_vllm_submit_ensemble_caption.py \
    --collect-jsons \
    --json-dir ./output/jsons \
    --base-sub 20250927_ens2_clip20.csv \
    --output-csv ./output/20250928_ens002_submit_ensemble.csv
else
  echo "JSONs are not complete yet. Skipping aggregation."
fi
