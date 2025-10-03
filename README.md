# kaggle 2COOOL 2nd Place Solution

- arxiv paper: https://arxiv.org/abs/XXX

- competition: https://www.kaggle.com/competitions/2coool

- ICCV2025 leaderboard: https://2coool.net/#leaderboard

![](./assets/ICCV_lb.png)

## Hardware

- Ubuntu 22.04.4 LTS
- Intel(R) Xeon(R) Platinum 8480C / Intel(R) Xeon(R) Platinum 8480+
- NVIDIA H100 80GB HBM3 / NVIDIA H200
- Memory 2.0TiB

## Software

- Python 3.10 and 3.12
- CUDA 12.4 / 12.8.1, NVIDIA driver 535.x / 570.x
- vLLM (GLM-4.5V / GPT-OSS-120B / Qwen3-VL-235B / Qwen3-Next-80B)
- FFmpeg (ffmpeg, ffprobe)

Note: Each `vllm_*` directory provisions its own virtualenv via `uv`. You can replace `module load` with your local CUDA setup if not on HPC.

---

## Repository structure

```
001_video2frames/            # Video → frames, heatmap/video vstack, utilities
002_frame_captioning/        # GLM-4.5V via vLLM: per-frame captions (stride configurable)
003_frame_detection/         # GPT-OSS-120B via vLLM: incident start-frame from captions CSV
004_description/             # Qwen3-VL-235B: multi-image reasoning → final JSON/CSV
005_ensemble/                # Qwen3-Next-80B: ensemble caption/reason/frame
vllm_glm45v/                 # env + server for GLM-4.5V
vllm_gpt-oss/                # env + server for GPT-OSS-120B
vllm_qwen3/                  # env + servers for Qwen3-VL / Qwen3-Next
```

---

## Models and local paths

Default model identifiers/paths expected by scripts:

- [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V): `/data/models/zai-org/GLM-4.5V`
- [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b): `/data/models/openai/gpt-oss-120b`
- [Qwen3-VL-235B-A22B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking): `/data/models/Qwen/Qwen3-VL-235B-A22B-Thinking`
- [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct): `/data/models/Qwen/Qwen3-Next-80B-A3B-Instruct`

Place model weights at the above paths, or update `run_server*.sh` to point to your local paths (or create symlinks under `/data/models`).

---

## Data layout (expected)

The preprocessing scripts assume these directories:

- Raw videos: `<Competition Data>/videos/*.mp4`
- Heatmap videos (if provided): `<Competition Data>/heatmaps/*.mp4`
- Output frames: `001_video2frames/gdrive_png/...`
- VStack outputs: `001_video2frames/mp4_vstack/*.mp4` and frames under `mp4_vstack_png/...`

Example (after [1. Video to Frames](https://github.com/riron1206/kaggle-2COOOL-Nth-Place-Solution?tab=readme-ov-file#solution-pipeline)):

```
001_video2frames/
  gdrive_png/
    videos/<video_id>/000001.png ...
  mp4_vstack/
    <name>.mp4
  mp4_vstack_png/
    mp4_vstack/<name>/000001.png ...
```

---

## Setup

1) Install prerequisites

    ```bash
    # FFmpeg (Ubuntu)
    sudo apt-get update && sudo apt-get install -y ffmpeg
    # uv (optional but used in setup.sh)
    pipx install uv || pip install uv
    ```

2) Prepare environments

    ```bash
    # GLM-4.5V
    cd vllm_glm45v
    bash setup.sh

    # GPT-OSS-120B
    cd vllm_gpt-oss
    bash setup.sh

    # Qwen3-VL-235B and Qwen3-Next-80B
    cd vllm_qwen3
    bash setup.sh
    ```

---

## Solution Pipeline

![](./assets/fig1_2coool.png)

![](./assets/fig2_2coool.png)

1. **Video to Frames**

    Convert videos into frames, and also prepare gaze heatmap images and vertically stacked videos with corresponding frames.

    ```bash
    # Python environment setup via `vllm_glm45v/setup.sh`
    source ./vllm_glm45v/.venv/bin/activate

    cd 001_video2frames

    # mp4 to png
    python ./src/mp4_to_png.py \
        --input-dirs <Competition Data> \
        --output-root gdrive_png

    # mp4 vstack
    python ./src/vstack_mp4_pairs_ffprobe.py \
        --left-dir /data/dataset/2coool/gdrive/heatmaps/ \
        --right-dir /data/dataset/2coool/gdrive/videos/ \
        --out-dir mp4_vstack

    # mp4 vstack png
    python ./src/mp4_to_png.py \
        --input-dirs mp4_vstack \
        --output-root mp4_vstack_png
    ```

2. **Frame Captioning (GLM-4.5V)**

    Every 10 frames, run inference with GLM-4.5V to generate captions for the image frames.

    Start vLLM Server
    ```bash
    cd ./vllm_glm45v
    bash ./run_server.sh
    ```

    Run
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    cd 002_frame_captioning

    bash ./run_glm45v_image_frames_infer_perception_vllm_server.sh
    ```

3. **Incident/Hazard Frame Detection (GPT-OSS-120B)**

    Use `gpt-oss-120b` to analyze the generated captions and identify incident or hazard frames.

    Start vLLM Server
    ```bash
    cd ./vllm_gpt-oss
    bash ./vllm_gpt-oss/run_server.sh
    ```

    Run
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    cd 003_frame_detection

    bash ./run_gpt-oss-120b_infer_vllm_sc_from_frames_csv.sh
    ```

4. **Incident/Hazard Description (GLM-4.5V / Qwen3-VL-235B)**

    Run inference with `GLM-4.5V` on about N frames around the detected frame to generate incident or hazard descriptions.

    Start vLLM Server
    ```bash
    cd ./vllm_glm45v
    bash ./run_server.sh
    ```

    Run
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    cd 004_description/scripts_glm45v

    bash ./run_glm45v_multi_image_select_frames_from_csv_infer_vllm.sh
    bash ./run_glm45v_multi_image_select_frames_from_gptoss_csv_infer_vllm_v2.sh

    bash ./merge_submit_2csv.sh
    ```

    Run inference with `Qwen3-VL-235B-A22B-Thinking` on about N frames around the detected frame to generate incident or hazard descriptions.

    Start vLLM Server
    ```bash
    cd ./vllm_qwen3
    bash ./run_server_qwen3vl.sh
    ```

    Run
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    cd 004_description/scripts_qwen3vl

    bash ./run_Qwen3VL_multi_image_select_frames_from_csv_infer_vllm_v2.sh

    bash ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v2.sh

    bash ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v3.sh
    bash ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v3_prompt_v2.sh

    bash ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v4.sh

    bash ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v5.sh

    bash ./merge_submit_2csv.sh

    bash ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v6.sh
    ```

5. **Ensemble submission.csv (Qwen3-Next-80B)**

    Ensemble multiple submission.csv files generated with `Qwen3-Next-80B-A3B-Instruct`.

    Copy the submission.csv file generated in Step `4. Incident/Hazard Description` into `/005_ensemble/sub_csv`. Since it has already been copied, no further action is required. To reproduce the results, place a submission.csv file with a similar filename in `/005_ensemble/sub_csv`.

    Start vLLM Server
    ```bash
    cd ./vllm_qwen3
    bash ./run_server_qwen3_next.sh
    ```

    Run
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    cd 005_ensemble

    bash ./run_qwen3_next_ensemble.sh
    ```

5. **Blind A/B Test**

    To determine the final submission, the submission.csv files are compared using [006_blind_test](./006_blind_test/) application.

---

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt).
