# kaggle 2COOOL Nth Place Solution

competition: https://www.kaggle.com/competitions/2coool

arxiv paper: https://arxiv.org/abs/XXX

solution summary: https://www.kaggle.com/competitions/2coool/discussion/XXX

## Hardware

- Ubuntu 22.04.4 LTS
- Intel(R) Xeon(R) Platinum 8480C / Intel(R) Xeon(R) Platinum 8480+
- NVIDIA H100 80GB HBM3 / NVIDIA H200
- Memory 2.0TiB

## Software

- Python 3.10 / 3.12
- CUDA Version 12.4 / 12.8.1
- nvidia Driver Version 535.183.01 / 570.172.08

Python package details change per working directory.

## Solution Pipeline

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
    python ./src/vstack_mp4_pairs_ffprobe.py

    # mp4 vstack png
    python ./src/mp4_to_png.py \
        --input-dirs mp4_vstack \
        --output-root mp4_vstack_png
    ```

2. **Frame Captioning**

    Every 10 frames, run inference with GLM-4.5V to generate captions for the image frames.

    Start vLLM Server
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    ./vllm_glm45v/run_server.sh
    ```

    Run
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    cd 002_frame_captioning

    ./run_glm45v_image_frames_infer_perception_vllm_server.sh
    ```

3. **Incident/Hazard Frame Detection**

    Use `gpt-oss-120b` to analyze the generated captions and identify incident or hazard frames.

    Start vLLM Server
    ```bash
    # Python environment setup via `vllm_gpt-oss/setup.sh`
    source ./vllm_gpt-oss/.venv/bin/activate

    ./vllm_gpt-oss/run_server.sh
    ```

    Run
    ```bash
    source ./vllm_gpt-oss/.venv/bin/activate

    cd 003_frame_detection

    ./run_gpt-oss-120b_infer_vllm_sc_from_frames_csv.sh
    ```

4. **Incident/Hazard Description**

    Run inference with `Qwen3-VL-235B-A22B-Thinking` on about N frames around the detected frame to generate incident or hazard descriptions.

    Start vLLM Server
    ```bash
    source ./vllm_qwen3/.venv/bin/activate

    ./vllm_qwen3/run_server_qwen3vl.sh
    ```

    Run
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    cd 004_description

    ./run_Qwen3VL_multi_image_select_frames_from_csv_infer_vllm_v2.sh

    ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v2.sh
    ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v3_prompt_v2.sh
    ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v3.sh
    ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v4.sh
    ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v5.sh
    ./run_Qwen3VL_multi_image_select_frames_from_gptoss_csv_infer_vllm_v6.sh

    ./run_Qwen3VL_multi_image_gptoss_csv_vllm_add_heatmap.sh
    ```

5. **Ensemble submission.csv**

    Ensemble multiple submission.csv files generated with `Qwen3-Next-80B-A3B-Instruct`.

    Start vLLM Server
    ```bash
    source ./vllm_qwen3/.venv/bin/activate

    ./vllm_qwen3/run_server_qwen3_next.sh
    ```

    Run
    ```bash
    source ./vllm_glm45v/.venv/bin/activate

    cd 005_ensemble

    ./run_qwen3_next_ensemble.sh
    ```
