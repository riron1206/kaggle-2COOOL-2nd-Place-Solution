import json
import glob
import argparse
import time
import pandas as pd
from openai import OpenAI
from typing import Dict, Any, List
from collections import Counter
from pathlib import Path
from tqdm import tqdm


SERVER = "http://localhost:8000/v1/chat/completions"
MODEL = "/data/models/openai/gpt-oss-120b"

INPUT_CSV_DIR = "/home/user_00006_821839/workspace/k/2coool/gaggle_cp/glm45v_image_frames_infer_perception_vllm_server"
OUTPUT_CSV = "/home/user_00006_821839/workspace/k/2coool/work3/results/gpt-oss-120b_infer_vllm_sc_from_frames_csv.csv"

SYSTEM = """Reasoning: high""".strip()
PROMPT_BASE = f"""
I will provide the per-frame inference results of a VLM on a dashcam video. Thoroughly analyze the content and answer from which frame the most severe accident or hazard begins.

The VLM inference results for each frame of the dashcam video are given in a tabular format as follows:
- The `frame` column is the video frame index.
- The `Scene_Description` column is a description of the frame image.
- The `Hazard_Clues` column lists hazard cues identifiable from the frame image.
- The `Gaze_Focus` column describes where in the image the driver's gaze is focused.
- The `Incident_Detection` column flags whether a traffic incident is present in the image:
  1 = accident, 0 = hazard, -1 = no incident.
- The `Crash_Severity` column indicates the severity of the accident or hazard inferred from the image.
- The `Ego_car_involved` column indicates whether the ego vehicle is involved in the accident or hazard:
  0 = not involved, 1 = involved.
- The `Label` column is the type of hazard or accident inferred from the image.
- The `Number_of_Bicyclists_Scooters` column is the number of bicyclists or scooters involved in the hazard or accident.
- The `Number_of_Animals` column is the number of animals involved.
- The `Number_of_Pedestrians` column is the number of pedestrians involved.
- The `Number_of_Vehicles_excluding_ego` column is the number of vehicles other than the ego vehicle involved.

Below are the VLM inference results for each frame of the dashcam video:
""".strip()


def chat_json(
    server_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    n_samples: int = 8,
) -> List[Dict[str, Any]]:
    schema_hint = (
        'Return ONLY valid JSON. Schema: {"frame": int}'
    )

    base_url = server_url
    if "/chat/completions" in base_url:
        base_url = base_url.split("/chat/completions")[0]
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{schema_hint}\n\n{user_prompt}"},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n_samples,
        extra_body={
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        },
        timeout=60.0,
    )

    results = []
    for choice in result.choices:
        try:
            content = choice.message.content
            results.append(json.loads(content))
        except Exception as e:
            print(f"[WARN] JSON parse failed: {e} → {choice}")
    return results


def aggregate_self_consistency(results: List[dict]) -> dict:
    """
    Self-Consistency: Aggregate frame values by majority vote or median.
    """
    frames = [r["frame"] for r in results if "frame" in r]
    if not frames:
        return {}
    counter = Counter(frames)
    most_common_frame, _ = counter.most_common(1)[0]
    sorted_frames = sorted(frames)
    median_frame = sorted_frames[len(sorted_frames) // 2]
    return {
        "frames": frames,
        "majority_vote": most_common_frame,
        "median": median_frame,
    }


def collect_self_consistency(
    user_prompt: str,
    total: int,
    per_call: int,
    server_url: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> dict:
    """
    Split into multiple calls to chat_json and aggregate frame candidates.
    per_call is the number of n_samples per call; total is the total number of candidates.
    Even if --n-samples is large, it will be internally split into chunks of size per_call (e.g., 4) and invoked multiple times.
    """
    if per_call <= 0:
        per_call = 1
    if total <= 0:
        return {"candidates": [], "majority_vote": None, "median": None}

    all_samples: List[Dict[str, Any]] = []
    rounds = (total + per_call - 1) // per_call
    for i in range(rounds):
        cur_n = min(per_call, total - per_call * i)
        try:
            batch = chat_json(
                server_url=server_url,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                n_samples=cur_n,
            )
        except Exception as e:
            print(f"[ERROR] chat_json split call failed (round {i+1}/{rounds}): {e}")
            batch = []
        all_samples.extend(batch)

    frames = [s["frame"] for s in all_samples if isinstance(s, dict) and "frame" in s]
    frames = [f for f in frames if isinstance(f, int)]
    frames.sort()
    if not frames:
        return {"candidates": [], "majority_vote": None, "median": None}
    majority = Counter(frames).most_common(1)[0][0]
    median = frames[len(frames) // 2]
    return {"candidates": frames, "majority_vote": majority, "median": median}


def run_from_frames_csv(
    input_csv_dir: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    n_samples: int,
    max_retries: int,
    per_call: int = 1,
) -> pd.DataFrame:

    video_names = []
    frames = []
    csv_paths = sorted(glob.glob(f"{input_csv_dir}/*.csv"))
    for csv_path in tqdm(csv_paths):

        attempt = 0
        majority_vote_frame = None
        median_frame = None
        while attempt < max_retries:
            df = pd.read_csv(csv_path)
            user_prompt = PROMPT_BASE + "\n\n" + df.to_markdown(index=False)

            try:
                summary = collect_self_consistency(
                    user_prompt=user_prompt,
                    total=n_samples,
                    per_call=min(per_call, max(1, n_samples)),
                    server_url=SERVER,
                    model=MODEL,
                    system_prompt=SYSTEM,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
            except Exception as e:
                print(f"[ERROR] collect_self_consistency failed on {csv_path}: {e}")
                attempt += 1
                print(f"[INFO] Retry {attempt}/{max_retries} for {csv_path} due to exception in collect_self_consistency")
                if attempt < max_retries:
                    time.sleep(3)
                continue

            print("\n=== Aggregated (Self-Consistency, split sampling) ===")
            print(json.dumps(summary, ensure_ascii=False, indent=2))

            majority_vote_value = summary.get("majority_vote") if summary else None
            median_value = summary.get("median") if summary else None
            if isinstance(majority_vote_value, int):
                majority_vote_frame = majority_vote_value
                median_frame = median_value
                break

            attempt += 1
            print(f"[INFO] Retry {attempt}/{max_retries} for {csv_path} due to invalid majority_vote_frame: {majority_vote_value}")
            if attempt < max_retries:
                time.sleep(3)

        video_name = Path(csv_path).stem
        if majority_vote_frame is None:
            print(f"[ERROR] Failed to get majority_vote_frame for {csv_path}")
            continue
        selected_frame = (
            majority_vote_frame if isinstance(majority_vote_frame, int) and majority_vote_frame != -1
            else (median_frame if isinstance(median_frame, int) and median_frame != -1 else 1)
        )
        frames.append(int(selected_frame))
        video_names.append(video_name)

    df_result = pd.DataFrame(
        {
            "video": video_names,
            "frame": frames,
        },
    )

    return df_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate VLM frame-level inferences with self-consistency.")
    parser.add_argument("--input-dir", type=str, default=INPUT_CSV_DIR, help="Input CSV directory")
    parser.add_argument("--output-dir", type=str, default=str(Path(OUTPUT_CSV).parent), help="Output CSV directory")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=2*6)
    parser.add_argument("--max-retries", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / Path(OUTPUT_CSV).name

    df_result = run_from_frames_csv(
        input_csv_dir=args.input_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        n_samples=args.n_samples,
        max_retries=args.max_retries,
    )
    df_result.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
