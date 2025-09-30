import re
import json
import csv
import argparse
import time
import pathlib
import requests
from typing import List, Sequence, Tuple, Optional

SERVER = "http://localhost:8000/v1/chat/completions"
MODEL = "/data/models/zai-org/GLM-4.5V"

PROMPT_MIN = """
You are an assistant analyzing dashcam frames with gaze heatmaps.
Work in ULTRATHINK mode: carefully inspect the image before answering; scan left→center→right and near→far; cross-check the top heatmap with the bottom dashcam. Think step by step internally but DO NOT output your reasoning — output only the final JSON.

Each input image is a single frame composed of two vertically stacked parts:
- Top: gaze heatmap (white marks indicate the driver’s focus)
- Bottom: original dashcam image (use this for scene and hazard descriptions)

Return PURE JSON with the following fields exactly:

{
  "Scene_Description": "<objective description of the road, vehicles, pedestrians, signs, lighting, occlusions, etc.>",
  "Hazard_Clues": "<potential risk factors that could lead to accidents; if none, write 'none'>",
  "Gaze_Focus": "<what object/region the white heatmap marks overlap with and its relative position (e.g., 'right parked car', 'center ahead crosswalk')>",
  "Number_of_Bicyclists_Scooters": <integer>,
  "Number_of_Animals": <integer>,
  "Number_of_Pedestrians": <integer>,
  "Number_of_Vehicles_excluding_ego": <integer>,
  "Incident_Detection": <integer>,
  "Crash_Severity": "<one of the predefined severity strings>",
  "Ego_car_involved": <integer>,
  "Label": "<one of the 19 predefined incident classes>"
}

Guidelines:
- Be concise and factual. Describe only what is visible; if uncertain, use "unknown".
- Prioritize hazard-relevant details (moving/parked vehicles, pedestrians/cyclists, traffic controls, road geometry, occlusions, poor visibility, unusual obstacles).
- Use the TOP heatmap ONLY to determine Gaze_Focus; verify it matches the BOTTOM dashcam content.
- For counting fields:
  * Count only clearly visible instances.
  * Exclude the ego-car (the car with dashcam) from vehicle counts.
  * If none are visible, set the number to 0.
- For "Incident_Detection":
  * 1 = Accident (collision happened in this frame)
  * 0 = Hazard (dangerous situation but no collision in this frame)
  * -1 = No incident visible
- For "Crash_Severity": choose exactly one from:
  "1. Ego-car collided but did not stop"
  "2. Ego-car collided and could not continue moving"
  "3. Ego-car collided with at-least one person or cyclist"
  "4. Other cars collided with person/car/object but ego-car is ok"
  "5. Multiple vehicles collided with ego-car"
  "6. One or Multiple vehicles collided but ego-car is fine"
  If no collision, write "unknown".
- For "Ego_car_involved": 1 if the ego-car (dashcam vehicle) is part of the collision, else 0.
- For "Label": choose exactly one from these 19 classes:
  "ego-car hits barrier", "flying object hit the car", "ego-car hit an animal",
  "many cars/pedestrians/cyclists collided", "car hits barrier",
  "ego-car hits a pedestrian", "animal on the road", "car flipped over",
  "ego-car hits a crossing cyclist", "vehicle drives into another vehicle",
  "ego-car loses control", "scooter on the road", "bicycle on road",
  "pedestrian is crossing the street", "pedestrian on the road",
  "vehicle hits ego-car", "ego-car hits a vehicle", "vehicle overtakes", "unknown"
- Output MUST be valid JSON with no extra text, no comments, and no backticks.
""".strip()


def extract_frame_number(path: pathlib.Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def list_frames_sorted(
    dir_path: str,
    patterns: Sequence[str] = ("*.png", "*.jpg", "*.jpeg"),
) -> List[pathlib.Path]:
    p = pathlib.Path(dir_path)
    files: List[pathlib.Path] = []
    for pat in patterns:
        files.extend(p.glob(pat))
    files = sorted(files, key=lambda x: extract_frame_number(x))
    return files


def sample_uniform(
    paths: List[pathlib.Path], k: int, include_ends: bool = True
) -> List[pathlib.Path]:
    n = len(paths)
    if k <= 0:
        return []
    if k >= n:
        return paths
    if include_ends:
        idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    else:
        idxs = [round((i + 1) * (n - 1) / (k + 1)) for i in range(k)]

    seen, uniq = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            uniq.append(i)

    j = 0
    while len(uniq) < k:
        cand = j % n
        if cand not in seen:
            seen.add(cand)
            uniq.append(cand)
        j += 1
    uniq.sort()
    return [paths[i] for i in uniq]


def build_frame_contents(paths: List[pathlib.Path]) -> List[dict]:
    content = []
    for path in paths:
        n = extract_frame_number(path)
        content.append({"type": "text", "text": f"Frame {n}"})
        content.append(
            {"type": "image_url", "image_url": {"url": f"file://{path.resolve()}"}}
        )
    return content


def parse_start_frame(text: str) -> Optional[int]:
    m = re.search(r"Start_Frame\s*[:=]\s*(\d+|none)", text, flags=re.I)
    if not m:
        return None
    val = m.group(1).lower()
    return None if val == "none" else int(val)


def infer_window(
    frame_paths: List[pathlib.Path],
    prompt: str = PROMPT_MIN,
    num_frames: int = 6,
    temperature: float = 0.05,
    max_tokens: int = 2048,
    timeout: int = 120,
    enable_thinking: bool = True,
) -> Tuple[Optional[str], str, dict, List[int]]:
    """
    Uniformly sample num_frames from the given frame list and run VLM inference.
    Returns: (reasoning_content, final_text, raw_json, list of used frame numbers)
    """
    if not frame_paths:
        raise ValueError("empty frame_paths")

    sampled = sample_uniform(frame_paths, num_frames, include_ends=True)
    print("[Debug]:", sampled)
    content = [{"type": "text", "text": prompt}] + build_frame_contents(sampled)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}
        },
    }

    resp = requests.post(SERVER, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    thoughts = msg.get("reasoning_content")
    final = msg.get("content")
    used_numbers = [extract_frame_number(p) for p in sampled]
    return thoughts, final, data, used_numbers


def scan_incident_windows(
    dir_path: str,
    window_len: int = 60,
    step: int = 15,
    num_frames: int = 6,
    prompt: str = PROMPT_MIN,
    temperature: float = 0.05,
    max_tokens: int = 2048,
    timeout: int = 120,
    enable_thinking: bool = True,
    early_stop: bool = True,
    sleep_sec: float = 0.0,
) -> dict:
    """
    Slide a window across sequential frames to find the suspicious/accident "start frame".

    - window_len: length of consecutive frames per window (e.g., 60)
    - step: window shift size (e.g., 15)
    - num_frames: number of uniformly sampled frames per window (e.g., 6)
    - early_stop: stop when the model returns 'Start_Frame: <N>' (True)
    Returns a dict:
      {
        "first_hit": {"start_frame": int|None, "window_index": int|None, "used_frames": [...], "text": str} or None,
        "windows": [
            {"index": i, "range": [startN, endN], "used_frames": [...], "start_frame": int|None, "text": "..."}
            ...
        ]
      }
    """
    all_paths = list_frames_sorted(dir_path)
    if not all_paths:
        raise FileNotFoundError(f"No frames found in: {dir_path}")

    n = len(all_paths)
    results = {"first_hit": None, "windows": []}
    i = 0
    widx = 0

    while i < n:
        j = min(i + window_len, n)
        window_paths = all_paths[i:j]
        if not window_paths:
            break

        try:
            thoughts, final, raw, used = infer_window(
                window_paths,
                prompt=prompt,
                num_frames=num_frames,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                enable_thinking=enable_thinking,
            )
        except requests.RequestException as e:
            results["windows"].append(
                {
                    "index": widx,
                    "range": [
                        extract_frame_number(window_paths[0]),
                        extract_frame_number(window_paths[-1]),
                    ],
                    "used_frames": [
                        extract_frame_number(p)
                        for p in sample_uniform(window_paths, num_frames)
                    ],
                    "start_frame": None,
                    "text": f"[Request error] {e}",
                }
            )
            i += step
            widx += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue

        startN = parse_start_frame(final)
        rec = {
            "index": widx,
            "range": [
                extract_frame_number(window_paths[0]),
                extract_frame_number(window_paths[-1]),
            ],
            "used_frames": used,
            "start_frame": startN,
            "text": final,
        }
        results["windows"].append(rec)

        if startN is not None and results["first_hit"] is None:
            results["first_hit"] = {
                "start_frame": startN,
                "window_index": widx,
                "used_frames": used,
                "text": final,
            }
            if early_stop:
                break

        i += step
        widx += 1
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return results


def infer_all_frames_sequential(
    dir_path: str,
    prompt: str = PROMPT_MIN,
    temperature: float = 0.05,
    max_tokens: int = 2048,
    timeout: int = 120,
    enable_thinking: bool = True,
    sleep_sec: float = 0.0,
    frame_stride: int = 1,
    limit_n: Optional[int] = None,
    out_json_path: Optional[str] = None,
) -> dict:
    """
    Send each frame to the VLM one by one and return results as JSON.
    If frame_stride is specified, process every N-th frame (e.g., 5 -> 0,5,10,...).
    If out_json_path is given, also save to file.
    Returns: {"dir_path": str, "model": str, "results": [ ... ]}
    """
    all_paths = list_frames_sorted(dir_path)
    if not all_paths:
        raise FileNotFoundError(f"No frames found in: {dir_path}")

    results = []
    added = 0
    if frame_stride <= 0:
        raise ValueError("frame_stride must be >= 1")
    for idx_in_all in range(0, len(all_paths), frame_stride):
        path = all_paths[idx_in_all]
        try:
            thoughts, final, raw, used = infer_window(
                [path],
                prompt=prompt,
                num_frames=1,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                enable_thinking=enable_thinking,
            )
            results.append(
                {
                    "index": idx_in_all,
                    "frame": extract_frame_number(path),
                    "path": str(path.resolve()),
                    "used_frames": used,
                    "text": final,
                    "reasoning": thoughts,
                }
            )
        except requests.RequestException as e:
            results.append(
                {
                    "index": idx_in_all,
                    "frame": extract_frame_number(path),
                    "path": str(path.resolve()),
                    "used_frames": [extract_frame_number(path)],
                    "text": f"[Request error] {e}",
                    "reasoning": None,
                }
            )

        added += 1
        if limit_n is not None and added >= limit_n:
            break

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    out = {"dir_path": dir_path, "model": MODEL, "results": results}
    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def infer_single_frame(
    dir_path: str,
    frame_number: int,
    prompt: str = PROMPT_MIN,
    temperature: float = 0.05,
    max_tokens: int = 2048,
    timeout: int = 120,
    enable_thinking: bool = True,
    out_json_path: Optional[str] = None,
) -> dict:
    """
    Run VLM inference for only the frame whose filename contains the given number.
    Returns: {"dir_path": str, "model": str, "results": [ ...1 item... ]}
    """
    all_paths = list_frames_sorted(dir_path)
    if not all_paths:
        raise FileNotFoundError(f"No frames found in: {dir_path}")

    target_paths = [p for p in all_paths if extract_frame_number(p) == frame_number]
    if not target_paths:
        raise FileNotFoundError(f"Frame number {frame_number} not found in: {dir_path}")

    path = target_paths[0]
    idx_in_dir = all_paths.index(path)

    thoughts, final, raw, used = infer_window(
        [path],
        prompt=prompt,
        num_frames=1,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )

    result = {
        "index": idx_in_dir,
        "frame": extract_frame_number(path),
        "path": str(path.resolve()),
        "used_frames": used,
        "text": final,
        "reasoning": thoughts,
    }

    out = {"dir_path": dir_path, "model": MODEL, "results": [result]}
    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def _parse_text_json_safely(text: str) -> Optional[dict]:
    """Safely parse a JSON string inside the text field into a dict; return None on failure.
    If not pure JSON, extract the outermost {...} and retry."""
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None


def export_text_json_to_csv(
    results: List[dict],
    csv_path: str,
    preferred_keys: Optional[List[str]] = None,
    meta_keys: Optional[List[str]] = None,
) -> int:
    """
    Parse JSON strings contained in results[*]['text'] and write a CSV to csv_path
    with keys as column headers. Returns the number of rows written (excluding header).
    Rows that fail to parse are skipped.
    """
    preferred = preferred_keys or [
        "Scene_Description",
        "Hazard_Clues",
        "Gaze_Focus",
        "Incident_Detection",
        "Crash_Severity",
        "Ego_car_involved",
        "Label",
        "Number_of_Bicyclists_Scooters",
        "Number_of_Animals",
        "Number_of_Pedestrians",
        "Number_of_Vehicles_excluding_ego",
    ]
    meta_keys = meta_keys or ["frame", "path"]

    parsed_rows: List[dict] = []
    key_union: List[str] = []
    key_seen = set()

    for rec in results:
        text = rec.get("text")
        if not isinstance(text, str):
            continue
        obj = _parse_text_json_safely(text)
        if obj is None or not isinstance(obj, dict):
            continue
        parsed_rows.append(obj)
        for k in obj.keys():
            if k not in key_seen:
                key_seen.add(k)
                key_union.append(k)

    header: List[str] = []
    for mk in meta_keys:
        if mk not in header:
            header.append(mk)
    for k in preferred:
        if k in key_seen:
            header.append(k)
    for k in key_union:
        if k not in header:
            header.append(k)

    if not header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            pass
        return 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        pr_iter = iter(parsed_rows)
        for rec in results:
            text = rec.get("text")
            if not isinstance(text, str):
                continue
            obj = _parse_text_json_safely(text)
            if obj is None or not isinstance(obj, dict):
                continue
            row = next(pr_iter)
            out_row = {}

            for mk in meta_keys:
                if mk == "frame":
                    out_row[mk] = rec.get("frame", "")
                else:
                    out_row[mk] = rec.get(mk, "")

            for k in header:
                if k in meta_keys:
                    continue
                out_row[k] = row.get(k, "")
            writer.writerow(out_row)
    return len(parsed_rows)


def infer_frame_groups(
    dir_path: str,
    group_size: int = 6,
    group_step: Optional[int] = None,
    prompt: str = PROMPT_MIN,
    temperature: float = 0.05,
    max_tokens: int = 2048,
    timeout: int = 120,
    enable_thinking: bool = True,
    sleep_sec: float = 0.0,
    limit_n: Optional[int] = None,
    out_json_path: Optional[str] = None,
    include_partial_last: bool = True,
) -> dict:
    """
    Split all frames into consecutive groups of N and infer each group in one VLM call.
    If group_step is omitted, it equals group_size (non-overlapping).
    Returns: {"dir_path": str, "model": str, "results": [ ... ]}
    """
    all_paths = list_frames_sorted(dir_path)
    if not all_paths:
        raise FileNotFoundError(f"No frames found in: {dir_path}")

    if group_size <= 0:
        raise ValueError("group_size must be >= 1")
    step = group_step if group_step is not None else group_size
    if step <= 0:
        raise ValueError("group_step must be >= 1")

    results = []
    n = len(all_paths)
    gidx = 0
    i = 0
    while i < n:
        j = min(i + group_size, n)
        group_paths = all_paths[i:j]
        if not group_paths:
            break
        if len(group_paths) < group_size and not include_partial_last:
            break

        try:
            thoughts, final, raw, used = infer_window(
                group_paths,
                prompt=prompt,
                num_frames=len(group_paths),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                enable_thinking=enable_thinking,
            )
        except requests.RequestException as e:
            results.append(
                {
                    "index": gidx,
                    "range": [
                        extract_frame_number(group_paths[0]),
                        extract_frame_number(group_paths[-1]),
                    ],
                    "frames": [extract_frame_number(p) for p in group_paths],
                    "used_frames": [extract_frame_number(p) for p in group_paths],
                    "text": f"[Request error] {e}",
                    "reasoning": None,
                }
            )
            if limit_n is not None and len(results) >= limit_n:
                break
            i += step
            gidx += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue

        results.append(
            {
                "index": gidx,
                "range": [
                    extract_frame_number(group_paths[0]),
                    extract_frame_number(group_paths[-1]),
                ],
                "frames": [extract_frame_number(p) for p in group_paths],
                "used_frames": used,
                "text": final,
                "reasoning": thoughts,
            }
        )
        if limit_n is not None and len(results) >= limit_n:
            break

        i += step
        gidx += 1
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    out = {"dir_path": dir_path, "model": MODEL, "results": results}
    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential frame inference (VLM per frame, save JSON)"
    )
    parser.add_argument("--dir", "-d", required=True, help="Directory of input frames")
    parser.add_argument(
        "--out_dir",
        "-o",
        default=".",
        help="Output directory. Filename will be <input-dir-name>.json",
    )
    parser.add_argument("--temperature", type=float, default=0.01, help="Generation temperature")
    parser.add_argument(
        "--max_tokens", type=int, default=4096 * 2, help="Maximum output tokens"
    )
    parser.add_argument(
        "--sleep_sec", type=float, default=0.0, help="Seconds to sleep between requests"
    )
    parser.add_argument(
        "--disable_thinking", action="store_true", help="Disable reasoning content"
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Frame stride for sequential inference (skip every N frames)",
    )
    parser.add_argument(
        "--group_size", type=int, default=1, help="Infer every N frames as a group. 1 = per-frame"
    )
    parser.add_argument(
        "--group_step",
        type=int,
        default=None,
        help="Group shift width. Defaults to group_size",
    )
    parser.add_argument(
        "--debug_n", type=int, default=None, help="Debug: infer only the first N items then exit"
    )
    parser.add_argument(
        "--single_frame",
        type=int,
        default=None,
        help="Infer only the frame with the given frame number",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip inference if the output file already exists",
    )
    args = parser.parse_args()


    in_base = pathlib.Path(args.dir).resolve().name
    if args.single_frame is not None:
        out_path = (
            pathlib.Path(args.out_dir).resolve()
            / f"{in_base}_frame_{args.single_frame}.json"
        )
        out_csv_path = (
            pathlib.Path(args.out_dir).resolve()
            / f"{in_base}_frame_{args.single_frame}.csv"
        )
    elif args.debug_n is not None:
        out_path = (
            pathlib.Path(args.out_dir).resolve()
            / f"{in_base}_debugN{args.debug_n}.json"
        )
        out_csv_path = (
            pathlib.Path(args.out_dir).resolve() / f"{in_base}_debugN{args.debug_n}.csv"
        )
    else:
        out_path = pathlib.Path(args.out_dir).resolve() / f"{in_base}.json"
        out_csv_path = pathlib.Path(args.out_dir).resolve() / f"{in_base}.csv"
    print(f"in_base: {in_base}")
    print(f"out_path: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_if_exists and out_path.exists():
        print(f"[SKIP] JSON exists: {out_path}")
        if not pathlib.Path(str(out_csv_path)).exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict) and "results" in existing:
                    export_text_json_to_csv(existing["results"], str(out_csv_path))
                    print(f"[CSV] generated from existing JSON: {out_csv_path}")
            except Exception as e:
                print(f"[WARN] failed to load existing JSON for CSV regeneration: {e}")
        raise SystemExit(0)

    if args.single_frame is not None:
        out_seq = infer_single_frame(
            dir_path=args.dir,
            frame_number=args.single_frame,
            prompt=PROMPT_MIN,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=not args.disable_thinking,
            out_json_path=str(out_path),
        )
        print(f"[ONE] saved 1 record (frame {args.single_frame}) to {out_path}")
        export_text_json_to_csv(out_seq["results"], str(out_csv_path))
        print(f"[CSV] saved to {out_csv_path}")
    elif args.group_size and args.group_size > 1:
        step = args.group_step if args.group_step is not None else args.group_size
        out_seq = infer_frame_groups(
            dir_path=args.dir,
            group_size=args.group_size,
            group_step=step,
            prompt=PROMPT_MIN,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=not args.disable_thinking,
            sleep_sec=args.sleep_sec,
            limit_n=args.debug_n,
            out_json_path=str(out_path),
        )
        print(f"[GROUP] saved {len(out_seq['results'])} groups to {out_path}")
        export_text_json_to_csv(out_seq["results"], str(out_csv_path))
        print(f"[CSV] saved to {out_csv_path}")
    else:
        out_seq = infer_all_frames_sequential(
            dir_path=args.dir,
            prompt=PROMPT_MIN,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=not args.disable_thinking,
            sleep_sec=args.sleep_sec,
            frame_stride=args.frame_stride,
            limit_n=args.debug_n,
            out_json_path=str(out_path),
        )
        print(f"[SEQ] saved {len(out_seq['results'])} records to {out_path}")
        export_text_json_to_csv(out_seq["results"], str(out_csv_path))
        print(f"[CSV] saved to {out_csv_path}")
