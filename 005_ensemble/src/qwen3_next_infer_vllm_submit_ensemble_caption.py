import re, math, glob, os, json
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import argparse
import csv

MODEL = "/data/models/Qwen/Qwen3-Next-80B-A3B-Instruct"
BASE_URL = "http://localhost:22002/v1"
API_KEY = "EMPTY"
CANONICAL_EGO = "ego-car"
MAX_WORDS_BEFORE = 22
MAX_WORDS_REASON = 22

SYS = "Answer in English only. Strictly follow the Side‑Safe Direction Policy: never output 'left' or 'right' (including LHS/RHS and similar), and express lateral relations with side‑neutral or road‑structure terms."

PROMPT_PAIR_SINGLE = """
Ensemble the following per-csv PAIRS (caption_before, reason, frame) into a SINGLE best triple.
Return ONLY the final results in the specified XML tags. Do not include any other text.

PAIRS (in the same order as provided):
{pairs_block}

Task:
1) Cluster pairs by event type primarily using the 'reason' text and optionally confirm with 'before' text.
   - Choose the MAJORITY cluster as the incident being described.
   - If there is a tie, choose the cluster whose frame is the smallest.
2) Directional & Side‑Safe Language Policy (MUST FOLLOW for all outputs):
   - Forbidden tokens: left, right, Left, Right, LHS, RHS, port, starboard, left-turn, right-turn, ←, →.
   - When inputs contain these, rewrite using side‑neutral / road‑structure terms, for example:
     • adjacent lane, same‑direction lane, oncoming lane
     • curbside lane (nearest road edge), median‑side lane (nearest centerline/median)
     • roadside shoulder, median, centerline, cross traffic, opposite-direction traffic
     • “turns across {canonical_ego}'s path”, “turns toward the curbside/median‑side”
   - If a lateral side cannot be stated without left/right, omit the side and describe only the relation
     (e.g., “a vehicle merges into {canonical_ego}'s lane”).
   - Do not invent compass directions (north/south/east/west) or lane counts.
3) Compose:
   a) <caption_before>: ONE sentence describing the scene immediately BEFORE the incident.
      - Use only facts supported by the majority cluster (entities, locations, states).
      - Avoid causality/outcome words (e.g., force/forcing/causing/sudden/lane change/disruption/cuts in).
      - Present tense, active voice, ≤ {max_words_before} words; use '{canonical_ego}' for the ego reference; English only.
      - Do not include numbers/timestamps/frames.
      - Apply the Side‑Safe Policy for any lateral/lane phrasing.
   b) <reason>: ONE sentence describing the REASON OF INCIDENT with explicit causality and effect.
      - Make entities, relations, and cause–effect explicit (agent acts → effect on {canonical_ego}/traffic).
      - Prefer consensus n-grams (CIDEr), allow light synonyms (METEOR), keep clear relations (SPICE).
      - Present tense, active voice, ≤ {max_words_reason} words; English only.
      - Do not include numbers/timestamps/frames.
      - Apply the Side‑Safe Policy for any lateral/lane phrasing.
3) Output the selected incident frame as a plain integer inside <incident_start_frame>.
   - From the chosen cluster, output the SMALLEST frame value among its members.

Hard constraints:
- Do not invent unseen objects, counts, orientations, directions, or maneuvers.
- Use '{canonical_ego}' exactly for the ego reference.
- Replace any left/right mentions with Side‑Safe terms; never output the words “left” or “right”.

OUTPUT FORMAT (strict; no extra text, no explanations):
<caption_before>...</caption_before>
<reason>...</reason>
<incident_start_frame>...</incident_start_frame>
""".strip()

def _normalize_text(s):
    if not isinstance(s, str):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return ""
        s = str(s)
    s = s.strip()
    s = re.sub(r"^\s*\d+\s+", "", s)
    s = s.replace("ego-car", "ego car").replace("ego-vehicle", "ego vehicle")
    s = re.sub(r"\s+", " ", s)
    return s


def _build_pairs_block(pairs: List[Tuple[int, str, str]]) -> str:
    lines = []
    for idx, (frame, before, reason) in enumerate(pairs, 1):
        b = before.replace("<", "(").replace(">", ")")
        r = reason.replace("<", "(").replace(">", ")")
        lines.append(f"{idx}) frame={int(frame)}\n   before: {b}\n   reason: {r}")
    return "\n".join(lines)


def _contains_non_english(text: Optional[str]) -> bool:
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False

    return bool(re.search(r"[^\x00-\x7F]", s))


def _parse_final_fields(content: str):
    def gx(pat):
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    before = gx(r"<caption_before>\s*(.*?)\s*</caption_before>")
    reason = gx(r"<reason>\s*(.*?)\s*</reason>")
    frame_txt = gx(r"<incident_start_frame>\s*(.*?)\s*</incident_start_frame>")
    frame_val = None
    if frame_txt:
        m = re.search(r"(-?\d+)", frame_txt)
        if m:
            try:
                frame_val = int(m.group(1))
            except:
                frame_val = None
    return before, reason, frame_val


def _choose_frame_consensus(frames: List[int]) -> int:
    from collections import Counter

    cnt = Counter(frames)
    max_freq = max(cnt.values())
    cands = [f for f, c in cnt.items() if c == max_freq]
    return min(cands)


def build_prompt_pair_single(pairs: List[Tuple[int, str, str]]):
    return PROMPT_PAIR_SINGLE.format(
        pairs_block=_build_pairs_block(pairs),
        canonical_ego=CANONICAL_EGO,
        max_words_before=MAX_WORDS_BEFORE,
        max_words_reason=MAX_WORDS_REASON,
    )


def infer_incident_fields_single(
    prompt: str,
    sys: str = SYS,
    base_url: str = BASE_URL,
    model: str = MODEL,
    *,
    # https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    max_tokens: int = 1024 * 4,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    seed: Optional[int] = 42,
):
    client = OpenAI(
        base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY")
    )
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        extra_body={
            "top_k": top_k,
            "min_p": min_p,
        },
    )
    if seed is not None:
        kwargs["seed"] = seed
    result = client.chat.completions.create(**kwargs)
    content = result.choices[0].message.content or ""
    before, reason, frame_val = _parse_final_fields(content)
    return {
        "caption_before": before,
        "reason": reason,
        "incident_start_frame": frame_val,
        "raw": content,
    }


def run_prompt_pair_single(
    dfs: List[pd.DataFrame], i: int, *, base_url: str = BASE_URL, model: str = MODEL
):
    pairs, frames_for_consensus = [], []
    videos = []
    for df in dfs:
        video = int(df.iloc[i]["video"]) if "video" in df.columns else None
        if video is not None:
            videos.append(video)
        frame = int(df.iloc[i]["Incident window start frame"])
        before = _normalize_text(df.iloc[i]["Caption Before Incident"])
        reason = _normalize_text(df.iloc[i]["Reason of Incident"])
        pairs.append((frame, before, reason))
        frames_for_consensus.append(frame)

    if videos:
        if len(set(videos)) != 1:
            raise ValueError(f"Inconsistent video ids at row {i}: {videos}")

    prompt = build_prompt_pair_single(pairs)
    out = infer_incident_fields_single(prompt, base_url=base_url, model=model)

    frame = out["incident_start_frame"]
    if frame is None or frame not in frames_for_consensus:
        frame = _choose_frame_consensus(frames_for_consensus)

    video_id = videos[0] if videos else None

    return {
        "prompt": prompt,
        "video": video_id,
        "Incident window start frame": frame,
        "Caption Before Incident": out["caption_before"],
        "Reason of Incident": out["reason"],
    }


def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _save_json_atomic(target_path: str, data: Dict[str, Any]):
    tmp_path = f"{target_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, target_path)


def save_row_result_json(json_dir: str, row_result: Dict[str, Any]):
    """Save single row inference result as JSON (atomically)."""
    _ensure_dir(json_dir)
    video_id = row_result.get("video")
    if video_id is None:
        raise ValueError("'video' is required in row_result to save JSON")
    file_path = os.path.join(json_dir, f"{int(video_id)}.json")
    _save_json_atomic(file_path, row_result)


def collect_jsons_to_csv(json_dir: str, base_sub: Optional[str], output_csv: str):
    """Collect per-video JSONs from json_dir and write final CSV (optionally aligned with base_sub)."""
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        raise ValueError(f"No JSON files found in: {json_dir}")

    rows: List[Dict[str, Any]] = []
    seen_videos: Dict[int, Dict[str, Any]] = {}
    for jp in json_files:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            continue
        v = data.get("video")
        if v is None:
            continue
        v_int = int(v)
        prev = seen_videos.get(v_int)
        if prev is None or os.path.getmtime(jp) >= os.path.getmtime(
            os.path.join(json_dir, f"{v_int}.json")
        ):
            seen_videos[v_int] = {
                "video": v_int,
                "Caption Before Incident": data.get("Caption Before Incident", ""),
                "Reason of Incident": data.get("Reason of Incident", ""),
                "Incident window start frame": data.get(
                    "Incident window start frame", None
                ),
            }

    rows = list(seen_videos.values())
    if not rows:
        raise ValueError("No valid rows collected from JSONs.")

    df_json = pd.DataFrame(rows)

    if base_sub and os.path.exists(base_sub):
        df_base = pd.read_csv(base_sub)
        required = [
            "video",
            "Caption Before Incident",
            "Reason of Incident",
            "Incident window start frame",
        ]
        missing = [c for c in required if c not in df_base.columns]
        if missing:
            raise ValueError(f"Missing columns in base_sub: {missing}")

        df_merged = df_json.merge(
            df_base, on="video", how="right", suffixes=("", "_base")
        )
        for col in [
            "Caption Before Incident",
            "Reason of Incident",
            "Incident window start frame",
        ]:
            base_col = f"{col}_base"
            if base_col in df_merged.columns:
                df_merged[col] = df_merged[col].where(
                    df_merged[col].notna(), df_merged[base_col]
                )
                df_merged.drop(columns=[base_col], inplace=True)

        df_out = df_merged[df_base.columns]
    else:
        df_out = df_json[
            [
                "video",
                "Caption Before Incident",
                "Reason of Incident",
                "Incident window start frame",
            ]
        ]

    df_out.to_csv(
        output_csv,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        quotechar='"',
        doublequote=True,
    )
    print(f"OUTPUT(from JSONs): {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble caption/reason/frame from multiple CSVs."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model identifier for vLLM/OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        help="Base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--base-sub",
        type=str,
        default="/home/user_00006_821839/workspace/k/2coool/gaggle_cp/sub_csv/exp004_and_exp005_n_sample4_0.11560.csv",
        help="Reference submission CSV path (optional)",
    )
    parser.add_argument(
        "--input-csvs",
        type=str,
        default="/home/user_00006_821839/workspace/k/2coool/gaggle_cp/sub_csv/*.csv",
        help="Glob pattern for input CSVs",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/home/user_00006_821839/workspace/k/2coool/work3/results/gpt-oss-120b_infer_vllm_submit_ensemble_caption.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default=None,
        help="Directory to save per-row JSON results (enables JSON mode)",
    )
    parser.add_argument(
        "--collect-jsons",
        action="store_true",
        help="Collect JSONs from --json-dir and write CSV",
    )
    parser.add_argument(
        "--skip-existing-json",
        dest="skip_existing_json",
        action="store_true",
        default=False,
        help="Skip rows whose JSON already exists in --json-dir",
    )
    parser.add_argument(
        "--no-skip-existing-json",
        dest="skip_existing_json",
        action="store_false",
        help="Do not skip even if JSON exists (recompute)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process rows in reverse order (max -> 0)",
    )
    parser.add_argument(
        "--aggregate-after",
        action="store_true",
        help="After inference to JSONs, aggregate them into CSV",
    )
    parser.add_argument(
        "--max-i",
        type=int,
        default=None,
        help="Override number of rows to process (default: min length of inputs)",
    )
    args = parser.parse_args()

    base_sub = args.base_sub
    input_csvs = args.input_csvs
    output_csv = args.output_csv
    json_dir = args.json_dir

    dfs = [pd.read_csv(p) for p in sorted(glob.glob(input_csvs))]

    if args.collect_jsons:
        if not json_dir:
            raise ValueError("--json-dir is required when using --collect-jsons")
        collect_jsons_to_csv(json_dir, base_sub, output_csv)
        raise SystemExit(0)

    if os.path.exists(output_csv) and not json_dir:
        df_out = pd.read_csv(output_csv)

        required_out_cols = [
            "video",
            "Caption Before Incident",
            "Reason of Incident",
            "Incident window start frame",
        ]
        missing_out = [c for c in required_out_cols if c not in df_out.columns]
        if missing_out:
            raise ValueError(f"Missing columns in existing output: {missing_out}")

        mask_non_en = df_out["Caption Before Incident"].apply(
            _contains_non_english
        ) | df_out["Reason of Incident"].apply(_contains_non_english)
        videos_to_update = df_out.loc[mask_non_en, "video"].tolist()

        if videos_to_update:
            df0 = dfs[0]
            video_to_idx = {int(df0.iloc[i]["video"]): i for i in range(len(df0))}
            missing_vids = [v for v in videos_to_update if v not in video_to_idx]
            if missing_vids:
                raise ValueError(
                    f"Videos not found in inputs for update: {missing_vids[:10]} (total {len(missing_vids)})"
                )

            updated_rows = []
            for v in tqdm(videos_to_update):
                i = video_to_idx[int(v)]
                updated_rows.append(
                    run_prompt_pair_single(
                        dfs, i, base_url=args.base_url, model=args.model
                    )
                )
            df_updates = pd.DataFrame(updated_rows)

            df_out = df_out.merge(
                df_updates[
                    [
                        "video",
                        "Incident window start frame",
                        "Caption Before Incident",
                        "Reason of Incident",
                    ]
                ],
                on="video",
                how="left",
                suffixes=("", "_new"),
            )
            for col in [
                "Caption Before Incident",
                "Reason of Incident",
                "Incident window start frame",
            ]:
                new_col = f"{col}_new"
                if new_col in df_out.columns:
                    df_out[col] = df_out[new_col].where(
                        df_out[new_col].notna(), df_out[col]
                    )
                    df_out.drop(columns=[new_col], inplace=True)

            df_out = df_out[df_out.columns]
            df_out.to_csv(
                output_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                quotechar='"',
                doublequote=True,
            )
            print(f"OUTPUT(updated): {output_csv}")
        else:
            df_out.to_csv(
                output_csv,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                quotechar='"',
                doublequote=True,
            )
            print(f"OUTPUT(no-change): {output_csv}")
        raise SystemExit(0)

    if args.max_i is not None:
        max_i = int(args.max_i)
    else:
        max_i = min(len(df) for df in dfs)

    if json_dir:
        _ensure_dir(json_dir)
        indices = range(max_i)
        if args.reverse:
            indices = range(max_i - 1, -1, -1)
        for i in tqdm(indices):
            v = int(dfs[0].iloc[i]["video"]) if "video" in dfs[0].columns else i
            fp = os.path.join(json_dir, f"{int(v)}.json")
            if args.skip_existing_json and os.path.exists(fp):
                continue
            row_result = run_prompt_pair_single(
                dfs, i, base_url=args.base_url, model=args.model
            )

            row_result["row_index"] = i
            save_row_result_json(json_dir, row_result)

        if args.aggregate_after:
            collect_jsons_to_csv(json_dir, base_sub, output_csv)
        raise SystemExit(0)

    result_rows = [
        run_prompt_pair_single(dfs, i, base_url=args.base_url, model=args.model)
        for i in tqdm(range(max_i))
    ]
    df_out = pd.DataFrame(result_rows)

    if base_sub and os.path.exists(base_sub):
        df_base = pd.read_csv(base_sub)
        required = [
            "video",
            "Caption Before Incident",
            "Reason of Incident",
            "Incident window start frame",
        ]
        missing = [c for c in required if c not in df_base.columns]
        if missing:
            raise ValueError(f"Missing columns in base_sub: {missing}")

        if "video" not in df_out.columns:
            raise ValueError(
                "'video' column not found in df_out for alignment with base_sub"
            )

        df_merged = df_out.merge(
            df_base, on="video", how="left", suffixes=("", "_base")
        )

        for col in [
            "Caption Before Incident",
            "Reason of Incident",
            "Incident window start frame",
        ]:
            base_col = f"{col}_base"
            if base_col in df_merged.columns:
                df_merged[col] = df_merged[col].where(
                    df_merged[col].notna(), df_merged[base_col]
                )
                df_merged.drop(columns=[base_col], inplace=True)

        df_out = df_merged[df_base.columns]

    df_out.to_csv(
        output_csv,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        quotechar='"',
        doublequote=True,
    )
    print(f"OUTPUT: {output_csv}")
