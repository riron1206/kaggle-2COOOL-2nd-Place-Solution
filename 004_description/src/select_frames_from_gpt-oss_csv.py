from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast


DEFAULT_EXTS: Tuple[str, ...] = (".png", ".jpg")


def _to_int_or_none(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _read_csv(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            vid = _to_int_or_none(row.get("video"))
            frame = _to_int_or_none(row.get("frame"))
            rows.append({"video": vid, "frame": frame})

    rows = [r for r in rows if r["frame"] is not None and r.get("video") is not None]
    rows.sort(key=lambda r: int(r["frame"]))
    return rows


def _infer_video_id_from_images_dir(images_dir: Path) -> Optional[int]:
    try:
        return int(images_dir.name)
    except Exception:
        return None


def _detect_min_max_frame_in_images_dir(
    images_dir: Path, exts: Sequence[str]
) -> Optional[Tuple[int, int]]:
    min_frame: Optional[int] = None
    max_frame: Optional[int] = None
    for ext in exts:
        for p in images_dir.glob(f"*{ext}"):
            stem = p.stem
            if not stem.isdigit():
                continue
            v = int(stem)
            if min_frame is None or v < min_frame:
                min_frame = v
            if max_frame is None or v > max_frame:
                max_frame = v
    if min_frame is None or max_frame is None:
        return None
    return (min_frame, max_frame)


def _resolve_image_path(
    images_dir: Path, frame: int, path_field: Optional[str], exts: Sequence[str]
) -> Optional[Path]:
    candidates: List[Path] = []
    for ext in exts:
        candidates.append(images_dir / f"{frame}{ext}")
        candidates.append(images_dir / f"{frame:06d}{ext}")

    for c in candidates:
        if c.exists():
            return c.resolve()

    if path_field:
        p = Path(str(path_field))
        if p.exists():
            return p.resolve()

    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enumerate images over a frame range based on CSV conditions"
    )
    ap.add_argument("--csv", required=True, help="Path to the input CSV")
    ap.add_argument(
        "--images_dir", required=True, help="Directory containing the corresponding images"
    )
    ap.add_argument(
        "--video_id",
        type=int,
        default=None,
        help="Video ID corresponding to the 'video' column in a gpt-oss-format CSV (if omitted, inferred from images_dir name)",
    )
    ap.add_argument(
        "--pre_frames",
        type=int,
        default=6,
        help="Number of frames before the trigger frame to include",
    )
    ap.add_argument(
        "--post_frames",
        type=int,
        default=5,
        help="Number of frames after the trigger frame to include (0 = up to the CSV max frame)",
    )
    ap.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Frame sampling step (>= 1)",
    )
    ap.add_argument(
        "--ext",
        nargs="*",
        default=None,
        help="File extensions to search (e.g., png jpg). Default: .png, .jpg",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output file (default: STDOUT). One absolute path per line",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    images_dir = Path(args.images_dir).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    exts: Tuple[str, ...]
    if args.ext:
        exts = tuple(f".{e.lower().lstrip('.')}" for e in args.ext)
    else:
        exts = DEFAULT_EXTS

    rows = _read_csv(csv_path)
    if not rows:
        raise SystemExit("No valid frame rows in the CSV")

    trigger_idx: int
    video_id: Optional[int] = cast(Optional[int], args.video_id)
    if video_id is None:
        video_id = _infer_video_id_from_images_dir(images_dir)
    if video_id is None:
        raise SystemExit(
            "CSV is expected to contain only 'video' and 'frame'. Provide --video_id or make the last segment of images_dir a numeric ID"
        )
    candidates = [i for i, r in enumerate(rows) if r.get("video") == video_id]
    if not candidates:
        raise SystemExit(f"No rows with video={video_id} found in CSV: {csv_path}")
    trigger_idx = candidates[0]
    pre = max(0, int(args.pre_frames))
    post = int(args.post_frames)
    step = max(1, int(args.frame_step))

    trigger_frame = cast(int, rows[trigger_idx]["frame"])
    print(f"[DEBUG] trigger_frame: {trigger_frame}")
    csv_min_frame = cast(int, rows[0]["frame"])
    csv_max_frame = cast(int, rows[-1]["frame"])
    mm = _detect_min_max_frame_in_images_dir(images_dir, exts)
    if mm is not None:
        min_frame_in_dir, max_frame_in_dir = mm
    else:
        min_frame_in_dir, max_frame_in_dir = csv_min_frame, csv_max_frame

    desired_count = pre + post + 1
    total_frames_in_dir = max_frame_in_dir - min_frame_in_dir + 1

    if desired_count > total_frames_in_dir:
        start_frame = min_frame_in_dir
        end_frame = max_frame_in_dir
        eff_step = 1
    else:
        eff_step = max(1, int(step))
        required_width = eff_step * (desired_count - 1)
        available_width = max_frame_in_dir - min_frame_in_dir

        if required_width > available_width:
            eff_step = (
                max(1, available_width // (desired_count - 1))
                if desired_count > 1
                else 1
            )
            required_width = eff_step * (desired_count - 1)

        start_frame = trigger_frame - pre * eff_step
        end_frame = start_frame + required_width

        if start_frame < min_frame_in_dir:
            start_frame = min_frame_in_dir
            end_frame = start_frame + required_width
        if end_frame > max_frame_in_dir:
            end_frame = max_frame_in_dir
            start_frame = end_frame - required_width

    frame_sequence: List[int] = list(range(start_frame, end_frame + 1, eff_step))

    out_lines: List[str] = []
    for frame_val in frame_sequence:
        p = _resolve_image_path(images_dir, frame_val, None, exts)
        if p is None:
            raise SystemExit(
                f"Image not found: frame={frame_val} images_dir={images_dir}"
            )
        out_lines.append(str(p))

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    else:
        print("\n".join(out_lines))


if __name__ == "__main__":
    main()
