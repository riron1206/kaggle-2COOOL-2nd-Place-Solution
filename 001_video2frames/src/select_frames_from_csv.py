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


def _is_none_like(s: object) -> bool:
    if s is None:
        return True
    t = str(s).strip().lower()
    return t in {"", "none", "null", "na", "n/a", "node"}


def _read_csv(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            frame = _to_int_or_none(row.get("frame"))
            inc = _to_int_or_none(row.get("Incident_Detection"))
            hazard = row.get("Hazard_Clues")
            path_field = row.get("path")
            bicy = _to_int_or_none(row.get("Number_of_Bicyclists_Scooters")) or 0
            anim = _to_int_or_none(row.get("Number_of_Animals")) or 0
            ped = _to_int_or_none(row.get("Number_of_Pedestrians")) or 0
            veh = _to_int_or_none(row.get("Number_of_Vehicles_excluding_ego")) or 0

            rows.append(
                {
                    "frame": frame,
                    "Incident_Detection": inc,
                    "Hazard_Clues": hazard,
                    "path": path_field,
                    "_sum_counts": bicy + anim + ped + veh,
                    "_raw": row,
                }
            )

    rows = [r for r in rows if r["frame"] is not None]
    rows.sort(key=lambda r: int(r["frame"]))
    return rows


def _find_trigger_index(rows: Sequence[Dict[str, Any]]) -> int:
    for idx, r in enumerate(rows):
        inc = cast(Optional[int], r.get("Incident_Detection"))
        if inc is not None and inc != -1:
            return idx

    for idx, r in enumerate(rows):
        if not _is_none_like(r.get("Hazard_Clues")):
            return idx

    best_idx = 0
    best_sum = -1
    for idx, r in enumerate(rows):
        s = cast(int, r.get("_sum_counts", 0))
        if s > best_sum:
            best_sum = s
            best_idx = idx
    return best_idx


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
        description="List images for frame ranges based on CSV conditions"
    )
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument(
        "--images_dir", required=True, help="Directory containing corresponding images"
    )
    ap.add_argument(
        "--pre_frames", type=int, default=6, help="How many rows before the trigger to start"
    )
    ap.add_argument(
        "--post_frames",
        type=int,
        default=5,
        help="How many rows after the start to include (0 = to the end)",
    )
    ap.add_argument(
        "--ext",
        nargs="*",
        default=None,
        help="File extensions to search (e.g., png jpg). Default: png/jpg/jpeg/webp/bmp",
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
        raise SystemExit("No valid 'frame' rows in CSV")

    trigger_idx = _find_trigger_index(rows)
    pre = max(0, int(args.pre_frames))
    post = int(args.post_frames)

    rows_len = len(rows)
    if post > 0:
        required = pre + post + 1
        if rows_len < required:
            selected = rows[:]
        else:
            start_idx = trigger_idx - pre
            end_idx = trigger_idx + post

            if start_idx < 0:
                shift = -start_idx
                start_idx = 0
                end_idx = end_idx + shift

            if end_idx >= rows_len:
                shift = end_idx - (rows_len - 1)
                end_idx = rows_len - 1
                start_idx = start_idx - shift

            current = end_idx - start_idx + 1
            if current > required:
                start_idx = end_idx - required + 1
            elif current < required:
                need = required - current
                start_idx = max(0, start_idx - need)
                end_idx = min(rows_len - 1, start_idx + required - 1)

            selected = rows[start_idx : end_idx + 1]
    else:
        start_idx = max(0, trigger_idx - pre)
        selected = rows[start_idx:]

    out_lines: List[str] = []
    for r in selected:
        frame_val = cast(Optional[int], r.get("frame"))
        if frame_val is None:
            continue
        path_field = cast(Optional[str], r.get("path"))
        p = _resolve_image_path(images_dir, frame_val, path_field, exts)
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
