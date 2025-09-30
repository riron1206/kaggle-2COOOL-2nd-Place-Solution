from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2


DEFAULT_INPUT_DIRS: Tuple[str, str] = (
    "<Competition Data>/heatmaps",
    "<Competition Data>/videos",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export mp4 to PNG images (default: all frames)"
    )
    parser.add_argument(
        "--input-dirs",
        nargs="*",
        default=list(DEFAULT_INPUT_DIRS),
        help="Directories to search for mp4 files (space-separated for multiple)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Base output directory. If not specified, create <stem>_frames/ "
            "alongside each mp4"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help=(
            "Sampling FPS. 0 or omitted to export all source frames. "
            "e.g., 2 -> save 2 images per second"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively (default: recursive)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Search only the top level of input directories",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files if they already exist",
    )
    return parser.parse_args(argv)


def iter_mp4_files(
    input_dirs: Iterable[Path], recursive: bool = True
) -> Iterable[Path]:
    for d in input_dirs:
        if not d.exists():
            continue
        yield from d.rglob("*.mp4") if recursive else d.glob("*.mp4")


def _relative_to_any(path: Path, roots: Iterable[Path]) -> Optional[Tuple[Path, Path]]:
    for root in roots:
        try:
            rel = path.relative_to(root)
            return root, rel
        except ValueError:
            continue
    return None


def resolve_output_dir(
    mp4_path: Path, output_root: Optional[Path], input_roots: Iterable[Path]
) -> Path:
    if output_root is None:
        return mp4_path.parent / f"{mp4_path.stem}_frames"

    rel_info = _relative_to_any(mp4_path, input_roots)
    if rel_info is not None:
        root, rel = rel_info
        return output_root / root.name / rel.parent / mp4_path.stem

    return output_root / mp4_path.stem


def compute_stride(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        return 1
    if source_fps <= 0:
        return 1
    stride = round(max(source_fps / target_fps, 1.0))
    return max(int(stride), 1)


def extract_frames(
    mp4_path: Path, output_dir: Path, target_fps: float, overwrite: bool
) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[WARN] Failed: could not open -> {mp4_path}")
        return 0, 0

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    stride = compute_stride(source_fps, target_fps)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    total = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        should_save = frame_idx % stride == 0
        if should_save:
            filename = f"{frame_idx + 1:06d}.png"
            out_path = output_dir / filename
            if out_path.exists() and not overwrite:
                pass
            else:
                ok_write = cv2.imwrite(str(out_path), frame)
                if ok_write:
                    saved += 1
                else:
                    print(f"[WARN] Failed to write: {out_path}")
        frame_idx += 1
        total += 1

    cap.release()
    return saved, total


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_dirs = [Path(p).expanduser() for p in args.input_dirs]
    output_root = Path(args.output_root).expanduser() if args.output_root else None

    existing_dirs = [d for d in input_dirs if d.exists()]
    if not existing_dirs:
        print(
            "[ERROR] No valid input directories found. Please check --input-dirs."
        )
        return 2

    mp4_files = list(iter_mp4_files(existing_dirs, recursive=args.recursive))
    if not mp4_files:
        print(
            "[INFO] No mp4 files found. Check search targets and recursion settings."
        )
        return 0

    print(
        f"[INFO] Target mp4 count: {len(mp4_files)}, output: {'alongside each mp4' if output_root is None else str(output_root)}"
    )
    if args.fps and args.fps > 0:
        print(f"[INFO] Sampling FPS: {args.fps}")
    else:
        print("[INFO] Exporting all frames")

    grand_saved = 0
    grand_total = 0
    for i, mp4 in enumerate(sorted(mp4_files)):
        out_dir = resolve_output_dir(mp4, output_root, existing_dirs)
        print(f"[{i + 1}/{len(mp4_files)}] {mp4} → {out_dir}")
        saved, total = extract_frames(mp4, out_dir, args.fps, args.overwrite)
        print(f"    Saved: {saved} / Read: {total}")
        grand_saved += saved
        grand_total += total

    print(f"[DONE] Total saved: {grand_saved} / Total read frames: {grand_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
