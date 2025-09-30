from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_LEFT_DIR = "<Competition Data>/heatmaps"
DEFAULT_RIGHT_DIR = "<Competition Data>/videos"
DEFAULT_OUT_DIR = "<output>/mp4_vstack"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertically stack mp4 files with the same name (ffmpeg required, hardened)"
    )
    parser.add_argument("--left-dir", type=Path, default=Path(DEFAULT_LEFT_DIR),
                        help="Directory containing top (heatmap) mp4 files")
    parser.add_argument("--right-dir", type=Path, default=Path(DEFAULT_RIGHT_DIR),
                        help="Directory containing bottom (video) mp4 files")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR),
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--width", type=int, default=None,
                        help="When specified, scale both videos to this width. If omitted, auto-adjust may occur (disable with --no-auto-width)")
    parser.add_argument("--fps", type=str, default=None,
                        help="Normalize output frame rate (e.g., 30, 29.97)")
    parser.add_argument("--crf", type=int, default=None,
                        help="x264 CRF value. If omitted, preserve source bitrate")
    parser.add_argument("--preset", type=str, default="veryfast",
                        help="x264 preset (ultrafast to veryslow, default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output file if it already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show targets only without execution")
    parser.add_argument("--validate", choices=["none", "ffprobe", "decode"], default="ffprobe",
                        help="Validation method for output mp4 (default: ffprobe)")
    parser.add_argument("--retries", type=int, default=0,
                        help="Number of retries when ffmpeg fails (default: 0)")
    parser.add_argument("--ffmpeg-loglevel", type=str, default="error",
                        help="ffmpeg log level (error, warning, info, debug, etc.)")
    parser.add_argument("--no-auto-width", dest="auto_width", action="store_false", default=True,
                        help="Disable auto width alignment when width is unspecified")
    return parser.parse_args()


def find_mp4_map(directory: Path) -> Dict[str, Path]:
    mp4_map: Dict[str, Path] = {}
    if not directory.exists():
        return mp4_map
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".mp4":
            mp4_map[path.name] = path
    return mp4_map


def run_subprocess(cmd: List[str]) -> Tuple[int, str]:
    try:
        completed = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True
        )
        return completed.returncode, completed.stdout
    except FileNotFoundError as exc:
        return 127, str(exc)

def ffprobe_json(path: Path) -> Optional[dict]:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    code, out = run_subprocess(cmd)
    if code != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None

def probe_width_height(path: Path) -> Tuple[Optional[int], Optional[int]]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path),
    ]
    code, out = run_subprocess(cmd)
    if code != 0:
        return None, None
    try:
        w_str, h_str = out.strip().split(",")
        w = int(w_str); h = int(h_str)
        return w, h
    except Exception:
        return None, None

def probe_video_bitrate_bps(video_path: Path) -> Optional[int]:
    r = subprocess.run(
        ["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=bit_rate",
         "-of","default=noprint_wrappers=1:nokey=1", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
    )
    val = r.stdout.strip()
    if val.isdigit() and int(val) > 0:
        return int(val)

    r2 = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=bit_rate",
         "-of","default=noprint_wrappers=1:nokey=1", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
    )
    val2 = r2.stdout.strip()
    if val2.isdigit() and int(val2) > 0:
        return int(val2)
    return None


def build_filter_complex(target_width: Optional[int], fps: Optional[str]) -> str:
    def chain(input_idx: int, out_label: str) -> str:
        steps = []
        if target_width is not None:
            steps.append(f"scale={target_width}:-1:flags=lanczos")
        steps.append("setsar=1")
        steps.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        if fps is not None:
            steps.append(f"fps=fps={fps}")
        return f"[{input_idx}:v]" + ",".join(steps) + f"[{out_label}]"

    c0 = chain(0, "v0")
    c1 = chain(1, "v1")
    return f"{c0};{c1};[v0][v1]vstack=inputs=2[v]"

def choose_target_width(args_width: Optional[int], auto_width: bool,
                        top_path: Path, bottom_path: Path) -> Optional[int]:
    if args_width is not None:
        return args_width
    if not auto_width:
        return None
    w0, _ = probe_width_height(top_path)
    w1, _ = probe_width_height(bottom_path)
    if w0 is not None and w1 is not None and w0 != w1:
        return min(w0, w1)
    return None

def build_ffmpeg_cmd(
    top_path: Path,
    bottom_path: Path,
    out_path: Path,
    target_width: Optional[int],
    crf: Optional[int],
    preset: str,
    ffmpeg_loglevel: str,
    fps: Optional[str],
) -> List[str]:
    filter_complex = build_filter_complex(target_width, fps)

    video_args: List[str] = [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
    ]
    if crf is not None:
        video_args += ["-crf", str(crf)]
    else:
        source_bitrate_bps = probe_video_bitrate_bps(bottom_path)
        if source_bitrate_bps and source_bitrate_bps > 0:
            video_args += ["-b:v", str(source_bitrate_bps)]
        else:
            video_args += ["-crf", "18"]

    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", ffmpeg_loglevel,
        "-fflags", "+genpts", "-i", str(top_path),
        "-fflags", "+genpts", "-i", str(bottom_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a?",
        *video_args,
        "-c:a", "aac",
        "-b:a", "192k",
        "-vsync", "vfr",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        "-shortest",
        "-f", "mp4",
        str(out_path),
    ]
    return cmd


def validate_mp4(path: Path, method: str) -> Tuple[bool, str]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return False, "Output file does not exist"
    if size < 1024:
        return False, f"File size is too small ({size} bytes)"

    if method == "none":
        return True, "skipped"

    if method == "ffprobe":
        info = ffprobe_json(path)
        if not info:
            return False, "ffprobe parsing failed"
        fmt = (info.get("format") or {}).get("format_name", "")
        if "mp4" not in fmt and "mov" not in fmt:
            return False, f"format_name={fmt} is not mp4/mov"
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not vstreams:
            return False, "No video stream found"
        v0 = vstreams[0]
        w = int(v0.get("width", 0) or 0)
        h = int(v0.get("height", 0) or 0)
        if w <= 0 or h <= 0:
            return False, f"Invalid dimensions width={w}, height={h}"
        dur = info.get("format", {}).get("duration")
        if dur is not None:
            try:
                if float(dur) <= 0.2:
                    return False, f"Duration is too short ({dur}s)"
            except Exception:
                pass
        return True, "ok"

    if method == "decode":
        code, out = run_subprocess(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
        if code == 0:
            return True, "ok"
        return False, f"Decode validation failed (exit={code}): {out}"

    return True, "unknown-method"


def main() -> int:
    args = parse_args()

    top_dir: Path = args.left_dir
    bottom_dir: Path = args.right_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    top_map = find_mp4_map(top_dir)
    bottom_map = find_mp4_map(bottom_dir)

    common_names = sorted(set(top_map.keys()) & set(bottom_map.keys()))
    if not common_names:
        print("No matching .mp4 filenames found.", file=sys.stderr)
        print(f"top-dir   : {top_dir}")
        print(f"bottom-dir: {bottom_dir}")
        return 1

    print(f"Target files: {len(common_names)}")
    if args.dry_run:
        for name in common_names:
            print(f"- {name}")
        return 0

    success = skipped = failed = 0

    for name in common_names:
        top_path = top_map[name]
        bottom_path = bottom_map[name]
        out_path = out_dir / name
        tmp_path = out_dir / Path(name).with_suffix(".tmp.mp4")

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Already exists (overwrite with --overwrite): {out_path}")
            skipped += 1
            continue

        target_width = choose_target_width(args.width, args.auto_width, top_path, bottom_path)

        attempts = args.retries + 1
        last_err = ""
        for attempt in range(1, attempts + 1):
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

            cmd = build_ffmpeg_cmd(
                top_path=top_path,
                bottom_path=bottom_path,
                out_path=tmp_path,
                target_width=target_width,
                crf=args.crf,
                preset=args.preset,
                ffmpeg_loglevel=args.ffmpeg_loglevel,
                fps=args.fps,
            )

            print(f"[RUN ] {name} (try {attempt}/{attempts})")
            code, output = run_subprocess(cmd)
            if code != 0:
                last_err = f"ffmpeg failed (exit={code})\n{output}"
                print(f"[FAIL] {name}: {last_err}")
                continue

            ok, reason = validate_mp4(tmp_path, args.validate)
            if not ok:
                last_err = f"Validation failed: {reason}"
                print(f"[FAIL] {name}: {last_err}")
                continue

            try:
                if out_path.exists():
                    out_path.unlink()
                tmp_path.replace(out_path)
            except Exception as e:
                last_err = f"rename failed: {e}"
                print(f"[FAIL] {name}: {last_err}")
                continue

            print(f"[DONE] {out_path}")
            success += 1
            break

        else:
            failed += 1
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            print(f"[FAIL] {name}: final failure. last error: {last_err}")

    print(f"Done: success={success}, skipped={skipped}, failed={failed}, out_dir={out_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
