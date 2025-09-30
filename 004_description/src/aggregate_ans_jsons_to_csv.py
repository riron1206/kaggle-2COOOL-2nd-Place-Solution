from __future__ import annotations
import argparse
from pathlib import Path
import json
import csv
import glob
from typing import Any, List, Sequence, Union


HEADER: Sequence[str] = [
    "video",
    "Incident window start frame",
    "Incident Detection",
    "Crash Severity",
    "Ego-car involved",
    "Label",
    "Number of Bicyclists/Scooters",
    "Number of animals involved",
    "Number of pedestrians involved",
    "Number of vehicles involved (excluding ego-car)",
    "Caption Before Incident",
    "Reason of Incident",
]


NUMERIC_KEYS = {
    "video",
    "Incident window start frame",
    "Incident Detection",
    "Ego-car involved",
    "Number of Bicyclists/Scooters",
    "Number of animals involved",
    "Number of pedestrians involved",
    "Number of vehicles involved (excluding ego-car)",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_value(value: Any) -> Union[str, int, float]:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def coerce_value_by_key(key: str, value: Any) -> Union[str, int, float]:
    if key in NUMERIC_KEYS:
        if value is None:
            return ""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                return ""
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                try:
                    return int(s)
                except Exception:
                    pass
            try:
                return float(s)
            except Exception:
                return s
        return json.dumps(value, ensure_ascii=False)
    return coerce_value(value)


def ensure_required_keys(data: dict, path: Path) -> None:
    missing = [k for k in HEADER if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in {path}: {missing}")


def collect_rows(root_spec: str) -> List[list]:
    if any(ch in root_spec for ch in "*?[]"):
        root_dirs = [Path(p) for p in glob.glob(root_spec, recursive=True)]
        root_dirs = [d for d in root_dirs if d.is_dir()]
    else:
        d = Path(root_spec)
        root_dirs = [d] if d.is_dir() else []

    files: List[Path] = []
    for rd in sorted(set(root_dirs)):
        files.extend(rd.rglob("ans.json"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No ans.json found under: {root_spec}")

    rows: List[list] = []
    for p in files:
        try:
            data = read_json(p)
        except Exception as e:
            raise ValueError(f"Failed to read JSON: {p}") from e
        ensure_required_keys(data, p)
        row = [coerce_value_by_key(k, data[k]) for k in HEADER]
        rows.append(row)

    def _video_key(v: Any):
        try:
            if isinstance(v, str):
                sv = v.strip()
                if sv.isdigit():
                    return (0, int(sv))
                return (1, sv)
            if isinstance(v, (int, float)):
                return (0, int(v))
            return (2, str(v))
        except Exception:
            return (3, str(v))

    rows.sort(key=lambda r: _video_key(r[0]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/shingo_yokoi/workspace/k/2coool_v2/sandbox/results/run_glm45v_multi_image_select_frames_from_csv_infer_vllm/**/",
        help="Root directory to recursively search for ans.json",
    )
    ap.add_argument(
        "--output_csv",
        default="/home/shingo_yokoi/workspace/k/2coool_v2/sandbox/results/run_glm45v_multi_image_select_frames_from_csv_infer_vllm/submit.csv",
        help="Path to output CSV",
    )
    args = ap.parse_args()

    out_csv = Path(args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args.root)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
