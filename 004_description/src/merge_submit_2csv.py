import argparse
from pathlib import Path

import pandas as pd


def merge_by_incident_start_frame(
    sample_csv_path: Path, full_csv_path: Path, output_csv_path: Path
) -> int:

    df_sample = pd.read_csv(sample_csv_path)
    df_full = pd.read_csv(full_csv_path)

    if "video" not in df_sample.columns or "video" not in df_full.columns:
        raise ValueError("Both CSVs must contain a 'video' column.")

    df_sample = df_sample.set_index("video", drop=False)
    df_full = df_full.set_index("video", drop=False)

    for col in df_sample.columns:
        if col not in df_full.columns:
            df_full[col] = pd.NA
    for col in df_full.columns:
        if col not in df_sample.columns:
            df_sample[col] = pd.NA
    df_full = df_full[df_sample.columns]

    col_sf = "Incident window start frame"
    col_id = "Incident Detection"
    for required in (col_sf, col_id):
        if required not in df_sample.columns or required not in df_full.columns:
            raise ValueError(f"Column '{required}' must exist in both CSVs.")

    sample_mask = (df_sample[col_sf] == 1) & (df_sample[col_id] == -1)

    full_diff_mask = ~((df_full[col_sf] == 1) & (df_full[col_id] == -1))
    full_diff_mask = full_diff_mask.reindex(df_sample.index).fillna(False)

    replace_mask = sample_mask & full_diff_mask
    replace_index = df_sample.index[replace_mask]

    if len(replace_index) > 0:
        df_sample.loc[replace_index] = df_full.loc[replace_index].values

    df_sample.to_csv(output_csv_path, index=False)

    return int(len(replace_index))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace rows in sample CSV (where Incident window start frame == 1) "
            "with rows from full CSV when full CSV has that value != 1, matched by 'video'."
        )
    )
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="Path to exp001_n_sample.csv",
    )
    parser.add_argument(
        "--full",
        type=Path,
        required=True,
        help="Path to exp001.csv (source of replacements)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        help="Output CSV path (default: <sample>.merged.csv)",
    )

    args = parser.parse_args()
    output_path = args.out or args.sample.with_suffix(".merged.csv")

    replaced = merge_by_incident_start_frame(args.sample, args.full, output_path)
    print(
        f"Wrote {output_path} (replaced {replaced} rows where (Incident window start frame, Incident Detection) changed from (1, -1))"
    )


if __name__ == "__main__":
    main()
