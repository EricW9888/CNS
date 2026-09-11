"""Audit the raw MaleCNS body-annotation schema without guessing a denominator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def counts(frame: pd.DataFrame, column: str, limit: int = 30) -> dict[str, int]:
    values = frame[column].astype("object").where(frame[column].notna(), "<NA>")
    return {str(key): int(value) for key, value in values.value_counts().head(limit).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/body-annotations.feather")
    parser.add_argument("--published-count", type=int, default=166_691)
    args = parser.parse_args()

    frame = pd.read_feather(args.annotations)
    if "bodyId" not in frame.columns:
        raise ValueError("The annotation export has no bodyId field")
    frame["bodyId"] = frame["bodyId"].astype("int64")
    required = ["status", "statusLabel", "superclass", "class", "type", "instance"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected classification fields: {missing}")

    status_type = (
        frame.assign(type_nonnull=frame["type"].notna())
        .groupby(["status", "type_nonnull"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    report = {
        "file": str(Path(args.annotations)),
        "rows": int(len(frame)),
        "unique_bodyId": int(frame["bodyId"].nunique()),
        "duplicate_bodyId_rows": int(frame["bodyId"].duplicated().sum()),
        "published_count_used_for_comparison": args.published_count,
        "raw_rows_minus_published_count": int(len(frame) - args.published_count),
        "null_counts": {column: int(frame[column].isna().sum()) for column in required},
        "value_counts": {column: counts(frame, column) for column in required},
        "status_by_type_presence": status_type.to_dict(orient="records"),
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
