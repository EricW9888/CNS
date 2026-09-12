"""Verify ignored MaleCNS source files and materialized graph bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.malecns_io import load_edges
from src.provenance import sha256_file, validate_materialization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    registry = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = list(registry["sources"])
    for materialization in registry["materializations"]:
        entries.extend(materialization["files"])

    checked: list[str] = []
    missing: list[str] = []
    for entry in entries:
        path = Path(entry["path"])
        if not path.exists():
            missing.append(str(path))
            continue
        actual_size = path.stat().st_size
        if actual_size != int(entry["bytes"]):
            raise SystemExit(
                f"size mismatch for {path}: {actual_size} != {entry['bytes']}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            raise SystemExit(
                f"SHA-256 mismatch for {path}: {actual_hash} != {entry['sha256']}"
            )
        checked.append(str(path))

    graph_contracts = 0
    for path in sorted(Path("data").glob("**/manifest.json")):
        nodes_path = path.with_name("nodes.parquet")
        edges_path = path.with_name("edges.parquet")
        if not nodes_path.exists() or not edges_path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_materialization(
            manifest,
            pd.read_parquet(nodes_path),
            load_edges(edges_path),
        )
        graph_contracts += 1

    if missing and not args.allow_missing:
        raise SystemExit(
            "missing audited data files; rerun with --allow-missing for a partial "
            f"installation: {missing}"
        )
    print(
        json.dumps(
            {
                "dataset": registry["dataset"],
                "files_checked": len(checked),
                "graph_contracts_checked": graph_contracts,
                "missing": missing,
                "status": "ok" if not missing else "partial",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
