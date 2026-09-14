"""Create a compact, content-addressed manifest for the EXP-011 archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "results/exp011_sensors_direct"
DEFAULT_OUTPUT = ROOT / "experiments/EXP-011-acceptance-convergence/archive-manifest.json"
FILES = tuple(sorted(ARCHIVE_ROOT.glob("*.npz")))


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument("--release-tag", default="exp011-canonical-archives-2026-09-14")
    parser.add_argument("--repository", default="EricW9888/CNS")
    parser.add_argument("--release-asset", default="exp011-canonical-archives-direct.tar")
    args = parser.parse_args()
    if len(FILES) != 15:
        raise RuntimeError(f"expected 15 canonical archives, found {len(FILES)}")

    entries = []
    for path in FILES:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(data["metadata"].item())
            entries.append({
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "fields": sorted(data.files),
                "shapes": {key: list(data[key].shape) for key in data.files},
                "dtypes": {key: str(data[key].dtype) for key in data.files},
                "light_min": float(data["light"].min()),
                "light_max": float(data["light"].max()),
                "blocked_mass_min": float(data["blocked_mass"].min()),
                "blocked_mass_max": float(data["blocked_mass"].max()),
                "blocked_mass_count_above_one": int(np.count_nonzero(data["blocked_mass"] > 1.0)),
                "metadata": metadata,
            })

    total = sum(entry["bytes"] for entry in entries)
    bundle = None
    if args.bundle_path is not None:
        bundle = {
            "path": str(args.bundle_path),
            "name": args.release_asset,
            "bytes": args.bundle_path.stat().st_size,
            "sha256": sha256(args.bundle_path),
        }
    report = {
        "experiment_id": "EXP-011-acceptance-convergence",
        "purpose": "Immutable review manifest for the exact canonical direct-execution sensor archives; binaries are not committed to ordinary Git history.",
        "archive_count": len(entries),
        "total_bytes": total,
        "total_gib": total / (1024 ** 3),
        "git_retention_decision": "archive binaries omitted from Git because the set is 779139958 bytes and three individual files exceed GitHub's ordinary 100 MiB file limit",
        "source_commit": "5e9ec94d9c982615441ab67f1ffca9f9d783c636",
        "archive_root": str(ARCHIVE_ROOT),
        "release_storage": {
            "repository": args.repository,
            "release_tag": args.release_tag,
            "asset_name": args.release_asset,
            "asset_url": f"https://github.com/{args.repository}/releases/download/{args.release_tag}/{args.release_asset}",
            "bundle": bundle,
        },
        "archives": entries,
        "compact_diagnostics": [
            "visibility-diagnosis.json",
            "canonical-local-result.json",
            "direct-path-preflight.json",
            "direct-execution-benchmark.json",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"archive_count": len(entries), "total_bytes": total, "bundle": bundle}, indent=2))


if __name__ == "__main__":
    main()
