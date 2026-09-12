"""Fast tracked-tree hygiene checks for local use and CI."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATA = {"data/README.md", "data/manifest.json"}
ALLOWED_RESULTS = {"results/.gitkeep"}
FORBIDDEN_SUFFIXES = {
    ".feather",
    ".parquet",
    ".xlsx",
    ".xls",
    ".csv",
    ".mp4",
    ".gif",
    ".key",
    ".pem",
    ".token",
}
PERSONAL_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\[^\\]+", re.IGNORECASE),
    re.compile(r"ericw9888@gmail\.com", re.IGNORECASE),
)
MAX_TRACKED_BYTES = 5 * 1024 * 1024


def tracked_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [path for path in output.split("\0") if path]


def main() -> None:
    errors: list[str] = []
    paths = tracked_files()
    for relative in paths:
        path = ROOT / relative
        normalized = relative.replace("\\", "/")
        if normalized.startswith("data/") and normalized not in ALLOWED_DATA:
            errors.append(f"tracked data artifact: {normalized}")
        if normalized.startswith("results/") and normalized not in ALLOWED_RESULTS:
            errors.append(f"tracked bulk result: {normalized}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden tracked suffix: {normalized}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 5 MiB: {normalized}")
        if path.suffix.lower() in {".md", ".py", ".json", ".txt", ".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8")
            for pattern in PERSONAL_PATTERNS:
                if pattern.search(text):
                    errors.append(f"personal path or email in {normalized}")
                    break

    for record_path in sorted((ROOT / "experiments").glob("*/record.json")):
        try:
            json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid experiment record {record_path}: {exc}")

    for forbidden in (ROOT / "src/exp004.py", ROOT / "scripts/run_exp004.py"):
        if forbidden.exists():
            errors.append(f"invalid provisional EXP-004 is active: {forbidden}")

    if errors:
        raise SystemExit("repository policy violations:\n- " + "\n- ".join(errors))
    print(f"repository policy: ok ({len(paths)} tracked files)")


if __name__ == "__main__":
    main()
