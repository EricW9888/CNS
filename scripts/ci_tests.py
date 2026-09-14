"""Partition existing test files exactly once; no extra platform/matrix."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "frozen-integrity": {"test_repository_records.py", "test_provenance.py", "test_exp004_sources.py",
                         "test_exp004_record.py", "test_exp005_record.py", "test_exp006_record.py", "test_experiment_registry.py"},
    "replay-smoke": {"test_exp008_observer.py", "test_observability.py"}}


def partition():
    files = {p.name for p in (ROOT/"tests").glob("test_*.py")}
    named = set().union(*GROUPS.values())
    if not named <= files or len(named) != sum(len(group) for group in GROUPS.values()):
        raise ValueError("CI test groups contain missing or duplicate files")
    return {"model-invariants": sorted(files-named), **{key: sorted(value) for key, value in GROUPS.items()}}


if __name__ == "__main__":
    groups = partition()
    if len(sys.argv) != 2 or sys.argv[1] not in groups:
        raise SystemExit(f"usage: ci_tests.py {'|'.join(groups)}")
    subprocess.run([sys.executable, "-m", "pytest", "-q", *[str(ROOT/"tests"/p) for p in groups[sys.argv[1]]]], cwd=ROOT, check=True)
