import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_all_experiment_records_are_valid_and_uniquely_identified():
    records = []
    for path in sorted((ROOT / "experiments").glob("*/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["experiment_id"]
        assert record["status"]
        records.append(record)
    identifiers = [record["experiment_id"] for record in records]
    assert len(records) == 9
    assert len(identifiers) == len(set(identifiers))


def test_historical_failure_and_invalid_attempt_remain_explicit():
    exp001 = json.loads(
        (ROOT / "experiments/EXP-001-adjacent-column-motion/record.json").read_text()
    )
    invalid = json.loads(
        (ROOT / "experiments/EXP-004-preflight-failure/record.json").read_text()
    )
    assert exp001["status"] == "inconclusive"
    assert exp001["result"]["t4"].startswith("No T4 neuron")
    assert invalid["status"] == "INVALID_IMPLEMENTATION"
    assert invalid["preflight"]["zero_input_decay_passed"] is False
    assert invalid["preflight"]["electrical_coupling_semantics_passed"] is False


def test_invalid_exp004_is_not_an_active_source_module():
    assert not (ROOT / "src/exp004.py").exists()
    assert not (ROOT / "scripts/run_exp004.py").exists()
    assert (ROOT / "experiments/EXP-004-preflight-failure/archive/exp004.py").exists()


def test_reader_facing_docs_use_research_facing_names():
    reader_paths = [ROOT / "README.md", *sorted((ROOT / "reports").glob("*.md"))]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in reader_paths)
    assert "PROJECT_AUDIT" not in combined
    assert "claim ledger" not in combined.lower()
    assert not list((ROOT / "reports").glob("*audit*"))


def test_data_registry_has_unique_paths_and_sha256_digests():
    registry = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    entries = list(registry["sources"])
    for materialization in registry["materializations"]:
        entries.extend(materialization["files"])
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    assert all(path.startswith("data/") for path in paths)
    assert all(len(entry["sha256"]) == 64 for entry in entries)
    assert all(int(entry["bytes"]) > 0 for entry in entries)


def test_local_markdown_links_resolve():
    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    missing = []
    for path in sorted(ROOT.glob("**/*.md")):
        if any(
            part in {".git", ".venv", ".local", "data", "results"}
            for part in path.parts[:-1]
        ):
            continue
        for target in link_pattern.findall(path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).resolve().exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not missing, missing
