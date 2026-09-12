import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_experiment_records_are_valid_and_uniquely_identified():
    records = []
    for path in sorted((ROOT / "experiments").glob("*/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["experiment_id"]
        assert record["status"]
        records.append(record)
    identifiers = [record["experiment_id"] for record in records]
    assert len(records) == 6
    assert len(identifiers) == len(set(identifiers))


def test_historical_failure_and_invalid_attempt_remain_explicit():
    exp001 = json.loads(
        (ROOT / "experiments/EXP-001-adjacent-column-motion/record.json").read_text()
    )
    invalid = json.loads(
        (ROOT / "experiments/EXP-004-invalid-provisional/record.json").read_text()
    )
    assert exp001["status"] == "inconclusive"
    assert exp001["result"]["t4"].startswith("No T4 neuron")
    assert invalid["status"] == "INVALID_IMPLEMENTATION"
    assert invalid["preflight"]["zero_input_decay_passed"] is False
    assert invalid["preflight"]["electrical_coupling_semantics_passed"] is False


def test_invalid_exp004_is_not_an_active_source_module():
    assert not (ROOT / "src/exp004.py").exists()
    assert not (ROOT / "scripts/run_exp004.py").exists()
    assert (ROOT / "experiments/EXP-004-invalid-provisional/archive/exp004.py").exists()
