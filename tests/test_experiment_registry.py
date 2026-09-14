from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.ci_tests import partition
from scripts.verify_experiment import (ROOT, evidence_assertion, integrity,
                                       ignored_local_artifact, pointer, repository_path, valid_test_reference, validate_registry)


def test_registry_and_headline_evidence_cover_frozen_results():
    entries = validate_registry()
    assert len(entries) == 13
    assert entries["EXP-009-retinotopic-eye"]["status"] == "retinotopic_chain_validation_incomplete"
    assert entries["EXP-010-retinotopic-refinement"]["status"] == "sampling_validation_incomplete"
    for entry in entries.values():
        assert integrity(entry, allow_missing=True)["replay"] == "not_executed"


@pytest.mark.parametrize("change", ["status", "question", "predecessor", "coverage", "cycle", "write_record"])
def test_registry_rejects_drift_and_mutating_verify_commands(change):
    registry = json.loads((ROOT/"experiments/registry.json").read_text())
    if change == "coverage":
        registry["experiments"].pop()
    elif change == "cycle":
        registry["experiments"][0]["predecessor"] = registry["experiments"][0]["experiment_id"]
    elif change == "write_record":
        next(e for e in registry["experiments"] if e["verification"]["kind"] == "exact_replay")["verification"]["replay_command"].append("--write-record")
    else:
        registry["experiments"][0][change] = "invented"
    with pytest.raises(ValueError):
        validate_registry(registry)


def test_claims_cannot_turn_failed_validation_into_success_or_weaken_evidence():
    claims = json.loads((ROOT/"experiments/evidence.json").read_text())
    original = deepcopy(claims)
    claim = next(c for c in claims["claims"] if c["claim_id"] == "EXP-009.full_eye_retinotopy_validated")
    claim["value"] = True
    with pytest.raises(ValueError):
        validate_registry(claims_document=claims)
    claim = next(c for c in original["claims"] if c["claim_id"] == "EXP-010.temporal_refinement_validates_interface")
    claim["evidence"][0]["greater_than"] = .5
    with pytest.raises(ValueError):
        validate_registry(claims_document=original)


def test_json_pointer_escape_list_and_boolean_contract():
    assert pointer({"a/b": {"~c": [False, 3]}}, "/a~1b/~0c/1") == 3
    assert evidence_assertion(False, {"equals": False})
    assert not evidence_assertion(0, {"equals": False})
    assert evidence_assertion(0., {"equals": 0})
    assert not evidence_assertion(float("nan"), {"greater_than": .05})
    with pytest.raises(ValueError):
        pointer({}, "a")
    with pytest.raises(KeyError):
        pointer({}, "/missing")
    with pytest.raises(ValueError):
        repository_path("../outside.json")
    assert not valid_test_reference("tests/test_experiment_registry.py::missing_test")


def test_integrity_detects_changed_record_before_replay(tmp_path, monkeypatch):
    import scripts.verify_experiment as verifier
    entry = validate_registry()["EXP-010-retinotopic-refinement"]
    path = tmp_path/"record.json"
    path.write_text(repository_path(entry["record"]).read_text().replace('"validation_passed": false', '"validation_passed": true'))
    original = verifier.repository_path
    monkeypatch.setattr(verifier, "repository_path", lambda name: path if name == entry["record"] else original(name))
    with pytest.raises(ValueError, match="frozen artifact changed"):
        integrity(entry)


def test_partial_install_reports_absent_ignored_results_without_claiming_replay(tmp_path, monkeypatch):
    import scripts.verify_experiment as verifier
    entry = validate_registry()["EXP-007-CvNA2-motor-boundary"]
    source = "results/exp006_neck/stage_traces.npz"
    original = verifier.repository_path
    monkeypatch.setattr(verifier, "repository_path", lambda name: tmp_path/"absent.npz" if name == source else original(name))
    result = integrity(entry, allow_missing=True)
    assert source in result["missing_local_artifacts"]
    assert result["replay"] == "not_executed"
    with pytest.raises(FileNotFoundError, match="required source absent"):
        integrity(entry)
    assert ignored_local_artifact(source)
    assert ignored_local_artifact("data/absent-source.feather")
    assert not ignored_local_artifact("data/manifest.json")
    assert not ignored_local_artifact("src/absent.py")


def test_partial_install_never_skips_present_but_corrupt_ignored_results(tmp_path, monkeypatch):
    import scripts.verify_experiment as verifier
    entry = validate_registry()["EXP-007-CvNA2-motor-boundary"]
    source = "results/exp006_neck/stage_traces.npz"
    corrupt = tmp_path/source
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not the frozen trace")
    original = verifier.repository_path
    digest = verifier.frozen_entry_digest
    monkeypatch.setattr(verifier, "repository_path", lambda name: corrupt if name == source else original(name))
    monkeypatch.setattr(verifier, "frozen_entry_digest", lambda item: digest(item, root=tmp_path) if item["path"] == source else digest(item))
    with pytest.raises(ValueError, match="source fingerprint differs"):
        integrity(entry, allow_missing=True)


def test_partial_install_handles_clean_checkout_without_local_bundles(tmp_path, monkeypatch):
    import scripts.verify_experiment as verifier
    original = verifier.repository_path
    monkeypatch.setattr(verifier, "repository_path", lambda name: tmp_path/name if ignored_local_artifact(name) else original(name))
    for entry in validate_registry().values():
        assert integrity(entry, allow_missing=True)["replay"] == "not_executed"


def test_ci_partitions_every_test_file_exactly_once():
    groups = partition()
    all_files = [file for group in groups.values() for file in group]
    assert len(all_files) == len(set(all_files))
    assert set(all_files) == {p.name for p in (ROOT/"tests").glob("test_*.py")}
    assert all(groups.values())


def test_template_contains_no_generated_science_or_results():
    template = json.loads((ROOT/"experiments/successor-template.json").read_text())
    assert template["template_only"]
    assert template["question"] is None and template["experiment_id"] is None
    assert template["evidence"]["free_parameters"] == {}
    assert template["record_contract"]["status"] is None
    assert template["acceptance"]["primary_criterion"] is None
