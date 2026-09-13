import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "experiments/EXP-005-sensory-to-DNp15"


def test_new_record_preserves_provisional_scope_and_negative_mechanism_result():
    record = json.loads((FOLDER / "record.json").read_text())
    spec = json.loads((FOLDER / "specification.json").read_text())
    metrics = record["metrics"]
    assert record["status"] == "completed_observable_provisional_chain"
    assert not any(record["scope"].values())
    assert spec["parent_commit"] == "15547c9"
    assert metrics["specification_sha256"] == hashlib.sha256((FOLDER / "specification.json").read_bytes()).hexdigest()
    assert spec["central_parameters"] == json.loads((ROOT / spec["central_specification"]["path"]).read_text())["parameters"]
    assert metrics["chain_anatomy"] == {"node_count": 7599, "edge_count": 24233, "visual_edges": 23873, "frozen_central_edges": 360}
    assert metrics["input_eye_verification"]["verified_bodies"] == 7556
    assert metrics["input_eye_verification"]["dendritic_ROI_absent_body_ids"] == [152777]
    assert metrics["sensory_to_both_DNp15_connected"]
    assert metrics["qualitative_target"]["limited_neural_transformation_gate"]
    assert not metrics["qualitative_target"]["recurrent_enhancement_contract_passed"]
    assert metrics["qualitative_target"]["full_reduces_both_DN_ratios_vs_feedforward"] == [False, False]
    assert all(v["published_preference_preserved"] for v in metrics["HS_H2_preferences"].values())
    assert not metrics["external_comparison"]["H2"]["inside_descriptive_roi_interval"]
    assert not metrics["external_comparison"]["H2rn"]["inside_descriptive_roi_interval"]
    assert (ROOT / record["outputs"]["figure"]).is_file()


def test_each_control_and_timestep_meets_numerical_and_survivor_contracts():
    spec = json.loads((FOLDER / "specification.json").read_text())
    metrics = json.loads((FOLDER / "record.json").read_text())["metrics"]
    assert tuple(metrics["preflight"]) == tuple(spec["controls"])
    assert metrics["deterministic_repeats"]
    for control, p in metrics["preflight"].items():
        assert p["surviving_weights_unchanged"]
        for stage in ["central", "half_dt_central", "LPi", "half_dt_LPi"]:
            assert p[stage]["passed"]
            assert p[stage]["global_euler_lipschitz_inf_bound"] < 1
            assert max(p[stage]["zero_input_2000ms_final_initial_inf_ratios"]) < 1e-8
        assert p["timestep_convergence"]["passed"]
        assert p["timestep_convergence"]["max_abs_late_index_difference"] <= .02
        for stages in p["timestep_convergence"]["relative_trace_errors"].values():
            assert set(stages) == {"channels", "LPi", "drive", "stimulus_luminance", "central"}
            assert max(stages.values()) <= .05
        assert metrics["responses"][control]["zero_condition_max_abs_state"] == 0
    disconnected = metrics["responses"]["visual_disconnected"]
    for row in disconnected["DNp15_per_body"]:
        assert row["di"] is None and row["magnitude_di"] is None
        for condition in spec["stimuli"]:
            assert row[condition] == 0


def test_source_and_trace_regressions():
    metrics = json.loads((FOLDER / "record.json").read_text())["metrics"]
    for path, digest in metrics["implementation_LF_sha256"].items():
        # Source hashes use canonical LF, surviving normal Windows checkout.
        assert hashlib.sha256((ROOT / path).read_text(encoding="utf8").replace("\r\n", "\n").encode()).hexdigest() == digest
    assert metrics["trace_arrays_sha256"] == "cfb72baa3599b6de5132dc9cfefd924ad2392d0d983b80f629032c9621934857"
    full = metrics["responses"]["full"]
    np.testing.assert_allclose(full["populations"]["DNp15"]["mean_di"], -.4818956278082758, atol=1e-14)
    assert len(metrics["stage_late_means"]["full"]["yaw_L_F"]["channels"]) == 16
    assert len(metrics["stage_late_means"]["full"]["yaw_L_F"]["LPi"]) == 6
