import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"experiments/EXP-006-DNp15-neck-motor"


def records():
    return (json.loads((FOLDER/"specification.json").read_text()),
            json.loads((FOLDER/"record.json").read_text()))


def test_record_preserves_identity_provenance_exclusions_and_provisional_claim_boundary():
    spec,record=records()
    assert spec["parent_commit"]=="4e36468"
    assert record["status"]=="completed_provisional_open_loop_head_motor_chain"
    assert not any(record["scope"].values())
    assert spec["anatomy"]["node_count"]==6 and spec["anatomy"]["edge_count"]==6
    identity={r["bodyId"]:r for r in spec["identity"]}
    for body,manc,axon in [(813696,17653,"L"),(804343,14549,"R")]:
        row=identity[body]
        assert row["manc_body_id"]==manc and row["axon_side"]==axon
        assert row["manc_instance"].endswith("_"+axon)
        assert row["somaSide"]!=axon and row["muscle"]=="TH2"
        assert row["paper_cell"]=="CvNA2"
        assert "Table 2" in row["identity_source"] and row["muscle_source"]
    excluded=spec["structural_edges_not_dynamically_modeled"]
    assert {(r["pre_body"],r["post_body"],r["synapse_count"]) for r in excluded}=={(801678,813696,1),(804343,903152,1)}
    assert "unclear" in spec["exclusion_reason"]
    assert spec["parameters"]=={"dt_ms":.5,"tau_ms":20.,"chemical_gain":.5,
        "torque_native_per_activity":1.,"paper_body_pitch_degrees":40.}
    assert spec["body_parameters"]["actuated_joint"]=="joint_Head_roll"
    assert spec["external_target"]["quantitative_status"].startswith("No source numerical")
    assert (ROOT/record["outputs"]["figure"]).exists()
    assert (ROOT/record["outputs"]["report"]).exists()


def test_every_control_passes_preflight_and_preserves_surviving_weights():
    spec,record=records();m=record["metrics"]
    assert m["fresh_EXP005_DN_late_responses_match"]
    assert m["deterministic_repeats"] and m["disconnected_physical_matches_zero_command"]
    assert m["physical_sign_check_passed"]
    assert m["physical_sign_pulse_300ms_mean_yaw_delta_rad"][0]>0
    assert m["physical_sign_pulse_300ms_mean_yaw_delta_rad"][1]<0
    for upstream in m["frozen_upstream_preflight"].values():
        assert all(c["passed"] for c in upstream.values())
    for control in spec["controls"]:
        checks=m["preflight"][control]
        assert checks["surviving_weights_unchanged"]
        for p in [checks,checks["half_dt"]]:
            assert p["passed"] and p["motor_transition_spectral_radius_and_contraction_bound"]<1
            assert p["zero_input_2000ms_final_initial_inf_ratio"]<1e-8
        assert max(m["motor_timestep_relative_errors"][control].values())<=.02
    for body in m["body_checks"].values():
        assert body["deterministic_repeat"] and body["no_controller_or_sensory_feedback"]
        assert body["half_dt_baseline_subtracted_yaw_relative_error"]<=.02
        np.testing.assert_array_equal(body["axis_at_initial_pose"],[0,0,1])
        np.testing.assert_allclose(body["finite_angle_head_azimuth_rad"],.001,atol=1e-12)


def test_causal_control_and_recorded_response_regressions():
    spec,record=records();m=record["metrics"];r=m["responses"]
    for condition in spec["stimuli"]:
        off=r["DN_disconnected"][condition]
        assert off["motor_late"]==[0]*4 and off["mean_torque_native"]==0
        assert r["motor_disconnected"][condition]["motor_late"]==r["full"][condition]["motor_late"]
        assert r["motor_disconnected"][condition]["mean_torque_native"]==0
        assert m["body_response"][condition]["mean_torque_and_late_head_yaw_same_sign"]
    np.testing.assert_allclose(r["full"]["yaw_L_F"]["mean_torque_native"],-1.0617769598956418e-5,atol=1e-17)
    np.testing.assert_allclose(r["full"]["yaw_R_F"]["mean_torque_native"],6.198976589962221e-6,atol=1e-17)
    assert abs(r["DN_L_disconnected"]["yaw_L_F"]["mean_torque_native"])<1e-20
    assert r["DN_L_disconnected"]["yaw_R_F"]["mean_torque_native"]==r["full"]["yaw_R_F"]["mean_torque_native"]
    assert abs(r["DN_R_disconnected"]["yaw_R_F"]["mean_torque_native"])<1e-20
    assert r["DN_R_disconnected"]["yaw_L_F"]["mean_torque_native"]==r["full"]["yaw_L_F"]["mean_torque_native"]
    np.testing.assert_allclose(m["body_response"]["yaw_L_F"]["late_mean_head_yaw_delta_degrees"],-.008869024607979729,atol=1e-12)
    np.testing.assert_allclose(m["body_response"]["yaw_R_F"]["late_mean_head_yaw_delta_degrees"],.005178285072558711,atol=1e-12)
    assert m["trace_arrays_sha256"]=="169b753d284ba74c9d6102b09c800c8febbf8ab445b88114927cedfa85f6ca74"


def test_frozen_upstream_and_new_implementation_hashes():
    spec,record=records()
    assert record["metrics"]["specification_sha256"]==hashlib.sha256((FOLDER/"specification.json").read_bytes()).hexdigest()
    for entry in spec["baseline_files"]:
        if not entry["path"].endswith(".py"):
            assert hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest()==entry["sha256"]
    old=json.loads((ROOT/"experiments/EXP-005-sensory-to-DNp15/record.json").read_text())
    for hashes in [old["metrics"]["implementation_LF_sha256"],record["metrics"]["implementation_LF_sha256"]]:
        for path,digest in hashes.items():
            assert hashlib.sha256((ROOT/path).read_text(encoding="utf8").encode()).hexdigest()==digest


def test_optional_body_axis_and_raw_torque_sign_independently():
    pytest.importorskip("flygym")
    from scripts.run_exp006_neck import open_loop_body
    spec,_=records()
    torque=np.full(200,1e-4)
    zero,_=open_loop_body(np.zeros_like(torque),.5,spec["body_parameters"])
    positive,axis=open_loop_body(torque,.5,spec["body_parameters"])
    negative,_=open_loop_body(-torque,.5,spec["body_parameters"])
    assert (positive-zero)[-1,0]>0 and (negative-zero)[-1,0]<0
    np.testing.assert_allclose(axis["finite_angle_head_azimuth_rad"],.001,atol=1e-12)
