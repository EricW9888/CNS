import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.motor_boundary import extract_target, indistinguishable_led_family, trace_panel, window_mask

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "experiments/EXP-007-CvNA2-motor-boundary"


def synthetic_plot():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[30:33, 12:90] = 0  # known constant grand-mean velocity
    image[50, 12:16] = 0  # thin axis tick, not a mean trace
    image[60, 12:90] = 128  # grey animal curve, not a mean trace
    panel = {"x_pixels":[10,90],"y_pixels":[10,90],"time_ms":[-100,300],
             "velocity_deg_s":[60,-80],"trace_columns":[12,89]}
    contract = {"panels":{"yaw":panel},"trace_y_rows":[12,89],
        "minimum_run_thickness_pixels":3,"maximum_run_thickness_pixels":24,
        "dark_threshold":64,"threshold_sensitivity":[48,80],
        "sample_time_ms":[-80,280],"sample_step_ms":8,
        "baseline_window_ms_inclusive":[-80,-8],"response_window_ms_inclusive":[0,280]}
    return image,panel,contract


def test_axes_units_and_thin_tick_rejection_independently():
    image,panel,contract = synthetic_plot()
    t,v,error = trace_panel(image,panel,contract,64)
    np.testing.assert_allclose(t,np.arange(-90,300,5))
    np.testing.assert_allclose(v,23.25)  # 60 - (31-10)*140/80
    np.testing.assert_allclose(error,3.5)  # (half stroke + 1 px)*140/80


def test_deterministic_measurement_integrals_and_baseline_transform():
    image,_,contract = synthetic_plot()
    result = extract_target(image,contract)
    assert result == extract_target(image.copy(),copy.deepcopy(contract))
    c = result["components"]["yaw"]
    np.testing.assert_allclose(c["delta_velocity_deg_s"],0,atol=1e-14)
    assert c["summary"]["absolute_component_integral_degrees"] == pytest.approx(23.25*.280)
    assert c["summary"]["delta_component_integral_degrees"] == pytest.approx(0,abs=1e-14)
    assert c["summary"]["delta_mean_extraction_envelope_deg_s"][0] <= 0 <= c["summary"]["delta_mean_extraction_envelope_deg_s"][1]
    np.testing.assert_array_equal(np.diff(result["time_ms"]),8)


def test_two_thick_traces_fail_instead_of_selecting_desired_sign():
    image,panel,contract = synthetic_plot()
    image[70:73,30] = 0
    with pytest.raises(ValueError,match="ambiguous"):
        trace_panel(image,panel,contract,64)


def test_missing_column_is_not_interpolated_or_gap_filled():
    image,panel,contract = synthetic_plot()
    image[30:33,30] = 255
    with pytest.raises(ValueError,match="Missing"):
        trace_panel(image,panel,contract,64)


def test_thick_annotation_or_invalid_image_fails_closed():
    image,panel,contract = synthetic_plot()
    image[25:60,30] = 0
    with pytest.raises(ValueError,match="too thick"):
        trace_panel(image,panel,contract,64)
    with pytest.raises(ValueError,match="RGB uint8"):
        trace_panel(image.astype(float),panel,contract,64)
    with pytest.raises(ValueError,match="threshold"):
        trace_panel(image,panel,contract,float("nan"))


@pytest.mark.parametrize("window",[[2,1],[-1,1],[0,4],[1,1],[float("nan"),2]])
def test_invalid_or_undersampled_window_rejected(window):
    with pytest.raises(ValueError):
        window_mask(np.arange(4),window)


def test_unidentifiable_recruitment_even_if_positive_sign_is_assumed():
    led = np.linspace(-10,20,38)
    drive = np.arange(4,dtype=float)
    observed,predicted = indistinguishable_led_family(led,drive,[0,1,10])
    np.testing.assert_array_equal(observed,np.tile(led,(3,1)))
    np.testing.assert_array_equal(predicted[0],0)
    np.testing.assert_array_equal(predicted[2],10*predicted[1])
    _,disconnected = indistinguishable_led_family(led,np.zeros_like(drive),[0,1,10])
    np.testing.assert_array_equal(disconnected,0)
    for gain in [1e-6,1000,1e12]:
        unchanged,arbitrary = indistinguishable_led_family(led,drive,[1,gain])
        np.testing.assert_array_equal(unchanged[0],unchanged[1])
        np.testing.assert_array_equal(arbitrary[1],gain*arbitrary[0])


def test_no_finite_biological_gain_bound_is_inferred_from_witness():
    with pytest.raises(ValueError):
        indistinguishable_led_family([1],[1],[-1])
    spec = json.loads((FOLDER/"specification.json").read_text())
    assert not spec["anatomical_evidence"]["functional_strength_from_synapse_count"]
    assert spec["calibration_contract"]["witness_coefficients"] == [0,1,10]
    assert "not fitted gains" in spec["calibration_contract"]["identifiability_witness"]


def test_record_preserves_blocked_calibration_without_rewriting_baselines():
    spec = json.loads((FOLDER/"specification.json").read_text())
    r = json.loads((FOLDER/"record.json").read_text())
    assert spec["parent_commit"] == "6f44b9e" and spec["frozen_EXP006_commit"] == "9db9e90"
    assert r["status"] == "completed_movement_target_recruitment_unidentified"
    assert not any(r["scope"].values())
    assert not r["calibration_gate"]["ready_to_replace_EXP006_bridge"]
    assert not r["calibration_gate"]["new_bridge_evaluated"]
    assert not r["calibration_gate"]["CvN7_calibration_substituted"]
    assert r["calibration_gate"]["CvNA2_movement_target_digitized"]
    assert r["metrics"]["target_tendency_check_passed"]
    assert r["metrics"]["identical_LED_data_do_not_identify_DN_gain"]
    assert spec["measurement_contract"]["trial_structure"] == {
        "validation_trials":677,"flies":11,"only_validation_half_plotted":True,
        "grand_mean":"Caption-reported grand mean of flies from per-fly trial means; exact weighting code not recovered. Grey curves are not independent trials."}
    expected = {813696:(17653,"L","R"),804343:(14549,"R","L"),
                801678:(20860,"L","R"),903152:(10407,"R","L")}
    for row in r["identity"]:
        assert (row["manc_body_id"],row["axon_side"],row["somaSide"]) == expected[row["bodyId"]]
    for entry in spec["baseline_files"][:2]:
        assert hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert (ROOT/r["outputs"]["figure"]).exists()
    assert (ROOT/r["outputs"]["report"]).exists()


def test_quantitative_regression_retains_absolute_and_delta_targets():
    r = json.loads((FOLDER/"record.json").read_text())
    m = r["metrics"]["target_summary"]
    np.testing.assert_allclose([m[c]["absolute_mean_deg_s"] for c in ["roll","pitch","yaw"]],
                               [3.292237141829514,-17.449952044119414,10.582484114614555],atol=1e-12)
    assert m["yaw"]["delta_mean_extraction_envelope_deg_s"][0] > 0
    assert m["pitch"]["delta_mean_extraction_envelope_deg_s"][1] < 0
    assert m["yaw"]["largest_abs_delta_time_ms"] == 32
    assert m["pitch"]["largest_abs_delta_time_ms"] == 80
    assert len(m["yaw"]["extraction_compatible_extremum_sample_times_ms"]) > 1
    for c in m.values():
        assert c["threshold_center_max_difference_deg_s"] < .17
        assert c["extremum_time_resolution_plus_axis_error_ms"] > 8


def test_all_saved_motor_controls_preserve_numerical_and_causal_contracts():
    r = json.loads((FOLDER/"record.json").read_text())
    controls = r["metrics"]["frozen_replay"]
    assert set(controls) == {"full","DN_disconnected","DN_L_disconnected","DN_R_disconnected","motor_disconnected"}
    for control,c in controls.items():
        assert c["preflight"]["passed"] and c["half_dt_preflight"]["passed"]
        for response in c["conditions"].values():
            assert response["deterministic"]
            assert response["motor_replay_relative_error"] == 0
            assert response["torque_replay_relative_error"] == 0
            assert response["held_input_half_dt_relative_error"] < 1e-12
            if control == "DN_disconnected":
                assert response["motor_late"] == [0]*4


def test_frozen_specification_and_implementation_digests():
    r = json.loads((FOLDER/"record.json").read_text())
    assert hashlib.sha256((FOLDER/"specification.json").read_bytes()).hexdigest() == r["metrics"]["specification_sha256"]
    for path,digest in r["metrics"]["implementation_LF_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_text(encoding="utf8").encode()).hexdigest() == digest


def test_optional_primary_raster_reproduces_recorded_target():
    path = ROOT/"data/exp007_motor_boundary/CvNA2_velocity_native.png"
    if not path.exists():
        pytest.skip("Ignored primary PDF-derived bitmap is not installed")
    from PIL import Image
    spec = json.loads((FOLDER/"specification.json").read_text())
    r = json.loads((FOLDER/"record.json").read_text())
    with Image.open(path) as image:
        target = extract_target(image,spec["measurement_contract"])
    encoded = json.dumps(target,indent=2,allow_nan=False)+"\n"
    assert hashlib.sha256(encoded.encode()).hexdigest() == r["metrics"]["target_sha256"]


def test_optional_replay_uses_saved_DN_not_upstream_or_body():
    if not (ROOT/"results/exp006_neck/stage_traces.npz").exists():
        pytest.skip("Ignored frozen EXP-006 stage arrays are not installed")
    from scripts.run_exp007_boundary import frozen_replay
    spec = json.loads((FOLDER/"specification.json").read_text())
    r = json.loads((FOLDER/"record.json").read_text())
    actual,_,_,_ = frozen_replay(spec)
    assert actual == r["metrics"]["frozen_replay"]
