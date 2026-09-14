from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.retinotopy_diagnostics import error_field, filter_states, correlators
from src.sensory_chain import SensoryParameters

ROOT = Path(__file__).resolve().parents[1]


def test_filter_transfer_functions_match_independent_scalar_before_update():
    p = SensoryParameters()
    light = np.random.default_rng(48).uniform(.1, .9, (200, 7))
    adaptation, q, delayed = filter_states(light, p)
    for facet in range(light.shape[1]):
        a, d = light[0, facet], np.zeros(2)
        for k, row in enumerate(light):
            hp = row[facet]-a
            current = np.array([max(hp, 0), max(-hp, 0)])
            np.testing.assert_allclose(adaptation[k, facet], a, atol=1e-14, rtol=0)
            np.testing.assert_allclose(q[k, facet], current, atol=1e-14, rtol=0)
            np.testing.assert_allclose(delayed[k, facet], d, atol=1e-14, rtol=0)
            a += p.dt_ms/p.highpass_tau_ms*(row[facet]-a)
            d += p.dt_ms/p.delay_tau_ms*(current-d)


def test_error_field_has_shared_denominator_and_correct_concentration():
    a = np.zeros((3, 100, 2))
    b = a.copy()
    a[0, 5, 1] = 2
    b[0, 5, 1] = 1.8
    result, peaks, energy = error_field(a, b)
    assert result["relative_error"] == pytest.approx(.1)
    assert result["top_1pct_energy_fraction"] == 1
    assert result["entities_above_5pct_global"] == 1
    assert peaks[5] == pytest.approx(.2)
    assert energy.sum() == pytest.approx(.04)
    assert error_field(a, a)[0]["relative_error"] == 0
    with pytest.raises(ValueError):
        error_field(a, b[:2])
    with pytest.raises(ValueError):
        error_field(a, b*np.nan)


def test_correlator_diagnostic_reconstructs_frozen_emission_and_polarity():
    # Reuse the test's valid measured-style hexagon geometry, not source data.
    from tests.test_retinotopic_eye import analytic_geometry
    from src.retinotopic_eye import RetinotopicReadout
    g = analytic_geometry()
    r = RetinotopicReadout.compile(g)
    p = SensoryParameters()
    light = np.random.default_rng(19).uniform(.1, .9, (80, len(g.left)))
    channels, local = r.responses(light, p)
    raw = correlators(light, r, p, stride=1)
    bd = raw["signed_bd"]
    signed = np.stack([-bd[..., 0], bd[..., 0], -bd[..., 1], bd[..., 1]], axis=-1)
    state = np.zeros_like(local[0])
    for k in range(len(light)):
        np.testing.assert_allclose(local[k], state, atol=1e-14, rtol=0)
        state += p.dt_ms/p.emission_tau_ms*(np.maximum(signed[k], 0)-state)
    np.testing.assert_array_equal(r.pool(local), channels)


def test_fixed_candidate_does_not_change_biology_or_move_threshold():
    spec = json.loads((ROOT/"experiments/EXP-010-retinotopic-refinement/specification.json").read_text())
    old = json.loads((ROOT/"experiments/EXP-009-retinotopic-eye/specification.json").read_text())
    assert spec["acceptance"]["maximum_stage_relative_error"] == old["acceptance"]["timestep_stage_relative_error"] == .05
    assert spec["free_biological_parameters_changed"] == []
    c = spec["candidate_declared_before_downstream_evaluation"]
    assert c["production_sensor_dt_ms"] == 1 and c["reference_sensor_dt_ms"] == .5
    assert c["production_neural_dt_ms"] == old["parameters"]["dt_ms"] == .5
    assert c["reference_neural_dt_ms"] == .25


def test_original_exp009_failure_remains_frozen():
    record = json.loads((ROOT/"experiments/EXP-009-retinotopic-eye/record.json").read_text())
    assert record["validation_passed"] is False
    assert len(record["failed_checks"]) == 2
    for c in ("translation_positive_X", "translation_negative_X"):
        assert record["convergence"][c]["sensor_2_to_1ms"]["local"] > .05


def test_successor_preserves_negative_result_and_identity_boundary():
    import hashlib
    folder = ROOT/"experiments/EXP-010-retinotopic-refinement"
    record = json.loads((folder/"record.json").read_text())
    assert record["status"] == "sampling_validation_incomplete"
    assert not record["validation_passed"]
    assert not record["local_validation_passed"]
    assert not record["candidate_downstream_validation_evaluated"]
    assert not record["individual_facet_to_MaleCNS_identity_resolved"]
    assert record["biological_parameter_changes"] == []
    assert record["anatomical_edge_changes"] == 0
    assert record["convergence"]["translation_negative_X"]["sensor_1_to_half_ms"]["local"] > .05
    assert all(v["passed"] for v in record["preflight"].values())
    assert record["specification_sha256"] == hashlib.sha256((folder/"specification.json").read_bytes()).hexdigest()
    assert record["identity_evidence_sha256"] == hashlib.sha256((folder/"identity-evidence.json").read_bytes()).hexdigest()
    for c in ("translation_positive_X", "translation_negative_X"):
        d = record["original_translation_diagnostics"][c]
        assert d["coincident_physical_light_max_difference"] == 0
        assert d["stages"]["causal_held_light"]["relative_error"] < .05
        assert d["stages"]["signed_bd"]["relative_error"] > .05
        assert d["stages"]["local_emissions"]["top_1pct_energy_fraction"] > .7
        assert record["neighborhood_interventions"][c]["no_occluder"]["local_2_to_1ms_relative_error"] < .05


def test_published_column_assignments_are_not_individual_facet_matches():
    evidence = json.loads((ROOT/"experiments/EXP-010-retinotopic-refinement/identity-evidence.json").read_text())
    assert evidence["column_assignments"]["exact_current_body_ids"] == 13267
    assert evidence["column_assignments"]["exact_type_agreement"] == 13267
    assert not evidence["individual_measured_facet_to_MaleCNS_mapping"]
    for subtype, data in evidence["published_Mi1_T4_alignment"].items():
        assert data["exact_current_subtype"] == data["published_assignments"]
        assert data["Mi1_ids_with_released_hex_addresses"] == data["published_assignments"]
        assert data["current_instances_R"] == data["published_assignments"]
        assert data["rows_used_in_published_analysis"] < data["published_assignments"]


def test_identity_join_rejects_ambiguous_ids_and_reports_mismatch_without_substitution():
    import pandas as pd
    from scripts.prepare_exp010_identity import summarize_assignments
    cols = pd.DataFrame({"bodyId": [1], "neuron_type": ["Mi1"], "assigned_hex1": [18], "assigned_hex2": [19]})
    ann = pd.DataFrame({"bodyId": [1, 2], "type": ["Mi1", "T4b"], "instance": ["Mi1_R", "T4b_R"],
                        "assignedOlHex1": [None, None], "assignedOlHex2": [None, None]})
    alignment = pd.DataFrame({"Mi1 bodyId": [1], "T4a bodyId": [2], "T4b bodyId": [2],
                              "T4c bodyId": [None], "T4d bodyId": [None], "Used in analysis": [1]})
    result = summarize_assignments(cols, alignment, ann)
    assert result["published_Mi1_T4_alignment"]["T4a"]["exact_current_subtype"] == 0
    assert result["published_Mi1_T4_alignment"]["T4b"]["exact_current_subtype"] == 1
    with pytest.raises(ValueError):
        summarize_assignments(cols, alignment, pd.concat([ann, ann.iloc[:1]]))
