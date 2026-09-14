from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.compound_eye import EyeGeometry
from src.retinotopic_eye import RetinotopicReadout, source_field_comparison, tangent_unit
from src.sensory_chain import SensoryParameters

ROOT = Path(__file__).resolve().parents[1]


def analytic_geometry():
    # Analytic hex patch ONLY for tests, never substituted for primary anatomy.
    offsets = np.array([[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]])
    grid = np.array([(p, q) for p in range(-2, 3) for q in range(-2, 3)
                     if max(abs(p), abs(q), abs(p-q)) <= 2])
    ids = {tuple(pq): i for i, pq in enumerate(grid)}
    neighbor = np.array([[ids.get(tuple(pq+delta), -1) for delta in offsets] for pq in grid])
    x, z = .05*(grid[:, 1]-grid[:, 0]), .03*grid.sum(axis=1)
    u = np.concatenate([np.column_stack([x, np.ones(len(x)), z]),
                        np.column_stack([x, -np.ones(len(x)), z])])
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return EyeGeometry(np.zeros_like(u), u, np.repeat([True, False], len(grid)),
                       np.concatenate([neighbor, np.where(neighbor >= 0, neighbor+len(grid), -1)]),
                       np.tile(grid, (2, 1)))


def moving_light(geometry, dt=.5, duration=800, direction=(1., 0., 0.)):
    t = np.arange(round(duration/dt))*dt/1000
    phase = 2*np.pi*(geometry.directions@np.asarray(direction))/.5
    return .5+.3*np.cos(phase[None, :]-2*np.pi*2*t[:, None])


def scalar_reference(light, readout, p):
    a, d = light[0].copy(), np.zeros((light.shape[1], 2))
    state = np.zeros((len(readout.centers), 2, 4))
    out = []
    for sample in light:
        out.append(state.copy())
        hp = sample-a
        q = np.array([[max(h, 0), max(-h, 0)] for h in hp])
        target = np.empty_like(state)
        for i, center in enumerate(readout.centers):
            for pol in range(2):
                c = []
                for neighbor in readout.neighbors[i]:
                    c.append(d[neighbor, pol]*q[center, pol]-q[neighbor, pol]*d[center, pol])
                b = (c[0]+c[5]-c[2]-c[3])/4
                v = (c[1]-c[4])/2
                target[i, pol] = [max(-b, 0), max(b, 0), max(-v, 0), max(v, 0)]
        state += p.dt_ms/p.emission_tau_ms*(target-state)
        a += p.dt_ms/p.highpass_tau_ms*(sample-a)
        d += p.dt_ms/p.delay_tau_ms*(q-d)
    return np.asarray(out)


def test_vector_kernel_equals_independent_scalar_reference():
    g = analytic_geometry()
    r, p = RetinotopicReadout.compile(g), SensoryParameters()
    light = moving_light(g, duration=15)
    channels, local = r.responses(light, p)
    reference = scalar_reference(light, r, p)
    np.testing.assert_allclose(local, reference, atol=1e-18, rtol=1e-13)
    np.testing.assert_array_equal(channels, r.pool(local))


def test_random_light_reference_and_facet_renumbering_invariance():
    g, p = analytic_geometry(), SensoryParameters()
    rng = np.random.default_rng(71)
    light = rng.uniform(.1, .9, (20, len(g.left)))
    r = RetinotopicReadout.compile(g)
    channels, local = r.responses(light, p)
    np.testing.assert_allclose(local, scalar_reference(light, r, p), rtol=1e-13, atol=1e-18)
    permutation = rng.permutation(len(g.left))
    inverse = np.argsort(permutation)
    old_neighbors = g.neighbors[permutation]
    new_neighbors = np.where(old_neighbors >= 0, inverse[np.maximum(old_neighbors, 0)], -1)
    new = EyeGeometry(g.lens_mm[permutation], g.directions[permutation], g.left[permutation],
                      new_neighbors, g.grid[permutation])
    renumbered = RetinotopicReadout.compile(new)
    nc, nl = renumbered.responses(light[:, permutation], p)
    np.testing.assert_allclose(channels, nc, atol=1e-18, rtol=1e-13)
    old_order = np.argsort(permutation[renumbered.centers])
    np.testing.assert_array_equal(local, nl[:, old_order])
    np.testing.assert_array_equal(r.preferred_directions, renumbered.preferred_directions[old_order])


def test_geometry_unit_tangent_antiparallel_and_preserved_nonorthogonality():
    g, r = analytic_geometry(), RetinotopicReadout.compile(analytic_geometry())
    np.testing.assert_allclose(np.linalg.norm(r.preferred_directions, axis=-1), 1, atol=1e-15)
    np.testing.assert_allclose(np.sum(r.preferred_directions*g.directions[r.centers, None], axis=-1), 0, atol=1e-15)
    np.testing.assert_array_equal(r.preferred_directions[:, 0], -r.preferred_directions[:, 1])
    np.testing.assert_array_equal(r.preferred_directions[:, 2], -r.preferred_directions[:, 3])
    assert np.any(abs(np.sum(r.preferred_directions[:, 1]*r.preferred_directions[:, 3], axis=-1)) > 1e-5)
    assert not r.preferred_directions.flags.writeable


@pytest.mark.parametrize("flicker", [False, True])
def test_static_or_uniform_flicker_has_exactly_zero_directed_motion(flicker):
    g = analytic_geometry()
    r = RetinotopicReadout.compile(g)
    light = np.full((80, len(g.left)), .5)
    if flicker:
        light[:] = np.linspace(.1, .9, 80)[:, None]
    channels, local = r.responses(light, SensoryParameters())
    assert not np.any(channels) and not np.any(local)


@pytest.mark.parametrize("direction,positive,negative", [((1, 0, 0), 1, 0), ((0, 0, -1), 3, 2)])
def test_scalar_motion_reversal_changes_local_preference(direction, positive, negative):
    g = analytic_geometry()
    r = RetinotopicReadout.compile(g)
    p = SensoryParameters()
    _, local = r.responses(moving_light(g, direction=direction), p)
    _, reversed_local = r.responses(moving_light(g, direction=-np.asarray(direction)), p)
    response = local[800:].mean(axis=(0, 1, 2))
    reverse = reversed_local[800:].mean(axis=(0, 1, 2))
    assert response[positive] > response[negative]
    assert reverse[negative] > reverse[positive]


def test_contrast_inversion_swaps_ON_OFF_exactly_and_eye_exchange_is_equivariant():
    g, p = analytic_geometry(), SensoryParameters()
    r = RetinotopicReadout.compile(g)
    light = moving_light(g, duration=80)
    c, local = r.responses(light, p)
    inverse_c, inverse = r.responses(1-light, p)
    np.testing.assert_allclose(inverse, local[:, :, ::-1], atol=1e-17, rtol=1e-12)
    swap = replace(r, left=~r.left)
    np.testing.assert_array_equal(swap.responses(light, p)[0], c.reshape(len(c), 2, 8)[:, ::-1].reshape(len(c), 16))


def test_causal_prefix_determinism_no_mutation_and_storage_decimation():
    g, p = analytic_geometry(), SensoryParameters()
    r = RetinotopicReadout.compile(g)
    light = moving_light(g, duration=50)
    copy = light.copy()
    c, local = r.responses(light, p)
    c2, l2 = r.responses(light, p, save_stride=4)
    pc, pl = r.responses(light[:37], p)
    np.testing.assert_array_equal(c, c2)
    np.testing.assert_array_equal(local[::4], l2)
    np.testing.assert_array_equal(c[:37], pc)
    np.testing.assert_array_equal(local[:37], pl)
    np.testing.assert_array_equal(light, copy)
    np.testing.assert_array_equal(r.responses(light, p)[1], local)


def test_pool_preserves_opposing_local_emissions_instead_of_canceling_before_split():
    r = RetinotopicReadout.compile(analytic_geometry())
    local = np.zeros((len(r.centers), 2, 4))
    indices = np.flatnonzero(r.left)
    local[indices[0], 0, 0] = 1
    local[indices[1], 0, 1] = 1
    pooled = r.pool(local)
    assert pooled[0] == pooled[1] == 1/len(indices)
    assert pooled[8:].sum() == 0
    with pytest.raises(ValueError):
        r.pool(-local)


def test_bad_light_and_degenerate_basis_rejected():
    r = RetinotopicReadout.compile(analytic_geometry())
    for light in (np.zeros((1, r.facet_count-1)), np.full((2, r.facet_count), np.nan),
                  np.full((1, r.facet_count), 1.1), np.zeros((0, r.facet_count))):
        with pytest.raises(ValueError):
            r.responses(light, SensoryParameters())
    with pytest.raises(ValueError):
        tangent_unit([[1, 0, 0]], [[1, 0, 0]])


def test_analytic_timestep_refinement():
    g, p = analytic_geometry(), SensoryParameters()
    r = RetinotopicReadout.compile(g)
    c, local = r.responses(moving_light(g, dt=p.dt_ms, duration=400), p)
    fine_p = replace(p, dt_ms=p.dt_ms/2)
    fc, fl = r.responses(moving_light(g, dt=fine_p.dt_ms, duration=400), fine_p)
    for a, b in [(c, fc[::2]), (local, fl[::2])]:
        assert np.max(abs(a-b))/max(np.max(abs(a)), np.max(abs(b))) < .05


def test_constant_input_washes_out_motion_state_and_observer_is_not_executed(monkeypatch):
    import src.compound_eye as eye
    def forbidden(*args, **kwargs):
        raise AssertionError("observer rendering entered sensory transform")
    monkeypatch.setattr(eye, "observer_camera", forbidden)
    g, p = analytic_geometry(), SensoryParameters()
    moving = moving_light(g, duration=200)
    light = np.concatenate([moving, np.repeat(moving[-1:], 2400, axis=0)])
    c, local = RetinotopicReadout.compile(g).responses(light, p)
    assert local.max() > 0
    assert local[-1].max()/local.max() < 1e-8
    assert not np.any(c < 0)
    assert local.max() <= 1  # bounded radiance and convex filters


def test_measured_registration_field_orientation_and_coverage():
    path = ROOT/"data/exp009_retinotopy/reference.npz"
    if not path.exists():
        pytest.skip("ignored primary registration not installed")
    g = EyeGeometry.load(ROOT/"data/exp008_eye/geometry.npz")
    r = RetinotopicReadout.compile(g)
    with np.load(path, allow_pickle=False) as reference:
        result = source_field_comparison(r, g, reference)
    assert result["registered_right_facets"] == 778
    assert result["compared_complete_right_facets"] == 738
    assert result["all_local_orientations_same_hemiplane"]
    assert [int(sum(r.left)), int(sum(~r.left))] == [753, 747]
    offsets = np.array([[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]])
    np.testing.assert_array_equal(g.grid[r.neighbors]-g.grid[r.centers, None], np.broadcast_to(offsets, r.neighbors.shape+(2,)))
    # Source registration metadata itself cannot be silently permuted.
    with np.load(path, allow_pickle=False) as ref:
        changed = dict(ref)
    changed["viewing_axes"] = changed["viewing_axes"][::-1]
    with pytest.raises(ValueError):
        source_field_comparison(r, g, changed)


def test_experiment_specification_freezes_boundary_and_no_output_fit():
    spec = json.loads((ROOT/"experiments/EXP-009-retinotopic-eye/specification.json").read_text())
    old = json.loads((ROOT/"experiments/EXP-008-compound-eye/specification.json").read_text())
    assert spec["parameters"] == old["early_vision"]["parameters"]
    assert spec["conditions"] == old["pose_conditions"]
    assert "NOT MaleCNS body IDs" in spec["anatomical_mapping"]
    assert "Lossy" in spec["adapter"]
    assert spec["MaleCNS_identity_evidence"]["motion_assignedOlHex1_nonmissing"] == 0
    assert spec["MaleCNS_identity_evidence"]["motion_assignedOlHex2_nonmissing"] == 0
    assert spec["source_registered_grid_minus_EXP008_grid"] == [1, 0]


def test_published_registration_extraction_reproduces_pinned_arrays():
    folder = ROOT/"data/exp009_retinotopy"
    if not (folder/"reference.npz").exists():
        pytest.skip("ignored primary retinotopy data not installed")
    rdata = pytest.importorskip("rdata")
    from scripts.prepare_exp009_retinotopy import extract_reference
    g = EyeGeometry.load(ROOT/"data/exp008_eye/geometry.npz")
    eye, field, med = [rdata.read_rda(folder/f) for f in ("eyemap.RData", "T4_RF_pred.RData", "med_ixy.RData")]
    arrays = extract_reference(eye, field, med, g)
    with np.load(folder/"reference.npz", allow_pickle=False) as stored:
        assert set(stored.files) == set(arrays)
        for k, v in arrays.items():
            np.testing.assert_array_equal(v, stored[k])
    bad = dict(eye)
    bad["eyemap"] = eye["eyemap"].copy()
    bad["eyemap"][0, 0] = 0
    with pytest.raises(ValueError, match="medulla index"):
        extract_reference(bad, field, med, g)


def test_record_keeps_failed_local_refinement_separate_from_propagation():
    record = json.loads((ROOT/"experiments/EXP-009-retinotopic-eye/record.json").read_text())
    spec = json.loads((ROOT/"experiments/EXP-009-retinotopic-eye/specification.json").read_text())
    assert record["status"] == "retinotopic_chain_validation_incomplete"
    assert record["validation_passed"] is False
    assert record["propagation_to_both_DNp15"] is True
    assert len(record["failed_checks"]) == 2
    for c, banks in record["convergence"].items():
        assert max(banks["neural_half_dt"].values()) < spec["acceptance"]["timestep_stage_relative_error"]
        if c.startswith("translation"):
            assert banks["sensor_2_to_1ms"]["local"] > .05
            assert max(v for k, v in banks["sensor_2_to_1ms"].items() if k != "local") < .05
        if c != "static":
            assert record["condition_metrics"][c]["both_eyes_ON_OFF_nonzero"]
            assert record["condition_metrics"][c]["HS_H2_all_projected_inputs_nonzero"]
        for k in ("eye_to_retinotopy_max_abs_activity", "retinotopy_to_T4_T5_max_abs_downstream", "visual_to_central_max_abs_central"):
            assert record["controls"][c][k] == 0
    assert record["anatomical_edge_changes"] == 0
    assert record["coverage"]["local_neighborhoods_L_R"] == [753, 747]
    assert record["coverage"]["distinct_used_facets_L_R"] == [857, 852]


def test_record_source_and_frozen_source_fingerprints():
    from scripts.prepare_exp008_eye import evidence_sha256
    from src.provenance import sha256_file
    spec_path = ROOT/"experiments/EXP-009-retinotopic-eye/specification.json"
    spec = json.loads(spec_path.read_text())
    record = json.loads(spec_path.with_name("record.json").read_text())
    assert record["specification_sha256"] == sha256_file(spec_path)
    for path, digest in record["implementation_sha256"].items():
        assert evidence_sha256(ROOT/path) == digest
    for entry in spec["frozen_files"]:
        if not entry["path"].startswith("data/"):
            assert evidence_sha256(ROOT/entry["path"]) == entry["sha256"]
    for entry in spec["source_files"][:-1]:
        assert spec["source_commit"] in entry["url"]
        assert entry["redistributed"] is False


def test_hotspot_local_axes_and_time_do_not_get_confused_with_body_identity():
    from scripts.run_exp009_retinotopy import refinement_hotspot
    g = analytic_geometry()
    r = RetinotopicReadout.compile(g)
    coarse = np.zeros((3, len(r.centers), 2, 4))
    fine = coarse.copy()
    fine[2, 1, 1, 3] = .05
    result = refinement_hotspot(coarse, fine, r, g)
    assert result["local_time_ms"] == 4
    assert result["measured_facet_index_0based"] == r.centers[1]
    assert result["polarity"] == "OFF" and result["subtype_proxy"] == "d"
    assert result["maximum_absolute_local_difference"] == .05


def test_observer_eye_grouping_uses_metadata_not_assumed_lens_order():
    from scripts.run_exp009_retinotopy import eye_display_order
    left = np.array([False, True, False, True, True])
    order = eye_display_order(left)
    np.testing.assert_array_equal(order, [1, 3, 4, 0, 2])
    assert left[order[:3]].all() and not left[order[3:]].any()
