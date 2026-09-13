from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.compound_eye import (EyeGeometry, EyeSampler, LocalReadout, Scene,
                              observer_camera, pose_rays, rotation_z, sphere_distance)
from src.sensory_chain import SensoryParameters

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/exp008_eye/geometry.npz"


def tiny_eye():
    # Synthetic rays for independent analytic tests only; never production geometry.
    axes = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
    return EyeGeometry(np.zeros((4, 3)), axes, np.array([True, True, False, False]),
                       np.full((4, 6), -1, dtype=int), np.zeros((4, 2), dtype=int))


@pytest.fixture
def measured():
    if not SOURCE.exists():
        pytest.skip("ignored primary geometry not installed")
    return EyeGeometry.load(SOURCE)


def test_rigid_pose_matches_hand_coordinates_and_rejects_reflection():
    o, d = pose_rays(np.array([[1., 0, 0]]), np.array([[0., 1., 0]]), rotation_z(np.pi/2), np.array([2., 0, 0]))
    np.testing.assert_allclose(o, [[2., 1., 0]], atol=1e-15)
    np.testing.assert_allclose(d, [[-1., 0., 0]], atol=1e-15)
    with pytest.raises(ValueError):
        pose_rays(o, d, np.diag([1., -1., 1.]), np.zeros(3))


def test_sphere_intersections_independent_cases():
    o = np.array([[0., 0, 0], [3., 0, 0], [3., 0, 0], [2., 1, 0]])
    d = np.array([[1., 0, 0], [-1., 0, 0], [1., 0, 0], [-1., 0, 0]])
    np.testing.assert_allclose(sphere_distance(o, d, [0., 0, 0], 1.), [1., 2., np.inf, 2.])
    # Non-unit directions retain correct ray-parameter distance.
    assert sphere_distance(np.zeros(3), np.array([2., 0, 0]), np.zeros(3), 2.) == 1.


def test_occlusion_nearest_surface_and_off_axis_visibility():
    scene = Scene(occluder_center=(3., 0, 0), occluder_radius_mm=1.)
    d = np.array([[1., 0, 0], [0., 1., 0], [-1., 0, 0]])
    light = scene.radiance(np.zeros_like(d), d)
    assert light[0] == scene.occluder_light
    np.testing.assert_allclose(light[1:], [.9, .9])
    behind_wall = replace(scene, occluder_center=(20., 0, 0))
    assert behind_wall.radiance(np.zeros((1, 3)), d[:1])[0] == .9


def test_gaussian_unit_rays_normalization_and_constant_radiance():
    sampler = EyeSampler.compile(tiny_eye(), radial=5, azimuthal=24)
    np.testing.assert_allclose(np.linalg.norm(sampler.local_rays, axis=-1), 1, atol=1e-15)
    assert np.all(sampler.weights > 0)
    np.testing.assert_allclose(sampler.weights.sum(), 1, atol=1e-15)
    for pose in (0., .2, -.6):
        np.testing.assert_allclose(sampler.sample(Scene(amplitude=0, occluder_radius_mm=0), rotation_z(pose)), .5, atol=1e-15)


def test_eye_geometry_rejects_bad_units_axes_and_cross_eye_neighbors():
    eye = tiny_eye()
    with pytest.raises(ValueError):
        replace(eye, directions=eye.directions*2)
    neighbors = eye.neighbors.copy()
    neighbors[0, 0] = 2
    with pytest.raises(ValueError):
        replace(eye, neighbors=neighbors)
    with pytest.raises(ValueError):
        replace(eye, lens_mm=eye.lens_mm*np.nan)


def test_sampling_pose_inverse_equivalence_and_translation_causality():
    eye = tiny_eye()
    sampler = EyeSampler.compile(eye)
    r, t = rotation_z(.2), np.array([.7, -.2, .1])
    moved_o, moved_d = pose_rays(eye.lens_mm[:, None]+np.zeros_like(sampler.local_rays), sampler.local_rays, r, t)
    direct = Scene().radiance(moved_o, moved_d) @ sampler.weights
    np.testing.assert_array_equal(direct, sampler.sample(Scene(), r, t))
    assert np.max(abs(direct-sampler.sample(Scene()))) > .01
    # A constant world conserves light under translation too.
    np.testing.assert_allclose(sampler.sample(Scene(amplitude=0, occluder_radius_mm=0), r, t), .5)


def test_measured_counts_grid_and_asymmetry_preserved(measured):
    assert measured.left.sum() == 857 and (~measured.left).sum() == 852
    assert np.all(measured.lens_mm[measured.left, 1] > 0)
    assert np.all(measured.lens_mm[~measured.left, 1] < 0)
    for mask in (measured.left, ~measured.left):
        assert len(np.unique(measured.grid[mask], axis=0)) == mask.sum()
    # Distinct counts and geometry are kept, not forced onto a reflected template.
    assert measured.left.sum() != (~measured.left).sum()


def test_measured_primary_extraction_reproducible(measured):
    rdata = pytest.importorskip("rdata")
    from scripts.prepare_exp008_eye import extract
    regenerated = extract(rdata.read_rda(SOURCE.parent/"20240701.RData"), rdata.read_rda(SOURCE.parent/"20240701_nb.RData"))
    for key in measured.__dict__:
        np.testing.assert_array_equal(getattr(measured, key), getattr(regenerated, key))


def test_measured_sampling_determinism_and_quadrature(measured):
    low = EyeSampler.compile(measured, radial=5, azimuthal=24)
    high = EyeSampler.compile(measured, radial=9, azimuthal=64)
    for r, t in [(rotation_z(0), np.zeros(3)), (rotation_z(.5), np.zeros(3)), (rotation_z(0), np.array([1., 0, 0]))]:
        a = low.sample(Scene(), r, t)
        np.testing.assert_array_equal(a, low.sample(Scene(), r, t))
        assert np.max(abs(a-high.sample(Scene(), r, t))) < .05


def test_measured_field_of_view_and_independent_reflection_consistency(measured):
    from scipy.spatial import cKDTree
    u = measured.directions
    az, el = np.rad2deg(np.arctan2(u[:, 1], u[:, 0])), np.rad2deg(np.arcsin(u[:, 2]))
    assert el.min() < -70 and el.max() > 85
    for mask in (measured.left, ~measured.left):
        eq = abs(az[mask & (abs(el) < 5)])
        assert eq.max() > 145 and eq.max() < 160
    dist = cKDTree(u[~measured.left]).query(u[measured.left]*[1, -1, 1])[0]
    angle = np.rad2deg(2*np.arcsin(dist/2))
    assert 0 < np.median(angle) < 3 and np.percentile(angle, 95) < 5


def synthetic_readout():
    # Each eye uses four real-valued test signals as opposing endpoints.
    ep = np.repeat(np.arange(4).reshape(1, 4, 1), 2, axis=2)
    return LocalReadout((np.array([0]), np.array([4])), (ep, ep+4))


def test_local_filter_scalar_reference_static_flicker_reversal_and_eye_swap():
    readout, p = synthetic_readout(), SensoryParameters()
    time = np.arange(600)*p.dt_ms
    # A rearward-moving sinusoid reaches rear endpoint after front endpoint.
    signal = .5+.3*np.cos(2*np.pi*(np.arange(4)[None, :]-time[:, None]/30)/4)
    light = np.concatenate([signal, signal], axis=1)
    observed = readout.channels(light, p)
    # Independently written scalar loops, including pre-update sample semantics.
    adapted, delay, state = light[0].copy(), np.zeros((2, 8)), np.zeros(16)
    expected = []
    for row in light:
        expected.append(state.copy())
        hp = row-adapted
        q = np.array([[max(float(v), 0) for v in hp], [max(float(-v), 0) for v in hp]])
        for side in range(2):
            for polarity in range(2):
                for axis in range(2):
                    a, b = 4*side+2*axis, 4*side+2*axis+1
                    raw = delay[polarity, a]*q[polarity, b]-q[polarity, a]*delay[polarity, b]
                    for sign in range(2):
                        i = 8*side+4*polarity+2*axis+sign
                        state[i] += p.dt_ms/p.emission_tau_ms*(max(raw if sign == 0 else -raw, 0)-state[i])
        adapted += p.dt_ms/p.highpass_tau_ms*(row-adapted)
        delay += p.dt_ms/p.delay_tau_ms*(q-delay)
    np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=1e-13)
    np.testing.assert_array_equal(readout.channels(np.repeat(light[:1], 600, axis=0), p), 0)
    np.testing.assert_array_equal(readout.channels(np.repeat(signal[:, :1], 8, axis=1), p), 0)
    reverse = light[:, [1, 0, 3, 2, 5, 4, 7, 6]]
    np.testing.assert_array_equal(readout.channels(reverse, p), observed[:, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]])
    np.testing.assert_array_equal(readout.channels(np.concatenate([light[:, 4:], light[:, :4]], axis=1), p), observed[:, list(range(8, 16))+list(range(8))])


def test_local_filter_causal_bounded_decay_and_contrast_inversion():
    readout, p = synthetic_readout(), SensoryParameters()
    rng = np.random.default_rng(18)
    light = rng.uniform(size=(5000, 8))
    light[1000:] = light[0]
    a = readout.channels(light, p)
    np.testing.assert_array_equal(a[:500], readout.channels(light[:500], p))
    assert a.min() >= 0 and a.max() <= 1 and abs(a[-1]).max() < 1e-12
    np.testing.assert_allclose(readout.channels(1-light, p), a[:, [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]], atol=1e-15)
    with pytest.raises(ValueError):
        readout.channels(light*2, p)


def test_readout_excludes_periphery_and_orients_physical_endpoints(measured):
    r = LocalReadout.compile(measured)
    assert list(map(len, r.centers)) == [65, 66]
    assert sum(map(len, r.centers)) < len(measured.directions)/10
    az = abs(np.rad2deg(np.arctan2(measured.directions[:, 1], measured.directions[:, 0])))
    el = np.rad2deg(np.arcsin(measured.directions[:, 2]))
    for endpoints in r.endpoints:
        assert np.all(az[endpoints[:, 1]].mean(axis=1) > az[endpoints[:, 0]].mean(axis=1))
        assert np.all(el[endpoints[:, 3]].mean(axis=1) > el[endpoints[:, 2]].mean(axis=1))
    for ids, endpoints in zip(r.centers, r.endpoints):
        np.testing.assert_array_equal(endpoints[:, 1, 0], ids)
        np.testing.assert_array_equal(endpoints[:, 3, 0], ids)
        # True adjacent neighbors, rather than a diametric jump over the center.
        assert np.max(az[endpoints[:, 1]].mean(axis=1)-az[endpoints[:, 0]].mean(axis=1)) < 9


def test_observer_camera_not_needed_for_sensor(monkeypatch):
    import src.compound_eye as module
    def forbidden(*args, **kwargs):
        raise AssertionError("observer entered fly input path")
    monkeypatch.setattr(module, "observer_camera", forbidden)
    assert EyeSampler.compile(tiny_eye()).sample(Scene()).shape == (4,)


def test_observer_camera_renders_only_debug_view():
    frame = observer_camera(Scene(), pixels=8)
    assert frame.shape == (8, 8) and 0 <= frame.min() <= frame.max() <= 1


def test_physical_scene_reflection_without_symmetrizing_measured_eyes(measured):
    mirror = replace(measured, lens_mm=measured.lens_mm*[1, -1, 1],
                     directions=measured.directions*[1, -1, 1], left=~measured.left)
    a = EyeSampler.compile(measured, radial=5, azimuthal=24)
    b = EyeSampler.compile(mirror, radial=5, azimuthal=24)
    # Reflect the entire existing geometry/pose, not one eye onto the other.
    np.testing.assert_allclose(a.sample(Scene(), rotation_z(.3), np.array([.2, .1, 0])),
                               b.sample(Scene(), rotation_z(-.3), np.array([.2, -.1, 0])), atol=2e-14)


def test_multineighbor_readout_matches_independent_facet_filter_reference():
    p = SensoryParameters()
    rng = np.random.default_rng(15)
    light = rng.uniform(size=(120, 16))
    ep = np.array([[[0, 1], [2, 2], [3, 3], [2, 2]],
                   [[4, 5], [6, 6], [7, 7], [6, 6]]])
    readout = LocalReadout((np.array([2, 6]), np.array([10, 14])), (ep, ep+8))
    state, delay, adaptation = np.zeros(16), np.zeros((2, 16)), light[0].copy()
    expected = []
    for frame in light:
        expected.append(state.copy())
        hp = frame-adaptation
        q = np.array([np.maximum(hp, 0), np.maximum(-hp, 0)])
        for side in range(2):
            for polarity in range(2):
                for axis in range(2):
                    products = []
                    for endpoint in readout.endpoints[side]:
                        for pair in range(2):
                            a, b = endpoint[2*axis, pair], endpoint[2*axis+1, pair]
                            products.append(delay[polarity, a]*q[polarity, b]-q[polarity, a]*delay[polarity, b])
                    raw = sum(products)/len(products)
                    for direction in range(2):
                        index = 8*side+4*polarity+2*axis+direction
                        target = max(raw if direction == 0 else -raw, 0)
                        state[index] += p.dt_ms/p.emission_tau_ms*(target-state[index])
        delay += p.dt_ms/p.delay_tau_ms*(q-delay)
        adaptation += p.dt_ms/p.highpass_tau_ms*(frame-adaptation)
    np.testing.assert_allclose(readout.channels(light, p), expected, atol=1e-15, rtol=1e-13)


def test_causal_hold_does_not_interpolate_future_sensory_samples():
    from scripts.run_exp008_eye import held_light
    samples = np.arange(4).reshape(4, 1)
    np.testing.assert_array_equal(held_light(samples, 2, .5, 8), np.repeat(samples, 4, axis=0))
    changed = samples.copy()
    changed[2:] = 100
    np.testing.assert_array_equal(held_light(changed, 2, .5, 8)[:8], held_light(samples, 2, .5, 8)[:8])
    with pytest.raises(ValueError):
        held_light(samples, 0, .5, 8)


def test_static_pose_is_sampled_once_and_pose_change_is_causal():
    from scripts.run_exp008_eye import sensor_movie
    class CountingSampler:
        geometry = tiny_eye()
        calls = 0
        def sample(self, scene, rotation, translation):
            self.calls += 1
            return np.repeat(rotation[0, 0]+translation[0], 4)
    p = SensoryParameters(onset_ms=2, offset_ms=4, duration_ms=6)
    a = CountingSampler()
    _, static = sensor_movie(a, Scene(), p, [0, 0], 1)
    assert a.calls == 1
    np.testing.assert_array_equal(static, 1)
    b = CountingSampler()
    _, moved = sensor_movie(b, Scene(), p, [45, 1], 1)
    np.testing.assert_array_equal(moved[:3], static[:3])
    assert not np.array_equal(moved[3:], static[3:])


def test_permutation_not_zipped_or_silently_substituted():
    from scripts.prepare_exp008_eye import extract
    lens = np.array([[1., 1, 0], [1., 2, 0], [1., -1, 0], [1., -2, 0]])
    axes = np.repeat([[1., 0, 0]], 4, axis=0)
    mask = np.array([True, True, False, False])
    order = np.array([2, 0, 3, 1])
    data = {"lens": lens, "cone": (lens-axes)[order], "i_match": order+1,
            "ind_left_lens": mask, "ind_left_cone": mask[order],
            "ucl_rot": axes[order], "ucl_rot_sm": axes[order]}
    nb = np.column_stack([np.arange(1, 3), np.full((2, 6), np.nan)])
    xy = np.array([[1, 0, 0], [2, 1, 0]])
    neighborhood = {f"{key}_{side}": value for side in ("left", "right")
                    for key, value in [("nb_ind", nb), ("nb_dist_ucl", nb), ("ind_xy", xy)]}
    np.testing.assert_array_equal(extract(data, neighborhood).directions, axes)
    with pytest.raises(ValueError):
        extract({**data, "i_match": np.arange(1, 5)}, neighborhood)


def test_frozen_graph_hashes_and_result_contract():
    import json
    from scripts.run_exp008_eye import SPEC, RECORD
    from src.provenance import sha256_file
    from scripts.prepare_exp008_eye import evidence_sha256
    spec = json.loads(SPEC.read_text(encoding="utf8"))
    result = json.loads(RECORD.read_text(encoding="utf8"))
    assert result["specification_sha256"] == sha256_file(SPEC)
    for entry in spec["frozen_files"]:
        assert evidence_sha256(ROOT/entry["path"]) == entry["sha256"]
    for path, expected in result["implementation_sha256"].items():
        assert evidence_sha256(ROOT/path) == expected
    assert result["local_readout_counts_L_R"] == [65, 66]
    assert all(c["passed"] for c in result["numerical_preflight"].values())
    assert result["propagation_to_both_DNp15"]
    assert set(result["controls"]) == set(spec["pose_conditions"])
    assert all(v["eye_to_motion_disconnected_max_abs_DN"] == 0 for v in result["controls"].values())
    assert not any("motor" in name for name in spec["controls"])
    errors = result["convergence"]
    observed_max = max(v for c in errors["numerical_half_dt_stage_relative_errors"].values() for v in c.values())
    observed_max = max(observed_max, max(v for c in errors["sensor_2_to_1ms_stage_relative_errors"].values()
                                          for k, v in c.items() if k != "max_abs_light_error"))
    assert errors["passed"] == (observed_max <= spec["acceptance"]["timestep_stage_relative_error"])
    passes = errors["passed"] and result["propagation_to_both_DNp15"] and all(all(x.values()) for x in result["stage_interface_checks"].values())
    assert result["status"] == ("continuous_sensory_chain_with_provisional_local_readout" if passes else "sensory_chain_validation_incomplete")


def test_source_fingerprint_normalizes_newlines_but_not_data(tmp_path):
    from scripts.prepare_exp008_eye import evidence_sha256
    a, b = tmp_path/"a.py", tmp_path/"b.py"
    a.write_bytes(b"x = 1\n")
    b.write_bytes(b"x = 1\r\n")
    assert evidence_sha256(a) == evidence_sha256(b)
    a, b = tmp_path/"a.dat", tmp_path/"b.dat"
    a.write_bytes(b"x = 1\n")
    b.write_bytes(b"x = 1\r\n")
    assert evidence_sha256(a) != evidence_sha256(b)


def test_predeclared_specification_has_stable_lf_bytes(monkeypatch, tmp_path):
    import json
    import scripts.prepare_exp008_eye as preparation
    from src.provenance import sha256_file
    manifest = json.loads(preparation.SPEC.read_text(encoding="utf8"))["eye_source"]
    target = tmp_path/"specification.json"
    monkeypatch.setattr(preparation, "SPEC", target)
    # All source paths are already registered; this does not modify the registry.
    preparation.freeze_specification(manifest)
    assert b"\r" not in target.read_bytes()
    original = sha256_file(target)
    preparation.freeze_specification(manifest)
    assert sha256_file(target) == original


def test_geometry_metadata_roundtrips_json(measured):
    import json
    from scripts.run_exp008_eye import geometry_preflight
    sampler = EyeSampler.compile(measured, radial=5, azimuthal=24)
    spec = {"optics": {"fwhm_degrees": 5.}, "acceptance": {
        "quadrature_reference": {"radial": 9, "azimuthal": 64}, "quadrature_max_abs_light_error": .05}}
    metadata = geometry_preflight(measured, sampler, Scene(), spec)
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata


def test_sensor_replay_reads_only_light_and_validates_axes(tmp_path):
    from scripts.run_exp008_eye import replay_sensor
    p = SensoryParameters(onset_ms=2, offset_ms=4, duration_ms=6)
    light = np.full((3, 4), .5)
    np.savez(tmp_path/"static.npz", sensor_time_ms=np.arange(3)*2, light=light,
             channels=np.full((3, 16), np.nan))
    # Corrupt saved neural outputs are irrelevant: replay recomputes them.
    _, loaded = replay_sensor(tmp_path, "static", tiny_eye(), p, 2)
    np.testing.assert_array_equal(loaded, light)
    np.savez(tmp_path/"static.npz", sensor_time_ms=np.arange(3), light=light)
    with pytest.raises(ValueError):
        replay_sensor(tmp_path, "static", tiny_eye(), p, 2)


def test_replay_cannot_create_a_new_world_experiment_record(monkeypatch, tmp_path):
    from scripts.run_exp008_eye import main
    monkeypatch.setattr("sys.argv", ["run_exp008_eye.py", "--replay-sensors", str(tmp_path)])
    with pytest.raises(ValueError, match="verification-only"):
        main()


def test_motion_boundary_rejects_rgb_frames():
    with pytest.raises(ValueError, match="time x facet"):
        synthetic_readout().channels(np.zeros((20, 16, 16, 3)), SensoryParameters())


def test_eye_runtime_imports_without_optional_download_or_extraction_packages():
    import subprocess
    import sys
    code = """
import builtins
original_import = builtins.__import__
def core_only(name, *args, **kwargs):
    if name.split('.')[0] in {'requests', 'rdata'}:
        raise ImportError('optional package unavailable: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = core_only
import scripts.prepare_exp008_eye
import scripts.run_exp008_eye
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
