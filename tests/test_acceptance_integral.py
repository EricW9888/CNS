from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from src.acceptance_integral import (AcceptanceIntegral, SmoothWallTable,
                                    acceptance_density, blocker_arc, wall_radiance)
from src.compound_eye import EyeGeometry, EyeSampler, Scene, rotation_z


def eye(angles=(0., .12, .22, .5)):
    axes = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(len(angles))])
    return EyeGeometry(np.zeros_like(axes), axes, np.array([True, True, False, False]),
                       np.full((4, 6), -1, int), np.zeros((4, 2), int))


@pytest.mark.parametrize("order", [8, 16, 32])
def test_constant_radiance_conserved_at_clear_mixed_and_blocked_cones(order):
    scene = Scene(amplitude=0., mean=.5, occluder_light=.5)
    s = AcceptanceIntegral.compile(eye(), radial=order, azimuthal=2*order)
    light, diagnostic = s.sample(scene, diagnostics=True)
    tolerances = {8: 1e-4, 16: 1e-8, 32: 1e-10}
    np.testing.assert_allclose(light, .5, atol=tolerances[order], rtol=0)
    assert diagnostic["mixed"].any()
    assert (diagnostic["blocked_mass"] == 1).any()
    assert (diagnostic["blocked_mass"] == 0).any()
    assert abs(diagnostic["full_cone_mass_error"]) < 3e-12


def test_centered_blocker_matches_independent_gaussian_mass_integral():
    # Small blocker puts its edge inside the central cone (no full blockage).
    scene = Scene(amplitude=0., occluder_radius_mm=.15)
    s = AcceptanceIntegral.compile(eye(), radial=32, azimuthal=64)
    alpha = np.arcsin(scene.occluder_radius_mm/4.)
    total = quad(lambda t: acceptance_density(t, s.sigma), 0, 3*s.sigma, epsabs=1e-14)[0]
    mass = quad(lambda t: acceptance_density(t, s.sigma), 0, alpha, epsabs=1e-14)[0]/total
    light, d = s.sample(scene, diagnostics=True)
    assert light[0] == pytest.approx(scene.mean+(scene.occluder_light-scene.mean)*mass, abs=2e-14)
    assert d["blocked_mass"][0] == pytest.approx(mass, abs=2e-14)


def test_visibility_pieces_share_one_global_normalization():
    scene = Scene(amplitude=0., mean=.73, occluder_light=.11, occluder_radius_mm=.15)
    s = AcceptanceIntegral.compile(eye(), radial=32, azimuthal=64)
    light, d = s.sample(scene, diagnostics=True)
    assert d["mixed"].any()
    assert np.all(d["blocked_mass"] >= -1e-14)
    assert np.all(d["blocked_mass"] <= 1+1e-14)
    # A single denominator gives the clear cone its full value and preserves
    # the blocked-light mixture in a mixed cone; independent piece
    # renormalization would erase that mixture.
    clear = np.flatnonzero((~d["mixed"]) & (d["blocked_mass"] == 0))
    mixed = np.flatnonzero(d["mixed"])
    assert len(clear) and len(mixed)
    assert light[clear[0]] == pytest.approx(.73, abs=1e-10)
    i = mixed[0]
    expected = .73+(scene.occluder_light-.73)*d["blocked_mass"][i]
    assert light[i] == pytest.approx(expected, abs=2e-6)


def test_blocked_arc_tangencies_and_symmetry_against_direct_ray_discriminant():
    theta = np.linspace(.005, .1, 15)[:, None]
    beta, alpha = .2, np.arcsin(.8/4)
    half = blocker_arc(theta, beta, alpha)
    phi = np.arange(4096)[None, :]*2*np.pi/4096
    cosine = np.cos(theta)*np.cos(beta)+np.sin(theta)*np.sin(beta)*np.cos(phi)
    fraction = (cosine >= np.cos(alpha)).mean(axis=-1)
    np.testing.assert_allclose(half[:, 0]/np.pi, fraction, atol=1/4096)
    np.testing.assert_allclose(np.cos(phi), np.cos(-phi), atol=0)


def test_mixed_cone_convergence_vs_independent_dense_legacy_rays():
    g, scene = eye(), Scene()
    values = [AcceptanceIntegral.compile(g, radial=n, azimuthal=2*n).sample(scene) for n in (8, 16, 32)]
    assert abs(values[1]-values[2]).max() < abs(values[0]-values[1]).max()
    dense = EyeSampler.compile(g, radial=128, azimuthal=1024).sample(scene)
    np.testing.assert_allclose(values[2], dense, atol=1e-4, rtol=0)


def test_wall_expression_matches_frozen_scene_and_away_boundary_converges():
    scene = Scene(occluder_radius_mm=0., stripe_cycles=3)
    g = eye()
    sampler = EyeSampler.compile(g, radial=8, azimuthal=16)
    origins = np.broadcast_to(g.lens_mm[:, None], sampler.local_rays.shape)+[.3, .1, 0.]
    np.testing.assert_allclose(wall_radiance(scene, origins, sampler.local_rays),
                               scene.radiance(origins, sampler.local_rays), atol=2e-15)
    values = [AcceptanceIntegral.compile(g, radial=n, azimuthal=2*n).sample(scene) for n in (8, 16, 32)]
    assert abs(values[1]-values[2]).max() < 1e-8
    assert abs(values[1]-values[2]).max() <= abs(values[0]-values[1]).max()+1e-14


def test_clear_visibility_path_matches_frozen_sampler_exactly():
    g = eye()
    scene = Scene(occluder_radius_mm=0.)
    split = AcceptanceIntegral.compile(g, radial=16, azimuthal=32)
    frozen = EyeSampler.compile(g, radial=16, azimuthal=32)
    for rotation, translation in ((np.eye(3), [0., 0., 0.]),
                                  (rotation_z(.31), [.17, 0., 0.])):
        np.testing.assert_allclose(
            split.sample(scene, rotation, translation),
            frozen.sample(scene, rotation, translation),
            atol=2e-15,
            rtol=0,
        )


def test_visibility_transition_pose_causality_and_observer_isolation(monkeypatch):
    import src.compound_eye as module
    monkeypatch.setattr(module, "observer_camera", lambda *a, **k: pytest.fail("observer became sensory input"))
    s, scene = AcceptanceIntegral.compile(eye()), Scene()
    baseline, d = s.sample(scene, diagnostics=True)
    moved, md = s.sample(scene, rotation_z(.45), diagnostics=True)
    assert d["blocked_mass"][0] == 1
    assert md["blocked_mass"][0] == 0
    assert baseline[0] == scene.occluder_light
    assert moved[0] != baseline[0]
    np.testing.assert_array_equal(baseline, s.sample(scene))
    # Alternating pose requests do not mutate the sensor or retain future state.
    np.testing.assert_array_equal(moved, s.sample(scene, rotation_z(.45)))


def test_smooth_wall_cache_verified_off_nodes_and_never_caches_visibility():
    g, scene = eye(), Scene()
    sampler = EyeSampler.compile(g, radial=8, azimuthal=16)
    table = SmoothWallTable(sampler, scene, 0., 1.)
    s = AcceptanceIntegral.compile(g, radial=8, azimuthal=16)
    for x in (.137, .423, .951):
        direct = sampler.sample(replace(scene, occluder_radius_mm=0), translation=[x, 0, 0])
        np.testing.assert_allclose(table.evaluate(x), direct, atol=1e-9, rtol=0)
        accelerated = s.sample(scene, translation=[x, 0, 0], smooth_light=table.evaluate(x))
        np.testing.assert_allclose(accelerated, s.sample(scene, translation=[x, 0, 0]), atol=1e-9, rtol=0)
    with pytest.raises(ValueError):
        table.evaluate(1.1)
    assert table.max_checked_error <= 1e-10
    assert table.validate(np.linspace(0., 1., 9)) <= 1e-10


def test_nested_shared_physical_points_and_determinism():
    s, scene = AcceptanceIntegral.compile(eye()), Scene()
    fine = np.asarray([s.sample(scene, translation=[x, 0, 0]) for x in np.arange(9)*.001])
    coarse = np.asarray([s.sample(scene, translation=[x, 0, 0]) for x in np.arange(5)*.002])
    np.testing.assert_array_equal(coarse, fine[::2])


def test_rejects_unsupported_scene_and_bad_numerical_arguments():
    for kwargs in ({"radial": 0}, {"azimuthal": 2}, {"fwhm_degrees": np.nan}):
        with pytest.raises(ValueError):
            AcceptanceIntegral.compile(eye(), **kwargs)
    s = AcceptanceIntegral.compile(eye())
    with pytest.raises(ValueError):
        s.sample(Scene(occluder_center=(11., 0., 0.)))
    with pytest.raises(ValueError):
        s.sample(Scene(), translation=[4., 0., 0.])


def test_predeclared_successor_does_not_change_biology_or_historical_criterion():
    import json
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root/"experiments/EXP-011-acceptance-convergence/specification.json").read_text())
    assert spec["free_biological_parameters_changed"] == []
    assert spec["acceptance"]["maximum_local_stage_relative_error"] == .05
    assert [x["radial"] for x in spec["ordered_levels"]] == [8, 16, 32]
    assert spec["candidate"]["name"] == "production"
