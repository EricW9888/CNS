"""Independent numerical preflight for EXP-011 acceptance integration."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_exp009_retinotopy import load_resolved
from src.acceptance_integral import AcceptanceIntegral, acceptance_density, blocker_arc
from src.compound_eye import Scene, pose_rays, rotation_z


def _basis(axes):
    reference = np.where((abs(axes[:, 2]) < .9)[:, None], [0., 0., 1.], [1., 0., 0.])
    first = np.cross(reference, axes)
    first /= np.linalg.norm(first, axis=1)[:, None]
    return first, np.cross(axes, first)


def dense_sample(geometry, scene, fwhm_degrees, rotation, translation, ids,
                 radial=96, azimuthal=512):
    """Direct dense angular ray sampling, independent of visibility splitting."""
    sigma = np.deg2rad(fwhm_degrees)/np.sqrt(8*np.log(2))
    limit = 3*sigma
    nodes, weights = leggauss(radial)
    theta = (nodes+1)*limit/2
    radial_weights = weights*limit/2*acceptance_density(theta, sigma)
    phi = np.arange(azimuthal)*2*np.pi/azimuthal
    origins, axes = pose_rays(geometry.lens_mm[ids], geometry.directions[ids],
                              rotation, translation)
    first, second = _basis(axes)
    reference = AcceptanceIntegral.compile(geometry, fwhm_degrees=fwhm_degrees,
                                           radial=radial, azimuthal=azimuthal)
    values = np.empty(len(ids))
    numerators = np.empty(len(ids))
    for k in range(len(ids)):
        rays = (axes[k][None, None, :]*np.cos(theta)[:, None, None]
                + first[k][None, None, :]*(np.sin(theta)[:, None]*np.cos(phi)[None, :])[..., None]
                + second[k][None, None, :]*(np.sin(theta)[:, None]*np.sin(phi)[None, :])[..., None])
        sampled = scene.radiance(origins[k, None, None], rays)
        numerators[k] = np.sum(sampled.mean(axis=-1)*radial_weights)
        values[k] = numerators[k]/reference.normalization
    return values, numerators, reference


def _direct_reference(geometry, scene, fwhm_degrees, rotation, translation, facet):
    """High-accuracy nested adaptive angular reference for one measured cone."""
    sigma = np.deg2rad(fwhm_degrees)/np.sqrt(8*np.log(2))
    limit = 3*sigma
    origin, axis = pose_rays(geometry.lens_mm[facet][None], geometry.directions[facet][None],
                             rotation, translation)
    origin, axis = origin[0], axis[0]
    first, second = _basis(axis[None])
    first, second = first[0], second[0]
    delta = np.asarray(scene.occluder_center)-origin
    distance = np.linalg.norm(delta)
    beta = float(np.arccos(np.clip(np.dot(delta/distance, axis), -1, 1)))
    alpha = float(np.arcsin(scene.occluder_radius_mm/distance)) if scene.occluder_radius_mm else 0.
    reference_mass = quad(lambda theta: float(acceptance_density(theta, sigma)),
                          0., limit, epsabs=2e-13, epsrel=2e-13, limit=300)[0]

    def ray(theta, phi):
        return (axis*np.cos(theta)+first*np.sin(theta)*np.cos(phi)
                +second*np.sin(theta)*np.sin(phi))

    def phi_mean(theta):
        if scene.occluder_radius_mm <= 0.:
            intervals = [(0., 2*np.pi)]
        else:
            half = float(blocker_arc(np.asarray([theta]), beta, alpha)[0])
            if half >= np.pi-1e-14:
                return float(scene.occluder_light)
            endpoints = sorted({0., 2*np.pi, (float(np.arctan2(
                np.dot(delta/distance, second), np.dot(delta/distance, first)))
                -half) % (2*np.pi), (float(np.arctan2(
                np.dot(delta/distance, second), np.dot(delta/distance, first)))
                +half) % (2*np.pi)})
            intervals = list(zip(endpoints[:-1], endpoints[1:]))
        total = 0.
        for lo, hi in intervals:
            if hi <= lo:
                continue
            total += quad(lambda phi: float(scene.radiance(
                origin[None], ray(theta, phi)[None])[0]), lo, hi,
                epsabs=3e-12, epsrel=3e-12, limit=300)[0]
        return total/(2*np.pi)

    points = []
    if scene.occluder_radius_mm > 0.:
        points = [x for x in (abs(beta-alpha), beta+alpha) if 0. < x < limit]
    numerator = quad(lambda theta: float(acceptance_density(theta, sigma))*phi_mean(theta),
                     0., limit, points=points, epsabs=2e-11, epsrel=2e-11, limit=300)[0]
    return numerator, reference_mass, numerator/reference_mass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/exp011_preflight.json")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT/"results").resolve()):
        raise ValueError("preflight output must stay under ignored results")

    _, eye_spec, geometry, _ = load_resolved()
    scene = Scene(**eye_spec["scene"])
    fwhm = eye_spec["optics"]["fwhm_degrees"]
    poses = {
        "static": (rotation_z(0.), np.zeros(3)),
        "translation_positive_X": (rotation_z(0.), np.array([1.5, 0., 0.])),
        "yaw_positive_Z": (rotation_z(np.deg2rad(45.*1.0)), np.zeros(3)),
    }
    orders = ((8, 16), (16, 32), (32, 64))
    result = {"checks": {}, "passed": True}

    # Constant radiance checks cover every measured cone and every visibility class.
    constant = replace(scene, amplitude=0., mean=.37, occluder_light=.37)
    constant_rows = {}
    for radial, azimuthal in orders:
        sampler = AcceptanceIntegral.compile(geometry, fwhm_degrees=fwhm,
                                             radial=radial, azimuthal=azimuthal)
        light, diagnostic = sampler.sample(constant, diagnostics=True)
        error = float(abs(light-.37).max())
        constant_rows[f"{radial}x{azimuthal}"] = {
            "max_final_normalized_light_error": error,
            "full_cone_mass_error": float(sampler.normalization-sampler.reference_normalization),
            "mixed_cones": int(diagnostic["mixed"].sum()),
            "fully_blocked_cones": int(np.sum(diagnostic["blocked_mass"] > 1-1e-12)),
        }
    result["checks"]["constant_radiance"] = constant_rows

    selected = {}
    numerator_rows = []
    for label, (rotation, translation) in poses.items():
        base = AcceptanceIntegral.compile(geometry, fwhm_degrees=fwhm, radial=32, azimuthal=64)
        _, diagnostic = base.sample(scene, rotation, translation, diagnostics=True)
        mixed_ids = np.flatnonzero(diagnostic["mixed"])
        clear_ids = np.flatnonzero((~diagnostic["mixed"]) & (diagnostic["blocked_mass"] == 0))
        blocked_ids = np.flatnonzero(diagnostic["blocked_mass"] > 1-1e-12)
        selected[label] = {"mixed": mixed_ids[:8].tolist(), "clear": clear_ids[:8].tolist(),
                           "fully_blocked": blocked_ids[:8].tolist()}
        # Adaptive references are intentionally small and independent; dense checks below
        # cover the larger measured-eye population.
        for facet in np.r_[clear_ids[:2], mixed_ids[:4], blocked_ids[:1]]:
            truth_num, truth_mass, truth_light = _direct_reference(
                geometry, scene, fwhm, rotation, translation, int(facet))
            sampler = AcceptanceIntegral.compile(geometry, fwhm_degrees=fwhm,
                                                 radial=32, azimuthal=64)
            value, diagnostic = sampler.sample(scene, rotation, translation, diagnostics=True)
            i = int(facet)
            numerator_rows.append({"pose": label, "facet": i,
                                   "numerator_error": float(abs(diagnostic["numerator"][i]-truth_num)),
                                   "full_cone_mass_error": float(abs(sampler.normalization-truth_mass)),
                                   "final_normalized_light_error": float(abs(value[i]-truth_light))})
    result["checks"]["adaptive_direct_reference"] = numerator_rows
    result["checks"]["selected_measured_facets"] = selected
    result["passed"] &= max((x["final_normalized_light_error"] for x in numerator_rows), default=0.) <= 1e-4

    # Dense direct rays are independent of blocker-arc splitting. Check every mixed
    # measured-eye cone at two poses, with a fixed high-order direct ray grid.
    dense_rows = []
    for label in ("static", "translation_positive_X"):
        rotation, translation = poses[label]
        sampler = AcceptanceIntegral.compile(geometry, fwhm_degrees=fwhm, radial=32, azimuthal=64)
        split, diagnostic = sampler.sample(scene, rotation, translation, diagnostics=True)
        ids = np.flatnonzero(diagnostic["mixed"])
        dense, dense_num, dense_sampler = dense_sample(
            geometry, scene, fwhm, rotation, translation, ids, radial=256, azimuthal=4096)
        errors = abs(split[ids]-dense)
        dense_rows.append({"pose": label, "cones": int(len(ids)),
                           "max_numerator_error": float(abs(diagnostic["numerator"][ids]-dense_num).max()),
                           "max_final_normalized_light_error": float(errors.max()),
                           "dense_full_cone_mass_error": float(dense_sampler.normalization-dense_sampler.reference_normalization)})
    result["checks"]["dense_measured_eye_mixed_cones"] = dense_rows
    result["passed"] &= max((x["max_final_normalized_light_error"] for x in dense_rows), default=0.) <= 1e-4

    # Limiting visibility: zero visible fraction, zero blocker, and clear cones retain
    # the same global normalization rather than a visibility-piece average.
    sampler = AcceptanceIntegral.compile(geometry, fwhm_degrees=fwhm, radial=32, azimuthal=64)
    no_blocker, no_diag = sampler.sample(replace(constant, occluder_radius_mm=0.), diagnostics=True)
    large_scene = replace(scene, occluder_radius_mm=1.5)
    large, large_diag = sampler.sample(large_scene, diagnostics=True)
    full = large_diag["blocked_mass"] > 1-1e-12
    result["checks"]["visibility_limits"] = {
        "no_blocker_max_blocked_mass": float(no_diag["blocked_mass"].max()),
        "large_blocker_min_blocked_mass": float(large_diag["blocked_mass"].min()),
        "large_blocker_max_blocked_mass": float(large_diag["blocked_mass"].max()),
        "fully_blocked_cones": int(full.sum()),
        "clear_constant_light_error": float(abs(no_blocker-.37).max()),
        "fully_blocked_light_error": float(abs(large[full]-large_scene.occluder_light).max()) if full.any() else None,
    }
    result["passed"] &= no_diag["blocked_mass"].max() == 0.
    result["passed"] &= bool(full.any())
    result["passed"] &= result["checks"]["visibility_limits"]["clear_constant_light_error"] <= 1e-12
    result["passed"] &= result["checks"]["visibility_limits"]["fully_blocked_light_error"] <= 1e-12

    # Order convergence is evaluated against the adaptive references, without adding
    # a candidate level. Use the static mixed facets already selected above.
    convergence_rows = []
    facets = np.asarray(selected["static"]["mixed"][:4], dtype=int)
    for facet in facets:
        truth_num, truth_mass, truth_light = _direct_reference(
            geometry, scene, fwhm, poses["static"][0], poses["static"][1], int(facet))
        errors = []
        for radial, azimuthal in orders:
            sampler = AcceptanceIntegral.compile(geometry, fwhm_degrees=fwhm,
                                                 radial=radial, azimuthal=azimuthal)
            value = sampler.sample(scene, poses["static"][0], poses["static"][1])[int(facet)]
            errors.append(float(abs(value-truth_light)))
        convergence_rows.append({"facet": int(facet), "final_normalized_light_errors": errors,
                                 "nonincreasing": bool(all(a+1e-10 >= b for a,b in zip(errors, errors[1:])))})
    result["checks"]["order_convergence"] = convergence_rows
    result["passed"] &= all(x["nonincreasing"] for x in convergence_rows)

    result["passed"] = bool(result["passed"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit("EXP-011 analytical preflight failed")


if __name__ == "__main__":
    main()
