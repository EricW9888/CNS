"""Bounded diagnosis of the frozen EXP-011 smooth-wall Chebyshev cache.

This is a diagnostic replay only.  It does not construct sensor archives,
evaluate canonical translations, or alter the degree, tolerance, interval, or
subdivision-depth contract.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_exp009_retinotopy import load_resolved
from src.acceptance_integral import acceptance_density
from src.compound_eye import EyeSampler, Scene, sphere_distance


FACET = 1572
FWHM_DEGREES = 5.0
DEGREE = 16
TOLERANCE = 1e-10
MAX_DEPTH = 12
NODES = np.cos(np.arange(DEGREE + 1) * np.pi / DEGREE)
CHECKS = np.cos((np.arange(DEGREE) + .5) * np.pi / DEGREE)


def scalar_value(sampler, scene, geometry, facet, x):
    origin = geometry.lens_mm[facet] + np.array([float(x), 0., 0.])
    return float(scene.radiance(origin[None, None, :], sampler.local_rays[facet:facet + 1])[0]
                 @ sampler.weights)


def local_fit(sampler, scene, geometry, facet, a, b):
    values = np.array([scalar_value(sampler, scene, geometry, facet,
                                    (a + b) / 2 + (b - a) / 2 * x)
                       for x in NODES])
    coefficients = chebfit(NODES, values, DEGREE)
    direct = np.array([scalar_value(sampler, scene, geometry, facet,
                                    (a + b) / 2 + (b - a) / 2 * x)
                       for x in CHECKS])
    prediction = chebval(CHECKS, coefficients)
    return coefficients, float(np.max(abs(prediction - direct)))


def independent_direct(geometry, scene, facet, x, radial, azimuthal):
    """Independent high-order angular ray evaluation for one measured cone."""
    sigma = np.deg2rad(FWHM_DEGREES) / np.sqrt(8 * np.log(2))
    limit = 3 * sigma
    nodes, weights = leggauss(radial)
    theta = (nodes + 1) * limit / 2
    radial_weights = weights * limit / 2 * acceptance_density(theta, sigma)
    phi = np.arange(azimuthal) * 2 * np.pi / azimuthal
    axis = geometry.directions[facet]
    origin = geometry.lens_mm[facet] + np.array([float(x), 0., 0.])
    reference = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([1., 0., 0.])
    first = np.cross(reference, axis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    total = 0.
    for k, angle in enumerate(theta):
        rays = (axis[None, :] * np.cos(angle)
                + first[None, :] * (np.sin(angle) * np.cos(phi)[:, None])
                + second[None, :] * (np.sin(angle) * np.sin(phi)[:, None]))
        # Recompute the enclosing-wall intersection and painted radiance here;
        # do not call EyeSampler.sample or SmoothWallTable.evaluate.
        b = rays @ origin
        a = np.sum(rays * rays, axis=1)
        distance = (-b + np.sqrt(b * b + a * (scene.radius_mm ** 2 - np.sum(origin * origin)))) / a
        point = origin[None, :] + distance[:, None] * rays
        rho = np.hypot(point[:, 0], point[:, 1])
        azimuth = np.arctan2(point[:, 1], point[:, 0])
        values = scene.mean + scene.amplitude * (rho / scene.radius_mm) * np.cos(
            scene.stripe_cycles * azimuth)
        total += radial_weights[k] * float(values.mean())
    return total / float(radial_weights.sum())


def main():
    _, _, geometry, _ = load_resolved()
    sampler = EyeSampler.compile(geometry, fwhm_degrees=FWHM_DEGREES,
                                 radial=8, azimuthal=16)
    scene = Scene(occluder_radius_mm=0.)
    worst_a, worst_b = .53564453125, .5361328125
    worst_x = .5358647419090993
    coefficients, local_error = local_fit(sampler, scene, geometry, FACET,
                                           worst_a, worst_b)

    xs = np.linspace(worst_a - .002, worst_b + .002, 2001)
    values = np.array([scalar_value(sampler, scene, geometry, FACET, x) for x in xs])
    dx = float(xs[1] - xs[0])
    first = np.gradient(values, dx)
    second = np.gradient(first, dx)
    near_pole = (float("inf"), None, None)
    for x in xs:
        origin = geometry.lens_mm[FACET] + np.array([float(x), 0., 0.])
        rays = sampler.local_rays[FACET]
        wall = sphere_distance(origin[None, :], rays, np.zeros(3), scene.radius_mm)
        point = origin[None, :] + wall[:, None] * rays
        rho = np.hypot(point[:, 0], point[:, 1])
        k = int(np.argmin(rho))
        if float(rho[k]) < near_pole[0]:
            near_pole = (float(rho[k]), float(x), k)

    intervals = [
        (.53125, .5390625),
        (.53515625, .5390625),
        (.53515625, .537109375),
        (.53515625, .5361328125),
        (.53564453125, .5361328125),
    ]
    coefficient_decay = []
    for a, b in intervals:
        co, error = local_fit(sampler, scene, geometry, FACET, a, b)
        coefficient_decay.append({
            "a": a, "b": b, "max_check_error": error,
            "max_abs_coeff_c12_to_c16": float(np.max(abs(co[12:]))),
        })

    points = [worst_a, worst_x, worst_b]
    direct_low = {str(x): scalar_value(sampler, scene, geometry, FACET, x) for x in points}
    cache_co = chebfit(NODES, np.array([
        scalar_value(sampler, scene, geometry, FACET,
                     (worst_a + worst_b) / 2 + (worst_b - worst_a) / 2 * x)
        for x in NODES]), DEGREE)
    cache = {str(x): float(chebval((2 * x - worst_a - worst_b) / (worst_b - worst_a), cache_co))
             for x in points}

    mapped = (0. + 2.) / 2 + (2. - 0.) / 2 * np.r_[NODES, CHECKS]
    roundtrip = (2 * mapped - 0. - 2.) / (2. - 0.)
    report = {
        "scope": "preflight diagnosis only; no canonical +X/-X translation output",
        "frozen_contract": {"degree": DEGREE, "tolerance": TOLERANCE,
                            "maximum_subdivision_depth": MAX_DEPTH,
                            "interval": [0., 2.]},
        "component": {"measured_facet": FACET, "ray_count": 8 * 16,
                       "near_pole_min_wall_xy_radius_mm": near_pole[0],
                       "near_pole_x_mm": near_pole[1],
                       "near_pole_ray_index": near_pole[2]},
        "worst_leaf": {"a": worst_a, "b": worst_b, "check_x": worst_x,
                       "direct": direct_low[str(worst_x)],
                       "cache": cache[str(worst_x)],
                       "absolute_error": abs(cache[str(worst_x)] - direct_low[str(worst_x)]),
                       "relative_to_light": abs(cache[str(worst_x)] - direct_low[str(worst_x)]) / direct_low[str(worst_x)],
                       "local_check_error": local_error},
        "coefficient_decay": coefficient_decay,
        "continuity": {"x_range": [float(xs[0]), float(xs[-1])],
                        "value_range": float(values.max() - values.min()),
                        "max_abs_first_derivative": float(np.max(abs(first))),
                        "max_abs_second_derivative": float(np.max(abs(second))),
                        "max_adjacent_jump": float(np.max(abs(np.diff(values)))),
                        "max_second_derivative_x": float(xs[np.argmax(abs(second))])},
        "independent_direct": {
            str([radial, azimuthal]): {
                str(x): independent_direct(geometry, scene, FACET, x, radial, azimuthal)
                for x in points
            }
            for radial, azimuthal in [(64, 1024), (128, 2048), (256, 4096)]
        },
        "mapping_and_axes": {
            "values_shape": [17, len(geometry.directions)],
            "checks_shape": [16, len(geometry.directions)],
            "raw_chebval_shape": [len(geometry.directions), 16],
            "prediction_after_transpose_shape": [16, len(geometry.directions)],
            "mapping_roundtrip_max_abs": float(np.max(abs(roundtrip - np.r_[NODES, CHECKS]))),
            "endpoint_map": [float(mapped[0]), float(mapped[16])],
            "check_range": [float(mapped[17:].min()), float(mapped[17:].max())],
        },
        "interpretation": "genuine frozen degree-16/depth-12 accelerator limitation at one near-pole ray; no implementation correction identified",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
