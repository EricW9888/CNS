"""Read-only diagnosis of EXP-011 blocked-mass archive diagnostics.

This script never writes or rewrites canonical sensor archives.  It separates
the recorded visibility-mass diagnostic from the scalar light arrays and
checks the latter's provenance independently.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_exp008_eye import evidence_sha256
from scripts.render_exp011_sensors import inputs
from scripts.run_exp008_eye import trace_digest
from src.acceptance_integral import AcceptanceIntegral, acceptance_density, blocker_arc
from src.compound_eye import Scene, pose_rays, rotation_z, sphere_distance
from src.provenance import sha256_file
from src.sensory_chain import SensoryParameters


OUTPUT = ROOT / "results/exp011_sensors_direct"


def archive_metadata_check(target, spec, eye_spec, geometry, condition, level):
    with np.load(target, allow_pickle=False) as data:
        required = {"sensor_time_ms", "light", "mixed", "blocked_mass", "metadata"}
        if set(data.files) != required:
            return {"passed": False, "reason": "field_set", "fields": data.files}
        time = data["sensor_time_ms"].copy()
        light = data["light"].copy()
        mixed = data["mixed"].copy()
        mass = data["blocked_mass"].copy()
        metadata = json.loads(data["metadata"].item())
    p = SensoryParameters(**eye_spec["early_vision"]["parameters"])
    expected_time = np.arange(round(p.duration_ms / level["sensor_dt_ms"])) * level["sensor_dt_ms"]
    expected_impl = {f: evidence_sha256(ROOT / f)
                     for f in ("src/acceptance_integral.py", "scripts/render_exp011_sensors.py")}
    checks = {
        "specification_sha256": metadata.get("specification_sha256") == sha256_file(ROOT / "experiments/EXP-011-acceptance-convergence/specification.json"),
        "implementation_sha256": metadata.get("implementation_sha256") == expected_impl,
        "condition": metadata.get("condition") == condition,
        "level": metadata.get("level") == level,
        "time_grid": np.array_equal(time, expected_time),
        "shape": light.shape == (len(time), len(geometry.directions)) and mixed.shape == light.shape and mass.shape == light.shape,
        "dtypes": mixed.dtype == bool,
        "light_finite": bool(np.isfinite(light).all()),
        "light_bounded": bool(np.all((light >= -1e-12) & (light <= 1 + 1e-12))),
        "mass_finite": bool(np.isfinite(mass).all()),
        "geometry_digest": metadata.get("geometry_digest") == trace_digest(geometry.__dict__),
        "light_digest": metadata.get("light_digest") == trace_digest({"sensor_light": light}),
    }
    return {"passed_except_blocked_mass_bounds": bool(all(checks.values())),
            "checks": checks, "light": light, "mixed": mixed, "mass": mass,
            "time": time, "metadata": metadata}


def pose_index(eye_spec, level, seconds):
    p = SensoryParameters(**eye_spec["early_vision"]["parameters"])
    return int(round((p.onset_ms / 1000 + float(seconds)) * 1000 / level["sensor_dt_ms"]))


def split_mass_detail(geometry, scene, eye_spec, level, condition, seconds, facet):
    solver = AcceptanceIntegral.compile(
        geometry, fwhm_degrees=eye_spec["optics"]["fwhm_degrees"],
        radial=level["radial"], azimuthal=level["azimuthal"])
    rate, speed = eye_spec["pose_conditions"][condition]
    rotation = rotation_z(np.deg2rad(rate) * seconds)
    translation = np.array([speed * seconds, 0., 0.])
    origins, axes = pose_rays(geometry.lens_mm[[facet]], geometry.directions[[facet]], rotation, translation)
    origins, axes = origins[0], axes[0]
    delta = np.asarray(scene.occluder_center) - origins
    distance = np.linalg.norm(delta)
    v = delta / distance
    beta = float(np.arccos(np.clip(np.dot(v, axes), -1, 1)))
    alpha = float(np.arcsin(scene.occluder_radius_mm / distance))
    reference = np.array([0., 0., 1.]) if abs(geometry.directions[facet, 2]) < .9 else np.array([1., 0., 0.])
    first = np.cross(reference, geometry.directions[facet])
    first /= np.linalg.norm(first)
    first = rotation @ first
    second = np.cross(axes, first)
    limit = 3 * solver.sigma
    mixed = bool((beta + alpha > 0) and (abs(beta - alpha) < limit))
    fully_blocked = bool(alpha >= beta + limit)
    if fully_blocked:
        return {"mixed": False, "fully_blocked": True, "split_total_normalized": 1.0,
                "split_total_raw_mass": solver.normalization, "blocked_normalized": 1.0,
                "blocked_raw_mass": solver.normalization, "full_cone_denominator": solver.normalization}
    if not mixed:
        return {"mixed": False, "fully_blocked": False, "split_total_normalized": 1.0,
                "split_total_raw_mass": solver.normalization, "blocked_normalized": 0.0,
                "blocked_raw_mass": 0.0, "full_cone_denominator": solver.normalization}
    cuts = np.array([0., np.clip(abs(beta - alpha), 0, limit),
                     np.clip(beta + alpha, 0, limit), limit])
    nodes, weights = leggauss(level["radial"])
    s = (nodes + 1) * np.pi / 4
    length = np.diff(cuts)
    theta = cuts[:-1, None] + length[:, None] * np.sin(s)[None, :] ** 2
    weight = length[:, None] * (np.pi / 2 * np.sin(s) * np.cos(s))[None, :] * weights[None, :]
    weight *= acceptance_density(theta, solver.sigma) / solver.normalization
    half = blocker_arc(theta, beta, alpha)
    visible = 1 - half / np.pi
    total_normalized = float(weight.sum())
    blocked_normalized = float(np.sum(weight * (1 - visible)))
    return {"mixed": True, "fully_blocked": False,
            "split_total_normalized": total_normalized,
            "split_total_raw_mass": total_normalized * solver.normalization,
            "blocked_normalized": blocked_normalized,
            "blocked_raw_mass": blocked_normalized * solver.normalization,
            "full_cone_denominator": solver.normalization,
            "reference_full_cone_denominator": solver.reference_normalization,
            "beta": beta, "alpha": alpha}


def independent_dense_mass(geometry, scene, eye_spec, condition, seconds, facet,
                           radial=256, azimuthal=4096):
    sigma = np.deg2rad(eye_spec["optics"]["fwhm_degrees"])/np.sqrt(8*np.log(2))
    limit = 3 * sigma
    nodes, weights = leggauss(radial)
    theta = (nodes + 1) * limit / 2
    radial_weights = weights * limit / 2 * acceptance_density(theta, sigma)
    phi = np.arange(azimuthal) * 2 * np.pi / azimuthal
    rate, speed = eye_spec["pose_conditions"][condition]
    rotation = rotation_z(np.deg2rad(rate) * seconds)
    translation = np.array([speed * seconds, 0., 0.])
    origin, axis = pose_rays(geometry.lens_mm[[facet]], geometry.directions[[facet]], rotation, translation)
    origin, axis = origin[0], axis[0]
    reference = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([1., 0., 0.])
    first = np.cross(reference, axis); first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    blocked_total = 0.
    for k, angle in enumerate(theta):
        rays = (axis[None, :] * np.cos(angle)
                + first[None, :] * (np.sin(angle) * np.cos(phi)[:, None])
                + second[None, :] * (np.sin(angle) * np.sin(phi)[:, None]))
        wall = sphere_distance(origin[None, :], rays, np.zeros(3), scene.radius_mm)
        blocker = sphere_distance(origin[None, :], rays, scene.occluder_center, scene.occluder_radius_mm)
        blocked_total += radial_weights[k] * float(np.mean(blocker < wall))
    return float(blocked_total / radial_weights.sum())


def main():
    spec, eye_spec, geometry, _ = inputs()
    scene = Scene(**eye_spec["scene"])
    levels = {x["name"]: x for x in spec["ordered_levels"]}
    archives = {}
    checks = {}
    for condition in spec["conditions"]:
        archives[condition] = {}
        checks[condition] = {}
        for level_name, level in levels.items():
            target = OUTPUT / f"{condition}_{level_name}.npz"
            result = archive_metadata_check(target, spec, eye_spec, geometry, condition, level)
            checks[condition][level_name] = result["checks"]
            archives[condition][level_name] = result

    offenders = []
    for condition in spec["conditions"]:
        for level_name in levels:
            result = archives[condition][level_name]
            bad = result["mass"] > 1 + 1e-12
            if bad.any():
                ti, fi = np.where(bad)
                order = np.argsort(result["mass"][ti, fi])[::-1]
                top = int(order[0])
                offenders.append({
                    "condition": condition, "level": level_name,
                    "count": int(len(ti)), "maximum": float(result["mass"][ti[top], fi[top]]),
                    "maximum_time_index": int(ti[top]), "maximum_facet": int(fi[top]),
                    "maximum_mixed": bool(result["mixed"][ti[top], fi[top]]),
                    "all_offenders_mixed": bool(result["mixed"][ti, fi].all()),
                })

    # Compare every coarse offender at the exact coincident physical pose in
    # production/reference, then do the same for isolated production offenders.
    coincident = []
    for condition in spec["conditions"]:
        base = archives[condition]["coarse"]
        bad = base["mass"] > 1 + 1e-12
        if not bad.any():
            continue
        time_indices, facets = np.where(bad)
        p = SensoryParameters(**eye_spec["early_vision"]["parameters"])
        elapsed = np.clip(base["time"][time_indices] - p.onset_ms, 0,
                          p.offset_ms - p.onset_ms) / 1000
        rows = {"condition": condition, "source_level": "coarse", "source_count": int(len(facets))}
        for name in ("coarse", "production", "reference"):
            level = levels[name]
            idx = np.rint((p.onset_ms / 1000 + elapsed) * 1000 / level["sensor_dt_ms"]).astype(int)
            mass = archives[condition][name]["mass"][idx, facets]
            rows[name] = {"max_blocked_mass": float(mass.max()),
                          "max_overshoot": float(max(0., mass.max() - 1)),
                          "count_above_one": int(np.count_nonzero(mass > 1 + 1e-12))}
        prod = archives[condition]["production"]["light"][
            np.rint((p.onset_ms / 1000 + elapsed) * 1000 / levels["production"]["sensor_dt_ms"]).astype(int), facets]
        ref = archives[condition]["reference"]["light"][
            np.rint((p.onset_ms / 1000 + elapsed) * 1000 / levels["reference"]["sensor_dt_ms"]).astype(int), facets]
        coarse_light = base["light"][time_indices, facets]
        rows["coincident_light"] = {
            "coarse_production_max_abs": float(np.max(abs(coarse_light - prod))),
            "production_reference_max_abs": float(np.max(abs(prod - ref))),
        }
        coincident.append(rows)

    # Independent high-order mass at the worst offender from each affected archive.
    high_order = []
    for row in offenders:
        condition, name = row["condition"], row["level"]
        result = archives[condition][name]
        ti, fi = np.where(result["mass"] > 1 + 1e-12)
        j = int(np.argmax(result["mass"][ti, fi]))
        seconds = float(np.clip(result["time"][ti[j]] - SensoryParameters(**eye_spec["early_vision"]["parameters"]).onset_ms, 0,
                                SensoryParameters(**eye_spec["early_vision"]["parameters"]).offset_ms - SensoryParameters(**eye_spec["early_vision"]["parameters"]).onset_ms) / 1000)
        high_order.append({"condition": condition, "level": name, "facet": int(fi[j]),
                           "seconds": seconds, "split_detail": split_mass_detail(geometry, scene, eye_spec, levels[name], condition, seconds, int(fi[j])),
                           "independent_dense_mass_256x4096": independent_dense_mass(geometry, scene, eye_spec, condition, seconds, int(fi[j]))})

    metadata_pass = all(all(all(values.values()) for values in condition.values()) for condition in checks.values())
    report = {
        "scope": "read-only diagnosis of immutable canonical archives; no archive or implementation changes",
        "metadata_shape_source_digest_checks_except_blocked_mass_bounds": metadata_pass,
        "offenders": offenders,
        "coincident_level_comparison": coincident,
        "independent_high_order_mass": high_order,
        "interpretation_boundary": "Blocked-mass overshoot is retained as a verifier failure. The independent dense mass is diagnostic only and cannot rescue the archive gate or alter canonical data.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
