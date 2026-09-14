"""Produce bounded full-eye optical movies; verification reuses recorded light."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_exp008_eye import evidence_sha256
from scripts.prepare_exp009_retinotopy import frozen_entry_digest
from scripts.run_exp008_eye import trace_digest
from scripts.run_exp009_retinotopy import load_resolved
from src.acceptance_integral import AcceptanceIntegral
from src.compound_eye import EyeSampler, Scene, rotation_z
from src.provenance import sha256_file
from src.sensory_chain import SensoryParameters

SPEC = ROOT/"experiments/EXP-011-acceptance-convergence/specification.json"


def _metadata_value(value):
    """Decode a scalar string field from an npz archive."""
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError("metadata field must be scalar")
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf8")
    return value


def inputs():
    spec = json.loads(SPEC.read_text())
    for entry in spec["frozen_files"]:
        if frozen_entry_digest(entry) != entry["sha256"]:
            raise ValueError(f"frozen source changed: {entry['path']}")
    _, eye_spec, g, reference = load_resolved()
    return spec, eye_spec, g, reference


def verify_sensor_output(target, spec, eye_spec, geometry, condition, level):
    """Verify an existing ignored sensor archive before replaying it.

    Existing files are evidence only when their source, geometry, temporal
    grid, shape, bounds, and content digest still match the current run.
    Stale archives are rejected rather than silently reused.
    """
    with np.load(target, allow_pickle=False) as data:
        required = {"sensor_time_ms", "light", "mixed", "blocked_mass", "metadata"}
        if set(data.files) != required:
            raise ValueError(f"sensor archive fields differ: {target}")
        time = data["sensor_time_ms"].copy()
        light = data["light"].copy()
        mixed = data["mixed"].copy()
        blocked_mass = data["blocked_mass"].copy()
        metadata = json.loads(_metadata_value(data["metadata"]))
    if metadata.get("specification_sha256") != sha256_file(SPEC):
        raise ValueError(f"sensor archive specification is stale: {target}")
    expected_implementation = {
        f: evidence_sha256(ROOT/f)
        for f in ("src/acceptance_integral.py", "scripts/render_exp011_sensors.py")
    }
    if metadata.get("implementation_sha256") != expected_implementation:
        raise ValueError(f"sensor archive implementation is stale: {target}")
    if metadata.get("condition") != condition or metadata.get("level") != level:
        raise ValueError(f"sensor archive identity differs: {target}")
    expected_dt = level["sensor_dt_ms"]
    expected_time = np.arange(round(SensoryParameters(**eye_spec["early_vision"]["parameters"]).duration_ms/expected_dt))*expected_dt
    if not np.array_equal(time, expected_time):
        raise ValueError(f"sensor archive time grid differs: {target}")
    if light.shape != (len(time), len(geometry.directions)) or mixed.shape != light.shape or blocked_mass.shape != light.shape:
        raise ValueError(f"sensor archive shape differs: {target}")
    if (not np.isfinite(light).all()
            or np.any((light < -1e-12) | (light > 1+1e-12))
            or not np.isfinite(blocked_mass).all()
            or np.any((blocked_mass < -1e-12) | (blocked_mass > 1+1e-12))
            or mixed.dtype != bool):
        raise ValueError(f"sensor archive values are invalid: {target}")
    if metadata.get("geometry_digest") != trace_digest(geometry.__dict__):
        raise ValueError(f"sensor archive geometry differs: {target}")
    if metadata.get("light_digest") != trace_digest({"sensor_light": light}):
        raise ValueError(f"sensor archive content digest differs: {target}")
    return {"sensor_time_ms": time, "light": light, "mixed": mixed,
            "blocked_mass": blocked_mass, "metadata": metadata}


def sample_physical_pose(sampler, scene, rotation, translation, *, smooth_light=None):
    """Run one canonical optical pose, optionally with the exact Z cache."""
    kwargs = {"diagnostics": True}
    if smooth_light is not None:
        kwargs["smooth_light"] = smooth_light
    return sampler.sample(scene, rotation, translation, **kwargs)


def render_condition(condition, output):
    spec, eye_spec, g, _ = inputs()
    p = SensoryParameters(**eye_spec["early_vision"]["parameters"])
    scene = Scene(**eye_spec["scene"])
    rate, speed = eye_spec["pose_conditions"][condition]
    output = Path(output)
    for level in spec["ordered_levels"]:
        target = output/f"{condition}_{level['name']}.npz"
        if target.exists():
            verify_sensor_output(target, spec, eye_spec, g, condition, level)
            print(f"Existing {target.name}; metadata and digest verification passed", flush=True)
            continue
        sampler = AcceptanceIntegral.compile(g, fwhm_degrees=eye_spec["optics"]["fwhm_degrees"],
                                             radial=level["radial"], azimuthal=level["azimuthal"])
        smooth = EyeSampler.compile(g, radial=level["radial"], azimuthal=level["azimuthal"],
                                    fwhm_degrees=eye_spec["optics"]["fwhm_degrees"])
        wall = replace(scene, occluder_radius_mm=0.)
        table = None
        translation_direct = bool(speed)
        if speed:
            # The predeclared Chebyshev translation accelerator failed its
            # frozen contract before canonical translation evaluation.  The
            # direct sampler computes the identical optical integral and is
            # now the approved translation execution path.
            cache = None
        else:
            base = smooth.sample(wall)-scene.mean
            quarter = smooth.sample(wall, rotation_z(np.pi/(2*scene.stripe_cycles)))-scene.mean
            cache = lambda seconds: scene.mean+base*np.cos(scene.stripe_cycles*np.deg2rad(rate)*seconds)+quarter*np.sin(scene.stripe_cycles*np.deg2rad(rate)*seconds)
        validation_seconds = np.linspace(0, 2, 41)
        checked_error = None
        if not translation_direct:
            checked_error = 0.
            for seconds in validation_seconds:
                direct = smooth.sample(wall, rotation_z(np.deg2rad(rate)*seconds), [speed*seconds, 0., 0.])
                checked_error = max(checked_error, float(abs(cache(seconds)-direct).max()))
            if checked_error > spec["smooth_wall_acceleration"]["direct_verification_absolute_tolerance"]:
                raise RuntimeError("smooth-wall acceleration fails direct full-eye verification")
        dt = level["sensor_dt_ms"]
        time = np.arange(round(p.duration_ms/dt))*dt
        elapsed = np.clip(time-p.onset_ms, 0, p.offset_ms-p.onset_ms)/1000
        unique, inverse = np.unique(elapsed, return_inverse=True)
        if not rate and not speed:
            unique, inverse = np.array([0.]), np.zeros(len(elapsed), int)
        light = np.empty((len(unique), len(g.left)))
        mass = np.empty_like(light)
        mixed = np.empty(light.shape, bool)
        for k, seconds in enumerate(unique):
            light[k], diagnostic = sample_physical_pose(
                sampler, scene, rotation_z(np.deg2rad(rate)*seconds),
                [speed*seconds, 0., 0.],
                smooth_light=cache(seconds) if cache is not None else None)
            mass[k], mixed[k] = diagnostic["blocked_mass"], diagnostic["mixed"]
            if k % 500 == 0:
                print(f"{condition}/{level['name']}: {k}/{len(unique)} physical poses", flush=True)
        values = light[inverse]
        fresh_error = None
        if not translation_direct:
            fresh_error = 0.
            for seconds in validation_seconds:
                direct = sampler.sample(scene, rotation_z(np.deg2rad(rate)*seconds), [speed*seconds, 0., 0.])
                k = int(round((p.onset_ms/1000+seconds)*1000/level["sensor_dt_ms"]))
                fresh_error = max(fresh_error, float(abs(values[k]-direct).max()))
            if fresh_error > spec["smooth_wall_acceleration"]["direct_verification_absolute_tolerance"]:
                raise RuntimeError("accelerated compound-eye samples differ from direct integral")
        metadata = {"specification_sha256": sha256_file(SPEC),
                    "implementation_sha256": {f: evidence_sha256(ROOT/f) for f in ("src/acceptance_integral.py", "scripts/render_exp011_sensors.py")},
                    "condition": condition, "level": level,
                    "light_digest": trace_digest({"sensor_light": values}),
                    "geometry_digest": trace_digest(g.__dict__),
                    "full_cone_mass": sampler.normalization,
                    "reference_full_cone_mass": sampler.reference_normalization,
                    "full_cone_mass_error": sampler.normalization-sampler.reference_normalization,
                    "smooth_wall_direct_41_pose_max_abs_error": checked_error,
                    "fresh_full_integral_max_abs_error": fresh_error,
                    "execution_path": "direct_translation" if translation_direct else "exact_z_rotation_harmonic",
                    "cache_segments": len(table.pieces) if table else 0,
                    "cache_off_node_max_abs_error": table.max_checked_error if table else None}
        np.savez_compressed(target, sensor_time_ms=time, light=values, mixed=mixed[inverse],
                            blocked_mass=mass[inverse], metadata=json.dumps(metadata, sort_keys=True))
        print(f"Saved {target.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/exp011_sensors")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT/"results").resolve()) or not 1 <= args.workers <= 4:
        raise ValueError("ignored output and 1..4 workers required")
    spec, _, _, _ = inputs()
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(render_condition, spec["conditions"], [args.output]*len(spec["conditions"])))


if __name__ == "__main__":
    main()
