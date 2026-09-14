"""Measure EXP-011 direct optical execution without Chebyshev acceleration.

This benchmark never writes sensor archives and never evaluates neural or
downstream stages.  Each timed call is the full-eye canonical optical call
with ``smooth_light`` omitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_exp011_sensors import inputs, sample_physical_pose
from src.acceptance_integral import AcceptanceIntegral
from src.compound_eye import Scene, rotation_z
from src.sensory_chain import SensoryParameters


POSES_MM = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
REPLAY_POSE_MM = 0.5


def memory_bytes(process):
    info = process.memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def optical_call(solver, scene, translation_mm):
    """The full-eye direct canonical optical call, with no smooth_light cache."""
    return sample_physical_pose(solver, scene, rotation_z(0.0),
                                [translation_mm, 0.0, 0.0])


def digest(result):
    light, diagnostic = result
    payload = b"".join([
        np.ascontiguousarray(light).tobytes(),
        np.ascontiguousarray(diagnostic["blocked_mass"]).tobytes(),
        np.ascontiguousarray(diagnostic["mixed"]).tobytes(),
    ])
    return hashlib.sha256(payload).hexdigest()


def pose_counts(eye_spec, levels):
    parameters = SensoryParameters(**eye_spec["early_vision"]["parameters"])
    counts = {}
    for level in levels:
        dt = level["sensor_dt_ms"]
        time_ms = np.arange(round(parameters.duration_ms / dt)) * dt
        elapsed = np.clip(time_ms - parameters.onset_ms, 0,
                          parameters.offset_ms - parameters.onset_ms) / 1000
        counts[level["name"]] = int(len(np.unique(elapsed)))
    return counts


def main():
    spec, eye_spec, geometry, _ = inputs()
    scene = Scene(**eye_spec["scene"])
    process = psutil.Process()
    baseline_peak = memory_bytes(process)
    timing = []
    solvers = {}

    for level in spec["ordered_levels"]:
        name = level["name"]
        started = time.perf_counter()
        solver = AcceptanceIntegral.compile(
            geometry,
            fwhm_degrees=eye_spec["optics"]["fwhm_degrees"],
            radial=level["radial"],
            azimuthal=level["azimuthal"],
        )
        compile_seconds = time.perf_counter() - started
        solvers[name] = solver

        # One unrecorded warmup at the central translation.
        optical_call(solver, scene, 0.0)
        rows = []
        for x in POSES_MM:
            started = time.perf_counter()
            result = optical_call(solver, scene, x)
            elapsed = time.perf_counter() - started
            rows.append({"translation_mm": x, "seconds": elapsed,
                         "light_shape": list(result[0].shape)})
        timing.append({
            "level": name,
            "radial": level["radial"],
            "azimuthal": level["azimuthal"],
            "compile_seconds": compile_seconds,
            "poses": rows,
            "mean_seconds_per_pose": float(np.mean([r["seconds"] for r in rows])),
            "median_seconds_per_pose": float(np.median([r["seconds"] for r in rows])),
            "max_seconds_per_pose": float(max(r["seconds"] for r in rows)),
            "min_seconds_per_pose": float(min(r["seconds"] for r in rows)),
            "peak_process_memory_bytes": memory_bytes(process),
            "peak_process_memory_delta_bytes": memory_bytes(process) - baseline_peak,
        })

    replay_solver = solvers["production"]
    first = optical_call(replay_solver, scene, REPLAY_POSE_MM)
    second = optical_call(replay_solver, scene, REPLAY_POSE_MM)
    first_digest, second_digest = digest(first), digest(second)
    replay_light_delta = float(np.max(abs(first[0] - second[0])))
    replay_blocked_delta = float(np.max(abs(first[1]["blocked_mass"] - second[1]["blocked_mass"])))

    counts = pose_counts(eye_spec, spec["ordered_levels"])
    estimates = []
    for row in timing:
        n = counts[row["level"]]
        # Each translation condition executes the levels sequentially.
        base = row["compile_seconds"]
        estimates.append({
            "level": row["level"],
            "unique_physical_poses_per_translation_stream": n,
            "estimated_stream_seconds_mean": base + n * row["mean_seconds_per_pose"],
            "estimated_stream_seconds_median": base + n * row["median_seconds_per_pose"],
            "estimated_stream_seconds_conservative_max": base + n * row["max_seconds_per_pose"],
        })
    by_level = {row["level"]: row for row in estimates}
    plus_mean = sum(row["estimated_stream_seconds_mean"] for row in estimates)
    plus_median = sum(row["estimated_stream_seconds_median"] for row in estimates)
    plus_max = sum(row["estimated_stream_seconds_conservative_max"] for row in estimates)

    report = {
        "scope": "direct optical feasibility only; no canonical archives or neural/downstream evaluation",
        "implementation": "AcceptanceIntegral.sample with smooth_light omitted",
        "experiment_id": spec["experiment_id"],
        "geometry_facets": int(len(geometry.directions)),
        "scene_and_orders": {"fwhm_degrees": eye_spec["optics"]["fwhm_degrees"],
                             "poses_mm": list(POSES_MM),
                             "levels": [{"name": x["name"], "radial": x["radial"],
                                         "azimuthal": x["azimuthal"],
                                         "sensor_dt_ms": x["sensor_dt_ms"]}
                                        for x in spec["ordered_levels"]]},
        "timing": timing,
        "physical_pose_counts": counts,
        "stream_estimates": estimates,
        "complete_translation_stream_estimate_seconds": {
            "positive_X_mean": plus_mean,
            "negative_X_mean": plus_mean,
            "positive_X_median": plus_median,
            "negative_X_median": plus_median,
            "positive_X_conservative_max": plus_max,
            "negative_X_conservative_max": plus_max,
        },
        "four_worker_translation_pair_estimate_seconds": {
            "mean": plus_mean,
            "median": plus_median,
            "conservative_max": plus_max,
            "interpretation": "The +X and -X condition workers can run concurrently under the existing four-worker cap; this is the translation-pair estimate only and excludes unbenchmarked static/yaw workers and pool startup overhead.",
        },
        "deterministic_replay": {
            "level": "production",
            "translation_mm": REPLAY_POSE_MM,
            "first_sha256": first_digest,
            "second_sha256": second_digest,
            "exact_digest_match": first_digest == second_digest,
            "max_light_abs_difference": replay_light_delta,
            "max_blocked_mass_abs_difference": replay_blocked_delta,
        },
        "peak_process_memory_baseline_bytes": baseline_peak,
        "limitations": [
            "Timing estimates extrapolate representative full-eye pose calls; canonical archives were not generated.",
            "This benchmark does not measure optical convergence error or any local neural/downstream output.",
            "The failed degree-16/tolerance-1e-10/depth-12 Chebyshev contract remains abandoned and unchanged.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
