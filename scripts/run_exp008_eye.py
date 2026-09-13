"""Prescribed 3D head poses -> measured eyes -> frozen visual/central projections."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time as clock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_exp005_sensory import load_specification, relative_error
from scripts.prepare_exp008_eye import evidence_sha256
from src.binocular_physiology import BinocularNetwork, PhysiologyParameters, POPULATIONS
from src.compound_eye import EyeGeometry, EyeSampler, LocalReadout, Scene, observer_camera, rotation_z
from src.provenance import sha256_file
from src.sensory_chain import CHANNELS, SensoryParameters, VisualProjection, central_drive

SPEC = ROOT/"experiments/EXP-008-compound-eye/specification.json"
RECORD = SPEC.with_name("record.json")
IMPLEMENTATION = ["src/compound_eye.py", "scripts/prepare_exp008_eye.py", "scripts/run_exp008_eye.py"]


def load_frozen_eye():
    spec = json.loads(SPEC.read_text(encoding="utf8"))
    for entry in spec["frozen_files"]+spec["eye_source"]["files"]:
        if evidence_sha256(ROOT/entry["path"]) != entry["sha256"]:
            raise ValueError(f"frozen evidence changed: {entry['path']}")
    geometry = EyeGeometry.load(ROOT/spec["eye_source"]["files"][-1]["path"])
    return spec, geometry


def sensor_movie(sampler, scene, p, condition_pose, dt_ms):
    if not np.isclose(p.duration_ms/dt_ms, round(p.duration_ms/dt_ms)):
        raise ValueError("sensor duration is not integral")
    time = np.arange(round(p.duration_ms/dt_ms))*dt_ms
    elapsed = np.clip(time-p.onset_ms, 0, p.offset_ms-p.onset_ms)/1000
    # Cache unchanged physical poses, not stimulus-derived neural answers.
    unique, inverse = np.unique(elapsed, return_inverse=True)
    if np.array_equal(condition_pose, [0., 0.]):
        unique, inverse = np.array([0.]), np.zeros(len(elapsed), dtype=int)
    samples = np.empty((len(unique), len(sampler.geometry.directions)))
    for k, seconds in enumerate(unique):
        rotation = rotation_z(np.deg2rad(condition_pose[0])*seconds)
        translation = np.array([condition_pose[1]*seconds, 0., 0.])
        samples[k] = sampler.sample(scene, rotation, translation)
        if k % 250 == 0:
            print(f"  ray sampling: {k}/{len(unique)} distinct poses, dt={dt_ms} ms", flush=True)
    return time, samples[inverse]


def replay_sensor(folder, condition, geometry, p, dt_ms, *, refinement=False):
    """Recorded per-facet light only; never consume saved neural outputs."""
    suffix = "_sensor_refinement" if refinement else ""
    with np.load(folder/f"{condition}{suffix}.npz", allow_pickle=False) as data:
        time, light = data["sensor_time_ms"].copy(), data["light"].copy()
    expected_time = np.arange(round(p.duration_ms/dt_ms))*dt_ms
    if (not np.array_equal(time, expected_time)
            or light.shape != (len(time), len(geometry.directions))
            or not np.isfinite(light).all() or np.any((light < 0) | (light > 1))):
        raise ValueError("invalid recorded compound-eye samples/axes")
    return time, light


def held_light(samples, sensor_dt, neural_dt, duration):
    if min(sensor_dt, neural_dt, duration) <= 0 or not np.isfinite([sensor_dt, neural_dt, duration]).all():
        raise ValueError("sampling times must be finite and positive")
    count = round(duration/neural_dt)
    index = np.floor(np.arange(count)*neural_dt/sensor_dt+1e-12).astype(int)
    if index[-1] >= len(samples):
        raise ValueError("sensor does not cover neural interval")
    return samples[index]


def propagate(samples, sensor_dt, readout, visual, central, p):
    light = held_light(samples, sensor_dt, p.dt_ms, p.duration_ms)
    channels = readout.channels(light, p)
    LPi, drive = visual.simulate(channels)
    states = central.simulate(central_drive(visual.input_ids, drive, central.nodes.bodyId))
    return {"channels": channels, "LPi": LPi, "HS_H2_drive": drive, "central": states}


def trace_digest(arrays):
    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        a = np.asarray(array, dtype="<f8", order="C")
        digest.update(key.encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def summarize(states, central, p):
    t = np.arange(len(states))*p.dt_ms
    late = (t >= 3500) & (t < 4000)
    rows = []
    for i, row in enumerate(central.nodes.itertuples()):
        rows.append({"body_id": int(row.bodyId), "type": row.type,
                     "stage": row.stage, "side": row.somaSide,
                     "late_mean": float(states[late, i].mean()),
                     "min_state": float(states[:, i].min()), "max_state": float(states[:, i].max())})
    return rows


def geometry_preflight(geometry, sampler, scene, spec):
    fine = EyeSampler.compile(geometry, fwhm_degrees=spec["optics"]["fwhm_degrees"],
                             **spec["acceptance"]["quadrature_reference"])
    errors = []
    poses = [(0., 0.), (.5, 0.), (-.5, 0.), (0., 1.), (0., -1.),
             (np.pi/2, 0.), (-np.pi/2, 0.), (0., 2.), (0., -2.)]
    for angle, x in poses:
        r, t = rotation_z(angle), np.array([x, 0., 0.])
        a = sampler.sample(scene, r, t)
        if not np.array_equal(a, sampler.sample(scene, r, t)):
            raise RuntimeError("nondeterministic sensory sampling")
        errors.append(float(abs(a-fine.sample(scene, r, t)).max()))
    reflected = geometry.directions[geometry.left]*[1, -1, 1]
    distances = cKDTree(geometry.directions[~geometry.left]).query(reflected)[0]
    degrees = np.rad2deg(2*np.arcsin(distances/2))
    az = np.rad2deg(np.arctan2(geometry.directions[:, 1], geometry.directions[:, 0]))
    el = np.rad2deg(np.arcsin(geometry.directions[:, 2]))
    fov = {}
    for side, mask in [("L", geometry.left), ("R", ~geometry.left)]:
        eq = mask & (abs(el) < 5)
        fov[side] = {"equatorial_azimuth_min_max_degrees": [float(az[eq].min()), float(az[eq].max())],
                     "elevation_min_max_degrees": [float(el[mask].min()), float(el[mask].max())]}
    return {"passed": max(errors) <= spec["acceptance"]["quadrature_max_abs_light_error"],
            "quadrature_max_abs_light_errors_by_pose": errors,
            "quadrature_checked_poses_angle_radians_translation_mm": [list(pose) for pose in poses],
            "mirrored_nearest_direction_median_p95_degrees": np.percentile(degrees, [50, 95]).tolist(),
            "symmetry_scope": "nearest-direction consistency only; no mirrored template or enforced equal counts",
            "field_of_view": fov, "left_count": int(geometry.left.sum()), "right_count": int((~geometry.left).sum())}


def figure(spec, geometry, readout, scene, sensor_time, sensors, time, stages, summary, output):
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), layout="constrained")
    ax = axes[0, 0]
    ax.imshow(observer_camera(scene), cmap="gray", vmin=0, vmax=1)
    ax.set(xticks=[], yticks=[], title="A  Human/debug pinhole view only — NOT neural input")
    ax.set_xlabel("Finite 10-mm painted spherical wall; nearer opaque sphere\nPrescribed head poses; no body or sensory-feedback loop")
    ax = axes[0, 1]
    u = geometry.directions
    az, el = np.rad2deg(np.arctan2(u[:, 1], u[:, 0])), np.rad2deg(np.arcsin(u[:, 2]))
    light = sensors["static"][0]
    ax.scatter(az, el, c=light, cmap="gray", vmin=0, vmax=1, s=10)
    for ids, color, label in zip(readout.centers, ["#0b96a5", "#d55138"], ["L readout (65)", "R readout (66)"]):
        ax.scatter(az[ids], el[ids], facecolors="none", edgecolors=color, s=24, linewidths=.8, label=label)
    ax.set(xlabel="Head-frame azimuth (degrees; positive = left)", ylabel="Elevation (degrees)",
           title="B  Actual facet viewing axes / received light: L 857, R 852", xlim=(-180, 180), ylim=(-90, 90))
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    # Human visualization sorts metadata; numerical sensor axes remain lens order.
    order = np.concatenate([np.flatnonzero(geometry.left), np.flatnonzero(~geometry.left)])
    ax.imshow(sensors["yaw_positive_Z"][:, order].T, extent=[0, 6, len(order), 0],
              cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.axhline(857, color="#0b96a5", lw=1)
    ax.set(xlabel="Time (s)", ylabel="Measured facet index, grouped L / R",
           title="C  Every ommatidium's scalar input — prescribed +Z head rotation")
    ax = axes[1, 1]
    for condition, color in [("yaw_positive_Z", "#0b96a5"), ("yaw_negative_Z", "#d55138")]:
        channels = stages[condition]["channels"]
        for channel, style in [("L_T4a", "-"), ("L_T4b", "--"), ("R_T5a", ":"), ("R_T5b", "-.")]:
            ax.plot(time/1000, channels[:, CHANNELS.index(channel)], style, color=color, lw=1,
                    label=f"{'+' if condition.endswith('positive_Z') else '−'}Z: {channel}")
    ax.set(xlabel="Time (s)", ylabel="Dimensionless motion proxy", title="D  Local real-neighbor ON/OFF readout → frozen type channels")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[2, 0]
    conditions = list(spec["pose_conditions"])[1:]
    rows = [(pop, side) for pop in POPULATIONS for side in ("L", "R")]
    values = np.array([[np.mean([r["late_mean"] for r in summary[c] if r["stage"] == pop and r["side"] == side])
                        for c in conditions] for pop, side in rows])
    limit = max(abs(values).max(), 1e-12)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set(yticks=range(len(rows)), yticklabels=[f"{pop} {side}" for pop, side in rows],
           xticks=range(4), xticklabels=["Head +Z", "Head −Z", "World +X", "World −X"],
           title="E  Frozen central stages: signed late state, 3.5–4 s")
    for row in range(2, len(rows), 2):
        ax.axhline(row-.5, color="white", lw=1.5)
    fig.colorbar(im, ax=ax, label="Dimensionless state (one shared scale)", shrink=.75)
    ax = axes[2, 1]
    for condition, color in [("yaw_positive_Z", "#0b96a5"), ("yaw_negative_Z", "#d55138")]:
        for side, style in [("L", "-"), ("R", "--")]:
            i = next(i for i, row in enumerate(summary[condition]) if row["stage"] == "DNp15" and row["side"] == side)
            ax.plot(time/1000, stages[condition]["central"][:, i], style, color=color,
                    label=f"{'+' if condition.endswith('positive_Z') else '−'}Z: DNp15 {side}")
    ax.axhline(0, color="black", lw=1, label="Frozen eye / eye disconnect / central disconnect: zero DN")
    ax.set(xlabel="Time (s)", ylabel="Signed dimensionless DN state", title="F  Sensory-to-DNp15 propagation and causal null controls")
    ax.legend(fontsize=8)
    for ax in [axes[1, 0], axes[1, 1], axes[2, 1]]:
        ax.axvspan(2, 4, color="#555555", alpha=.08)
    fig.suptitle("EXP-008 · 3D world → measured bilateral compound eyes → provisional local motion → frozen DNp15 pathway\nMeasured geometry; assumed 5° optics; central-only type interface; no motor/behavior claim", fontsize=13)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/exp008_eye")
    parser.add_argument("--write-record", action="store_true")
    parser.add_argument("--check-record", action="store_true")
    parser.add_argument("--promote-figure", action="store_true")
    parser.add_argument("--replay-sensors", type=Path,
                        help="recompute all neural stages from an existing raw-facet bundle; --check-record verifies its full trace digest")
    args = parser.parse_args()
    if args.write_record and (args.check_record or RECORD.exists()):
        raise ValueError("records cannot be overwritten")
    if args.replay_sensors is not None and not args.check_record:
        raise ValueError("sensor replay is verification-only and requires --check-record")
    started = clock.perf_counter()
    spec, geometry = load_frozen_eye()
    _, central_spec, visual_graph, central_graph = load_specification()
    p = SensoryParameters(**spec["early_vision"]["parameters"])
    cp = PhysiologyParameters(**central_spec["parameters"])
    visual = VisualProjection.compile(visual_graph, p)
    central = BinocularNetwork.compile(central_graph, central_spec["electrical_pairs"], cp)
    fine_visual = VisualProjection.compile(visual_graph, replace(p, dt_ms=p.dt_ms/2))
    fine_central = BinocularNetwork.compile(central_graph, central_spec["electrical_pairs"], replace(cp, dt_ms=cp.dt_ms/2))
    optics = {k: spec["optics"][k] for k in ("fwhm_degrees", "radial", "azimuthal")}
    sampler, scene = EyeSampler.compile(geometry, **optics), Scene(**spec["scene"])
    readout = LocalReadout.compile(geometry, **{k: spec["early_vision"][k] for k in ("azimuth_range", "elevation_limit")})
    checks = {"geometry": geometry_preflight(geometry, sampler, scene, spec),
              "LPi": visual.preflight(), "central": central.preflight(),
              "LPi_half_dt": fine_visual.preflight(), "central_half_dt": fine_central.preflight()}
    if not all(check["passed"] for check in checks.values()):
        raise RuntimeError(f"pre-integration numerical/geometry failure: {checks}")
    print("Geometry and frozen-graph numerical preflight passed; beginning integrated evaluation.", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    sensors, stages, summaries, metrics, digests = {}, {}, {}, {}, {}
    numerical_errors, sensor_errors = {}, {}
    refinement_digests = {}
    control_results = {}
    for condition, pose in spec["pose_conditions"].items():
        print(f"Condition: {condition}", flush=True)
        sensor_time, light = (sensor_movie(sampler, scene, p, pose, spec["sensor_dt_ms"])
                              if args.replay_sensors is None else
                              replay_sensor(args.replay_sensors, condition, geometry, p, spec["sensor_dt_ms"]))
        stages[condition] = propagate(light, spec["sensor_dt_ms"], readout, visual, central, p)
        repeated = propagate(light, spec["sensor_dt_ms"], readout, visual, central, p)
        if any(not np.array_equal(stages[condition][k], repeated[k]) for k in repeated):
            raise RuntimeError("nondeterministic stage output")
        baseline_steps = round(p.onset_ms/p.dt_ms)
        if any(np.any(v[:baseline_steps]) for v in repeated.values()):
            raise RuntimeError("activity precedes prescribed pose change")
        digests[condition] = trace_digest({"sensor_light": light, **stages[condition]})
        summaries[condition] = summarize(stages[condition]["central"], central, p)
        metrics[condition] = {"light_min_max": [float(light.min()), float(light.max())],
                              "max_abs_light_change_from_initial_pose": float(abs(light-light[0]).max()),
                              "motion_peak_by_channel": dict(zip(CHANNELS, stages[condition]["channels"].max(axis=0).tolist()))}
        fine_num = propagate(light, spec["sensor_dt_ms"], readout, fine_visual, fine_central, replace(p, dt_ms=p.dt_ms/2))
        numerical_errors[condition] = {k: relative_error(stages[condition][k], fine_num[k][::2]) for k in repeated}
        if condition != "static":
            _, finer_light = (sensor_movie(sampler, scene, p, pose, spec["sensor_refinement_dt_ms"])
                               if args.replay_sensors is None else
                               replay_sensor(args.replay_sensors, condition, geometry, p, spec["sensor_refinement_dt_ms"], refinement=True))
            finer = propagate(finer_light, spec["sensor_refinement_dt_ms"], readout, visual, central, p)
            refinement_digests[condition] = trace_digest({"sensor_light": finer_light})
            sensor_errors[condition] = {k: relative_error(stages[condition][k], finer[k]) for k in finer}
            sensor_errors[condition]["max_abs_light_error"] = float(abs(light[np.arange(len(finer_light))//2]-finer_light).max())
            np.savez_compressed(args.output/f"{condition}_sensor_refinement.npz", sensor_time_ms=np.arange(len(finer_light))*spec["sensor_refinement_dt_ms"], light=finer_light)
        else:
            if any(np.any(v) for v in stages[condition].values()):
                raise RuntimeError("static scene generated neural activity")
        # Three stage-boundary disconnections; surviving weights remain identical.
        zero = np.zeros_like(stages[condition]["channels"])
        disconnected_LPi, disconnected_drive = visual.simulate(zero)
        disconnected = central.simulate(central_drive(visual.input_ids, disconnected_drive, central.nodes.bodyId))
        central_only_disconnect = central.simulate(np.zeros_like(stages[condition]["central"]))
        frozen_eye = propagate(np.repeat(light[:1], len(light), axis=0), spec["sensor_dt_ms"], readout, visual, central, p)
        if any(np.any(v) for v in [disconnected_LPi, disconnected_drive, disconnected, central_only_disconnect, *frozen_eye.values()]):
            raise RuntimeError("causal null control failed")
        control_results[condition] = {"eye_frozen_at_initial_pose_max_abs_DN": 0.,
                                      "eye_to_motion_disconnected_max_abs_DN": 0.,
                                      "visual_to_central_disconnected_max_abs_DN": 0.,
                                      "central_disconnect_preserves_visual_stage": True,
                                      "same_weights_no_renormalization": True}
        sensors[condition] = light
        time = np.arange(round(p.duration_ms/p.dt_ms))*p.dt_ms
        np.savez_compressed(args.output/f"{condition}.npz", sensor_time_ms=sensor_time, light=light,
                            time_ms=time, channel_names=np.array(CHANNELS), LPi_body_ids=visual.LPi_ids,
                            HS_H2_body_ids=visual.input_ids, central_body_ids=central.nodes.bodyId.to_numpy(),
                            **stages[condition])
    worst_num = max(v for c in numerical_errors.values() for v in c.values())
    worst_sensor = max(v for c in sensor_errors.values() for k, v in c.items() if k != "max_abs_light_error")
    # These predeclared checks may fail; never change defaults based on DN output.
    convergence = {"passed": max(worst_num, worst_sensor) <= spec["acceptance"]["timestep_stage_relative_error"],
                   "numerical_half_dt_stage_relative_errors": numerical_errors,
                   "sensor_2_to_1ms_stage_relative_errors": sensor_errors}
    propagation = all(any(r["max_state"] > 0 or r["min_state"] < 0 for r in summaries[c] if r["stage"] == "DNp15" and r["side"] == side)
                      for c in list(spec["pose_conditions"])[1:] for side in ("L", "R"))
    interfaces = {}
    for condition in list(spec["pose_conditions"])[1:]:
        interfaces[condition] = {
            "both_eyes_ON_and_OFF_nonzero": all(np.any(stages[condition]["channels"][:, a:a+4]) for a in (0, 4, 8, 12)),
            "HS_H2_drive_nonzero": bool(np.all(np.max(abs(stages[condition]["HS_H2_drive"]), axis=0) > 0)),
            "all_central_populations_nonzero": all(any(r["max_state"] > 0 or r["min_state"] < 0 for r in summaries[condition] if r["stage"] == pop) for pop in POPULATIONS)}
    interface_passed = all(all(values.values()) for values in interfaces.values())
    result = {"experiment_id": spec["experiment_id"],
              "status": "continuous_sensory_chain_with_provisional_local_readout" if convergence["passed"] and propagation and interface_passed else "sensory_chain_validation_incomplete",
              "specification_sha256": sha256_file(SPEC),
              "implementation_sha256": {f: evidence_sha256(ROOT/f) for f in IMPLEMENTATION},
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "geometry": checks["geometry"], "local_readout_counts_L_R": list(map(len, readout.centers)),
              "numerical_preflight": checks, "convergence": convergence, "propagation_to_both_DNp15": propagation,
              "stage_interface_checks": interfaces,
              "condition_metrics": metrics, "central_per_body": summaries,
              "controls": control_results, "deterministic_trace_digests": digests,
              "sensor_refinement_digests": refinement_digests,
              "evidence_boundary": "Measured female bilateral lens/tip geometry and grid; assumed achromatic optics; central-local correlators; shared MaleCNS eye/type population assignment, not individual retinotopy. Frozen central physiology discrepancies remain. No body/motor run."}
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    (args.output/"metrics.json").write_text(text, encoding="utf8")
    np.savez_compressed(args.output/"sensor_axes.npz", **geometry.__dict__,
                        left_readout_centers=readout.centers[0], right_readout_centers=readout.centers[1],
                        left_readout_endpoints=readout.endpoints[0], right_readout_endpoints=readout.endpoints[1])
    figure(spec, geometry, readout, scene, sensor_time, sensors, time, stages, summaries, args.output/"EXP-008-compound-eye.png")
    if args.check_record and json.loads(RECORD.read_text(encoding="utf8")) != result:
        raise RuntimeError("run differs from committed experiment record")
    if args.write_record:
        RECORD.write_text(text, encoding="utf8")
    if args.promote_figure:
        shutil.copyfile(args.output/"EXP-008-compound-eye.png", ROOT/"figures/EXP-008-compound-eye.png")
    print(f"Result: {result['status']}; worst numerical error={worst_num:.4%}, sensor error={worst_sensor:.4%}; elapsed={clock.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
