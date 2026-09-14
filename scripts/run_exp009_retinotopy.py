"""Validate the measured-grid stage, then replay scalar light into frozen DNp15.

No sensory/registration choice uses downstream output. EXP-008 world samples are
read as light only, checked against its frozen record and independent ray replay.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_exp008_eye import evidence_sha256
from scripts.prepare_exp009_retinotopy import SPEC
from scripts.run_exp005_sensory import load_specification, relative_error
from scripts.run_exp008_eye import held_light, load_frozen_eye, replay_sensor, summarize, trace_digest
from src.binocular_physiology import BinocularNetwork, PhysiologyParameters, POPULATIONS
from src.compound_eye import EyeSampler, LocalReadout, Scene, rotation_z
from src.provenance import sha256_file
from src.retinotopic_eye import RetinotopicReadout, source_field_comparison
from src.sensory_chain import CHANNELS, SensoryParameters, VisualProjection, central_drive

RECORD = SPEC.with_name("record.json")
IMPLEMENTATION = ["src/retinotopic_eye.py", "scripts/prepare_exp009_retinotopy.py", "scripts/run_exp009_retinotopy.py"]


def load_resolved():
    spec = json.loads(SPEC.read_text(encoding="utf8"))
    for entry in spec["frozen_files"]+spec["source_files"]:
        path = ROOT/entry["path"]
        if evidence_sha256(path) != entry["sha256"]:
            raise ValueError(f"frozen evidence differs: {entry['path']}")
    eye_spec, geometry = load_frozen_eye()
    with np.load(ROOT/spec["source_files"][-1]["path"], allow_pickle=False) as data:
        reference = {k: data[k].copy() for k in data.files}
    return spec, eye_spec, geometry, reference


def verified_trial(bundle, condition, geometry, eye_spec):
    """Verify full frozen trace, but deliver only scalar light to the new kernel."""
    old_record = json.loads((ROOT/"experiments/EXP-008-compound-eye/record.json").read_text(encoding="utf8"))
    with np.load(bundle/f"{condition}.npz", allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in ("light", "channels", "LPi", "HS_H2_drive", "central")}
        digest = trace_digest({"sensor_light": arrays["light"], **{k: v for k, v in arrays.items() if k != "light"}})
        if digest != old_record["deterministic_trace_digests"][condition]:
            raise ValueError("baseline scalar-light/neural trace differs from frozen EXP-008")
    with np.load(bundle/"sensor_axes.npz", allow_pickle=False) as saved:
        if any(not np.array_equal(saved[k], v) for k, v in geometry.__dict__.items()):
            raise ValueError("recorded eye axes differ from measured geometry")
    p = SensoryParameters(**eye_spec["early_vision"]["parameters"])
    _, light = replay_sensor(bundle, condition, geometry, p, eye_spec["sensor_dt_ms"])
    if not np.array_equal(light, arrays.pop("light")):
        raise ValueError("sensor replay changed scalar samples")
    sampler = EyeSampler.compile(geometry, **{k: eye_spec["optics"][k] for k in ("fwhm_degrees", "radial", "azimuthal")})
    scene = Scene(**eye_spec["scene"])
    rate, speed = eye_spec["pose_conditions"][condition]
    for milliseconds in (0., 3000., 4500.):
        seconds = np.clip(milliseconds-p.onset_ms, 0, p.offset_ms-p.onset_ms)/1000
        sample = sampler.sample(scene, rotation_z(np.deg2rad(rate)*seconds), [speed*seconds, 0., 0.])
        if not np.array_equal(sample, light[round(milliseconds/eye_spec["sensor_dt_ms"])]):
            raise ValueError("direct world ray replay differs from saved facet samples")
    return light, arrays


def evaluate_local(light, readout, p, sensor_dt):
    sample_stride = round(sensor_dt/p.dt_ms)
    if sample_stride < 1 or not np.isclose(sample_stride*p.dt_ms, sensor_dt):
        raise ValueError("local storage must coincide with numerical samples")
    c, local = readout.responses(held_light(light, sensor_dt, p.dt_ms, p.duration_ms), p,
                                save_stride=sample_stride)
    if not np.array_equal(readout.pool(local), c[::sample_stride]):
        raise RuntimeError("stored local emissions and every coincident channel sample disagree")
    return {"channels": c, "local": local}


def propagate_channels(channels, visual, central):
    LPi, drive = visual.simulate(channels)
    states = central.simulate(central_drive(visual.input_ids, drive, central.nodes.bodyId))
    return {"channels": channels, "LPi": LPi, "HS_H2_drive": drive, "central": states}


def local_preflight(geometry, readout, reference, p):
    comparison = source_field_comparison(readout, geometry, reference)
    offsets = np.array([[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]])
    grid_ok = np.array_equal(geometry.grid[readout.neighbors]-geometry.grid[readout.centers, None],
                             np.broadcast_to(offsets, readout.neighbors.shape+(2,)))
    t = np.arange(400)*p.dt_ms/1000
    light = .5+.3*np.cos(2*np.pi*geometry.directions[:, 0][None, :]+2*np.pi*3*t[:, None])
    channels, local = readout.responses(light, p, save_stride=4)
    repeat_c, repeat_l = readout.responses(light, p, save_stride=4)
    prefix_c, prefix_l = readout.responses(light[:201], p, save_stride=4)
    inverse_c, inverse_l = readout.responses(1-light, p, save_stride=4)
    static = readout.responses(np.repeat(light[:1], len(light), axis=0), p)
    flicker = readout.responses(np.broadcast_to(np.linspace(.1, .9, 400)[:, None], light.shape), p)
    coefficients = p.dt_ms/np.array([p.highpass_tau_ms, p.delay_tau_ms, p.emission_tau_ms])
    result = {"source_field": comparison, "primary_neighbor_slot_addresses": grid_ok,
              "determinism": bool(np.array_equal(channels, repeat_c) and np.array_equal(local, repeat_l)),
              "causal_prefix": bool(np.array_equal(channels[:201], prefix_c) and np.array_equal(local[:51], prefix_l)),
              "contrast_ON_OFF_exchange": bool(np.allclose(inverse_l, local[:, :, ::-1], atol=1e-16, rtol=1e-11)),
              "static_null": all(not np.any(x) for x in static),
              "uniform_flicker_null": all(not np.any(x) for x in flicker),
              "convex_filter_coefficients": coefficients.tolist(),
              "bounded_filter_updates": bool(np.all((coefficients > 0) & (coefficients <= 1)))}
    result["passed"] = comparison["all_local_orientations_same_hemiplane"] and all(
        result[k] for k in ("primary_neighbor_slot_addresses", "determinism", "causal_prefix", "contrast_ON_OFF_exchange",
                           "static_null", "uniform_flicker_null", "bounded_filter_updates"))
    return result


def refinement_hotspot(coarse, fine, readout, geometry):
    difference = abs(coarse-fine)
    time, column, polarity, subtype = np.unravel_index(np.argmax(difference), difference.shape)
    facet = int(readout.centers[column])
    return {"local_time_ms": float(time*2), "measured_facet_index_0based": facet,
            "eye": "L" if geometry.left[facet] else "R", "measured_grid_p_q": geometry.grid[facet].tolist(),
            "polarity": "ON" if polarity == 0 else "OFF", "subtype_proxy": "abcd"[subtype],
            "maximum_absolute_local_difference": float(difference[time, column, polarity, subtype]),
            "coarse_local_state": float(coarse[time, column, polarity, subtype]),
            "fine_local_state": float(fine[time, column, polarity, subtype]),
            "local_stage_global_max": float(max(coarse.max(), fine.max()))}


def eye_display_order(left):
    """Observer-only grouping; measured lens order can interleave the eyes."""
    return np.concatenate([np.flatnonzero(left), np.flatnonzero(~left)])


def make_figure(g, readout, ref_check, figures, metrics, central, output):
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), layout="constrained")
    u = g.directions[readout.centers]
    az, el = np.rad2deg(np.arctan2(u[:, 1], u[:, 0])), np.rad2deg(np.arcsin(u[:, 2]))
    ax = axes[0, 0]
    ax.scatter(az, el, s=3, c=np.where(readout.left, "#1693a1", "#d26837"), alpha=.45)
    ids = np.arange(0, len(u), 10)
    # Display tangent arrows only; no angular projection enters the kernel.
    endpoint = u[ids]+.035*readout.preferred_directions[ids, 1]
    endpoint /= np.linalg.norm(endpoint, axis=1, keepdims=True)
    dx = (np.rad2deg(np.arctan2(endpoint[:, 1], endpoint[:, 0]))-az[ids]+180)%360-180
    dy = np.rad2deg(np.arcsin(endpoint[:, 2]))-el[ids]
    ax.quiver(az[ids], el[ids], dx, dy, angles="xy", scale_units="xy", scale=1, width=.002)
    ax.set(title="A  Measured-grid local b bases, not global cardinal axes", xlabel="Head-frame azimuth (degrees; positive = left)",
           ylabel="Elevation (degrees)", xlim=(-180, 180), ylim=(-90, 90))
    ax.text(.02, .02, "753 L / 747 R neighborhoods; 1,709 facets sampled", transform=ax.transAxes, fontsize=9)
    ax = axes[0, 1]
    for s, color in [("b", "#1693a1"), ("d", "#d26837")]:
        median, p95, maximum = ref_check["angles_degrees_p50_p95_max"][s]
        ax.plot([median, p95, maximum], [s]*3, "o-", color=color, label=f"{s}: median / 95th / max")
    ax.set(title="B  Grid rule vs. released right-eye T4 predictions", xlabel="Tangent orientation difference (degrees)", xlim=(0, 90))
    ax.set_ylim(-.6, 1.6)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(.02, .05, "738 comparable columns; anatomy-predicted, not calcium\nNo MaleCNS body registration; left/T5 field extension provisional",
            transform=ax.transAxes, fontsize=9)
    ax = axes[1, 0]
    values = figures["local_opponency"]
    limit = max(abs(values).max(), 1e-12)
    im = ax.imshow(values[:, eye_display_order(readout.left)].T, extent=[0, 6, len(u), 0], aspect="auto", interpolation="nearest",
                   cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.axhline(sum(readout.left), color="black", lw=1)
    ax.set(title="C  Every local channel retained: +Z rotation", xlabel="Time (s)", ylabel="Measured neighborhood (L / R; not body IDs)")
    fig.colorbar(im, ax=ax, shrink=.8, label="Local a−b, ON + OFF activity proxy")
    ax = axes[1, 1]
    current = figures["peaks"]
    baseline = figures["baseline_peaks"]
    x = np.arange(16)
    ax.bar(x-.18, baseline, width=.36, color="#bbbbbb", label="EXP-008 central-local")
    ax.bar(x+.18, current, width=.36, color="#1693a1", label="EXP-009 measured-grid")
    ax.set(xticks=x, xticklabels=[n.replace("_", "\n") for n in CHANNELS], ylabel="Peak activity proxy",
           title="D  Explicit lossy type adapter: +Z rotation")
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=9)
    ax = axes[2, 0]
    stages = ["HS", "H2", "H2rn", "uLPTCrn", "bIPS", "DNp15"]
    values = []
    for pop in stages:
        for side in "LR":
            values.append([np.mean([r["late_mean"] for r in metrics[c]["central_per_body"] if r["stage"] == pop and r["side"] == side])
                           for c in ["yaw_positive_Z", "yaw_negative_Z", "translation_positive_X", "translation_negative_X"]])
    values = np.asarray(values)
    limit = max(abs(values).max(), 1e-12)
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title="E  Unchanged central stages: 3.5–4 s signed means", xticks=range(4), xticklabels=["Head +Z", "Head −Z", "World +X", "World −X"],
           yticks=range(12), yticklabels=[f"{pop} {side}" for pop in stages for side in "LR"])
    for i in range(2, 12, 2):
        ax.axhline(i-.5, color="white", lw=1.5)
    fig.colorbar(im, ax=ax, shrink=.8, label="Dimensionless state; one shared scale")
    ax = axes[2, 1]
    t = np.arange(len(figures["yaw_positive_Z"]))*.5/1000
    for c, color in [("yaw_positive_Z", "#1693a1"), ("yaw_negative_Z", "#d26837")]:
        for i, side in enumerate("LR"):
            ax.plot(t, figures[c][:, i], "-" if i == 0 else "--", color=color,
                    label=f"{'+' if c == 'yaw_positive_Z' else '−'}Z DNp15 {side}")
    ax.axhline(0, color="black", lw=1, label="Eye / T4-T5 / central disconnection: zero DN")
    ax.axvspan(2, 4, color="gray", alpha=.08)
    ax.set(title="F  Continuous causal propagation, not calibrated physiology", xlabel="Time (s)", ylabel="Signed dimensionless DNp15 state")
    ax.legend(fontsize=8)
    fig.suptitle("EXP-009 · Measured eyes → spatial local-direction proxies → frozen MaleCNS visual/central pathway\n"
                 "Incomplete validation: local translation refinement 6.98–9.27% exceeds the frozen 5% criterion\n"
                 "No individual sensory-body crosswalk; no motor work or new DNp15 physiology claim", fontsize=12)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", type=Path, default=ROOT/"results/exp008_eye_final")
    parser.add_argument("--output", type=Path, default=ROOT/"results/exp009_retinotopy")
    parser.add_argument("--write-record", action="store_true")
    parser.add_argument("--check-record", action="store_true")
    parser.add_argument("--promote-figure", action="store_true")
    args = parser.parse_args()
    if args.write_record and (RECORD.exists() or args.check_record):
        raise ValueError("records cannot be overwritten")
    if not args.output.resolve().is_relative_to((ROOT/"results").resolve()):
        raise ValueError("bulk output must stay under ignored results")
    spec, eye_spec, g, reference = load_resolved()
    readout = RetinotopicReadout.compile(g)
    p = SensoryParameters(**spec["parameters"])
    local_checks = local_preflight(g, readout, reference, p)
    if not local_checks["passed"]:
        raise RuntimeError(f"local-stage preflight failed before DN evaluation: {local_checks}")
    _, cp_spec, visual_graph, central_graph = load_specification()
    cp = PhysiologyParameters(**cp_spec["parameters"])
    visual, central = VisualProjection.compile(visual_graph, p), BinocularNetwork.compile(central_graph, cp_spec["electrical_pairs"], cp)
    fp = replace(p, dt_ms=p.dt_ms/2)
    fine_v, fine_c = VisualProjection.compile(visual_graph, fp), BinocularNetwork.compile(central_graph, cp_spec["electrical_pairs"], replace(cp, dt_ms=cp.dt_ms/2))
    preflight = {"local": local_checks, "local_half_dt": local_preflight(g, readout, reference, fp),
                 "LPi": visual.preflight(), "central": central.preflight(),
                 "LPi_half_dt": fine_v.preflight(), "central_half_dt": fine_c.preflight()}
    if not all(v["passed"] for v in preflight.values()):
        raise RuntimeError("final resolved stage numerical preflight failed")
    print("Frozen source/local/numerical gates passed; downstream evaluation begins.", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, digests, convergence, controls, figures = {}, {}, {}, {}, {}
    dn_columns = [int(np.flatnonzero(central.nodes.bodyId.to_numpy() == i)[0]) for i in (12069, 11215)]
    old_record = json.loads((ROOT/"experiments/EXP-008-compound-eye/record.json").read_text(encoding="utf8"))
    for condition in spec["conditions"]:
        print(f"Condition {condition}: verified world samples -> local stage", flush=True)
        light, baseline = verified_trial(args.sensors, condition, g, eye_spec)
        local = evaluate_local(light, readout, p, spec["sensor_dt_ms"])
        stages = propagate_channels(local["channels"], visual, central)
        repeated = evaluate_local(light, readout, p, spec["sensor_dt_ms"])
        if any(not np.array_equal(local[k], repeated[k]) for k in local):
            raise RuntimeError("local execution is nondeterministic")
        del repeated
        repeated_stages = propagate_channels(stages["channels"], visual, central)
        if any(not np.array_equal(stages[k], repeated_stages[k]) for k in stages):
            raise RuntimeError("downstream execution is nondeterministic")
        del repeated_stages
        fine_local = evaluate_local(light, readout, fp, spec["sensor_dt_ms"])
        fine = propagate_channels(fine_local["channels"], fine_v, fine_c)
        errors = {"local": relative_error(local["local"], fine_local["local"]),
                  **{k: relative_error(v, fine[k][::2]) for k, v in stages.items()}}
        del fine_local, fine
        if condition != "static":
            _, finer_light = replay_sensor(args.sensors, condition, g, p, 1., refinement=True)
            if trace_digest({"sensor_light": finer_light}) != old_record["sensor_refinement_digests"][condition]:
                raise ValueError("refined scalar-light input differs from frozen baseline")
            sensor_local = evaluate_local(finer_light, readout, p, 1.)
            sensor_fine = propagate_channels(sensor_local["channels"], visual, central)
            sensor_errors = {"local": relative_error(local["local"], sensor_local["local"][::2]),
                             **{k: relative_error(v, sensor_fine[k]) for k, v in stages.items()}}
            hotspot = refinement_hotspot(local["local"], sensor_local["local"][::2], readout, g)
            del finer_light, sensor_local, sensor_fine
        else:
            sensor_errors = {k: 0. for k in errors}
            hotspot = None
        convergence[condition] = {"neural_half_dt": errors, "sensor_2_to_1ms": sensor_errors}
        if any(np.any(v[:round(p.onset_ms/p.dt_ms)]) for v in stages.values()) or np.any(local["local"][:1000]):
            raise RuntimeError("neural response precedes pose change")
        if condition == "static" and (any(np.any(v) for v in stages.values()) or np.any(local["local"])):
            raise RuntimeError("static-scene null failed")
        prefix = evaluate_local(light[:1551], readout, replace(p, duration_ms=3102., offset_ms=3000.), 2.)
        prefix_stages = propagate_channels(prefix["channels"], visual, central)
        if (not np.array_equal(prefix["local"], local["local"][:1551])
                or any(not np.array_equal(v, stages[k][:len(v)]) for k, v in prefix_stages.items())):
            raise RuntimeError("integrated causal prefix failed")
        del prefix, prefix_stages
        zeros = propagate_channels(np.zeros_like(stages["channels"]), visual, central)
        eye_disconnect = evaluate_local(np.repeat(light[:1], len(light), axis=0), readout, p, 2.)
        central_disconnect = central.simulate(np.zeros_like(stages["central"]))
        if any(np.any(v) for v in [*zeros.values(), *eye_disconnect.values(), central_disconnect]):
            raise RuntimeError("causal disconnection control failed")
        controls[condition] = {"eye_to_retinotopy_max_abs_activity": 0., "retinotopy_to_T4_T5_max_abs_downstream": 0.,
                               "visual_to_central_max_abs_central": 0., "retinotopy_disconnect_preserves_local": True,
                               "central_disconnect_preserves_visual": True, "same_surviving_operators_no_renormalization": True}
        del zeros, eye_disconnect, central_disconnect
        # Quantify the intentionally lossy adapter, not a decoder of body motion.
        channels_by_eye = stages["channels"].reshape(-1, 2, 2, 4)
        metric = {"channel_peaks": stages["channels"].max(axis=0).tolist(),
                  "EXP008_channel_peaks": baseline["channels"].max(axis=0).tolist(),
                  "both_eyes_ON_OFF_nonzero": bool(np.all(channels_by_eye.max(axis=(0, 3)) > 0)) if condition != "static" else None,
                  "HS_H2_all_projected_inputs_nonzero": bool(np.all(abs(stages["HS_H2_drive"]).max(axis=0) > 0)),
                  "central_per_body": summarize(stages["central"], central, p),
                  "local_max": float(local["local"].max()), "local_centers_nonzero": int(np.count_nonzero(local["local"].max(axis=(0, 2, 3)) > 0)),
                  "sensor_refinement_hotspot": hotspot,
                  "baseline_exp008_trace_digest": old_record["deterministic_trace_digests"][condition]}
        parity = g.grid[readout.centers, 0] % 2
        subset_sensitivity = []
        for bank in (0, 1):
            subset = replace(readout, centers=readout.centers[parity == bank], neighbors=readout.neighbors[parity == bank],
                             left=readout.left[parity == bank], preferred_directions=readout.preferred_directions[parity == bank])
            pooled = subset.pool(local["local"][:, parity == bank])
            subset_sensitivity.append(relative_error(stages["channels"][::4], pooled))
        metric["p_parity_subset_pool_relative_errors_descriptive"] = subset_sensitivity
        metrics[condition] = metric
        digests[condition] = trace_digest({"local": local["local"], **stages})
        time = np.arange(len(stages["channels"]))*p.dt_ms
        np.savez_compressed(args.output/f"{condition}.npz", time_ms=time, sensor_time_ms=np.arange(len(light))*2.,
                            local_time_ms=np.arange(len(local["local"]))*2., local=local["local"], light=light,
                            channel_names=np.asarray(CHANNELS), central_body_ids=central.nodes.bodyId.to_numpy(),
                            LPi_body_ids=visual.LPi_ids, HS_H2_body_ids=visual.input_ids, **stages)
        if condition.startswith("yaw"):
            figures[condition] = stages["central"][:, dn_columns].copy()
        if condition == "yaw_positive_Z":
            figures["local_opponency"] = (local["local"][:, :, :, 0]-local["local"][:, :, :, 1]).sum(axis=-1)
            figures["peaks"], figures["baseline_peaks"] = metric["channel_peaks"], metric["EXP008_channel_peaks"]
        print(f"  local responsive centers={metric['local_centers_nonzero']}; stage refinements recorded", flush=True)
        del local, stages, light, baseline
    worst = max(v for c in convergence.values() for bank in c.values() for v in bank.values())
    moving = [c for c in metrics if c != "static"]
    propagated = all(all(any(r["max_state"] > 0 or r["min_state"] < 0 for r in metrics[c]["central_per_body"]
                               if r["stage"] == "DNp15" and r["side"] == side) for side in "LR") for c in moving)
    passed = worst <= spec["acceptance"]["timestep_stage_relative_error"] and propagated
    result = {"experiment_id": spec["experiment_id"], "status": "broader_measured_grid_chain_with_provisional_population_adapter" if passed else "retinotopic_chain_validation_incomplete",
              "specification_sha256": sha256_file(SPEC), "implementation_sha256": {f: evidence_sha256(ROOT/f) for f in IMPLEMENTATION},
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "coverage": {"sampled_facets_L_R": [int(g.left.sum()), int((~g.left).sum())],
                           "local_neighborhoods_L_R": [int(readout.left.sum()), int((~readout.left).sum())],
                           "distinct_used_facets_L_R": [int(np.sum(g.left[np.unique(np.r_[readout.centers, readout.neighbors.ravel()])])),
                                                        int(np.sum(~g.left[np.unique(np.r_[readout.centers, readout.neighbors.ravel()])]))],
                           "EXP008_neighborhoods_L_R": [65, 66], "omitted_centers": "incomplete boundary neighborhoods only; no synthetic padding or output-selected aperture"},
              "preflight": preflight, "convergence": convergence, "worst_stage_refinement_relative_error": worst,
              "validation_passed": passed,
              "failed_checks": [f"{c}/{bank}/{stage}: {error:.8f} > 0.05" for c, banks in convergence.items()
                                for bank, errors in banks.items() for stage, error in errors.items()
                                if error > spec["acceptance"]["timestep_stage_relative_error"]],
              "propagation_to_both_DNp15": propagated, "condition_metrics": metrics, "controls": controls,
              "deterministic_trace_digests": digests,
              "anatomical_edge_changes": 0,
              "evidence_boundary": "Broader measured female eye grid and independently checked anatomy-predicted right T4 orientation; left/T5 class-level extension and local correlator physiology provisional. Spatial trace -> shared MaleCNS type adapter is explicitly lossy; no individual sensory-body registration, photoreceptor superposition reconstruction, new central-physiology agreement or motor result."}
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    (args.output/"metrics.json").write_text(text, encoding="utf8", newline="\n")
    np.savez_compressed(args.output/"retinotopy_axes.npz", centers=readout.centers, neighbors=readout.neighbors,
                        left=readout.left, preferred_directions=readout.preferred_directions, grid=g.grid[readout.centers])
    make_figure(g, readout, local_checks["source_field"], figures, metrics, central, args.output/"EXP-009-retinotopic-eye.png")
    if args.check_record and json.loads(RECORD.read_text(encoding="utf8")) != result:
        raise RuntimeError("reproduction differs from committed EXP-009 record")
    if args.write_record:
        RECORD.write_text(text, encoding="utf8", newline="\n")
    if args.promote_figure:
        shutil.copyfile(args.output/"EXP-009-retinotopic-eye.png", ROOT/"figures/EXP-009-retinotopic-eye.png")
    print(f"Result {result['status']}; worst refinement error={worst:.4%}", flush=True)


if __name__ == "__main__":
    main()
