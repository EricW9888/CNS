"""Diagnose frozen EXP-009, validate one predeclared sampling successor, replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_exp008_eye import evidence_sha256
from scripts.render_exp010_reference import SPEC
from scripts.run_exp005_sensory import load_specification, relative_error
from scripts.run_exp008_eye import held_light, replay_sensor, sensor_movie, summarize, trace_digest
from scripts.run_exp009_retinotopy import load_resolved, local_preflight, propagate_channels, verified_trial
from src.binocular_physiology import BinocularNetwork, PhysiologyParameters
from src.compound_eye import EyeGeometry, EyeSampler, Scene, rotation_z
from src.provenance import sha256_file
from src.retinotopic_eye import RetinotopicReadout
from src.retinotopy_diagnostics import correlators, error_field, weighted_occlusion, possible_occlusion
from src.sensory_chain import SensoryParameters, VisualProjection

RECORD = SPEC.with_name("record.json")
IMPLEMENTATION = ["src/retinotopy_diagnostics.py", "scripts/render_exp010_reference.py", "scripts/run_exp010_refinement.py", "scripts/prepare_exp010_identity.py"]


def local_response(light, sensor_dt, readout, p):
    return readout.responses(held_light(light, sensor_dt, p.dt_ms, p.duration_ms), p,
                             save_stride=round(2 / p.dt_ms))


def original_diagnostic(coarse, fine, coarse_local, fine_local, readout, g, p, speed):
    inputs = [held_light(coarse, 2., p.dt_ms, p.duration_ms), held_light(fine, 1., p.dt_ms, p.duration_ms)]
    light_error, _, _ = error_field(*inputs)
    components = [correlators(x, readout, p) for x in inputs]
    stages = {"causal_held_light": light_error}
    for key in components[0]:
        stages[key] = error_field(components[0][key], components[1][key])[0]
    del components, inputs
    local_error, peaks, energy = error_field(coarse_local, fine_local)
    stages["local_emissions"] = local_error
    difference = abs(coarse_local-fine_local)
    order = np.argsort(-peaks, kind="stable")
    u, n = g.directions[readout.centers], g.directions[readout.neighbors]
    spacing = np.rad2deg(np.arccos(np.clip((u[:, None]*n).sum(axis=-1), -1, 1))).max(axis=-1)
    skew = np.rad2deg(np.arccos(np.clip((readout.preferred_directions[:, 1]*readout.preferred_directions[:, 3]).sum(axis=-1), -1, 1)))
    rows = []
    for i in order[:10]:
        time, polarity, subtype = np.unravel_index(np.argmax(difference[:, i]), difference[:, i].shape)
        facet = int(readout.centers[i])
        rows.append({"facet_index": facet, "eye": "L" if readout.left[i] else "R", "grid": g.grid[facet].tolist(),
                     "time_ms": int(time*2), "polarity": ["ON", "OFF"][polarity], "subtype_proxy": "abcd"[subtype],
                     "peak_error": float(peaks[i]), "energy_fraction": float(energy[i]/max(energy.sum(), 1e-30)),
                     "max_neighbor_angle_degrees": float(spacing[i]), "b_d_angle_degrees": float(skew[i])})
    # Light jumps are input diagnostics, not motion labels supplied to the model.
    jumps = abs(np.diff(fine, axis=0)).max(axis=0)
    jump_per_center = np.maximum(jumps[readout.centers], jumps[readout.neighbors].max(axis=-1))
    correlations = {"max_neighbor_spacing": float(np.corrcoef(peaks, spacing)[0, 1]),
                    "local_basis_skew": float(np.corrcoef(peaks, abs(skew-90))[0, 1]),
                    "neighborhood_light_jump": float(np.corrcoef(peaks, jump_per_center)[0, 1])}
    occlusion = possible_occlusion(g, Scene(), speed)
    near_boundary = occlusion[readout.centers] | occlusion[readout.neighbors].any(axis=-1)
    smooth_wall = replace(Scene(), occluder_radius_mm=0.)
    smooth = np.asarray([smooth_wall.radiance(g.lens_mm+[speed*s, 0., 0.], g.directions) for s in (0., .5, 1., 1.5, 2.)])
    contrast = abs(smooth[:, readout.neighbors]-smooth[:, readout.centers, None]).max(axis=(0, 2))
    correlations["smooth_wall_neighbor_contrast_five_poses"] = float(np.corrcoef(peaks, contrast)[0, 1])
    location = {"neighborhoods_with_possible_acceptance_cone_occlusion": int(near_boundary.sum()),
                "local_error_energy_fraction_in_possible_occlusion_region": float(energy[near_boundary].sum()/energy.sum()),
                "top10_peak_neighborhoods_in_possible_occlusion_region": int(near_boundary[order[:10]].sum()),
                "method": "Analytic 3sigma acceptance-cone/sphere overlap at 2001 poses, conservative potential visibility, not neural input"}
    return {"stages": stages, "coincident_physical_light_max_difference": float(abs(coarse-fine[::2]).max()),
            "top_neighborhoods": rows, "spatial_correlations_descriptive": correlations, "occlusion_localization": location}, peaks, energy


def neighborhood_interventions(g, readout, eye_spec, p):
    """Fixed original hotspots, not neighborhoods selected from successor/DN output."""
    original_ids = np.array([400, 711])
    rows = np.array([np.flatnonzero(readout.centers == i)[0] for i in original_ids])
    ids = np.unique(np.r_[original_ids, readout.neighbors[rows].ravel()])
    lookup = np.full(len(g.left), -1, int)
    lookup[ids] = np.arange(len(ids))
    neighbors = g.neighbors[ids].copy()
    valid = neighbors >= 0
    neighbors[valid] = lookup[neighbors[valid]]
    subg = EyeGeometry(g.lens_mm[ids], g.directions[ids], g.left[ids], neighbors, g.grid[ids])
    subr = replace(readout, centers=lookup[original_ids], neighbors=lookup[readout.neighbors[rows]],
                   left=readout.left[rows], preferred_directions=readout.preferred_directions[rows], facet_count=len(ids))
    scene = Scene(**eye_spec["scene"])
    result = {}
    for condition in ("translation_positive_X", "translation_negative_X"):
        result[condition] = {}
        for label, geom, sc, radial, azimuthal in [
            ("original_5x24", subg, scene, 5, 24),
            ("quadrature_9x64", subg, scene, 9, 64),
            ("quadrature_17x128", subg, scene, 17, 128),
            ("no_occluder", subg, replace(scene, occluder_radius_mm=0.), 5, 24),
            ("common_lens_origin", replace(subg, lens_mm=np.zeros_like(subg.lens_mm)), scene, 5, 24)]:
            print(f"Diagnostic-only {condition}/{label}", flush=True)
            sampler = EyeSampler.compile(geom, radial=radial, azimuthal=azimuthal, fwhm_degrees=5.)
            _, light = sensor_movie(sampler, sc, p, eye_spec["pose_conditions"][condition], 1.)
            _, a = local_response(light[::2], 2., subr, p)
            _, b = local_response(light, 1., subr, p)
            delta = abs(a-b).max(axis=(0, 2, 3))
            entry = {"local_2_to_1ms_relative_error": relative_error(a, b),
                     "max_local_differences_at_facets_400_711": delta.tolist(),
                     "maximum_1ms_light_jump": float(abs(np.diff(light, axis=0)).max())}
            if label == "original_5x24":
                base = light.copy()
                time = np.arange(len(light))
                x = np.clip(time-p.onset_ms, 0, p.offset_ms-p.onset_ms)/1000*eye_spec["pose_conditions"][condition][1]
                visibility = weighted_occlusion(sampler, sc, np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]))
                active = np.max(visibility, axis=0) > 0
                entry["facets_with_any_occlusion_in_hotspot_neighborhoods"] = ids[active].tolist()
                entry["largest_occlusion_weight_jump"] = float(abs(np.diff(visibility, axis=0)).max())
                entry["hotspot_center_occlusion_max"] = visibility[:, lookup[original_ids]].max(axis=0).tolist()
            else:
                entry["light_max_difference_from_original"] = float(abs(light-base).max())
            result[condition][label] = entry
    return result


def make_figure(g, readout, fields, diagnostics, convergence, metrics, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    u = g.directions[readout.centers]
    az, el = np.rad2deg(np.arctan2(u[:, 1], u[:, 0])), np.rad2deg(np.arcsin(u[:, 2]))
    for ax, c in zip(axes[0], ("translation_positive_X", "translation_negative_X")):
        peaks = fields[c]
        im = ax.scatter(az, el, c=peaks/diagnostics[c]["stages"]["local_emissions"]["global_normalizer"]*100,
                        s=8, cmap="magma", vmin=0, vmax=10)
        hot = diagnostics[c]["top_neighborhoods"][0]["facet_index"]
        i = np.flatnonzero(readout.centers == hot)[0]
        ax.scatter([az[i]], [el[i]], s=90, facecolor="none", edgecolor="#00b5ad")
        ax.set(title=f"EXP-009 {'+X' if 'positive' in c else '−X'} · 2→1 ms local error field",
               xlabel="Head-frame azimuth (degrees; +left)", ylabel="Elevation (degrees)", xlim=(-180, 180), ylim=(-90, 90))
        fig.colorbar(im, ax=ax, label="Peak local difference / shared stage maximum (%)", shrink=.8)
    ax = axes[1, 0]
    keys = list(convergence)
    x = np.arange(len(keys))
    ax.bar(x-.18, [diagnostics[k]["stages"]["local_emissions"]["relative_error"]*100 if k in diagnostics else 0 for k in keys],
           width=.36, color="#aaaaaa", label="Frozen EXP-009 2→1 ms (translations)")
    ax.bar(x+.18, [convergence[k]["sensor_1_to_half_ms"]["local"]*100 for k in keys], width=.36, color="#1693a1", label="Fixed successor 1→0.5 ms")
    ax.axhline(5, color="#ad3434", linestyle="--", label="Unchanged 5% criterion")
    ax.set(title="Local criterion, before downstream evaluation", ylabel="Stage max-norm difference (%)", xticks=x,
           xticklabels=["Static", "+Z", "−Z", "+X", "−X"])
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    if metrics:
        for c, color in [("yaw_positive_Z", "#1693a1"), ("yaw_negative_Z", "#d26837")]:
            for side, style in [("L", "-"), ("R", "--")]:
                row = next(r for r in metrics[c] if r["stage"] == "DNp15" and r["side"] == side)
                ax.plot([0, 1], [row["min_state"], row["max_state"]], style, color=color, label=f"{c.replace('yaw_', '')} DNp15 {side}")
        ax.set(xticks=[0, 1], xticklabels=["Minimum", "Maximum"], ylabel="Signed dimensionless state", title="Unretuned downstream propagation (not physiology validation)")
        ax.legend(fontsize=8)
    else:
        names = ["causal_held_light", "delayed", "neighbor_correlators", "signed_bd", "local_emissions"]
        x = np.arange(len(names))
        for offset, c, color in [(-.18, "translation_positive_X", "#1693a1"), (.18, "translation_negative_X", "#d26837")]:
            ax.bar(x+offset, [diagnostics[c]["stages"][n]["relative_error"]*100 for n in names], width=.36, color=color,
                   label="+X" if offset < 0 else "−X")
        ax.axhline(5, color="#ad3434", linestyle="--")
        ax.set(xticks=x, xticklabels=["Held\nlight", "Delay", "Neighbor\nproducts", "Signed\nstencil", "Local\nemission"],
               ylabel="Relative stage max-norm error (%)", title="EXP-009 discrepancy develops before pooling")
        ax.legend(fontsize=8)
    fig.suptitle("EXP-010 · Temporal refinement of the measured-grid sensory interface\nFrozen EXP-009 remains incomplete; numerical checks do not establish sensory-body identities", fontsize=12)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", type=Path, default=ROOT / "results/exp008_eye_final")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/exp010_reference")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/exp009_portable")
    parser.add_argument("--output", type=Path, default=ROOT / "results/exp010_refinement")
    parser.add_argument("--write-record", action="store_true")
    parser.add_argument("--check-record", action="store_true")
    parser.add_argument("--promote-figure", action="store_true")
    args = parser.parse_args()
    if args.write_record and (RECORD.exists() or args.check_record):
        raise ValueError("canonical records cannot be overwritten")
    if not args.output.resolve().is_relative_to((ROOT / "results").resolve()):
        raise ValueError("bulk output must stay under ignored results")
    spec = json.loads(SPEC.read_text())
    spec009, eye_spec, g, reference = load_resolved()
    old = json.loads((ROOT / "experiments/EXP-009-retinotopic-eye/record.json").read_text())
    identity_path = SPEC.with_name("identity-evidence.json")
    identity = json.loads(identity_path.read_text())
    for entry in identity["source_files"]:
        if sha256_file(ROOT/entry["path"]) != entry["sha256"]:
            raise ValueError("identity evidence source changed")
    for f, digest in old["implementation_sha256"].items():
        if evidence_sha256(ROOT/f) != digest:
            raise ValueError(f"frozen EXP-009 implementation changed: {f}")
    if sha256_file(ROOT / "experiments/EXP-009-retinotopic-eye/specification.json") != old["specification_sha256"]:
        raise ValueError("frozen EXP-009 specification changed")
    p = SensoryParameters(**spec009["parameters"])
    fp = replace(p, dt_ms=.25)
    readout = RetinotopicReadout.compile(g)
    preflight = {"local": local_preflight(g, readout, reference, p), "local_half_dt": local_preflight(g, readout, reference, fp)}
    if not all(x["passed"] for x in preflight.values()):
        raise RuntimeError("local preflight failed")
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics, fields, convergence, input_digests, channels, digests, metrics, controls = {}, {}, {}, {}, {}, {}, {}, {}
    old008 = json.loads((ROOT / "experiments/EXP-008-compound-eye/record.json").read_text())
    # Finish all input/local checks before inspecting any new downstream output.
    for c in spec["conditions"]:
        print(f"Local validation {c}", flush=True)
        coarse, _ = verified_trial(args.sensors, c, g, eye_spec)
        if c == "static":
            light = np.repeat(coarse[:1], 6000, axis=0)
            fine_light = np.repeat(coarse[:1], 12000, axis=0)
        else:
            _, light = replay_sensor(args.sensors, c, g, p, 1., refinement=True)
            if trace_digest({"sensor_light": light}) != old008["sensor_refinement_digests"][c]:
                raise ValueError("1ms frozen sensory input changed")
            _, fine_light = replay_sensor(args.reference, c, g, p, .5)
            if not np.array_equal(fine_light[::2], light):
                raise ValueError("reference changed coincident physical samples")
            sampler = EyeSampler.compile(g, radial=5, azimuthal=24, fwhm_degrees=5.)
            for k in (4001, 7773):
                seconds = np.clip(k*.5-p.onset_ms, 0, p.offset_ms-p.onset_ms)/1000
                rate, speed = eye_spec["pose_conditions"][c]
                if not np.array_equal(sampler.sample(Scene(**eye_spec["scene"]), rotation_z(np.deg2rad(rate)*seconds), [speed*seconds, 0, 0]), fine_light[k]):
                    raise ValueError("fresh reference ray replay differs")
        a, al = local_response(light, 1., readout, p)
        b, bl = local_response(fine_light, .5, readout, p)
        fc, fl = local_response(light, 1., readout, fp)
        repeated_c, repeated_l = local_response(light, 1., readout, p)
        if not np.array_equal(a, repeated_c) or not np.array_equal(al, repeated_l):
            raise RuntimeError("local determinism failed")
        del repeated_c, repeated_l
        convergence[c] = {"sensor_1_to_half_ms": {"local": relative_error(al, bl), "channels": relative_error(a, b)},
                          "neural_half_dt": {"local": relative_error(al, fl), "channels": relative_error(a, fc[::2])}}
        if c.startswith("translation"):
            with np.load(args.baseline/f"{c}.npz", allow_pickle=False) as saved:
                baseline_arrays = {k: saved[k] for k in ("local", "channels", "LPi", "HS_H2_drive", "central")}
            if trace_digest(baseline_arrays) != old["deterministic_trace_digests"][c]:
                raise ValueError("saved EXP-009 stage arrays changed")
            diagnostics[c], fields[c], _ = original_diagnostic(coarse, light, baseline_arrays["local"], al, readout, g, p, eye_spec["pose_conditions"][c][1])
            diagnostics[c]["downstream_sensor_2_to_1ms"] = old["convergence"][c]["sensor_2_to_1ms"]
            del baseline_arrays
        input_digests[c] = {"production_light": trace_digest({"sensor_light": light}), "reference_light": trace_digest({"sensor_light": fine_light})}
        channels[c] = {"production": a, "sensor_reference": b, "neural_reference": fc,
                       "local_digest": trace_digest({"local": al}),
                       "both_eyes_ON_OFF_nonzero": bool(np.all(a.reshape(-1, 2, 2, 4).max(axis=(0, 3)) > 0)) if c != "static" else None}
        _, eye_null = local_response(np.repeat(light[:1], len(light), axis=0), 1., readout, p)
        if np.any(eye_null) or np.any(al[:1000]) or (c == "static" and np.any(al)):
            raise RuntimeError("static/causal/eye disconnection check failed")
        controls[c] = {"eye_disconnection_max_abs_local": 0., "response_before_pose_change_max_abs": 0.}
        del al, bl, fl, eye_null, light, fine_light, coarse
        print(f"  convergence {convergence[c]}", flush=True)
    interventions = neighborhood_interventions(g, readout, eye_spec, p)
    local_pass = all(v <= .05 for banks in convergence.values() for errors in banks.values() for v in errors.values())
    if local_pass:
        print("All fixed-candidate local checks pass; frozen downstream replay begins.", flush=True)
        _, cp_spec, vg, cg = load_specification()
        cp = PhysiologyParameters(**cp_spec["parameters"])
        visual, central = VisualProjection.compile(vg, p), BinocularNetwork.compile(cg, cp_spec["electrical_pairs"], cp)
        fv, fn = VisualProjection.compile(vg, fp), BinocularNetwork.compile(cg, cp_spec["electrical_pairs"], replace(cp, dt_ms=.25))
        preflight.update({"LPi": visual.preflight(), "central": central.preflight(), "LPi_half_dt": fv.preflight(), "central_half_dt": fn.preflight()})
        if not all(x["passed"] for x in preflight.values()):
            raise RuntimeError("downstream preflight failed")
        for c, bank in channels.items():
            a = propagate_channels(bank["production"], visual, central)
            b = propagate_channels(bank["sensor_reference"], visual, central)
            f = propagate_channels(bank["neural_reference"], fv, fn)
            repeated = propagate_channels(bank["production"], visual, central)
            if any(not np.array_equal(a[k], repeated[k]) for k in a):
                raise RuntimeError("downstream determinism failed")
            for k in ("LPi", "HS_H2_drive", "central"):
                convergence[c]["sensor_1_to_half_ms"][k] = relative_error(a[k], b[k])
                convergence[c]["neural_half_dt"][k] = relative_error(a[k], f[k][::2])
            dn = np.isin(central.nodes.bodyId, [12069, 11215])
            convergence[c]["sensor_1_to_half_ms"]["DNp15"] = relative_error(a["central"][:, dn], b["central"][:, dn])
            convergence[c]["neural_half_dt"]["DNp15"] = relative_error(a["central"][:, dn], f["central"][:, dn][::2])
            null = propagate_channels(np.zeros_like(bank["production"]), visual, central)
            if any(np.any(x) for x in null.values()) or np.any(central.simulate(np.zeros_like(a["central"]))):
                raise RuntimeError("downstream disconnection failed")
            controls[c].update({"motion_disconnection_max_abs_downstream": 0., "central_disconnection_max_abs": 0., "unchanged_surviving_operators": True})
            metrics[c] = summarize(a["central"], central, p)
            digests[c] = {"local": bank["local_digest"], "downstream": trace_digest(a)}
            np.savez_compressed(args.output/f"{c}.npz", time_ms=np.arange(len(a["central"]))*p.dt_ms, **a)
    else:
        # Requested diagnosis of the ORIGINAL 2->1ms downstream discrepancy.
        # This is not integration of the failed 0.5ms candidate or a validation
        # claim, and cannot choose any sensor/physiology parameter.
        _, cp_spec, vg, cg = load_specification()
        visual = VisualProjection.compile(vg, p)
        central = BinocularNetwork.compile(cg, cp_spec["electrical_pairs"], PhysiologyParameters(**cp_spec["parameters"]))
        preflight.update({"LPi": visual.preflight(), "central": central.preflight()})
        if not all(x["passed"] for x in preflight.values()):
            raise RuntimeError("frozen diagnosis operators fail preflight")
        for c in diagnostics:
            refined = propagate_channels(channels[c]["production"], visual, central)
            repeated = propagate_channels(channels[c]["production"], visual, central)
            if any(not np.array_equal(refined[k], repeated[k]) for k in refined):
                raise RuntimeError("original downstream diagnostic is nondeterministic")
            with np.load(args.baseline/f"{c}.npz", allow_pickle=False) as saved:
                old_central = saved["central"]
            diagnostics[c]["downstream_sensor_2_to_1ms"]["DNp15"] = relative_error(old_central[:, np.isin(central.nodes.bodyId, [12069, 11215])], refined["central"][:, np.isin(central.nodes.bodyId, [12069, 11215])])
            digests[c] = {"original_1ms_local": channels[c]["local_digest"], "original_refined_downstream": trace_digest(refined)}
            null = propagate_channels(np.zeros_like(channels[c]["production"]), visual, central)
            if any(np.any(v) for v in null.values()) or np.any(central.simulate(np.zeros_like(refined["central"]))):
                raise RuntimeError("original downstream diagnostic disconnection fails")
            controls[c].update({"original_motion_disconnect_max_abs_downstream": 0., "original_central_disconnect_max_abs": 0., "unchanged_surviving_operators": True})
            np.savez_compressed(args.output/f"{c}_original_1ms_diagnostic.npz", **refined)
    passed = local_pass and all(v <= .05 for banks in convergence.values() for errors in banks.values() for v in errors.values())
    result = {"experiment_id": spec["experiment_id"], "status": "numerically_validated_sampling_successor" if passed else "sampling_validation_incomplete",
              "validation_passed": passed, "local_validation_passed": local_pass,
              "specification_sha256": sha256_file(SPEC), "implementation_sha256": {f: evidence_sha256(ROOT/f) for f in IMPLEMENTATION},
              "frozen_predecessor_record_sha256": sha256_file(ROOT / "experiments/EXP-009-retinotopic-eye/record.json"),
              "identity_evidence_sha256": sha256_file(identity_path),
              "preflight": preflight, "original_translation_diagnostics": diagnostics, "neighborhood_interventions": interventions,
              "convergence": convergence, "input_digests": input_digests, "deterministic_trace_digests": digests,
              "controls": controls, "central_per_body": metrics,
              "both_eyes_ON_OFF_nonzero": {c: x["both_eyes_ON_OFF_nonzero"] for c, x in channels.items()},
              "anatomical_edge_changes": 0, "biological_parameter_changes": [],
              "individual_facet_to_MaleCNS_identity_resolved": False,
              "candidate_downstream_validation_evaluated": local_pass,
              "evidence_boundary": "Numerical sampling validation of an unchanged provisional local correlator/type adapter. No individual sensory-body registration, calibrated physiology, or motor result."}
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    (args.output/"metrics.json").write_text(text, encoding="utf8", newline="\n")
    make_figure(g, readout, fields, diagnostics, convergence, metrics, args.output/"EXP-010-retinotopic-refinement.png")
    if args.check_record and json.loads(RECORD.read_text()) != result:
        raise RuntimeError("exact record replay differs")
    if args.write_record:
        RECORD.write_text(text, encoding="utf8", newline="\n")
    if args.promote_figure:
        import shutil
        shutil.copyfile(args.output/"EXP-010-retinotopic-refinement.png", ROOT/"figures/EXP-010-retinotopic-refinement.png")
    print(f"Result: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
