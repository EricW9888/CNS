"""Run the frozen continuous sensory-to-DNp15 experiment, without fitting."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time as walltime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_exp004_physiology import CONDITIONS, compare_external, finite_json, load_frozen, summarize
from src.binocular_physiology import BinocularNetwork, POPULATIONS, PhysiologyParameters, evaluate_transformation, response_table
from src.malecns_io import ConnectomeGraph
from src.provenance import sha256_file, validate_materialization
from src.sensory_chain import CHANNELS, SensoryParameters, VisualProjection, central_drive, luminance_movie, motion_channels

SPEC = ROOT / "experiments/EXP-005-sensory-to-DNp15/specification.json"
RECORD = SPEC.with_name("record.json")
IMPLEMENTATION = ["src/sensory_chain.py", "scripts/prepare_exp005_sensory.py", "scripts/run_exp005_sensory.py"]


def load_specification():
    spec = json.loads(SPEC.read_text(encoding="utf8"))
    entry = spec["central_specification"]
    if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
        raise ValueError("frozen central specification changed")
    central_spec, central = load_frozen()
    for field, old_field in [("central_parameters", "parameters"), ("external_target", "external_target"), ("observation_contract", "observation_contract"), ("stimuli", "stimuli")]:
        if spec[field] != central_spec[old_field]:
            raise ValueError("successor changed a frozen central contract")
    for entry in spec["anatomy_files"]:
        path = ROOT / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("visual anatomy differs from frozen specification")
    folder = (ROOT / spec["anatomy_files"][0]["path"]).parent
    visual = ConnectomeGraph(pd.read_parquet(folder / "nodes.parquet"), pd.read_parquet(folder / "edges.parquet"))
    validate_materialization(spec["anatomy"], visual.nodes, visual.edges)
    eye_path = folder / "input_eye_verification.json"
    evidence = json.loads(eye_path.read_text(encoding="utf8"))
    evidence_rows = pd.DataFrame(evidence["rows"]).set_index("bodyId")
    motion = visual.nodes[visual.nodes.stage.isin(["T4", "T5"])].set_index("bodyId")
    if set(evidence_rows.index) != set(motion.index) or evidence_rows.index.duplicated().any():
        raise ValueError("incomplete sensory eye verification")
    for row in evidence_rows.itertuples():
        side = motion.loc[row.Index, "somaSide"]
        ipsi, contra = (row.L_post, row.R_post) if side == "L" else (row.R_post, row.L_post)
        if row.basis == "LOP_output_only_dendritic_ROI_absent" and ipsi == contra == 0:
            ipsi, contra = (row.L_LOP_pre, row.R_LOP_pre) if side == "L" else (row.R_LOP_pre, row.L_LOP_pre)
        elif row.basis != "sensory_dendritic_ROI":
            raise ValueError("unsupported sensory eye evidence basis")
        if row.assigned_eye != side or row.stage != motion.loc[row.Index, "stage"] or ipsi <= contra:
            raise ValueError("invalid sensory input-eye correspondence")
    return spec, central_spec, visual, central


def relative_error(coarse, fine):
    if coarse.shape != fine.shape:
        raise ValueError("timestep axes differ")
    return float(np.max(abs(coarse-fine))/max(float(np.max(abs(coarse))), float(np.max(abs(fine))), 1e-12))


def upstream(visual, p, stimuli):
    results = {}
    for condition, orders in stimuli.items():
        time, images = luminance_movie(p, orders)
        channels = motion_channels(images, p)
        if not np.array_equal(channels, motion_channels(images, p)):
            raise RuntimeError("sensory detector nondeterministic")
        LPi, drive = visual.simulate(channels)
        repeated = visual.simulate(channels)
        if not np.array_equal(LPi, repeated[0]) or not np.array_equal(drive, repeated[1]):
            raise RuntimeError("visual projection nondeterministic")
        results[condition] = {"channels": channels, "LPi": LPi, "drive": drive, "stimulus_luminance": images[:, :, 0, 0]}
        print(f"Sensory bank and visual projection completed: {condition}, dt={p.dt_ms} ms", flush=True)
    return time, results


def select_visual(visual, stages, control):
    channels = stages["channels"]
    if control in {"no_T4", "no_T5"}:
        channels = channels.copy()
        channels[:, [i for i, name in enumerate(CHANNELS) if ("T4" if control == "no_T4" else "T5") in name]] = 0
    if control in {"no_T4", "no_T5", "no_LPi_output"}:
        LPi, drive = visual.simulate(channels, no_LPi_output=control == "no_LPi_output")
    else:
        LPi, drive = stages["LPi"], stages["drive"]
    if control == "visual_disconnected":
        drive = np.zeros_like(drive)
    return {"channels": channels, "LPi": LPi, "drive": drive, "stimulus_luminance": stages["stimulus_luminance"]}


def figure(spec, summaries, table, stage_means, traces, time, destination):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16, 14), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    axes = np.array([[fig.add_subplot(grid[row, col]) if (row, col) != (0, 0) else None for col in range(2)] for row in range(3)], dtype=object)
    first = grid[0, 0].subgridspec(2, 1, height_ratios=[1, 1.2])
    axes[0, 0] = fig.add_subplot(first[0])
    labels = ["L F / R B", "L B / R F", "F / F", "B / B", "L F", "L B", "R F", "R B"]
    ax = axes[0, 0]
    orders = np.array([spec["stimuli"][c] for c in CONDITIONS[:-1]]).T
    ax.imshow(orders, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for eye in range(2):
        for k in range(8):
            ax.text(k, eye, {1: "F →", -1: "← B", 0: "still"}[orders[eye, k]], ha="center", va="center", color="white" if orders[eye, k] else "black")
    ax.set(xticks=range(8), xticklabels=labels, yticks=[0, 1], yticklabels=["Left image", "Right image"], title="A  Actual luminance gratings translated in stimulus-space")
    ax.set_xlabel("F/B: published per-eye directions; not workbook axes or body yaw")
    ax = fig.add_subplot(first[1])
    p = SensoryParameters(**spec["sensory_parameters"])
    _, movie = luminance_movie(p, [1, -1])
    for eye, color in [(0, "#347da0"), (1, "#b34f6c")]:
        ax.plot(time/1000, movie[:, eye, 0, 1], color=color, label=f"Eye {'LR'[eye]}, pixel x=1")
    ax.axvspan(2, 4, color="gray", alpha=.1)
    ax.set(xlabel="2 s static → 2 s motion (45°/s, 18° period) → 2 s static", ylabel="Luminance [0,1]", title="Example actual image samples: L F / R B")
    ax.legend(fontsize=8, ncol=2)
    ax = axes[0, 1]
    values = np.array([stage_means[c]["channels"] for c in CONDITIONS[:-1]]).T
    im = ax.imshow(values, cmap="viridis", aspect="auto", vmin=0)
    ax.set(xticks=range(8), xticklabels=labels, yticks=range(16), yticklabels=CHANNELS, title="B  Provisional ON/OFF motion detector → T4/T5 emissions")
    ax.axhline(7.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, label="Raw luminance-product activity (no fitted gain)", shrink=.75)
    ax = axes[1, 0]
    rows, names = [], []
    for population in ["LPi", *POPULATIONS]:
        for side in ["L", "R"]:
            if population == "LPi":
                ids = [i for i, n in enumerate(spec["LPi_identity"]) if n["somaSide"] == side]
                rows.append([np.mean(np.asarray(stage_means[c]["LPi"])[ids]) for c in CONDITIONS[:-1]])
            else:
                selected = table[table.stage.eq(population) & table.somaSide.eq(side)]
                rows.append([selected[c].mean() for c in CONDITIONS[:-1]])
            names.append(f"{population} {side}")
    values = np.array(rows)
    limit = max(float(np.max(abs(values))), 1e-12)
    im = ax.imshow(values, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    ax.set(xticks=range(8), xticklabels=labels, yticks=range(len(names)), yticklabels=names, title="C  MaleCNS LPi → frozen HS/H2 → intermediates → DNp15")
    for k in range(1, 7):
        ax.axhline(2*k-.5, color="white", lw=1.5)
    fig.colorbar(im, ax=ax, label="Signed state, late window 3.5–4 s (shared scale)", shrink=.8)
    ax = axes[1, 1]
    for condition, style in [("yaw_L_F", "-"), ("yaw_R_F", "--")]:
        for population, color in [("HS", "#347da0"), ("H2", "#9364a0"), ("DNp15", "#d07336")]:
            selected = table.stage.eq(population) & table.somaSide.eq("L")
            ax.plot(time/1000, traces[condition][:, selected].mean(axis=1), style, color=color, label=f"{population} L: {condition}")
    ax.axvspan(2, 4, alpha=.1, color="gray")
    ax.axhline(0, color="gray", lw=.6)
    ax.set(xlabel="Time (s)", ylabel="Signed dimensionless state", title="D  Directional neural traces, including subthreshold / negative states")
    ax.legend(fontsize=8, ncol=2)
    ax = axes[2, 0]
    observed = np.array([spec["external_target"]["observed"][p]["mean_di"] for p in POPULATIONS])
    modeled = [summaries["full"]["populations"][p]["mean_di"] for p in POPULATIONS]
    intervals = np.array([spec["external_target"]["observed"][p]["descriptive_roi_t95_interval"] for p in POPULATIONS]).T
    x = np.arange(6)
    ax.bar(x-.18, observed, .35, color="#77909c", label="Published calcium ROI mean")
    ax.errorbar(x-.18, observed, yerr=[observed-intervals[0], intervals[1]-observed], fmt="none", color="black", capsize=3)
    ax.bar(x+.18, modeled, .35, color="#d28b3d", label="Continuous chain (no fit)")
    ax.axhline(0, color="gray", lw=.6)
    ax.set(xticks=x, xticklabels=POPULATIONS, ylabel="Signed DI (negative = rotational preference)", title="E  Frozen external target: descriptive comparison, not calibration", ylim=(-1.15, 1.05))
    ax.legend(fontsize=8)
    ax = axes[2, 1]
    controls = list(spec["controls"])
    for side, color in [("L", "#347da0"), ("R", "#b34f6c")]:
        for key, style, label in [("DNp15_translation_yaw_ratio", "o-", "translation / yaw OF"), ("DNp15_common_mode_yaw_ratio", "s--", "common mode / yaw")]:
            ratios = [next(r[key] for r in summaries[c]["matched_upstream_magnitude_comparison"] if r["somaSide"] == side) for c in controls]
            ax.plot(range(len(controls)), ratios, style, color=color, label=f"DN {side}: {label}")
    ax.set(xticks=range(len(controls)), xticklabels=["Full", "Visual\noff", "LPi\noutput off", "T4\noff", "T5\noff", "Electrical\noff", "bIPS\nGABA off", "Feedforward"], ylabel="Magnitude ratio (lower = greater rotational preference)", title="F  Controls preserve survivor weights; missing ratio = zero signal")
    ax.legend(fontsize=8)
    fig.suptitle("EXP-005: continuous visual stimulus → motion channels → MaleCNS optic-flow circuit → DNp15\nProvisional sensory transduction; frozen EXP-004 recurrence; neural-only dimensionless activity", fontsize=14)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/exp005_sensory")
    parser.add_argument("--write-record", action="store_true")
    parser.add_argument("--check-record", action="store_true")
    parser.add_argument("--promote-figure", action="store_true")
    args = parser.parse_args()
    if args.write_record and (args.check_record or RECORD.exists()):
        parser.error("write only a new record; never overwrite frozen evidence")
    promoted = ROOT / "figures/EXP-005-sensory-to-DNp15.png"
    if args.promote_figure and promoted.exists():
        raise FileExistsError("refusing to overwrite promoted evidence")
    spec, central_spec, graph, central_graph = load_specification()
    p = SensoryParameters(**spec["sensory_parameters"])
    fine_p = replace(p, dt_ms=p.dt_ms/2)
    visual = VisualProjection.compile(graph, p)
    fine_visual = VisualProjection.compile(graph, fine_p)
    parameters = PhysiologyParameters(**spec["central_parameters"])
    if p.dt_ms != parameters.dt_ms:
        raise ValueError("cascade stages require the same sampling clock")
    controls = list(spec["controls"])
    networks, fine_networks, preflight = {}, {}, {}
    for control in controls:
        central_control = control if control in {"no_electrical", "no_bIPS_GABA_input", "feedforward_chemical_only"} else "full"
        network = BinocularNetwork.compile(central_graph, central_spec["electrical_pairs"], parameters, central_control)
        fine_network = BinocularNetwork.compile(central_graph, central_spec["electrical_pairs"], replace(parameters, dt_ms=parameters.dt_ms/2), central_control)
        networks[control], fine_networks[control] = network, fine_network
        tests = {"central": network.preflight(), "half_dt_central": fine_network.preflight(),
                 "LPi": visual.preflight(no_LPi_output=control == "no_LPi_output"), "half_dt_LPi": fine_visual.preflight(no_LPi_output=control == "no_LPi_output")}
        if not all(v["passed"] for v in tests.values()):
            raise RuntimeError("zero-input/effective transition preflight failed")
        preflight[control] = tests
    full_operator = networks["full"].chemical.toarray()
    for c, n in networks.items():
        values = n.chemical.toarray()
        if not np.array_equal(values[values != 0], full_operator[values != 0]):
            raise RuntimeError("control changed surviving central weights")
        preflight[c]["surviving_weights_unchanged"] = True
    print("All final-graph controls passed contraction, effective-transition and zero-input decay preflight", flush=True)
    started = walltime.perf_counter()
    time, base = upstream(visual, p, spec["stimuli"])
    fine_time, fine_base = upstream(fine_visual, fine_p, spec["stimuli"])
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, tables, stage_means = {}, [], {}
    trace_digest = hashlib.sha256()
    full_traces = None
    for control in controls:
        traces, fine_traces, convergence, stage_summary, saved = {}, {}, {}, {}, {}
        network, fine_network = networks[control], fine_networks[control]
        for condition in CONDITIONS:
            stages = select_visual(visual, base[condition], control)
            fine_stages = select_visual(fine_visual, fine_base[condition], control)
            drive = central_drive(visual.input_ids, stages["drive"], network.nodes.bodyId)
            fine_drive = central_drive(fine_visual.input_ids, fine_stages["drive"], fine_network.nodes.bodyId)
            trace = network.simulate(drive)
            fine_trace = fine_network.simulate(fine_drive)
            if not np.array_equal(trace, network.simulate(drive)) or not np.array_equal(fine_trace, fine_network.simulate(fine_drive)):
                raise RuntimeError("central execution nondeterministic")
            again = select_visual(visual, base[condition], control)
            if not all(np.array_equal(stages[k], again[k]) for k in stages):
                raise RuntimeError("visual execution nondeterministic")
            stages["central"] = trace
            fine_stages["central"] = fine_trace
            convergence[condition] = {k: relative_error(stages[k], fine_stages[k][::2]) for k in stages}
            if max(convergence[condition].values()) > .05:
                raise RuntimeError(f"stage timestep convergence failed: {control}/{condition}: {convergence[condition]}")
            traces[condition], fine_traces[condition] = trace, fine_trace
            mask = (time >= 3500) & (time < 4000)
            stage_summary[condition] = {k: v[mask].mean(axis=0).tolist() for k, v in stages.items() if k != "central"}
            for stage, values in stages.items():
                trace_digest.update(f"{control}/{condition}/{stage}".encode())
                trace_digest.update(np.asarray(values, dtype="<f8").tobytes())
                saved[f"{condition}_{stage}"] = values
        table = response_table(network, time, traces, spec["external_target"]["response_window_ms"])
        fine_table = response_table(fine_network, fine_time, fine_traces, spec["external_target"]["response_window_ms"])
        a, b = table[["di", "magnitude_di"]].to_numpy(), fine_table[["di", "magnitude_di"]].to_numpy()
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise RuntimeError("timestep changed defined observations")
        difference = abs(a-b)
        late_error = float(np.max(difference[np.isfinite(difference)])) if np.isfinite(difference).any() else 0.
        if late_error > .02:
            raise RuntimeError("late index convergence failed")
        preflight[control]["timestep_convergence"] = {"passed": True, "relative_trace_errors": convergence, "max_abs_late_index_difference": late_error}
        table.insert(0, "control", control)
        tables.append(table)
        summaries[control] = summarize(table, traces, time, spec["external_target"]["response_window_ms"])
        stage_means[control] = stage_summary
        np.savez_compressed(args.output / f"stages_{control}.npz", time_ms=time,
                            channel_names=np.asarray(CHANNELS), motion_body_ids=visual.motion_ids,
                            motion_body_channel_indices=visual.motion_channel_indices,
                            LPi_body_ids=visual.LPi_ids, HS_H2_input_ids=visual.input_ids,
                            central_body_ids=network.nodes.bodyId.to_numpy(), **saved)
        if control == "full":
            full_traces, full_table = traces, table
        print(f"Completed {control}: all stages, repeats and half-timestep comparisons", flush=True)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(args.output / "central_neuron_responses.csv", index=False, float_format="%.17g")
    full_stage_means = stage_means["full"]
    target = evaluate_transformation(full_table, epsilon=spec["observation_contract"]["numerical_epsilon"])
    ff = summaries["feedforward_chemical_only"]["matched_upstream_magnitude_comparison"]
    target["full_reduces_both_DN_ratios_vs_feedforward"] = [all(row[k]+1e-10 < next(r[k] for r in ff if r["bodyId"] == row["bodyId"]) for k in ["DNp15_translation_yaw_ratio", "DNp15_common_mode_yaw_ratio"]) for row in target["per_side"]]
    target["DNp15_bilateral_superposition_residual_max"] = max(summaries["full"]["bilateral_minus_sum_of_unilateral_max_abs_state"]["DNp15"].values())
    target["recurrent_enhancement_contract_passed"] = bool(target["limited_neural_transformation_gate"] and all(target["full_reduces_both_DN_ratios_vs_feedforward"]) and target["DNp15_bilateral_superposition_residual_max"] > 1e-10)
    preferences = {}
    for population in ["HS", "H2"]:
        for side in "LR":
            selected = full_table[full_table.stage.eq(population) & full_table.somaSide.eq(side)]
            f = selected["left_F" if side == "L" else "right_F"].mean()
            b = selected["left_B" if side == "L" else "right_B"].mean()
            preferences[f"{population}_{side}"] = {"F": float(f), "B": float(b), "published_preference_preserved": bool(f > b if population == "HS" else b > f)}
    union_ids = set(graph.nodes.bodyId) | set(central_graph.nodes.bodyId)
    chain_connected = bool(all(float(max(abs(row.of_yaw), abs(row.of_translation))) > 1e-10 for row in full_table[full_table.stage.eq("DNp15")].itertuples()))
    if summaries["visual_disconnected"]["zero_condition_max_abs_state"] != 0 or any(abs(v) > 0 for r in summaries["visual_disconnected"]["DNp15_per_body"] for c, v in r.items() if c in CONDITIONS):
        raise RuntimeError("disconnection did not abolish DN activity")
    source_hashes = {path: hashlib.sha256((ROOT / path).read_text(encoding="utf8").replace("\r\n", "\n").encode()).hexdigest() for path in IMPLEMENTATION}
    eye_path = ROOT / "data/malecns_exp005_sensory/input_eye_verification.json"
    eye_evidence = json.loads(eye_path.read_text(encoding="utf8"))
    metrics = finite_json({"specification_sha256": sha256_file(SPEC), "implementation_LF_sha256": source_hashes,
        "input_eye_verification": {"path": eye_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(eye_path), "verified_bodies": len(visual.motion_ids),
                                   "dendritic_ROI_absent_body_ids": eye_evidence["dendritic_ROI_absent_body_ids"],
                                   "basis": eye_evidence["method"]},
        "preflight": preflight, "responses": summaries, "stage_late_means": stage_means,
        "external_comparison": compare_external(summaries["full"], spec), "HS_H2_preferences": preferences,
        "qualitative_target": target, "sensory_to_both_DNp15_connected": chain_connected,
        "chain_anatomy": {"node_count": len(union_ids), "edge_count": len(graph.edges)+len(central_graph.edges),
                          "visual_edges": len(graph.edges), "frozen_central_edges": len(central_graph.edges)},
        "trace_arrays_sha256": trace_digest.hexdigest(), "deterministic_repeats": True,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}})
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    (args.output / "runtime.json").write_text(json.dumps({"simulation_analysis_seconds": walltime.perf_counter()-started})+"\n", encoding="utf8")
    image_path = args.output / "figures/EXP-005-sensory-to-DNp15.png"
    figure(spec, summaries, full_table, full_stage_means, full_traces, time, image_path)
    if args.check_record and json.loads(RECORD.read_text(encoding="utf8"))["metrics"] != metrics:
        raise ValueError("reproduction differs from committed result")
    if args.write_record:
        record = {"experiment_id": spec["experiment_id"], "title": "Continuous binocular luminance-to-DNp15 neural pathway",
            "status": "completed_observable_provisional_chain" if chain_connected else "completed_stable_propagation_failure",
            "parent_commit": spec["parent_commit"], "preserved_baselines": ["EXP-001", "EXP-002", "EXP-003", "EXP-004"],
            "question": "Can explicit binocular visual motion propagate through a physiological T4/T5 approximation and identified MaleCNS visual projections into the frozen central optic-flow network and DNp15, with errors localized by stage?",
            "scope": {"body_simulation": False, "parameter_search": False, "central_retuning": False, "elementary_motion_detection_from_connectome": False},
            "specification": SPEC.relative_to(ROOT).as_posix(), "anatomy": spec["anatomy"], "metrics": metrics,
            "conclusion_boundary": "An observable sensory-to-descending chain with an explicitly provisional luminance-to-T4/T5 approximation is not a validated retinal microcircuit, calcium prediction, behavior, or whole fly. The recurrent mechanism is tested separately against the frozen EXP-004 target; retain negative comparisons and all ablations.",
            "outputs": {"figure": promoted.relative_to(ROOT).as_posix(), "generated": "results/exp005_sensory/{metrics.json,central_neuron_responses.csv,stages_<control>.npz,figures/EXP-005-sensory-to-DNp15.png}", "bulk_ignored": True},
            "reproduce": "python scripts/prepare_exp005_sensory.py --check-specification; python scripts/run_exp005_sensory.py --check-record"}
        RECORD.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    if args.promote_figure:
        promoted.write_bytes(image_path.read_bytes())
    print(json.dumps({"sensory_to_both_DNp15_connected": chain_connected, "HS_H2_preferences": preferences,
                      "qualitative_target": metrics["qualitative_target"], "trace_arrays_sha256": trace_digest.hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
