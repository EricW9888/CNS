"""Run the frozen, neural-only binocular physiology experiment.

Preparation fixes anatomy, primary-source targets, inputs and parameters.
This runner cannot fit parameters, alter those inputs, or execute a body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time as walltime
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.binocular_physiology import (
    BinocularNetwork, CONTROLS, POPULATIONS, PhysiologyParameters,
    controlled_drive, response_table, matched_upstream_comparison, evaluate_transformation,
)
from src.malecns_io import ConnectomeGraph
from src.numerics import spectral_radius
from src.provenance import sha256_file, validate_materialization

SPEC_PATH = ROOT / "experiments/EXP-004-binocular-physiology/specification.json"
RECORD_PATH = SPEC_PATH.with_name("record.json")
CONDITIONS = ("yaw_L_F", "yaw_R_F", "translation_F", "translation_B",
              "left_F", "left_B", "right_F", "right_B", "zero")
SHORT_CONTROLS = ["Full", "No electrical", "bIPS output off", "bIPS GABA off", "DN GABA off", "Chemical feedback off", "Feed-forward only"]


def finite_json(value):
    """Encode undefined indices as null, never as manufactured zero scores."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_frozen():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf8"))
    for entry in spec["sources"] + spec["anatomy_files"]:
        path = ROOT / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"source/anatomy differs from frozen specification: {entry['path']}")
    folder = ROOT / Path(spec["anatomy_files"][0]["path"]).parent
    nodes = pd.read_parquet(folder / "nodes.parquet")
    edges = pd.read_parquet(folder / "edges.parquet")
    validate_materialization(spec["anatomy"], nodes, edges)
    if finite_json(nodes.replace({np.nan: None}).to_dict(orient="records")) != spec["nodes"]:
        raise ValueError("body metadata differs from frozen specification")
    if tuple(k for k in spec["controls"] if k != "normalization") != CONTROLS:
        raise ValueError("control implementations differ from predeclared controls")
    if tuple(spec["stimuli"]) != CONDITIONS:
        raise ValueError("stimulus implementations differ from predeclared conditions")
    return spec, ConnectomeGraph(nodes,edges)


def summarize(table: pd.DataFrame, traces: dict, time: np.ndarray, window: list) -> dict:
    populations = {}
    mask = (time>=window[0]) & (time<window[1])
    for population in POPULATIONS:
        selected = table[table.stage.eq(population)]
        populations[population] = {
            "n_bodies": len(selected), "defined_di_bodies": int(selected.di.notna().sum()),
            "mean_di": float(selected.di.mean()), "mean_magnitude_di": float(selected.magnitude_di.mean()),
            "mean_of_yaw": float(selected.of_yaw.mean()),
            "mean_of_translation": float(selected.of_translation.mean()),
            "condition_mean_signed_state": {c: float(selected[c].mean()) for c in CONDITIONS},
            "condition_mean_emission": {c: float(selected[c+"_emission"].mean()) for c in CONDITIONS},
        }
    nonlinear = {}
    for population in POPULATIONS:
        index = table.stage.eq(population).to_numpy()
        residuals = {
            "yaw_L_F": traces["yaw_L_F"]-traces["left_F"]-traces["right_B"],
            "yaw_R_F": traces["yaw_R_F"]-traces["left_B"]-traces["right_F"],
            "translation_F": traces["translation_F"]-traces["left_F"]-traces["right_F"],
            "translation_B": traces["translation_B"]-traces["left_B"]-traces["right_B"],
        }
        nonlinear[population] = {c: float(np.max(abs(residual[mask][:,index]))) for c,residual in residuals.items()}
    return {"populations": populations, "bilateral_minus_sum_of_unilateral_max_abs_state": nonlinear,
            "matched_upstream_magnitude_comparison": matched_upstream_comparison(table),
            "DNp15_per_body": table[table.stage.eq("DNp15")].to_dict(orient="records"),
            "washout_final_max_abs_state": max(float(np.max(abs(trace[-1]))) for trace in traces.values()),
            "zero_condition_max_abs_state": float(np.max(abs(traces["zero"])))}


def compare_external(summary: dict, spec: dict) -> dict:
    out = {}
    for population in POPULATIONS:
        model = summary["populations"][population]["mean_di"]
        observed = spec["external_target"]["observed"][population]
        lo,hi = observed["descriptive_roi_t95_interval"]
        out[population] = {"model_mean_di": model, "published_mean_di": observed["mean_di"],
            "model_minus_published": model-observed["mean_di"],
            "inside_descriptive_roi_interval": bool(lo<=model<=hi),
            "same_mean_preference_sign": bool(np.sign(model)==np.sign(observed["mean_di"]))}
    return out


def figure(spec: dict, summaries: dict, full_table: pd.DataFrame, filename: Path) -> None:
    """One shared-scale response figure; no per-condition output normalization."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16,11), layout="constrained")
    grid = fig.add_gridspec(2,2, height_ratios=[1,1.15])
    ax = fig.add_subplot(grid[0,0])
    motion = np.asarray([spec["stimuli"][c] for c in CONDITIONS[:-1]]).T
    ax.imshow(motion, vmin=-1,vmax=1,cmap="RdBu_r", aspect="auto")
    for eye in range(2):
        for k in range(8):
            ax.text(k,eye,{1:"F",-1:"B",0:"still"}[motion[eye,k]],ha="center",va="center",fontweight="bold",color="white" if motion[eye,k]!=0 else "black")
    labels = ["Yaw-like\nL F / R B", "Yaw-like\nL B / R F", "Translation-like\nF / F", "Translation-like\nB / B",
              "Left only\nF / still", "Left only\nB / still", "Right only\nstill / F", "Right only\nstill / B"]
    ax.set(xticks=range(8),xticklabels=labels,yticks=[0,1],yticklabels=["Left eye","Right eye"],title="A  Published stimulus-space classes (controlled input)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("2 s still → 2 s motion → 2 s washout; F = front-to-back, B = reverse\nHS drive = eye value; H2 drive = its negative; no T4/T5 or workbook coordinates")

    ax = fig.add_subplot(grid[0,1])
    rows = []
    rowlabels = []
    for population in POPULATIONS:
        for side in ["L","R"]:
            selected = full_table[full_table.stage.eq(population) & full_table.somaSide.eq(side)]
            rows.append([selected[c].mean() for c in CONDITIONS[:-1]])
            rowlabels.append(f"{population} {side}  (n={len(selected)})")
    values = np.asarray(rows)
    limit = max(float(np.max(abs(values))),1e-12)
    im = ax.imshow(values,vmin=-limit,vmax=limit,cmap="RdBu_r",aspect="auto")
    ax.set(yticks=range(12),yticklabels=rowlabels,xticks=range(8),xticklabels=["Yaw L F","Yaw R F","F / F","B / B","L F","L B","R F","R B"],title="B  Full circuit: signed late response by stage / side")
    for k in range(1,6):
        ax.axhline(2*k-.5,color="white",lw=2)
    fig.colorbar(im,ax=ax,label="Dimensionless state, mean over 3.5–4 s (one shared scale)",shrink=.8)

    ax = fig.add_subplot(grid[1,0])
    x = np.arange(6)
    actual = [summaries["full"]["populations"][p]["mean_di"] for p in POPULATIONS]
    obs = [spec["external_target"]["observed"][p]["mean_di"] for p in POPULATIONS]
    ci = np.asarray([spec["external_target"]["observed"][p]["descriptive_roi_t95_interval"] for p in POPULATIONS]).T
    ax.bar(x-.18,obs,.35,color="#5d798e",label="Published calcium: ROI mean ± descriptive t95")
    ax.errorbar(x-.18,obs,yerr=[np.asarray(obs)-ci[0],ci[1]-np.asarray(obs)],fmt="none",color="black",capsize=3)
    ax.bar(x+.18,actual,.35,color="#d28b3d",label="Frozen model: per-body DI mean")
    ax.axhline(0,color="#555555",lw=.8)
    ax.set(xticks=x,xticklabels=POPULATIONS,ylim=(-1.15,.7),ylabel="DI = (OF translation − OF yaw) / (|OF translation| + |OF yaw|)",title="C  External physiology target vs model (no fit)")
    ax.legend(loc="lower left",fontsize=8)
    ax.text(.02,.97,"More negative = rotational preference. Different measurement scales;\nROI intervals are descriptive, not calibrated acceptance thresholds.",transform=ax.transAxes,va="top",fontsize=9)

    sub = grid[1,1].subgridspec(2,1,height_ratios=[1,1.1])
    ax = fig.add_subplot(sub[0])
    for side,color in [("L","#237c8a"),("R","#9a3b55")]:
        yaw = []; translation = []
        for control in CONTROLS:
            row = next(r for r in summaries[control]["DNp15_per_body"] if r["somaSide"]==side)
            yaw.append(row["of_yaw"]); translation.append(row["of_translation"])
        ax.plot(range(len(CONTROLS)),yaw,"o-",color=color,label=f"DN {side}: yaw OF")
        ax.plot(range(len(CONTROLS)),translation,"s--",color=color,label=f"DN {side}: translation OF")
    ax.set(xticks=range(len(CONTROLS)),xticklabels=[],ylabel="Signed OF contrast\n(state proxy)",title="D  Mechanistic controls: DNp15 contrast and selectivity")
    ax.axhline(0,color="#777777",lw=.8)
    ax.legend(ncol=2,fontsize=8)
    ax = fig.add_subplot(sub[1])
    for side,color in [("L","#237c8a"),("R","#9a3b55")]:
        indices = [next(r["di"] for r in summaries[c]["DNp15_per_body"] if r["somaSide"]==side) for c in CONTROLS]
        ax.plot(range(len(CONTROLS)),indices,"o-",color=color,label=f"DN {side}")
    ax.axhline(spec["external_target"]["observed"]["DNp15"]["mean_di"],color="black",ls=":",label="Published DN mean")
    ax.axhline(0,color="#777777",lw=.8)
    ax.set(xticks=range(len(CONTROLS)),xticklabels=SHORT_CONTROLS,ylim=(-1.05,.5),ylabel="DNp15 DI")
    ax.tick_params(axis="x",rotation=22,labelsize=8)
    ax.legend(ncol=3,fontsize=8,loc="lower left")
    fig.suptitle("EXP-004 · Controlled HS/H2 → MaleCNS recurrent circuit → DNp15\nNeural physiology validation; stable default dynamics, no parameter search",fontsize=16)
    filename.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(filename,dpi=150,bbox_inches="tight",pad_inches=.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT / "results/exp004_physiology")
    parser.add_argument("--write-record",action="store_true",help="create the new experiment record after preflight; never overwrite")
    parser.add_argument("--check-record",action="store_true",help="require results to match the committed record")
    parser.add_argument("--promote-figure",action="store_true",help="copy this compact new figure into figures/ as selected evidence")
    args = parser.parse_args()
    if args.write_record and args.check_record:
        parser.error("choose write or check record, not both")
    spec,graph = load_frozen()
    params = PhysiologyParameters(**spec["parameters"])
    networks = {c: BinocularNetwork.compile(graph,spec["electrical_pairs"],params,c) for c in CONTROLS}
    preflight = {c: n.preflight() for c,n in networks.items()}
    full_operator = networks["full"].chemical.toarray()
    fine_networks = {c: BinocularNetwork.compile(graph,spec["electrical_pairs"],replace(params,dt_ms=params.dt_ms/2),c) for c in CONTROLS}
    for control,network in networks.items():
        operator = network.chemical.toarray()
        survivors = operator!=0
        if not np.array_equal(operator[survivors],full_operator[survivors]):
            raise RuntimeError("ablation altered surviving chemical weights")
        preflight[control]["surviving_chemical_weights_identical"] = True
        preflight[control]["half_dt_preflight"] = fine_networks[control].preflight()
    if not all(p["passed"] for p in preflight.values()):
        raise RuntimeError("preflight failed; no biological response record produced")
    if not all(p["half_dt_preflight"]["passed"] for p in preflight.values()):
        raise RuntimeError("half-dt preflight failed")
    args.output.mkdir(parents=True,exist_ok=True)
    summaries = {}; tables = []; digest = hashlib.sha256(); full_table = None
    started = walltime.perf_counter()
    for control,network in networks.items():
        traces = {}; masks = set()
        fine_traces = {}; convergence = {}
        for condition,motion in spec["stimuli"].items():
            time,drive = controlled_drive(network,motion)
            trace = network.simulate(drive)
            if not np.array_equal(trace,network.simulate(drive)):
                raise RuntimeError("nondeterministic execution")
            traces[condition] = trace
            fine_time,fine_drive = controlled_drive(fine_networks[control],motion)
            fine = fine_networks[control].simulate(fine_drive)
            if not np.array_equal(fine,fine_networks[control].simulate(fine_drive)):
                raise RuntimeError("half-dt execution is nondeterministic")
            fine_traces[condition] = fine
            convergence[condition] = float(np.max(abs(trace-fine[::2])))
            digest.update((control+"/"+condition).encode("ascii"))
            digest.update(np.asarray(trace,dtype="<f8").tobytes())
            masks.update(row.tobytes() for row in np.unique(trace>0,axis=0))
        preflight[control]["encountered_effective_transition_rho_max"] = max(
            spectral_radius(network.transition(np.frombuffer(m,dtype=bool))) for m in masks)
        preflight[control]["encountered_rectifier_masks"] = len(masks)
        if preflight[control]["encountered_effective_transition_rho_max"] >= 1:
            raise RuntimeError("trajectory Jacobian stability failed")
        table = response_table(network,time,traces,spec["external_target"]["response_window_ms"])
        fine_table = response_table(fine_networks[control],fine_time,fine_traces,spec["external_target"]["response_window_ms"])
        response_columns = list(CONDITIONS)+["of_yaw","of_translation","di","magnitude_di"]
        differences = abs(table[response_columns].to_numpy()-fine_table[response_columns].to_numpy())
        if not np.array_equal(np.isnan(table[response_columns]),np.isnan(fine_table[response_columns])):
            raise RuntimeError("dt halving changed which measurements are defined")
        finite_differences = differences[np.isfinite(differences)]
        late_error = float(np.max(finite_differences)) if len(finite_differences) else 0.
        convergence_passed = max(convergence.values())<=.01 and late_error<=1e-6
        preflight[control]["timestep_convergence"] = {"passed": bool(convergence_passed),
            "dt_ms": params.dt_ms,"half_dt_ms": params.dt_ms/2,
            "max_abs_trace_difference_by_condition": convergence,
            "max_abs_late_response_or_index_difference": late_error}
        if not convergence_passed:
            raise RuntimeError("timestep convergence failed; no experiment record produced")
        table.insert(0,"control",control); tables.append(table)
        if control=="full": full_table = table
        summaries[control] = summarize(table,traces,time,spec["external_target"]["response_window_ms"])
        np.savez_compressed(args.output / f"traces_{control}.npz",time_ms=time,body_ids=network.nodes.bodyId.to_numpy(),**traces)
        print(f"Completed {control}: stability, deterministic repeats and dt-halving passed",flush=True)
    combined = pd.concat(tables,ignore_index=True)
    combined.to_csv(args.output / "neuron_responses.csv",index=False,float_format="%.17g")
    external = compare_external(summaries["full"],spec)
    full = summaries["full"]
    target = evaluate_transformation(full_table,epsilon=spec["observation_contract"]["numerical_epsilon"])
    nonlinear = max(full["bilateral_minus_sum_of_unilateral_max_abs_state"]["DNp15"].values())
    target["DNp15_bilateral_superposition_residual_max"] = nonlinear
    feedforward = summaries["feedforward_chemical_only"]
    mechanism_checks = []
    for row in target["per_side"]:
        ff = next(r for r in feedforward["DNp15_per_body"] if r["bodyId"]==row["bodyId"])
        ff_magnitude = next(r for r in feedforward["matched_upstream_magnitude_comparison"] if r["bodyId"]==row["bodyId"])
        row["yaw_magnitude_full_over_feedforward"] = row["DNp15_yaw_of_magnitude"]/abs(ff["of_yaw"]) if ff["of_yaw"] else None
        row["translation_magnitude_full_over_feedforward"] = row["DNp15_translation_of_magnitude"]/abs(ff["of_translation"]) if ff["of_translation"] else None
        improves = all(row[key]+spec["observation_contract"]["numerical_epsilon"]<ff_magnitude[key] for key in ["DNp15_translation_yaw_ratio","DNp15_common_mode_yaw_ratio"])
        mechanism_checks.append({"bodyId": row["bodyId"],"full_reduces_both_ratios_vs_feedforward": bool(improves)})
    target["recurrent_mechanism_checks"] = mechanism_checks
    target["recurrent_enhancement_contract_passed"] = bool(target["limited_neural_transformation_gate"] and nonlinear>spec["observation_contract"]["numerical_epsilon"] and all(r["full_reduces_both_ratios_vs_feedforward"] for r in mechanism_checks))
    gate = target["recurrent_enhancement_contract_passed"]
    bips_control = spec["external_target"]["bIPS_GABA_receptor_control"]
    ablation_target = {"published_bIPS_control_mean_di": bips_control["bIPS control"]["mean_di"],
        "published_bIPS_Rdl_mean_di": bips_control["bIPS RdlFLPStop"]["mean_di"],
        "model_bIPS_control_mean_di": full["populations"]["bIPS"]["mean_di"],
        "model_bIPS_GABA_off_mean_di": summaries["no_bIPS_GABA_input"]["populations"]["bIPS"]["mean_di"]}
    metrics = finite_json({"specification_sha256": sha256_file(SPEC_PATH),
        "parameters": spec["parameters"],"preflight": preflight,"responses": summaries,
        "external_comparison": external,"trace_arrays_sha256": digest.hexdigest(),
        "deterministic_repeat_all_conditions": True,
        "qualitative_target": target,"published_ablation_comparison": ablation_target,
        "environment": {"python": platform.python_version(),"numpy": np.__version__,"scipy": scipy.__version__,"pandas": pd.__version__}})
    (args.output / "metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n",encoding="utf8")
    (args.output / "runtime.json").write_text(json.dumps({"simulation_analysis_seconds": walltime.perf_counter()-started})+"\n",encoding="utf8")
    figure(spec,summaries,full_table,args.output / "figures/EXP-004-binocular-physiology.png")
    if args.check_record:
        record = json.loads(RECORD_PATH.read_text(encoding="utf8"))
        if record["metrics"] != metrics:
            raise ValueError("results differ from committed experiment record")
    if args.write_record:
        if RECORD_PATH.exists(): raise FileExistsError("refusing to overwrite an experiment record")
        record = {"experiment_id": spec["experiment_id"],"title": "Controlled binocular HS/H2 recurrent network physiology validation",
            "status": "completed_stable_partial_match" if gate else "completed_stable_negative",
            "question": "Does the MaleCNS-constrained recurrent network enhance rotational/asymmetric selectivity at DNp15 relative to HS/H2 while suppressing symmetric translation-like responses?",
            "parent_commit": "705f266", "preserved_baselines": ["EXP-001","EXP-002","EXP-003","EXP-004 provisional preflight failure"],
            "model_scope": {"direct_controlled_HS_H2_input": True,"T4_T5_reconnected": False,"body_simulation": False,"parameter_search": False,"DNa02_included": False},
            "anatomy": spec["anatomy"],"electrical_pairs": spec["electrical_pairs"],
            "equations": "tau dx/dt = -x + Wchem max(x,0) + Gdiff x + u(t); Wchem[i,j]=chemical_gain * sign(j) * synapses(j,i) / sum_k synapses(k,i) in the full retained graph; Gdiff x=sum_pairs g*(other-self); explicit Euler; pre-update samples",
            "external_target": spec["external_target"],"controls": spec["controls"],
            "metrics": metrics,
            "conclusion_boundary": "The target gate tests a limited modeled transformation, not biological voltage, behavior, full physiological hierarchy or emergence from chemical connectivity alone. Report every population discrepancy and ablation; do not interpret a failed default model as falsifying binocular biology.",
            "outputs": {"figure": "figures/EXP-004-binocular-physiology.png","generated": "results/exp004_physiology/{metrics.json,neuron_responses.csv,traces_<control>.npz,figures/EXP-004-binocular-physiology.png}","bulk_results_ignored": True},
            "reproduce": "python scripts/prepare_exp004_physiology.py --download-sources --check-specification; python scripts/run_exp004_physiology.py --check-record"}
        RECORD_PATH.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf8")
    if args.promote_figure:
        destination = ROOT / "figures/EXP-004-binocular-physiology.png"
        if destination.exists(): raise FileExistsError("refusing to overwrite promoted evidence")
        destination.write_bytes((args.output / "figures/EXP-004-binocular-physiology.png").read_bytes())
    print(json.dumps({"external_comparison": external,"qualitative_target": metrics["qualitative_target"],
          "preflight_passed": all(p["passed"] for p in preflight.values()),"trace_arrays_sha256": digest.hexdigest()},indent=2))


if __name__=="__main__":
    main()
