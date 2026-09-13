"""Measure CvNA2 primary figure data and test, without replacing, frozen EXP-006.

No upstream simulation, new motor gains, pose fitting or body run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.malecns_io import ConnectomeGraph
from src.motor_boundary import extract_target, indistinguishable_led_family
from src.neck_motor import MotorParameters, NeckMotorPathway, head_torque
from src.provenance import sha256_file, validate_materialization

SPEC = ROOT / "experiments/EXP-007-CvNA2-motor-boundary/specification.json"
RECORD = SPEC.with_name("record.json")
FIGURE = ROOT / "figures/EXP-007-CvNA2-motor-boundary.png"


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def relative_error(actual, expected):
    scale = float(np.max(np.abs(expected)))
    error = float(np.max(np.abs(actual - expected)))
    return error / scale if scale else error


def frozen_replay(spec):
    old = read_json(ROOT / spec["baseline_files"][0]["path"])
    record = read_json(ROOT / spec["baseline_files"][1]["path"])
    for entry in old["anatomy_files"]:
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen motor anatomy changed")
    for path, digest in record["metrics"]["implementation_LF_sha256"].items():
        if hashlib.sha256((ROOT/path).read_text(encoding="utf8").encode()).hexdigest() != digest:
            raise ValueError("Frozen motor implementation changed")
    folder = ROOT / "data/malecns_exp006_neck"
    graph = ConnectomeGraph(pd.read_parquet(folder/"nodes.parquet"), pd.read_parquet(folder/"edges.parquet"))
    validate_materialization(old["anatomy"], graph.nodes, graph.edges)
    edges = {(int(r.pre_body), int(r.post_body), int(r.synapse_count)) for r in graph.edges.itertuples()}
    expected = {tuple(r) for r in spec["anatomical_evidence"]["direct_edges"] + spec["anatomical_evidence"]["unmodeled_motor_edges"]}
    if edges != expected:
        raise ValueError("Motor edge contract changed")
    p = MotorParameters(**old["parameters"])
    metrics, for_figure = {}, {}
    with np.load(ROOT/spec["baseline_files"][2]["path"], allow_pickle=False) as saved:
        digest = hashlib.sha256()
        for key in sorted(saved.files):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(saved[key], dtype="<f8").tobytes())
        if digest.hexdigest() != spec["frozen_trace_arrays_sha256"]:
            raise ValueError("Frozen stage-trace arrays changed")
        time = saved["time_ms"]
        if not np.allclose(np.diff(time), p.dt_ms, rtol=0, atol=1e-12):
            raise ValueError("Invalid frozen time axis")
        for control in old["controls"]:
            path = NeckMotorPathway.compile(graph, old["incoming_totals"], p, control)
            half = NeckMotorPathway.compile(graph, old["incoming_totals"], replace(p, dt_ms=p.dt_ms/2), control)
            if not np.array_equal(saved["DN_body_ids"], path.source_ids) or not np.array_equal(saved["motor_body_ids"], path.motor_ids):
                raise ValueError("Frozen body-ID axes changed")
            checks = {"preflight": path.preflight(), "half_dt_preflight": half.preflight(), "conditions": {}}
            for condition in old["stimuli"]:
                dn = saved[condition+"_DN"]
                motor = path.simulate(dn)
                torque = head_torque(motor, path.motor_ids, old["identity"], p, disconnected=control == "motor_disconnected")
                repeated = path.simulate(dn)
                finer = half.simulate(np.repeat(dn, 2, axis=0))[::2]
                replay_error = relative_error(motor, saved[f"{control}_{condition}_motor"])
                torque_error = relative_error(torque, saved[f"{control}_{condition}_torque"])
                dt_error = relative_error(finer, motor)
                if (max(replay_error, torque_error) > spec["calibration_contract"]["frozen_replay_relative_error_max"]
                    or dt_error > spec["calibration_contract"]["held_input_dt_relative_error_max"]
                    or not np.array_equal(motor, repeated)):
                    raise RuntimeError("Frozen replay / determinism / held-input timestep check failed")
                if control == "DN_disconnected" and (np.any(motor) or np.any(torque)):
                    raise RuntimeError("DN disconnection not causal")
                if control == "motor_disconnected" and np.any(torque):
                    raise RuntimeError("Motor disconnection failed")
                late = (time >= 3500) & (time < 4000)
                checks["conditions"][condition] = {"motor_replay_relative_error": replay_error,
                    "torque_replay_relative_error": torque_error, "held_input_half_dt_relative_error": dt_error,
                    "deterministic": True, "DN_late": dn[late].mean(axis=0).tolist(),
                    "motor_late": motor[late].mean(axis=0).tolist()}
                if control == "full":
                    for_figure[condition] = motor[:,[int(np.flatnonzero(path.motor_ids==body)[0]) for body in [813696,804343]]]
            if not checks["preflight"]["passed"] or not checks["half_dt_preflight"]["passed"]:
                raise RuntimeError("Frozen motor stability failed")
            metrics[control] = checks
        # Laterality check uses explicit axon crosswalk, never MN soma side.
        unit = np.eye(4)
        sign = head_torque(unit, saved["motor_body_ids"], old["identity"], p)
        side = {r["bodyId"]:r["axon_side"] for r in old["identity"] if r["paper_cell"]=="CvNA2"}
        for j, body in enumerate(saved["motor_body_ids"]):
            if int(body) in side and np.sign(sign[j]) != (1 if side[int(body)]=="R" else -1):
                raise RuntimeError("Frozen cervical-output torque mapping failed")
    return metrics, time, for_figure, old["identity"]


def make_figure(target, time, motor, path):
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])
    t = np.asarray(target["time_ms"])
    for i, (name, color) in enumerate(zip(["roll", "pitch", "yaw"], ["#7e57a3", "#cf6842", "#24778b"])):
        ax = fig.add_subplot(grid[0,2*i:2*i+2])
        c = target["components"][name]
        ax.axvspan(0,300,color="#e8edf1")
        ax.fill_between(t,c["absolute_low_deg_s"],c["absolute_high_deg_s"],color=color,alpha=.2)
        ax.plot(t,c["absolute_velocity_deg_s"],color=color,lw=2)
        ax.axhline(0,color=".6",lw=.7)
        ax.set(xlabel="Time from optogenetic onset (ms)", ylabel="Published component velocity (deg/s)",
               title=f"CvNA2: {name}", xlim=(-96,300), ylim=(-40,30))
        ax.spines[["top","right"]].set_visible(False)
    ax = fig.add_subplot(grid[1,:3]);ax.axis("off")
    ax.text(0,1,"Where calibration stops",fontsize=13,weight="bold",va="top")
    ax.text(0,.88,"DNp15 L 12069  →  CvNA2 813696 / CvN-left\nDNp15 R 11215  →  CvNA2 804343 / CvN-right\nMaleCNS contacts: 76 / 40; not functional gains.",fontsize=11,va="top",linespacing=1.6)
    ax.text(0,.56,"CvNA2 → TH2 muscle: primary anatomical identity.\nDirect CvNA2 activation → movement: measured.\nDNp15 → CvNA2 recruitment: not quantified.",fontsize=11,va="top",linespacing=1.6)
    ax.text(0,.27,"A posture-conditioned DN→head transform cannot\nbe identified from these grand means. No new\ntorque gain, pose fit or body simulation.",fontsize=11,va="top",linespacing=1.6,color="#9a3e32")
    ax = fig.add_subplot(grid[1,3:])
    for condition, color in [("yaw_L_F","#6355a6"),("yaw_R_F","#27947e")]:
        for j, style in enumerate(["-","--"]):
            ax.plot(time/1000,motor[condition][:,j],style,color=color,lw=1.6,
                    label=f"{condition}: CvN-{'left' if j==0 else 'right'}")
    ax.set(xlabel="Frozen sensory experiment time (s)",ylabel="Uncalibrated motor activity proxy",
           title="EXP-006 replay only; not optogenetic prediction")
    ax.legend(fontsize=8,loc="upper right"); ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("EXP-007: CvNA2 movement is measurable; DN recruitment remains unidentified\n"
                 "Gorko et al. Supplementary Fig. 10g-i: 677 validation trials / 11 flies; right-axon standardization",
                 fontsize=13)
    fig.supxlabel("Top: digitized grand means; shading is raster-extraction uncertainty, not biological confidence.\n"
                  "Lab frame, body pitched 40°; no calibrated DN-to-movement or posture-conditioned model is claimed.",fontsize=9)
    fig.savefig(path,dpi=160,metadata={"Software":"CNS EXP-007"});plt.close(fig)


def make_pixel_overlay(image, target, contract, path):
    """Independent visual check in native image coordinates, saved locally only."""
    fig, ax = plt.subplots(figsize=(12, 5.11), layout="constrained")
    ax.imshow(image)
    t = np.asarray(target["time_ms"])
    for name, panel in contract["panels"].items():
        x0,x1 = panel["x_pixels"]; y0,y1 = panel["y_pixels"]
        t0,t1 = panel["time_ms"]; v0,v1 = panel["velocity_deg_s"]
        x = x0 + (t-t0)*(x1-x0)/(t1-t0)
        y = y0 + (v0-np.asarray(target["components"][name]["absolute_velocity_deg_s"]))*(y1-y0)/(v0-v1)
        ax.scatter(x,y,s=7,color="#dc3030",zorder=3)
    ax.set(xlim=(0,image.width),ylim=(image.height,0));ax.axis("off")
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/exp007_motor_boundary")
    parser.add_argument("--write-record",action="store_true")
    parser.add_argument("--check-record",action="store_true")
    parser.add_argument("--promote-figure",action="store_true")
    args = parser.parse_args()
    if args.write_record and args.check_record:
        parser.error("Select record writing or checking, not both")
    spec = read_json(SPEC)
    for entry in spec["baseline_files"]+spec["source_files"]:
        if sha256_file(ROOT/entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen file changed: {entry['path']}")
    with Image.open(ROOT/spec["source_files"][2]["path"]) as image:
        target = extract_target(image, spec["measurement_contract"])
        if target != extract_target(image, spec["measurement_contract"]):
            raise RuntimeError("Target extraction is not deterministic")
        args.output.mkdir(parents=True,exist_ok=True)
        make_pixel_overlay(image,target,spec["measurement_contract"],args.output/"native_pixel_overlay.png")
    yaw = target["components"]["yaw"]["summary"]["delta_mean_extraction_envelope_deg_s"]
    pitch = target["components"]["pitch"]["summary"]["delta_mean_extraction_envelope_deg_s"]
    tendency = yaw[0] > 0 and pitch[1] < 0
    # Preserve a failed target check if present; it is never permission to retune.
    replay,time,motor,identity = frozen_replay(spec)
    led,dn = indistinguishable_led_family(target["components"]["yaw"]["absolute_velocity_deg_s"],
                                         np.ones(2),spec["calibration_contract"]["witness_coefficients"])
    witness = bool(np.array_equal(led[0],led[1]) and np.array_equal(led[1],led[2]) and not np.array_equal(dn[0],dn[1]))
    args.output.mkdir(parents=True,exist_ok=True)
    encoded = json.dumps(target,indent=2,allow_nan=False)+"\n"
    (args.output/"CvNA2_digitized_target.json").write_text(encoded,encoding="utf8",newline="\n")
    selected = args.output/"CvNA2_motor_boundary.png"
    make_figure(target,time,motor,selected)
    if args.promote_figure:
        FIGURE.write_bytes(selected.read_bytes())
    metrics = {"specification_sha256":sha256_file(SPEC),"target_sha256":hashlib.sha256(encoded.encode()).hexdigest(),
        "target_summary":{name:c["summary"] for name,c in target["components"].items()},
        "target_tendency_check_passed":tendency,"extraction_deterministic":True,
        "frozen_replay":replay,"cervical_output_side_contract_passed":True,
        "identical_LED_data_do_not_identify_DN_gain":witness,
        "implementation_LF_sha256":{path:hashlib.sha256((ROOT/path).read_text(encoding="utf8").encode()).hexdigest()
            for path in ["src/motor_boundary.py","scripts/prepare_exp007_target.py","scripts/run_exp007_boundary.py"]}}
    record = {"experiment_id":spec["experiment_id"],"status":"completed_movement_target_recruitment_unidentified",
        "question":spec["question"],"metrics":metrics,"identity":identity,
        "calibration_gate":{"new_bridge_evaluated":False,"DN_to_CvNA_gain_sign_latency_measured":False,
            "CvNA2_posture_coefficients_extracted":False,"CvNA2_movement_target_digitized":True,
            "CvN7_calibration_substituted":False,"ready_to_replace_EXP006_bridge":False},
        "scope":{"EXP001_through_EXP006_changed":False,"upstream_rerun":False,"body_rerun":False,
                 "parameter_fit_or_search":False,"closed_loop":False},
        "conclusion":"CvNA2-specific grand-mean movement can be quantified from primary figures. Neither DNp15 recruitment nor a posture-conditioned movement transform is identified. Anatomy provides direct contacts and cervical-output sides, not physiological gain/sign/latency. Frozen EXP-006 remains an uncalibrated torque approximation; no replacement or closed-loop gate is justified. This is a calibration boundary, not a negative test of biological function.",
        "minimum_next_evidence":["Identified DNp15 stimulation/recording with simultaneous CvNA1/CvNA2 voltage or spikes, independent units and cervical-output side, over controlled state/posture",
            "CvNA2-specific trial quaternions with pre-onset pose, fly IDs and training/validation split, plus authors' CvNA2 fit parameters or a held-out refit",
            "CvNA2-specific activation/firing-to-head transfer; LED expression gain alone cannot convert dimensionless DN activity into motor firing or torque"],
        "outputs":{"figure":FIGURE.relative_to(ROOT).as_posix(),"report":"reports/EXP-007-CvNA2-motor-boundary.md",
            "target":"results/exp007_motor_boundary/CvNA2_digitized_target.json","metrics":"results/exp007_motor_boundary/metrics.json"}}
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n",encoding="utf8",newline="\n")
    if args.write_record:
        if not FIGURE.exists() or FIGURE.read_bytes() != selected.read_bytes():
            raise RuntimeError("Promote this run's selected figure before recording")
        RECORD.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf8",newline="\n")
    if args.check_record and read_json(RECORD) != record:
        raise RuntimeError("Reproduced EXP-007 record differs")
    print(json.dumps({"target_summary":metrics["target_summary"],"target_tendency_check_passed":tendency,
                      "calibration_gate":record["calibration_gate"]},indent=2))


if __name__ == "__main__":
    main()
