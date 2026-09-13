"""Reproduce frozen visual DN drive, then test identified neck output open-loop."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_exp006_neck import SPEC
from scripts.run_exp005_sensory import load_specification, relative_error, upstream
from src.binocular_physiology import BinocularNetwork, PhysiologyParameters
from src.malecns_io import ConnectomeGraph
from src.neck_motor import MotorParameters, NeckMotorPathway, head_torque
from src.provenance import sha256_file, validate_materialization
from src.sensory_chain import SensoryParameters, VisualProjection, central_drive

RECORD = SPEC.with_name("record.json")
PROMOTED = ROOT / "figures/EXP-006-DNp15-neck-motor.png"


def load_motor_specification():
    spec = json.loads(SPEC.read_text(encoding="utf8"))
    for entry in spec["anatomy_files"]+spec["source_files"]:
        path = ROOT / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("motor anatomy/source differs from frozen specification")
    for entry in spec["baseline_files"]:
        # Canonical source regressions are checked against the historical record;
        # raw file sizes/digests are the original Windows freeze provenance only.
        if not entry["path"].endswith(".py") and sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("EXP-005 record/specification changed")
    old_record = json.loads((ROOT / "experiments/EXP-005-sensory-to-DNp15/record.json").read_text())
    for path, digest in old_record["metrics"]["implementation_LF_sha256"].items():
        if hashlib.sha256((ROOT / path).read_text(encoding="utf8").encode()).hexdigest() != digest:
            raise ValueError("frozen upstream implementation changed")
    folder = (ROOT / spec["anatomy_files"][0]["path"]).parent
    graph = ConnectomeGraph(pd.read_parquet(folder / "nodes.parquet"), pd.read_parquet(folder / "edges.parquet"))
    validate_materialization(spec["anatomy"], graph.nodes, graph.edges)
    return spec, graph


def sensory_descending(spec):
    """Fresh evaluation of unchanged EXP-005 kernels; no use of body state."""
    visual_spec, central_spec, visual_graph, central_graph = load_specification()
    p = SensoryParameters(**visual_spec["sensory_parameters"])
    cp = PhysiologyParameters(**visual_spec["central_parameters"])
    stimuli = {c: visual_spec["stimuli"][c] for c in spec["stimuli"]}
    runs, preflight = {}, {}
    for label, factor in [("recorded_dt", 1), ("half_dt", 2)]:
        vp = replace(p, dt_ms=p.dt_ms/factor)
        network = BinocularNetwork.compile(central_graph, central_spec["electrical_pairs"], replace(cp, dt_ms=cp.dt_ms/factor))
        visual = VisualProjection.compile(visual_graph, vp)
        checks = {"central": network.preflight(), "visual": visual.preflight()}
        if not all(v["passed"] for v in checks.values()):
            raise RuntimeError("frozen upstream preflight failed")
        time, stages = upstream(visual, vp, stimuli)
        descending = {}
        indexes = [int(np.flatnonzero(network.nodes.bodyId.to_numpy() == i)[0]) for i in [11215, 12069]]
        for condition in stimuli:
            drive = central_drive(visual.input_ids, stages[condition]["drive"], network.nodes.bodyId)
            states = network.simulate(drive)
            if not np.array_equal(states, network.simulate(drive)):
                raise RuntimeError("central nondeterminism")
            descending[condition] = states[:,indexes]
            if factor == 1:
                # Compare individual stage/body late responses against frozen evidence,
                # not just the bilateral sign. Saved bulk outputs are not a dependency.
                record = json.loads((ROOT / "experiments/EXP-005-sensory-to-DNp15/record.json").read_text())
                late = (time >= 3500) & (time < 4000)
                expected = record["metrics"]["responses"]["full"]["DNp15_per_body"]
                for j, body in enumerate([11215,12069]):
                    target = next(r[condition] for r in expected if r["bodyId"] == body)
                    if not np.isclose(descending[condition][late,j].mean(), target, atol=1e-15, rtol=0):
                        raise RuntimeError("fresh DN output differs from EXP-005")
        runs[label] = (time, descending)
        preflight[label] = checks
    return runs, preflight


def open_loop_body(torque, neural_dt_ms, body_spec, *, timestep_s=None):
    """Raw FlyGym/MuJoCo motor actuator, no gait/feedback/behavior controller.

    Neural sample k is held during its time interval, never interpolated using
    a future sample. The efficient execution path steps physics directly; FlyGym
    supplies the body, arena, actuator construction and initial state.
    """
    from flygym import Fly, SingleFlySimulation
    from flygym.arena import Tethered
    for package in ["flygym", "mujoco"]:
        if version(package) != body_spec[package]:
            raise RuntimeError("body-version freeze differs")
    dt = body_spec["timestep_s"] if timestep_s is None else timestep_s
    ratio = neural_dt_ms/1000/dt
    repeat = round(ratio)
    if repeat < 1 or not np.isclose(repeat, ratio):
        raise ValueError("body dt must subdivide neural clock")
    torque = np.asarray(torque, dtype=float)
    if torque.ndim != 1 or len(torque) < 1 or not np.isfinite(torque).all():
        raise ValueError("invalid open-loop torque")
    fly = Fly(name="NeckOpenLoop", actuated_joints=[body_spec["actuated_joint"]],
        monitored_joints=[body_spec["actuated_joint"]], control="motor", actuator_gain=None,
        joint_stiffness=body_spec["joint_stiffness"], joint_damping=body_spec["joint_damping"],
        neck_stiffness=body_spec["neck_stiffness"],
        non_actuated_joint_stiffness=body_spec["non_actuated_joint_stiffness"],
        non_actuated_joint_damping=body_spec["non_actuated_joint_damping"],
        actuator_forcerange=body_spec["actuator_forcerange"],
        spawn_orientation=tuple(body_spec["spawn_orientation"]),
        enable_vision=False, enable_adhesion=False, head_stabilization_model=None)
    simulation = SingleFlySimulation(fly=fly, arena=Tethered(), cameras=[], timestep=dt)
    try:
        simulation.reset(seed=body_spec["seed"])
        physics = simulation.physics
        physics.model.opt.gravity[:] = body_spec["gravity_mm_s2"]
        joint = fly.model.find("joint", body_spec["actuated_joint"])
        head = physics.bind(fly.model.find("body", "Head"))
        thorax = physics.bind(fly.model.find("body", "Thorax"))
        hinge = physics.bind(joint)
        actuator = physics.bind(fly.actuators)
        axis = np.array(hinge.xaxis)
        if not np.allclose(axis, [0,0,1], atol=1e-12):
            raise RuntimeError("native head actuator is not the declared physical Z hinge")
        # Finite-angle independent physical-coordinate test, then restore state.
        initial_q = np.array(hinge.qpos)
        hinge.qpos = initial_q+.001
        physics.forward()
        direction = np.array(head.xmat).reshape(3,3)[:,0]
        finite_yaw = float(np.arctan2(direction[1], direction[0]))
        hinge.qpos = initial_q
        physics.forward()
        if not np.isclose(finite_yaw, .001, atol=1e-12):
            raise RuntimeError("joint label does not match measured physical azimuth")
        output = np.empty((len(torque),3))
        for k, command in enumerate(torque):
            # Pre-update output aligned to neural t=k*dt.
            head_rotation = np.array(head.xmat).reshape(3,3)
            thorax_rotation = np.array(thorax.xmat).reshape(3,3)
            relative = thorax_rotation.T @ head_rotation
            output[k] = [np.arctan2(relative[1,0],relative[0,0]),
                         float(hinge.qpos[0]), np.arctan2(thorax_rotation[1,0],thorax_rotation[0,0])]
            actuator.ctrl = [command]
            for _ in range(repeat):
                physics.step()
        if not np.isfinite(output).all():
            raise FloatingPointError("nonfinite physical body result")
        return output, {"axis_at_initial_pose": axis.tolist(), "finite_angle_head_azimuth_rad": finite_yaw,
                        "actuator_gear": physics.model.actuator_gear[actuator.element_id].tolist(),
                        "timestep_s": dt, "neural_sample_hold_repeat": repeat,
                        "no_controller_or_sensory_feedback": True}
    finally:
        simulation.close()


def make_figure(spec, time, dn, motor, commands, body, baseline, destination):
    fig, axes = plt.subplots(3,2,figsize=(13,11),layout="constrained")
    ax = axes[0,0]
    ax.axis("off")
    ax.set_title("A  Primary-ID-matched direct neck pathway")
    ax.text(.02,.95,"DNp15 L 12069  →  ADNM2 813696 / MANC 17653\n"
        "76 chemical synapses     CvNA2 axon LEFT → TH2\n\n"
        "DNp15 R 11215  →  ADNM2 804343 / MANC 14549\n"
        "40 chemical synapses     CvNA2 axon RIGHT → TH2\n\n"
        "ADNM1 / CvNA1 → TH1: observed, not actuated\n"
        "150 and 82 DN synapses; two 1-synapse MN edges\n"
        "retained anatomically, effective action unresolved.\n\n"
        "Motor axon side ≠ motor soma side.\n"
        "Published CvNA2 effects depend on initial head pose.", va="top", fontsize=10)
    colors = {"yaw_L_F":"#317ca4", "yaw_R_F":"#bd5268"}
    for condition in spec["stimuli"]:
        color = colors[condition]
        for j, side in enumerate(["R","L"]):
            axes[0,1].plot(time/1000,dn[condition][:,j],color=color,ls="-" if side=="L" else "--",label=f"{condition} DN {side}")
        for row in spec["identity"]:
            if row["paper_cell"] == "CvNA2":
                j = [r["bodyId"] for r in spec["identity"]].index(row["bodyId"])
                axes[1,0].plot(time/1000,motor["full"][condition][:,j],color=color,
                    ls="-" if row["axon_side"]=="L" else "--",label=f"{condition} axon {row['axon_side']}")
        axes[1,1].plot(time/1000,commands["full"][condition],color=color,label=condition)
        axes[2,0].plot(time/1000,np.rad2deg(body[condition][:,0]-baseline[:,0]),color=color,label=condition)
    for ax in axes.flat[1:]:
        ax.axhline(0,color="gray",lw=.5)
        if ax is not axes[2,1]:
            ax.axvspan(2,4,alpha=.08,color="gray")
            ax.set_xlabel("Time (s); frozen images move from 2–4 s")
            ax.legend(fontsize=7,ncol=2)
    axes[0,1].set(title="B  Unchanged EXP-005 descending states",ylabel="Signed dimensionless activity")
    axes[1,0].set(title="C  Anatomical DN recruitment → CvNA2",ylabel="Dimensionless motor proxy (not spikes)")
    axes[1,1].set(title="D  Fixed provisional TH2 yaw-component bridge",ylabel="Generalized torque (native = nN m)")
    axes[2,0].plot(time/1000,np.zeros_like(time),color="black",ls=":",label="DN or motor disconnected")
    axes[2,0].set(title="E  Open-loop physical head azimuth vs zero command",ylabel="Head relative to thorax (degrees)")
    ax=axes[2,1]
    controls=spec["controls"]
    x=np.arange(len(controls))
    late=(time>=3500)&(time<4000)
    for j,condition in enumerate(spec["stimuli"]):
        ax.bar(x+(j-.5)*.36,[commands[c][condition][late].mean() for c in controls],.36,color=colors[condition],label=condition)
    ax.set(title="F  Causal interventions, unchanged surviving weights",ylabel="Mean torque, 3.5–4 s (native)",xticks=x,
        xticklabels=["Full","DNs off","DN L off","DN R off","Motor off"])
    ax.legend(fontsize=7)
    fig.suptitle("EXP-006  Frozen visual → DNp15 → identified neck MN → provisional open-loop head torque\n"
        "No walking steering, calibrated muscle force, pose-dependent reflex or sensory feedback",fontsize=13)
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/exp006_neck")
    parser.add_argument("--write-record",action="store_true")
    parser.add_argument("--check-record",action="store_true")
    parser.add_argument("--promote-figure",action="store_true")
    args=parser.parse_args()
    if args.write_record and (RECORD.exists() or args.check_record):
        parser.error("never overwrite frozen evidence")
    if args.promote_figure and PROMOTED.exists():
        raise FileExistsError("promoted evidence already exists")
    spec,graph=load_motor_specification()
    p=MotorParameters(**spec["parameters"])
    paths={c:NeckMotorPathway.compile(graph,spec["incoming_totals"],p,c) for c in spec["controls"]}
    preflight={c:network.preflight() for c,network in paths.items()}
    if not all(v["passed"] for v in preflight.values()):
        raise RuntimeError("motor stability preflight failed")
    for c,path in paths.items():
        selected=path.weights.toarray()!=0
        if not np.array_equal(path.weights.toarray()[selected],paths["full"].weights.toarray()[selected]):
            raise RuntimeError("control renormalized surviving connections")
        preflight[c]["surviving_weights_unchanged"]=True
    print("Motor graph, axon crosswalk and stable dynamics frozen; evaluating unchanged sensory baseline",flush=True)
    runs,upstream_preflight=sensory_descending(spec)
    time,dn=runs["recorded_dt"]
    _,fine_dn=runs["half_dt"]
    motor,commands,convergence={},{},{}
    for c,path in paths.items():
        fine_path=NeckMotorPathway.compile(graph,spec["incoming_totals"],replace(p,dt_ms=p.dt_ms/2),c)
        preflight[c]["half_dt"]=fine_path.preflight()
        motor[c],commands[c],convergence[c]={},{},{}
        for condition in spec["stimuli"]:
            m=path.simulate(dn[condition])
            fine_m=fine_path.simulate(fine_dn[condition])
            if not np.array_equal(m,path.simulate(dn[condition])):
                raise RuntimeError("motor nondeterminism")
            torque=head_torque(m,path.motor_ids,spec["identity"],p,disconnected=c=="motor_disconnected")
            error=relative_error(m,fine_m[::2])
            if error>spec["predeclared_checks"]["motor_dt_relative_error_max"]:
                raise RuntimeError("motor timestep convergence failed")
            if c=="DN_disconnected" and (np.any(m) or np.any(torque)):
                raise RuntimeError("disconnected pathway not zero")
            motor[c][condition],commands[c][condition]=m,torque
            convergence[c][condition]=error
    # Mechanical baseline is a matched zero-command test, not biological feedback.
    print("Motor numerical/causal gate passed. Testing fixed raw neck actuator open-loop",flush=True)
    baseline,axis_check=open_loop_body(np.zeros(len(time)),p.dt_ms,spec["body_parameters"])
    body,body_checks={},{}
    for condition in spec["stimuli"]:
        output,check=open_loop_body(commands["full"][condition],p.dt_ms,spec["body_parameters"])
        repeat,_=open_loop_body(commands["full"][condition],p.dt_ms,spec["body_parameters"])
        fine,_=open_loop_body(commands["full"][condition],p.dt_ms,spec["body_parameters"],timestep_s=spec["body_parameters"]["timestep_s"]/2)
        fine_zero,_=open_loop_body(np.zeros(len(time)),p.dt_ms,spec["body_parameters"],timestep_s=spec["body_parameters"]["timestep_s"]/2)
        error=relative_error((output-baseline)[:,:1],(fine-fine_zero)[:,:1])
        if not np.array_equal(output,repeat) or error>spec["predeclared_checks"]["body_dt_relative_error_max"]:
            raise RuntimeError("body deterministic/convergence gate failed")
        body[condition]=output
        check.update({"deterministic_repeat":True,"half_dt_baseline_subtracted_yaw_relative_error":error,
                      "max_abs_thorax_yaw_rad":float(np.max(abs(output[:,2])))})
        body_checks[condition]=check
        print(f"Open-loop physical run and half-step check completed: {condition}",flush=True)
    disconnected,_=open_loop_body(commands["DN_disconnected"][spec["stimuli"][0]],p.dt_ms,spec["body_parameters"])
    if not np.array_equal(disconnected,baseline):
        raise RuntimeError("physical disconnection did not match zero-command body")
    # Independent unit-sign physical calibration, not fitted to a sensory condition.
    pulse=np.where((time>=2000)&(time<2300),1e-4,0.)
    positive,_=open_loop_body(pulse,p.dt_ms,spec["body_parameters"])
    negative,_=open_loop_body(-pulse,p.dt_ms,spec["body_parameters"])
    pulse_window=(time>=2000)&(time<2300)
    pulse_values=[float((trace-baseline)[pulse_window,0].mean()) for trace in [positive,negative]]
    physical_sign=pulse_values[0]>0 and pulse_values[1]<0
    if not physical_sign:
        raise RuntimeError("raw physical hinge torque sign not supported by the body test")
    args.output.mkdir(parents=True,exist_ok=True)
    saved={"time_ms":time,"DN_body_ids":paths["full"].source_ids,"motor_body_ids":paths["full"].motor_ids,
           "body_zero_command":baseline,"body_DN_disconnected":disconnected}
    digest=hashlib.sha256()
    late=(time>=3500)&(time<4000)
    responses={}
    for c in spec["controls"]:
        responses[c]={}
        for condition in spec["stimuli"]:
            saved[f"{c}_{condition}_motor"]=motor[c][condition]
            saved[f"{c}_{condition}_torque"]=commands[c][condition]
            responses[c][condition]={"motor_late":motor[c][condition][late].mean(axis=0).tolist(),
                "mean_torque_native":float(commands[c][condition][late].mean()),
                "max_abs_torque_native":float(np.max(abs(commands[c][condition])))}
    body_metrics={}
    for condition in spec["stimuli"]:
        saved[f"{condition}_DN"]=dn[condition]
        saved[f"{condition}_body"]=body[condition]
        delta=body[condition][:,0]-baseline[:,0]
        body_metrics[condition]={"late_mean_head_yaw_delta_degrees":float(np.rad2deg(delta[late].mean())),
            "max_abs_head_yaw_delta_degrees":float(np.rad2deg(np.max(abs(delta)))),
            "final_head_yaw_delta_degrees":float(np.rad2deg(delta[-1])),
            "mean_torque_and_late_head_yaw_same_sign":bool(np.sign(delta[late].mean())==np.sign(commands["full"][condition][late].mean()))}
    for key in sorted(saved):
        array=np.ascontiguousarray(saved[key],dtype="<f8")
        digest.update(key.encode());digest.update(array.tobytes())
    np.savez_compressed(args.output/"stage_traces.npz",**saved)
    figure=args.output/"DNp15_neck_open_loop.png"
    make_figure(spec,time,dn,motor,commands,body,baseline,figure)
    if args.promote_figure:
        PROMOTED.write_bytes(figure.read_bytes())
    metrics={"specification_sha256":sha256_file(SPEC),"preflight":preflight,
        "frozen_upstream_preflight":upstream_preflight,"fresh_EXP005_DN_late_responses_match":True,
        "motor_timestep_relative_errors":convergence,"body_checks":body_checks,"axis_check":axis_check,
        "deterministic_repeats":True,"disconnected_physical_matches_zero_command":True,
        "physical_sign_pulse_300ms_mean_yaw_delta_rad":pulse_values,
        "physical_sign_check_passed":physical_sign,"responses":responses,"body_response":body_metrics,
        "trace_arrays_sha256":digest.hexdigest(),
        "implementation_LF_sha256":{p:hashlib.sha256((ROOT/p).read_text(encoding="utf8").encode()).hexdigest()
            for p in ["src/neck_motor.py","scripts/prepare_exp006_neck.py","scripts/run_exp006_neck.py"]}}
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n",encoding="utf8")
    record={"experiment_id":spec["experiment_id"],"status":"completed_provisional_open_loop_head_motor_chain",
        "question":"Can frozen sensory-derived DNp15 activity recruit identified neck motor cells and produce an interpretable open-loop physical motor tendency?",
        "metrics":metrics,"scope":{"walking_steering":False,"whole_body_yaw":False,"sensory_closed_loop":False,
            "calibrated_motor_physiology":False,"pose_targeted_reflex":False,"EXP005_changed":False},
        "conclusion":"Anatomical DN-to-CvNA2/TH2 recruitment and a fixed uncalibrated yaw-component transduction produce causal open-loop head azimuth. This is a provisional physical motor boundary, not a validation of physiological recruitment, magnitude or posture-dependent kinematics. No closed-loop gate is passed.",
        "outputs":{"figure":PROMOTED.relative_to(ROOT).as_posix(),"bulk":"results/exp006_neck/stage_traces.npz",
            "metrics":"results/exp006_neck/metrics.json","report":"reports/EXP-006-DNp15-neck-motor.md"}}
    if args.write_record:
        if not PROMOTED.exists():
            raise RuntimeError("promote selected evidence before recording")
        RECORD.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf8",newline="\n")
    if args.check_record and json.loads(RECORD.read_text(encoding="utf8"))!=record:
        raise RuntimeError("reproduced scientific record differs")
    print(json.dumps({"responses":responses["full"],"body":body_metrics,"digest":digest.hexdigest()},indent=2))


if __name__=="__main__":
    main()
