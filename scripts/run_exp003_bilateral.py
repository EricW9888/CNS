"""Run the bilateral EXP-003 optic-flow hypothesis test."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_exp003 import _label_frame, _make_body, _save_video, _yaw_from_observation  # noqa: E402
from src.exp003 import ClosedLoopMotionScene, OnlineGradedCircuit, SteeringBridge  # noqa: E402
from src.exp003_corrected import CorrectedDownstreamMatrices  # noqa: E402
from src.exp003_bilateral import BilateralDownstreamReadout  # noqa: E402
from src.graded_model import GradedParameters  # noqa: E402
from src.malecns_io import build_graph, load_edges  # noqa: E402


CONDITIONS = {
    "A_left_forward_right_reverse": {"left_reverse": False, "right_reverse": True},
    "B_left_reverse_right_forward": {"left_reverse": True, "right_reverse": False},
}


def _load_graph(bundle: Path):
    return build_graph(
        pd.read_parquet(bundle / "nodes.parquet"),
        load_edges(bundle / "edges.parquet"),
    )


def _make_side_circuits(left_graph, right_graph, left_columns, right_columns, params):
    return (
        OnlineGradedCircuit(left_graph, motion_columns=left_columns, params=params),
        OnlineGradedCircuit(right_graph, motion_columns=right_columns, params=params),
    )


def _run_open_loop(
    *,
    condition_name: str,
    left_graph,
    right_graph,
    left_columns: list[dict],
    right_columns: list[dict],
    readout: BilateralDownstreamReadout,
    params: GradedParameters,
    bridge: SteeringBridge,
    duration_s: float,
    ablate: bool,
) -> pd.DataFrame:
    left_circuit, right_circuit = _make_side_circuits(
        left_graph, right_graph, left_columns, right_columns, params
    )
    spec = CONDITIONS[condition_name]
    left_scene = ClosedLoopMotionScene(reverse=spec["left_reverse"])
    right_scene = ClosedLoopMotionScene(reverse=spec["right_reverse"])
    rows: list[dict] = []
    n_steps = int(round(duration_s * 1000.0 / params.dt_ms))
    for step in range(n_steps):
        time_ms = step * params.dt_ms
        left_drive = left_scene.column_drive(time_ms, 0.0)
        right_drive = right_scene.column_drive(time_ms, 0.0)
        left_t4 = left_circuit.step(left_drive)
        right_t4 = right_circuit.step(right_drive)
        activity = readout.activities(left_t4, right_t4, ablate=ablate)
        raw_command = 0.0 if ablate else activity["right_minus_left_DNp15"]
        action = bridge.action(raw_command)
        rows.append(
            {
                "condition": condition_name,
                "time_ms": time_ms,
                "left_column_01_07": float(left_drive[0]),
                "left_column_01_08": float(left_drive[1]),
                "right_column_01_07": float(right_drive[0]),
                "right_column_01_08": float(right_drive[1]),
                **activity,
                "raw_steering_command": float(raw_command),
                "normalized_steering_command": float(bridge.normalized_command(raw_command)),
                "controller_channel_0": float(action[0]),
                "controller_channel_1": float(action[1]),
            }
        )
    return pd.DataFrame(rows)


def _run_body(
    *,
    condition_name: str,
    left_graph,
    right_graph,
    left_columns: list[dict],
    right_columns: list[dict],
    readout: BilateralDownstreamReadout,
    params: GradedParameters,
    bridge: SteeringBridge,
    duration_s: float,
    ablate: bool,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    timestep_s = params.dt_ms / 1000.0
    simulation, _camera = _make_body(condition_name, timestep_s)
    observation, _ = simulation.reset(seed=0)
    left_circuit, right_circuit = _make_side_circuits(
        left_graph, right_graph, left_columns, right_columns, params
    )
    spec = CONDITIONS[condition_name]
    left_scene = ClosedLoopMotionScene(reverse=spec["left_reverse"])
    right_scene = ClosedLoopMotionScene(reverse=spec["right_reverse"])
    rows: list[dict] = []
    frames: list[np.ndarray] = []
    n_steps = int(round(duration_s / timestep_s))
    for step in range(n_steps):
        time_ms = step * params.dt_ms
        yaw_before = _yaw_from_observation(observation)
        left_drive = left_scene.column_drive(time_ms, yaw_before)
        right_drive = right_scene.column_drive(time_ms, yaw_before)
        left_t4 = left_circuit.step(left_drive)
        right_t4 = right_circuit.step(right_drive)
        activity = readout.activities(left_t4, right_t4, ablate=ablate)
        raw_command = 0.0 if ablate else activity["right_minus_left_DNp15"]
        action = bridge.action(raw_command)
        observation, _reward, terminated, truncated, _info = simulation.step(action)
        yaw_after = _yaw_from_observation(observation)
        rows.append(
            {
                "condition": condition_name,
                "ablated": bool(ablate),
                "time_ms": time_ms,
                "left_column_01_07": float(left_drive[0]),
                "left_column_01_08": float(left_drive[1]),
                "right_column_01_07": float(right_drive[0]),
                "right_column_01_08": float(right_drive[1]),
                **activity,
                "raw_steering_command": float(raw_command),
                "normalized_steering_command": float(bridge.normalized_command(raw_command)),
                "controller_channel_0": float(action[0]),
                "controller_channel_1": float(action[1]),
                "yaw_rad": float(yaw_after),
                "retinal_shift_ms": float(yaw_before * left_scene.yaw_feedback_ms_per_rad),
            }
        )
        rendered = simulation.render()[0]
        if rendered is not None:
            frames.append(np.asarray(Image.fromarray(rendered).resize((320, 240))))
        if terminated or truncated:
            break
    simulation.close()
    return pd.DataFrame(rows), frames


def _summary(frame: pd.DataFrame) -> dict:
    command = frame.raw_steering_command.to_numpy(dtype=float)
    return {
        "min": float(command.min()),
        "max": float(command.max()),
        "peak_abs": float(np.max(np.abs(command))),
        "signed_area_command_ms": float(np.trapezoid(command, frame.time_ms.to_numpy(dtype=float))),
        "positive_samples": bool(np.any(command > 1e-12)),
        "negative_samples": bool(np.any(command < -1e-12)),
    }


def _paired_order_summary(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for column in (
        "left_DNp15",
        "right_DNp15",
        "right_minus_left_DNp15",
        "left_DNa02",
        "right_DNa02",
        "right_minus_left_DNa02",
        "raw_steering_command",
    ):
        delta = first[column].to_numpy(dtype=float) - second[column].to_numpy(dtype=float)
        result[column] = {
            "max_abs_A_minus_B": float(np.max(np.abs(delta))),
            "signed_area_A_minus_B_ms": float(np.trapezoid(delta, first.time_ms.to_numpy(dtype=float))),
        }
    return result


def _plot_prebody(frames: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    colors = {"A_left_forward_right_reverse": "tab:blue", "B_left_reverse_right_forward": "tab:orange"}
    for name, frame in frames.items():
        color = colors[name]
        axes[0].plot(frame.time_ms, frame.left_DNp15, color=color, linestyle="-", label=f"left DNp15 {name[0]}")
        axes[0].plot(frame.time_ms, frame.right_DNp15, color=color, linestyle="--", label=f"right DNp15 {name[0]}")
        axes[1].plot(frame.time_ms, frame.right_minus_left_DNp15, color=color, label=f"right-left {name[0]}")
        axes[1].plot(frame.time_ms, frame.right_minus_left_DNa02, color=color, linestyle=":", label=f"DNa02 right-left {name[0]}")
        axes[2].plot(frame.time_ms, frame.raw_steering_command, color=color, label=f"command {name[0]}")
    axes[0].set_ylabel("selected DN activity")
    axes[1].set_ylabel("bilateral difference")
    axes[2].set_ylabel("raw steering command")
    axes[2].set_xlabel("time (ms)")
    axes[0].set_title("Bilateral optic-flow readout before body integration")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_body(frames: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    colors = {"A_left_forward_right_reverse": "tab:blue", "B_left_reverse_right_forward": "tab:orange"}
    for name, frame in frames.items():
        axes[0].plot(frame.time_ms, frame.raw_steering_command, color=colors[name], label=name)
        axes[1].plot(frame.time_ms, frame.yaw_rad, color=colors[name], label=name)
    axes[0].set_ylabel("raw command")
    axes[1].set_ylabel("body yaw (rad)")
    axes[1].set_xlabel("time (ms)")
    axes[0].set_title("Bilateral command and embodied yaw")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "data" / "malecns_exp003_bilateral")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "exp003_bilateral")
    parser.add_argument("--duration-s", type=float, default=0.25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    left_graph = _load_graph(args.bundle / "left_visual")
    right_graph = _load_graph(args.bundle / "right_visual")
    left_downstream = CorrectedDownstreamMatrices.from_graph(_load_graph(args.bundle / "left_downstream"))
    right_downstream = CorrectedDownstreamMatrices.from_graph(_load_graph(args.bundle / "right_downstream"))
    readout = BilateralDownstreamReadout.from_matrices(left_downstream, right_downstream)
    left_manifest = json.loads((args.bundle / "left_visual" / "manifest.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((args.bundle / "right_visual" / "manifest.json").read_text(encoding="utf-8"))
    left_columns = left_manifest["motion_columns"]
    right_columns = right_manifest["motion_columns"]
    params = GradedParameters()
    bridge = SteeringBridge()

    open_frames: dict[str, pd.DataFrame] = {}
    ablated_open_frames: dict[str, pd.DataFrame] = {}
    for condition_name in CONDITIONS:
        frame = _run_open_loop(
            condition_name=condition_name,
            left_graph=left_graph,
            right_graph=right_graph,
            left_columns=left_columns,
            right_columns=right_columns,
            readout=readout,
            params=params,
            bridge=bridge,
            duration_s=args.duration_s,
            ablate=False,
        )
        ablated = _run_open_loop(
            condition_name=condition_name,
            left_graph=left_graph,
            right_graph=right_graph,
            left_columns=left_columns,
            right_columns=right_columns,
            readout=readout,
            params=params,
            bridge=bridge,
            duration_s=args.duration_s,
            ablate=True,
        )
        open_frames[condition_name] = frame
        ablated_open_frames[condition_name] = ablated
        frame.to_csv(args.output / f"open_loop_{condition_name}.csv", index=False)
        ablated.to_csv(args.output / f"open_loop_ablated_{condition_name}.csv", index=False)

    _plot_prebody(open_frames, args.output / "bilateral_prebody_readout.png")
    areas = {name: _summary(frame)["signed_area_command_ms"] for name, frame in open_frames.items()}
    directionally_meaningful = areas["A_left_forward_right_reverse"] * areas["B_left_reverse_right_forward"] < 0.0
    first_condition = open_frames["A_left_forward_right_reverse"]
    second_condition = open_frames["B_left_reverse_right_forward"]
    left_dn_nodes = _load_graph(args.bundle / "left_downstream").nodes
    right_dn_nodes = _load_graph(args.bundle / "right_downstream").nodes
    left_primary_ids = left_dn_nodes.loc[
        left_dn_nodes["stage"].eq("descending_output") & left_dn_nodes["type"].eq("DNp15"), "bodyId"
    ].astype(int).tolist()
    right_primary_ids = right_dn_nodes.loc[
        right_dn_nodes["stage"].eq("descending_output") & right_dn_nodes["type"].eq("DNp15"), "bodyId"
    ].astype(int).tolist()
    left_secondary_ids = left_dn_nodes.loc[
        left_dn_nodes["stage"].eq("descending_output") & left_dn_nodes["type"].eq("DNa02"), "bodyId"
    ].astype(int).tolist()
    right_secondary_ids = right_dn_nodes.loc[
        right_dn_nodes["stage"].eq("descending_output") & right_dn_nodes["type"].eq("DNa02"), "bodyId"
    ].astype(int).tolist()
    left_downstream_manifest = json.loads((args.bundle / "left_downstream" / "manifest.json").read_text(encoding="utf-8"))
    right_downstream_manifest = json.loads((args.bundle / "right_downstream" / "manifest.json").read_text(encoding="utf-8"))

    body_frames: dict[str, pd.DataFrame] = {}
    ablated_body_frames: dict[str, pd.DataFrame] = {}
    rendered: dict[str, list[np.ndarray]] = {}
    if directionally_meaningful:
        for condition_name in CONDITIONS:
            frame, video_frames = _run_body(
                condition_name=condition_name,
                left_graph=left_graph,
                right_graph=right_graph,
                left_columns=left_columns,
                right_columns=right_columns,
                readout=readout,
                params=params,
                bridge=bridge,
                duration_s=args.duration_s,
                ablate=False,
            )
            ablated, _ = _run_body(
                condition_name=condition_name,
                left_graph=left_graph,
                right_graph=right_graph,
                left_columns=left_columns,
                right_columns=right_columns,
                readout=readout,
                params=params,
                bridge=bridge,
                duration_s=args.duration_s,
                ablate=True,
            )
            body_frames[condition_name] = frame
            ablated_body_frames[condition_name] = ablated
            rendered[condition_name] = video_frames
            frame.to_csv(args.output / f"body_{condition_name}.csv", index=False)
            ablated.to_csv(args.output / f"body_ablated_{condition_name}.csv", index=False)
            _save_video(
                [_label_frame(image, condition_name) for image in video_frames],
                args.output / f"{condition_name}.mp4",
            )
        _plot_body(body_frames, args.output / "bilateral_body_yaw.png")

    summary = {
        "experiment": "EXP-003-bilateral",
        "parent_frozen_commit": "529d42f",
        "preserved_prior_experiments": ["EXP-001", "EXP-002", "EXP-003", "EXP-003-downstream-audit"],
        "exp002_parameters_unchanged": True,
        "bridge": asdict(bridge),
        "stimulus": {
            "A": "left 01_07 -> 01_08 while right 01_08 -> 01_07",
            "B": "left 01_08 -> 01_07 while right 01_07 -> 01_08",
            "geometry_label": "workbook-axis order only; no canonical yaw/clockwise naming",
        },
        "pathways": {
            "left_visual": {"nodes": left_graph.n_nodes, "edges": left_graph.n_edges},
            "right_visual": {"nodes": right_graph.n_nodes, "edges": right_graph.n_edges},
            "left_downstream": {"nodes": left_downstream_manifest["node_count"], "edges": left_downstream_manifest["edge_count"]},
            "right_downstream": {"nodes": right_downstream_manifest["node_count"], "edges": right_downstream_manifest["edge_count"]},
            "primary_dn_type": "DNp15/DNHS1",
            "secondary_dn_type_reported": "DNa02",
            "primary_dn_body_ids": {"left": left_primary_ids, "right": right_primary_ids},
            "secondary_dn_body_ids": {"left": left_secondary_ids, "right": right_secondary_ids},
            "homology": "same exact MaleCNS type with opposite soma-side: DNp15_L vs DNp15_R; DNa02_L vs DNa02_R",
        },
        "pre_body": {
            name: _summary(frame) | {
                "mean_left_DNp15": float(frame.left_DNp15.mean()),
                "mean_right_DNp15": float(frame.right_DNp15.mean()),
                "mean_right_minus_left_DNp15": float(frame.right_minus_left_DNp15.mean()),
                "mean_right_minus_left_DNa02": float(frame.right_minus_left_DNa02.mean()),
            }
            for name, frame in open_frames.items()
        },
        "condition_order_contrast": _paired_order_summary(first_condition, second_condition),
        "ablation_pre_body": {name: _summary(frame) for name, frame in ablated_open_frames.items()},
        "directionally_meaningful": directionally_meaningful,
        "embodiment_run": directionally_meaningful,
        "body_final_yaw_rad": {
            name: float(frame.yaw_rad.iloc[-1]) for name, frame in body_frames.items()
        },
        "ablated_body_final_yaw_rad": {
            name: float(frame.yaw_rad.iloc[-1]) for name, frame in ablated_body_frames.items()
        },
        "interpretation": (
            "The bilateral hypothesis is supported at the pre-body command level only when the two condition signed areas have opposite signs; body results are reported only in that case."
            if directionally_meaningful
            else "The bilateral DNp15 right-minus-left command does not reverse sign between the two mirrored optic-flow conditions; embodiment was not run because the pre-body criterion failed."
        ),
        "literature_basis": {
            "primary_dn": "DNp15/DNHS1 is an HS-linked bilateral optic-flow DN with yaw sensitivity and binocular modulation.",
            "secondary_dn": "DNa02 is reported as a steering-related DN in the same binocular optic-flow circuit literature but is not combined into the primary command.",
        },
        "flygym_version": version("flygym"),
        "mujoco_version": version("mujoco"),
        "outputs": [str(path.relative_to(args.output)) for path in args.output.rglob("*") if path.is_file()],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
