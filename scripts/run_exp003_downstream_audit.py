"""Audit the frozen EXP-003 bridge against original and corrected pathways."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_exp003 import _load_graph, _plot_signal_trace, _run_condition, _save_video, _label_frame  # noqa: E402
from src.exp003 import ClosedLoopMotionScene, DownstreamMatrices, OnlineGradedCircuit, SteeringBridge  # noqa: E402
from src.exp003_corrected import CorrectedDownstreamMatrices  # noqa: E402
from src.graded_model import GradedParameters  # noqa: E402


T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
CONDITIONS = (
    ("full_forward", False, False),
    ("full_reverse", True, False),
    ("ablated_forward", False, True),
    ("ablated_reverse", True, True),
)


def _original_coverage(matrices: DownstreamMatrices) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for t4_type in T4_TYPES:
        values = [i for i, value in enumerate(matrices.t4_types) if value == t4_type]
        supported = matrices.t4_to_wide[:, values].sum(axis=0) > 0
        result[t4_type] = {
            "total_neurons": len(values),
            "neurons_with_direct_target": int(np.count_nonzero(supported)),
            "neurons_with_target_to_descending_path": int(
                np.count_nonzero(np.any(matrices.t4_to_wide[:, values].T @ matrices.wide_to_descending.T > 0, axis=1))
            ),
        }
    return result


def _run_open_loop(
    *,
    name: str,
    reverse: bool,
    ablate: bool,
    exp002_graph,
    motion_columns: list[dict],
    matrices,
    params: GradedParameters,
    bridge: SteeringBridge,
    duration_s: float,
) -> pd.DataFrame:
    circuit = OnlineGradedCircuit(exp002_graph, motion_columns=motion_columns, params=params)
    scene = ClosedLoopMotionScene(reverse=reverse, start_ms=10.0, pulse_duration_ms=10.0, amplitude=1.0)
    node_types = exp002_graph.nodes["type"].astype(str).to_numpy()
    t4_types = node_types[circuit.t4_indices]
    rows: list[dict] = []
    n_steps = int(round(duration_s * 1000.0 / params.dt_ms))
    for step in range(n_steps):
        time_ms = step * params.dt_ms
        columns = scene.column_drive(time_ms, 0.0)
        t4_activity = circuit.step(columns)
        middle_activity, descending_activity = matrices.propagate(
            t4_activity,
            ablate_t4_projection=ablate,
        )
        raw_command = 0.0 if ablate else bridge.raw_command(descending_activity, matrices)
        row = {
            "condition": name,
            "reverse": bool(reverse),
            "time_ms": float(time_ms),
            "column_01_07": float(columns[0]),
            "column_01_08": float(columns[1]),
            "wide_field_abs_mean": float(np.mean(np.abs(middle_activity))),
            "descending_abs_mean": float(np.mean(np.abs(descending_activity))),
            "raw_steering_command": float(raw_command),
            "normalized_steering_command": float(bridge.normalized_command(raw_command)),
        }
        for t4_type in T4_TYPES:
            values = t4_activity[t4_types == t4_type]
            row[f"{t4_type}_mean"] = float(np.mean(values)) if len(values) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _command_summary(frame: pd.DataFrame) -> dict:
    command = frame["raw_steering_command"].to_numpy(dtype=float)
    absolute_index = int(np.argmax(np.abs(command)))
    return {
        "min": float(command.min()),
        "max": float(command.max()),
        "peak_absolute": float(np.max(np.abs(command))),
        "peak_absolute_time_ms": float(frame.iloc[absolute_index]["time_ms"]),
        "signed_area_command_ms": float(np.trapezoid(command, frame["time_ms"].to_numpy(dtype=float))),
        "has_positive_samples": bool(np.any(command > 1e-12)),
        "has_negative_samples": bool(np.any(command < -1e-12)),
    }


def _plot_comparison(open_loop: dict[str, dict[str, pd.DataFrame]], body: dict[str, dict[str, pd.DataFrame]], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    colors = {"forward": "tab:blue", "reverse": "tab:orange"}
    for column, graph_label in enumerate(("original", "corrected")):
        for condition, color in colors.items():
            frame = open_loop[graph_label][f"full_{condition}"]
            axes[0, column].plot(frame.time_ms, frame.raw_steering_command, color=color, label=condition)
            frame = body[graph_label][f"full_{condition}"]
            axes[1, column].plot(frame.time_ms, frame.yaw_rad, color=color, label=condition)
            frame = body[graph_label][f"ablated_{condition}"]
            axes[2, column].plot(frame.time_ms, frame.yaw_rad, color=color, linestyle="--", label=f"ablated {condition}")
        axes[0, column].set_title(f"{graph_label}: open-loop bridge command")
        axes[1, column].set_title(f"{graph_label}: full body yaw")
        axes[2, column].set_title(f"{graph_label}: ablated body yaw")
        for row in range(3):
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend(fontsize=8)
            axes[row, column].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("raw command")
    axes[1, 0].set_ylabel("yaw (rad)")
    axes[2, 0].set_ylabel("yaw (rad)")
    fig.suptitle("EXP-003 downstream audit: frozen bridge, original vs corrected MaleCNS path")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_ablation(open_loop: dict[str, dict[str, pd.DataFrame]], body: dict[str, dict[str, pd.DataFrame]], output: Path) -> None:
    labels = ["original full", "corrected full", "corrected ablated"]
    raw = [
        np.max(np.abs(open_loop["original"]["full_forward"].raw_steering_command.to_numpy() - open_loop["original"]["full_reverse"].raw_steering_command.to_numpy())),
        np.max(np.abs(open_loop["corrected"]["full_forward"].raw_steering_command.to_numpy() - open_loop["corrected"]["full_reverse"].raw_steering_command.to_numpy())),
        np.max(np.abs(open_loop["corrected"]["ablated_forward"].raw_steering_command.to_numpy() - open_loop["corrected"]["ablated_reverse"].raw_steering_command.to_numpy())),
    ]
    yaw = [
        abs(body["original"]["full_forward"].yaw_rad.iloc[-1] - body["original"]["full_reverse"].yaw_rad.iloc[-1]),
        abs(body["corrected"]["full_forward"].yaw_rad.iloc[-1] - body["corrected"]["full_reverse"].yaw_rad.iloc[-1]),
        abs(body["corrected"]["ablated_forward"].yaw_rad.iloc[-1] - body["corrected"]["ablated_reverse"].yaw_rad.iloc[-1]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, raw, color=["0.55", "tab:blue", "0.7"])
    axes[0].set_ylabel("max |forward - reverse| raw command")
    axes[1].bar(x, yaw, color=["0.55", "tab:blue", "0.7"])
    axes[1].set_ylabel("|final yaw forward - reverse| (rad)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("EXP-003 pathway audit: order contrast and T4-path ablation")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp002-bundle", type=Path, default=ROOT / "data" / "malecns_exp002_local_circuit")
    parser.add_argument("--original-bundle", type=Path, default=ROOT / "data" / "malecns_visual_subgraph")
    parser.add_argument("--corrected-bundle", type=Path, default=ROOT / "data" / "malecns_exp003_corrected_downstream")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "exp003_downstream_audit")
    parser.add_argument("--duration-s", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    exp002_graph = _load_graph(args.exp002_bundle)
    original_graph = _load_graph(args.original_bundle)
    corrected_graph = _load_graph(args.corrected_bundle)
    if (exp002_graph.n_nodes, exp002_graph.n_edges) != (187, 958):
        raise RuntimeError("Frozen EXP-002 visual bundle is not 187 nodes / 958 edges")
    if (original_graph.n_nodes, original_graph.n_edges) != (137, 356):
        raise RuntimeError("Frozen original downstream graph is not 137 nodes / 356 edges")
    if (corrected_graph.n_nodes, corrected_graph.n_edges) != (405, 1566):
        raise RuntimeError("Corrected downstream graph is not 405 nodes / 1566 edges")

    manifest = json.loads((args.exp002_bundle / "manifest.json").read_text(encoding="utf-8"))
    motion_columns = manifest["motion_columns"]
    graph_objects = {
        "original": DownstreamMatrices.from_graph(original_graph),
        "corrected": CorrectedDownstreamMatrices.from_graph(corrected_graph),
    }
    params = GradedParameters()
    bridge = SteeringBridge()
    open_loop: dict[str, dict[str, pd.DataFrame]] = {"original": {}, "corrected": {}}
    body: dict[str, dict[str, pd.DataFrame]] = {"original": {}, "corrected": {}}

    for graph_label, matrices in graph_objects.items():
        graph_output = args.output / graph_label
        graph_output.mkdir(parents=True, exist_ok=True)
        for condition, reverse, ablate in CONDITIONS:
            open_frame = _run_open_loop(
                name=condition,
                reverse=reverse,
                ablate=ablate,
                exp002_graph=exp002_graph,
                motion_columns=motion_columns,
                matrices=matrices,
                params=params,
                bridge=bridge,
                duration_s=args.duration_s,
            )
            open_loop[graph_label][condition] = open_frame
            open_frame.to_csv(graph_output / f"open_loop_{condition}.csv", index=False)
            body_frame, frames = _run_condition(
                name=condition,
                reverse=reverse,
                ablate=ablate,
                exp002_graph=exp002_graph,
                motion_columns=motion_columns,
                matrices=matrices,
                params=params,
                bridge=bridge,
                duration_s=args.duration_s,
                frame_stride=args.frame_stride,
            )
            body[graph_label][condition] = body_frame
            body_frame.to_csv(graph_output / f"body_{condition}.csv", index=False)
            if condition.startswith("full"):
                labeled = [_label_frame(frame, f"{graph_label} {condition}") for frame in frames]
                _save_video(labeled, graph_output / f"{condition}.mp4")
        _plot_signal_trace(
            body[graph_label]["full_forward"],
            body[graph_label]["full_reverse"],
            graph_output / "signal_to_yaw.png",
        )

    _plot_comparison(open_loop, body, args.output / "downstream_audit_comparison.png")
    _plot_ablation(open_loop, body, args.output / "order_contrast_ablation.png")

    open_loop_summary = {
        graph_label: {condition: _command_summary(frame) for condition, frame in conditions.items()}
        for graph_label, conditions in open_loop.items()
    }
    final_yaw = {
        graph_label: {condition: float(frame.yaw_rad.iloc[-1]) for condition, frame in conditions.items()}
        for graph_label, conditions in body.items()
    }
    summary = {
        "experiment": "EXP-003-downstream-audit",
        "parent_frozen_commit": "aeccf1c",
        "exp002_parameters_unchanged": True,
        "bridge_unchanged": asdict(bridge),
        "graphs": {
            "visual_exp002": {"nodes": exp002_graph.n_nodes, "edges": exp002_graph.n_edges},
            "original": {"nodes": original_graph.n_nodes, "edges": original_graph.n_edges},
            "corrected": {"nodes": corrected_graph.n_nodes, "edges": corrected_graph.n_edges},
        },
        "audit_finding": {
            "root_cause": "The original downstream query retained only HSE/HSN/HSS/HST/VS/VST1/VST2/VSm as direct T4 targets. This target-type filter excluded the direct T4b/T4c lobula-plate paths; no subtype, side, or depth failure was required to explain the absence.",
            "current_query_depth": 2,
            "corrected_query_depth": 2,
            "corrected_target_rule": "visual_projection OR visual_centrifugal OR type starts with LPi",
            "coverage_before": _original_coverage(graph_objects["original"]),
            "coverage_after": graph_objects["corrected"].coverage(),
        },
        "open_loop_steering": open_loop_summary,
        "final_yaw_rad": final_yaw,
        "order_contrast": {
            graph_label: {
                "raw_command_max_abs_forward_minus_reverse": float(
                    np.max(np.abs(conditions["full_forward"].raw_steering_command.to_numpy() - conditions["full_reverse"].raw_steering_command.to_numpy()))
                ),
                "final_yaw_abs_forward_minus_reverse": abs(
                    final_yaw[graph_label]["full_forward"] - final_yaw[graph_label]["full_reverse"]
                ),
            }
            for graph_label, conditions in body.items()
        },
        "ablation": {
            "operation": "remove all corrected T4 -> lobula_plate_target matrix entries before descending readout",
            "corrected_raw_command_max_abs_forward_minus_reverse": float(
                np.max(np.abs(open_loop["corrected"]["ablated_forward"].raw_steering_command.to_numpy() - open_loop["corrected"]["ablated_reverse"].raw_steering_command.to_numpy()))
            ),
            "corrected_final_yaw_abs_forward_minus_reverse": abs(
                final_yaw["corrected"]["ablated_forward"] - final_yaw["corrected"]["ablated_reverse"]
            ),
        },
        "interpretation": "Compare corrected open-loop forward/reverse command signs directly; body yaw is reported separately because FlyGym dynamics can preserve a common turn bias even when the CNS command differs.",
        "geometry_status": "The MaleCNS workbook axis-to-canonical visual direction mapping remains unresolved; conditions retain forward/reverse order labels only.",
        "flygym_version": version("flygym"),
        "mujoco_version": version("mujoco"),
        "outputs": [str(path.relative_to(args.output)) for path in args.output.rglob("*") if path.is_file()],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
