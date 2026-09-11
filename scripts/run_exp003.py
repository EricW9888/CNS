"""Run the first embodied EXP-003 steering demonstration.

The script keeps the frozen EXP-002 graph/model as the visual computation,
then adds only a transparent MaleCNS downstream readout and an explicit
two-channel FlyGym turning bridge.  It intentionally does not import or
modify FlyGym's visual sensors: the current experiment uses the validated
two-column visual abstraction, with fly yaw feeding back into its retinal
timing.
"""

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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp003 import (  # noqa: E402
    ClosedLoopMotionScene,
    DownstreamMatrices,
    OnlineGradedCircuit,
    SteeringBridge,
)
from src.graded_model import GradedParameters  # noqa: E402
from src.malecns_io import build_graph, load_edges  # noqa: E402


CONTACT_SEGMENTS = ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
T4_TYPES = ("T4a", "T4b", "T4c", "T4d")


def _load_graph(bundle: Path):
    return build_graph(
        pd.read_parquet(bundle / "nodes.parquet"),
        load_edges(bundle / "edges.parquet"),
    )


def _yaw_from_observation(observation: dict) -> float:
    orientation = np.asarray(observation["fly_orientation"], dtype=np.float64)
    return float(np.arctan2(orientation[1], orientation[0]))


def _make_body(condition_name: str, timestep_s: float):
    try:
        from flygym import SingleFlySimulation, YawOnlyCamera
        from flygym.examples.locomotion.turning_fly import HybridTurningFly
    except ImportError as exc:  # pragma: no cover - exercised in environment setup
        raise RuntimeError(
            "EXP-003 requires the optional FlyGym 1.2.1 environment; "
            "install requirements-exp003-flygym.txt first"
        ) from exc

    contact_sensor_placements = [
        f"{leg}{segment}" for leg in LEGS for segment in CONTACT_SEGMENTS
    ]
    fly = HybridTurningFly(
        timestep=timestep_s,
        enable_adhesion=False,
        seed=0,
        contact_sensor_placements=contact_sensor_placements,
    )
    camera = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_right",
        targeted_fly_names=fly.name,
        play_speed=0.1,
    )
    simulation = SingleFlySimulation(
        fly=fly,
        cameras=[camera],
        timestep=timestep_s,
    )
    return simulation, camera


def _run_condition(
    *,
    name: str,
    reverse: bool,
    ablate: bool,
    exp002_graph,
    motion_columns: list[dict],
    matrices: DownstreamMatrices,
    params: GradedParameters,
    bridge: SteeringBridge,
    duration_s: float,
    frame_stride: int,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    timestep_s = params.dt_ms / 1000.0
    simulation, _camera = _make_body(name, timestep_s)
    observation, _ = simulation.reset(seed=0)
    circuit = OnlineGradedCircuit(
        exp002_graph,
        motion_columns=motion_columns,
        params=params,
    )
    scene = ClosedLoopMotionScene(
        reverse=reverse,
        start_ms=10.0,
        pulse_duration_ms=10.0,
        amplitude=1.0,
    )
    node_types = exp002_graph.nodes["type"].astype(str).to_numpy()
    t4_types = node_types[circuit.t4_indices]
    n_steps = int(round(duration_s / timestep_s))
    frames: list[np.ndarray] = []
    rows: list[dict] = []

    for step in range(n_steps):
        time_ms = step * params.dt_ms
        yaw_before = _yaw_from_observation(observation)
        columns = scene.column_drive(time_ms, yaw_before)
        t4_activity = circuit.step(columns)
        wide_activity, descending_activity = matrices.propagate(
            t4_activity,
            ablate_t4_projection=ablate,
        )
        raw_command = 0.0 if ablate else bridge.raw_command(descending_activity, matrices)
        action = bridge.action(raw_command)
        observation, _reward, terminated, truncated, _info = simulation.step(action)
        yaw_after = _yaw_from_observation(observation)
        row = {
            "condition": name,
            "reverse": bool(reverse),
            "ablated": bool(ablate),
            "time_ms": float(time_ms),
            "column_01_07": float(columns[0]),
            "column_01_08": float(columns[1]),
            "wide_field_abs_mean": float(np.mean(np.abs(wide_activity))),
            "descending_abs_mean": float(np.mean(np.abs(descending_activity))),
            "raw_steering_command": float(raw_command),
            "normalized_steering_command": float(bridge.normalized_command(raw_command)),
            "controller_channel_0": float(action[0]),
            "controller_channel_1": float(action[1]),
            "yaw_rad": float(yaw_after),
            "x_mm": float(observation["fly"][0, 0]),
            "y_mm": float(observation["fly"][0, 1]),
            "retinal_shift_ms": float(yaw_before * scene.yaw_feedback_ms_per_rad),
        }
        for t4_type in T4_TYPES:
            values = t4_activity[t4_types == t4_type]
            row[f"{t4_type}_mean"] = float(np.mean(values)) if len(values) else 0.0
        rows.append(row)

        # FlyGym cameras decide their own effective render interval.  They
        # return None between frames, so call render every physics step and
        # retain only completed frames.
        rendered = simulation.render()[0]
        if rendered is not None:
            frame = np.asarray(Image.fromarray(rendered).resize((320, 240)))
            frames.append(frame)
        if terminated or truncated:
            break

    simulation.close()
    return pd.DataFrame(rows), frames


def _save_video(frames: list[np.ndarray], path: Path, *, fps: int = 50) -> None:
    if not frames:
        raise RuntimeError(f"No rendered frames were produced for {path}")
    imageio.mimsave(path, frames, fps=fps)


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return np.asarray(image)


def _plot_signal_trace(forward: pd.DataFrame, reverse: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    for frame, label, color in ((forward, "forward order", "tab:blue"), (reverse, "reverse order", "tab:orange")):
        axes[0].plot(frame.time_ms, frame.column_01_07, color=color, linestyle="-", label=f"01_07 {label}")
        axes[0].plot(frame.time_ms, frame.column_01_08, color=color, linestyle="--", label=f"01_08 {label}")
        for t4_type in T4_TYPES:
            axes[1].plot(frame.time_ms, frame[f"{t4_type}_mean"], label=f"{t4_type} {label}", alpha=0.8)
        axes[2].plot(frame.time_ms, frame.wide_field_abs_mean, label=f"wide-field {label}", color=color)
        axes[2].plot(frame.time_ms, frame.descending_abs_mean, label=f"descending {label}", color=color, linestyle="--")
        axes[3].plot(frame.time_ms, frame.raw_steering_command, label=label, color=color)
        axes[3].plot(frame.time_ms, frame.normalized_steering_command, label=f"normalized {label}", color=color, linestyle="--")
        axes[4].plot(frame.time_ms, frame.yaw_rad, label=label, color=color)
    axes[0].set_ylabel("column drive")
    axes[1].set_ylabel("T4 activity")
    axes[2].set_ylabel("|activity proxy|")
    axes[3].set_ylabel("steering")
    axes[4].set_ylabel("body yaw (rad)")
    axes[4].set_xlabel("time (ms)")
    axes[0].set_title("EXP-003 closed-loop signal path: two-column input → T4 → downstream → steering → yaw")
    for axis in axes:
        axis.axvspan(10, 30, color="0.9", zorder=-2)
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_control_comparison(frames: dict[str, pd.DataFrame], output: Path) -> dict[str, float]:
    full_forward = frames["full_forward"]
    full_reverse = frames["full_reverse"]
    ablated_forward = frames["ablated_forward"]
    ablated_reverse = frames["ablated_reverse"]
    min_len = min(len(full_forward), len(full_reverse))
    min_ablated = min(len(ablated_forward), len(ablated_reverse))
    raw_contrast = float(
        np.max(
            np.abs(
                full_forward.raw_steering_command.to_numpy()[:min_len]
                - full_reverse.raw_steering_command.to_numpy()[:min_len]
            )
        )
    )
    raw_contrast_ablated = float(
        np.max(
            np.abs(
                ablated_forward.raw_steering_command.to_numpy()[:min_ablated]
                - ablated_reverse.raw_steering_command.to_numpy()[:min_ablated]
            )
        )
    )
    yaw_contrast = abs(float(full_forward.yaw_rad.iloc[-1] - full_reverse.yaw_rad.iloc[-1]))
    yaw_contrast_ablated = abs(
        float(ablated_forward.yaw_rad.iloc[-1] - ablated_reverse.yaw_rad.iloc[-1])
    )
    labels = ["full circuit", "T4→wide-field ablation"]
    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(x - width / 2, [raw_contrast, raw_contrast_ablated], width, label="max |forward − reverse|")
    axes[0].bar(x + width / 2, [
        float(np.mean(np.abs(full_forward.raw_steering_command))),
        float(np.mean(np.abs(ablated_forward.raw_steering_command))),
    ], width, label="mean |steering|")
    axes[0].set_ylabel("dimensionless bridge signal")
    axes[0].set_title("Downstream order contrast")
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [yaw_contrast, yaw_contrast_ablated], color=["tab:blue", "0.6"])
    axes[1].set_ylabel("|final yaw forward − reverse| (rad)")
    axes[1].set_title("Embodied order contrast")
    axes[1].set_xticks(x, labels, rotation=15)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {
        "raw_order_contrast_full": raw_contrast,
        "raw_order_contrast_ablated": raw_contrast_ablated,
        "yaw_order_contrast_full": yaw_contrast,
        "yaw_order_contrast_ablated": yaw_contrast_ablated,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp002-bundle", type=Path, default=ROOT / "data" / "malecns_exp002_local_circuit")
    parser.add_argument("--downstream-bundle", type=Path, default=ROOT / "data" / "malecns_visual_subgraph")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "exp003")
    parser.add_argument("--duration-s", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    exp002_graph = _load_graph(args.exp002_bundle)
    downstream_graph = _load_graph(args.downstream_bundle)
    if (exp002_graph.n_nodes, exp002_graph.n_edges) != (187, 958):
        raise RuntimeError("EXP-002 bundle is not the frozen 187-node/958-edge circuit")
    if (downstream_graph.n_nodes, downstream_graph.n_edges) != (137, 356):
        raise RuntimeError("Downstream bundle is not the 137-node/356-edge materialized graph")
    manifest = json.loads((args.exp002_bundle / "manifest.json").read_text(encoding="utf-8"))
    motion_columns = manifest["motion_columns"]
    matrices = DownstreamMatrices.from_graph(downstream_graph)
    params = GradedParameters()
    bridge = SteeringBridge()

    outputs: dict[str, pd.DataFrame] = {}
    rendered: dict[str, list[np.ndarray]] = {}
    for name, reverse, ablate in (
        ("full_forward", False, False),
        ("full_reverse", True, False),
        ("ablated_forward", False, True),
        ("ablated_reverse", True, True),
    ):
        frame, video_frames = _run_condition(
            name=name,
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
        outputs[name] = frame
        rendered[name] = video_frames
        frame.to_csv(args.output / f"{name}.csv", index=False)

    full_forward_frames = [_label_frame(frame, "forward order") for frame in rendered["full_forward"]]
    full_reverse_frames = [_label_frame(frame, "reverse order") for frame in rendered["full_reverse"]]
    _save_video(full_forward_frames, args.output / "exp003_full_forward.mp4")
    _save_video(full_reverse_frames, args.output / "exp003_full_reverse.mp4")
    n_frames = min(len(full_forward_frames), len(full_reverse_frames))
    combined = [
        np.concatenate([full_forward_frames[i], full_reverse_frames[i]], axis=1)
        for i in range(n_frames)
    ]
    _save_video(combined, args.output / "exp003_full_conditions.mp4")
    imageio.mimsave(args.output / "exp003_full_conditions.gif", combined, duration=1 / 50)

    _plot_signal_trace(
        outputs["full_forward"],
        outputs["full_reverse"],
        args.output / "exp003_signal_to_yaw.png",
    )
    contrast = _plot_control_comparison(outputs, args.output / "exp003_ablation_control.png")

    pd.DataFrame(
        {
            "bodyId": matrices.descending_ids,
            "type": matrices.descending_types,
            "t4a_path_weight": matrices.t4a_path,
            "t4d_path_weight": matrices.t4d_path,
            "bridge_weight": matrices.bridge_weights,
        }
    ).to_csv(args.output / "downstream_bridge_weights.csv", index=False)
    pathway_metadata = {
        "downstream_nodes": {
            "t4": len(matrices.t4_ids),
            "wide_field": len(matrices.wide_ids),
            "descending": len(matrices.descending_ids),
        },
        "t4_type_edge_support": {
            t4_type: {
                "neurons_with_outgoing_edges": int(
                    sum(
                        1
                        for cell_type, weights in zip(matrices.t4_types, matrices.t4_to_wide.T)
                        if cell_type == t4_type and np.any(weights > 0)
                    )
                ),
                "neurons_without_outgoing_edges": int(
                    sum(
                        1
                        for cell_type, weights in zip(matrices.t4_types, matrices.t4_to_wide.T)
                        if cell_type == t4_type and not np.any(weights > 0)
                    )
                ),
            }
            for t4_type in T4_TYPES
        },
        "readout": "descending activity weighted by normalized path mass from T4a minus T4d",
        "t4_to_wide_ablation": "all selected T4->wide-field matrix entries removed",
    }
    (args.output / "downstream_pathway_metadata.json").write_text(
        json.dumps(pathway_metadata, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment": "EXP-003",
        "flygym_version": version("flygym"),
        "mujoco_version": version("mujoco"),
        "exp002_graph": {"nodes": exp002_graph.n_nodes, "edges": exp002_graph.n_edges},
        "downstream_graph": {"nodes": downstream_graph.n_nodes, "edges": downstream_graph.n_edges},
        "parameters": asdict(params),
        "bridge": asdict(bridge),
        "visual_scene": {
            "columns": motion_columns,
            "start_ms": 10.0,
            "pulse_duration_ms": 10.0,
            "amplitude": 1.0,
            "yaw_feedback_ms_per_rad": 100.0,
            "geometry_status": "The workbook suffix-to-visual-axis mapping remains unresolved; conditions are named forward/reverse order only.",
        },
        "control": "T4->wide-field projection ablation",
        "contrast": contrast,
        "final_yaw_rad": {
            name: float(frame.yaw_rad.iloc[-1]) for name, frame in outputs.items()
        },
        "max_abs_raw_steering": {
            name: float(np.max(np.abs(frame.raw_steering_command)))
            for name, frame in outputs.items()
        },
        "outputs": [path.name for path in args.output.iterdir() if path.is_file()],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
