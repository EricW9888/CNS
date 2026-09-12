"""INVALID provisional EXP-004 runner retained for audit."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp003 import ClosedLoopMotionScene, OnlineGradedCircuit  # noqa: E402
from src.exp004 import BinocularNetwork, BinocularParameters  # noqa: E402
from src.graded_model import GradedParameters  # noqa: E402
from src.malecns_io import build_graph, load_edges  # noqa: E402


CONDITIONS = {
    "yaw_like_A": {"left_reverse": False, "right_reverse": True},
    "yaw_like_B": {"left_reverse": True, "right_reverse": False},
    "translation_like_A": {"left_reverse": False, "right_reverse": False},
    "translation_like_B": {"left_reverse": True, "right_reverse": True},
    "left_only_forward": {"left_reverse": False, "right_reverse": None},
    "left_only_reverse": {"left_reverse": True, "right_reverse": None},
    "right_only_forward": {"left_reverse": None, "right_reverse": False},
    "right_only_reverse": {"left_reverse": None, "right_reverse": True},
}

CONTROL_CONFIGS = {
    "full": {"electrical_coupling": True, "recurrent_chemical": True, "inhibitory_intermediates": True},
    "no_electrical": {"electrical_coupling": False, "recurrent_chemical": True, "inhibitory_intermediates": True},
    "no_recurrent": {"electrical_coupling": True, "recurrent_chemical": False, "inhibitory_intermediates": True},
    "no_lpi": {"electrical_coupling": True, "recurrent_chemical": True, "inhibitory_intermediates": False},
}


def _load_graph(bundle: Path):
    return build_graph(
        pd.read_parquet(bundle / "nodes.parquet"),
        load_edges(bundle / "edges.parquet"),
    )


def _drive(scene: ClosedLoopMotionScene | None, time_ms: float, yaw: float = 0.0) -> np.ndarray:
    return scene.column_drive(time_ms, yaw) if scene is not None else np.zeros(2, dtype=np.float64)


def _run_condition(
    *,
    condition_name: str,
    left_graph,
    right_graph,
    combined_graph,
    left_columns: list[dict],
    right_columns: list[dict],
    params: GradedParameters,
    downstream_params: BinocularParameters,
    duration_ms: float,
    control: dict[str, bool],
) -> pd.DataFrame:
    left_circuit = OnlineGradedCircuit(left_graph, motion_columns=left_columns, params=params)
    right_circuit = OnlineGradedCircuit(right_graph, motion_columns=right_columns, params=params)
    network = BinocularNetwork.from_graph(combined_graph, params=downstream_params)
    spec = CONDITIONS[condition_name]
    left_scene = ClosedLoopMotionScene(reverse=spec["left_reverse"]) if spec["left_reverse"] is not None else None
    right_scene = ClosedLoopMotionScene(reverse=spec["right_reverse"]) if spec["right_reverse"] is not None else None
    left_t4_ids = left_graph.nodes.loc[left_graph.nodes["stage"].eq("t4_motion"), "bodyId"].astype(int).tolist()
    right_t4_ids = right_graph.nodes.loc[right_graph.nodes["stage"].eq("t4_motion"), "bodyId"].astype(int).tolist()
    rows: list[dict] = []
    n_steps = int(round(duration_ms / params.dt_ms))
    for step in range(n_steps):
        time_ms = step * params.dt_ms
        left_drive = _drive(left_scene, time_ms)
        right_drive = _drive(right_scene, time_ms)
        left_t4 = left_circuit.step(left_drive)
        right_t4 = right_circuit.step(right_drive)
        combined_t4 = network.embed_side_t4(left_t4_ids, left_t4, right_t4_ids, right_t4)
        states = network.step(combined_t4, **control)
        population = network.population_means()
        side_population = network.side_population_means()
        rows.append(
            {
                "condition": condition_name,
                "time_ms": time_ms,
                "left_column_01_07": float(left_drive[0]),
                "left_column_01_08": float(left_drive[1]),
                "right_column_01_07": float(right_drive[0]),
                "right_column_01_08": float(right_drive[1]),
                "t4_left_mean": float(np.mean(left_t4)),
                "t4_right_mean": float(np.mean(right_t4)),
                "intermediate_abs_mean": float(np.mean(np.abs(states["intermediate"]))),
                "lptc_abs_mean": float(np.mean(np.abs(states["lptc"]))),
                "dn_abs_mean": float(np.mean(np.abs(states["dn"]))),
                **population,
                **side_population,
            }
        )
    return pd.DataFrame(rows)


def _area(frame: pd.DataFrame, column: str) -> float:
    return float(np.trapezoid(frame[column].to_numpy(dtype=float), frame.time_ms.to_numpy(dtype=float)))


def _metrics(frame: pd.DataFrame) -> dict[str, float | bool]:
    dn_diff = frame["DNp15_R"] - frame["DNp15_L"]
    dn_common = (frame["DNp15_R"] + frame["DNp15_L"]) / 2.0
    hs_diff = frame["HSE_R"] - frame["HSE_L"]
    h2_diff = frame["H2_R"] - frame["H2_L"]
    return {
        "DNp15_L_area": _area(frame, "DNp15_L"),
        "DNp15_R_area": _area(frame, "DNp15_R"),
        "DNp15_bilateral_difference_area": float(np.trapezoid(dn_diff, frame.time_ms)),
        "DNp15_bilateral_difference_peak_abs": float(np.max(np.abs(dn_diff))),
        "DNp15_common_mode_peak_abs": float(np.max(np.abs(dn_common))),
        "HSE_bilateral_difference_area": float(np.trapezoid(hs_diff, frame.time_ms)),
        "H2_bilateral_difference_area": float(np.trapezoid(h2_diff, frame.time_ms)),
        "DNa02_bilateral_difference_area": _area(frame, "DNa02_R") - _area(frame, "DNa02_L"),
    }


def _plot_primary(frames: dict[str, pd.DataFrame], metrics: dict[str, dict], output: Path) -> None:
    order = ["yaw_like_A", "yaw_like_B", "translation_like_A", "translation_like_B"]
    colors = {
        "yaw_like_A": "tab:blue",
        "yaw_like_B": "tab:orange",
        "translation_like_A": "tab:green",
        "translation_like_B": "tab:red",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].set_title("Stimulus-space conditions")
    for index, name in enumerate(order):
        y = index
        frame = frames[name]
        axes[0, 0].plot(frame.time_ms, frame.left_column_01_07 - frame.left_column_01_08, color=colors[name], label=f"L {name}")
        axes[0, 0].plot(frame.time_ms, frame.right_column_01_07 - frame.right_column_01_08, color=colors[name], linestyle="--", label=f"R {name}")
    axes[0, 0].set_ylabel("column 01_07 - 01_08")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].legend(fontsize=7, ncol=2)

    for name in order:
        frame = frames[name]
        axes[0, 1].plot(frame.time_ms, frame["HSE_L"], color=colors[name], alpha=0.8, label=f"HSE L {name}")
        axes[0, 1].plot(frame.time_ms, frame["HSE_R"], color=colors[name], linestyle="--", alpha=0.8, label=f"HSE R {name}")
        axes[0, 1].plot(frame.time_ms, frame["H2_L"], color=colors[name], linestyle=":", alpha=0.8, label=f"H2 L {name}")
        axes[0, 1].plot(frame.time_ms, frame["H2_R"], color=colors[name], linestyle="-.", alpha=0.8, label=f"H2 R {name}")
    axes[0, 1].set_title("HS/H2 stage")
    axes[0, 1].set_ylabel("graded activity proxy")
    axes[0, 1].set_xlabel("time (ms)")
    axes[0, 1].legend(fontsize=6, ncol=2)

    for name in order:
        frame = frames[name]
        axes[1, 0].plot(frame.time_ms, frame["DNp15_L"], color=colors[name], label=f"L {name}")
        axes[1, 0].plot(frame.time_ms, frame["DNp15_R"], color=colors[name], linestyle="--", label=f"R {name}")
    axes[1, 0].set_title("DNp15 stage")
    axes[1, 0].set_ylabel("graded activity proxy")
    axes[1, 0].set_xlabel("time (ms)")
    axes[1, 0].legend(fontsize=7, ncol=2)

    labels = ["yaw A", "yaw B", "trans A", "trans B"]
    values = [metrics[name]["DNp15_bilateral_difference_area"] for name in order]
    axes[1, 1].bar(labels, values, color=[colors[name] for name in order])
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("DNp15 bilateral difference")
    axes[1, 1].set_ylabel("signed area")
    axes[1, 1].tick_params(axis="x", rotation=20)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("EXP-004 MaleCNS binocular network: stimulus → HS/H2 → DNp15")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_controls(control_metrics: dict[str, dict[str, dict]], output: Path) -> None:
    order = ["yaw_like_A", "yaw_like_B", "translation_like_A", "translation_like_B"]
    controls = list(control_metrics)
    x = np.arange(len(order))
    width = 0.8 / len(controls)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for i, control in enumerate(controls):
        values = [control_metrics[control][name]["DNp15_bilateral_difference_peak_abs"] for name in order]
        axes[0].bar(x + (i - (len(controls) - 1) / 2) * width, values, width, label=control)
        values = [control_metrics[control][name]["DNp15_bilateral_difference_area"] for name in order]
        axes[1].bar(x + (i - (len(controls) - 1) / 2) * width, values, width, label=control)
    axes[0].set_title("Peak bilateral DNp15 difference")
    axes[1].set_title("Signed-area bilateral DNp15 difference")
    axes[0].set_ylabel("activity proxy")
    axes[1].set_ylabel("signed area")
    for axis in axes:
        axis.set_xticks(x, ["yaw A", "yaw B", "trans A", "trans B"], rotation=20)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("EXP-004 mechanistic controls")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-bundle", type=Path, default=ROOT / "data" / "malecns_exp004_binocular_network")
    parser.add_argument("--bilateral-bundle", type=Path, default=ROOT / "data" / "malecns_exp003_bilateral")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "exp004")
    parser.add_argument("--duration-ms", type=float, default=250.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    combined_graph = _load_graph(args.network_bundle)
    left_graph = _load_graph(args.bilateral_bundle / "left_visual")
    right_graph = _load_graph(args.bilateral_bundle / "right_visual")
    left_manifest = json.loads((args.bilateral_bundle / "left_visual" / "manifest.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((args.bilateral_bundle / "right_visual" / "manifest.json").read_text(encoding="utf-8"))
    left_columns = left_manifest["motion_columns"]
    right_columns = right_manifest["motion_columns"]
    visual_params = GradedParameters()
    downstream_params = BinocularParameters(dt_ms=visual_params.dt_ms)

    all_frames: dict[str, dict[str, pd.DataFrame]] = {}
    all_metrics: dict[str, dict[str, dict]] = {}
    for control_name, control in CONTROL_CONFIGS.items():
        control_dir = args.output / control_name
        control_dir.mkdir(parents=True, exist_ok=True)
        all_frames[control_name] = {}
        all_metrics[control_name] = {}
        for condition_name in CONDITIONS:
            frame = _run_condition(
                condition_name=condition_name,
                left_graph=left_graph,
                right_graph=right_graph,
                combined_graph=combined_graph,
                left_columns=left_columns,
                right_columns=right_columns,
                params=visual_params,
                downstream_params=downstream_params,
                duration_ms=args.duration_ms,
                control=control,
            )
            all_frames[control_name][condition_name] = frame
            all_metrics[control_name][condition_name] = _metrics(frame)
            frame.to_csv(control_dir / f"{condition_name}.csv", index=False)

    _plot_primary(all_frames["full"], all_metrics["full"], args.output / "exp004_binocular_responses.png")
    _plot_controls(all_metrics, args.output / "exp004_ablation_controls.png")
    pd.DataFrame(
        [
            {"control": control, "condition": condition, **metrics}
            for control, condition_metrics in all_metrics.items()
            for condition, metrics in condition_metrics.items()
        ]
    ).to_csv(args.output / "response_metrics.csv", index=False)

    full = all_metrics["full"]
    yaw_area = np.mean([abs(full["yaw_like_A"]["DNp15_bilateral_difference_area"]), abs(full["yaw_like_B"]["DNp15_bilateral_difference_area"])])
    translation_area = np.mean([abs(full["translation_like_A"]["DNp15_bilateral_difference_area"]), abs(full["translation_like_B"]["DNp15_bilateral_difference_area"])])
    summary = {
        "experiment": "EXP-004",
        "parent_frozen_commit": "af26f03",
        "preserved_prior_experiments": ["EXP-001", "EXP-002", "EXP-003", "EXP-003-downstream-audit", "EXP-003-bilateral"],
        "graph": {
            "nodes": combined_graph.n_nodes,
            "edges": combined_graph.n_edges,
            "stage_counts": combined_graph.nodes["stage"].value_counts().to_dict(),
            "chemical_only": True,
        },
        "visual_parameters_frozen": asdict(visual_params),
        "downstream_parameters": asdict(downstream_params),
        "literature_derived_physiology": {
            "electrical_coupling": "contralateral H2<->HSE/HSN/HSS term, enabled only in full control",
            "coupling_matrix": "row-normalized binary relation based on exact type and opposite somaSide, not MaleCNS anatomy",
        },
        "stimulus_classes": {
            "yaw_like_A": "left 01_07 -> 01_08; right 01_08 -> 01_07",
            "yaw_like_B": "left 01_08 -> 01_07; right 01_07 -> 01_08",
            "translation_like_A": "both eyes 01_07 -> 01_08",
            "translation_like_B": "both eyes 01_08 -> 01_07",
            "left_only_forward": "left 01_07 -> 01_08; right static",
            "left_only_reverse": "left 01_08 -> 01_07; right static",
            "right_only_forward": "right 01_07 -> 01_08; left static",
            "right_only_reverse": "right 01_08 -> 01_07; left static",
            "geometry_status": "Stimulus-space workbook order only; no canonical visual/body direction assigned.",
        },
        "full_response_metrics": full,
        "hierarchy_summary": {
            "yaw_DNp15_difference_area_mean_abs": float(yaw_area),
            "translation_DNp15_difference_area_mean_abs": float(translation_area),
            "yaw_to_translation_difference_ratio": float(yaw_area / max(translation_area, 1e-12)),
            "yaw_condition_difference_signs": [
                float(full["yaw_like_A"]["DNp15_bilateral_difference_area"]),
                float(full["yaw_like_B"]["DNp15_bilateral_difference_area"]),
            ],
            "translation_condition_difference_signs": [
                float(full["translation_like_A"]["DNp15_bilateral_difference_area"]),
                float(full["translation_like_B"]["DNp15_bilateral_difference_area"]),
            ],
        },
        "control_metrics": all_metrics,
        "interpretation": "EXP-004 evaluates neural response hierarchy only; no body or FlyGym result is included.",
        "outputs": [str(path.relative_to(args.output)) for path in args.output.rglob("*") if path.is_file()],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
