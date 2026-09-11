"""Run the first graded, local T4 direction-selectivity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graded_model import (
    EXCITATORY_T4_INPUTS,
    INHIBITORY_T4_INPUTS,
    GradedParameters,
    run_graded,
    t4_comparison,
)
from src.malecns_io import build_graph, load_edges, load_transmitters
from src.visual_stimulus import motion_events


def _load_graph(bundle: Path, transmitters: Path):
    nodes = pd.read_parquet(bundle / "nodes.parquet")
    edges = load_edges(bundle / "edges.parquet")
    tx = load_transmitters(transmitters) if transmitters.exists() else None
    return build_graph(nodes, edges, tx)


def _type_summary(trace: pd.DataFrame) -> pd.DataFrame:
    t4 = trace[trace["stage"] == "t4_motion"].copy()
    per_neuron = (
        t4.groupby(["bodyId", "type"], as_index=False)
        .agg(max_activity=("activity_proxy", "max"), min_activity=("activity_proxy", "min"))
    )
    return (
        per_neuron.groupby("type", as_index=False)
        .agg(
            neurons=("bodyId", "count"),
            mean_max_activity=("max_activity", "mean"),
            max_activity=("max_activity", "max"),
            min_activity=("min_activity", "min"),
        )
        .sort_values("type")
    )


def _heatmap_data(trace: pd.DataFrame, order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    t4 = trace[trace["stage"] == "t4_motion"]
    pivot = t4.pivot(index="bodyId", columns="time_ms", values="activity_proxy")
    return pivot.reindex(order).to_numpy(), pivot.columns.to_numpy()


def _plot_circuit(ax: plt.Axes, graph, motion_columns: list[dict]) -> None:
    type_x = {
        "L1": 0.0,
        "Mi1": 0.82,
        "Tm3": 0.98,
        "Mi4": 1.20,
        "Mi9": 1.36,
        "C3": 1.52,
        "CT1": 1.68,
        "T4a": 1.96,
        "T4b": 2.06,
        "T4c": 2.16,
        "T4d": 2.26,
    }
    nodes = graph.nodes.copy()
    type_order = {
        "visual_input": ["L1"],
        "medulla_excitation": ["Mi1", "Tm3"],
        "t4_input_inhibition": ["Mi4", "Mi9", "C3", "CT1"],
        "t4_motion": ["T4a", "T4b", "T4c", "T4d"],
    }
    positions: dict[int, tuple[float, float]] = {}
    for stage, types in type_order.items():
        for type_name in types:
            ids = nodes.loc[(nodes["stage"] == stage) & (nodes["type"] == type_name), "bodyId"].tolist()
            if not ids:
                continue
            ys = np.linspace(0.9, 0.1, len(ids))
            x = type_x[type_name]
            for body_id, y in zip(ids, ys):
                positions[int(body_id)] = (x, float(y))
            if stage != "t4_motion":
                ax.text(x, 1.03, f"{type_name}\n(n={len(ids)})", ha="center", va="bottom", fontsize=7)
    t4_count = int((nodes["stage"] == "t4_motion").sum())
    ax.text(2.1, 1.03, f"T4a-d\n(n={t4_count})", ha="center", va="bottom", fontsize=8)

    aggregate: dict[tuple[str, str], float] = {}
    for row in graph.edges.itertuples(index=False):
        key = (str(row.pre_type), str(row.post_type))
        aggregate[key] = aggregate.get(key, 0.0) + float(row.synapse_count)
    for (pre_type, post_type), synapses in aggregate.items():
        pre_ids = nodes.loc[nodes["type"] == pre_type, "bodyId"].astype(int).tolist()
        post_ids = nodes.loc[nodes["type"] == post_type, "bodyId"].astype(int).tolist()
        if not pre_ids or not post_ids or pre_ids[0] not in positions or post_ids[0] not in positions:
            continue
        x0, y0 = positions[pre_ids[0]]
        x1, y1 = positions[post_ids[0]]
        color = "#2ca02c" if pre_type in EXCITATORY_T4_INPUTS else "#d62728" if pre_type in INHIBITORY_T4_INPUTS else "#555555"
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "lw": 0.6 + np.log1p(synapses) / 2.0, "alpha": 0.35, "color": color},
        )
    for body_id, (x, y) in positions.items():
        node = nodes.loc[nodes["bodyId"] == body_id].iloc[0]
        color = "#1f77b4" if node["stage"] == "visual_input" else "#2ca02c" if node["type"] in EXCITATORY_T4_INPUTS else "#d62728" if node["type"] in INHIBITORY_T4_INPUTS else "#9467bd"
        ax.scatter([x], [y], s=12 if node["stage"] != "visual_input" else 35, color=color, zorder=3)
    ax.text(0, -0.05, f"L1 {motion_columns[0]['l1_body_id']} / {motion_columns[1]['l1_body_id']}", ha="center", fontsize=8)
    ax.set_xlim(-0.3, 2.55)
    ax.set_ylim(-0.15, 1.12)
    ax.set_xticks([0.0, 0.9, 1.45, 2.1], ["L1", "Mi1/Tm3", "inhibitory\ninputs", "T4"])
    ax.set_yticks([])
    ax.set_title("MaleCNS-derived local circuit\nedge width ∝ log(1 + synapse count)", fontsize=10)
    ax.grid(False)


def _plot_main(
    path: Path,
    graph,
    motion_columns: list[dict],
    forward: pd.DataFrame,
    reverse: pd.DataFrame,
    ablated: pd.DataFrame,
    reverse_ablated: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    order_frame = graph.nodes[graph.nodes["stage"] == "t4_motion"].sort_values(["type", "bodyId"])
    order = order_frame["bodyId"].astype(int).tolist()
    f_heat, times = _heatmap_data(forward, order)
    r_heat, _ = _heatmap_data(reverse, order)
    difference = f_heat - r_heat
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    _plot_circuit(axes[0, 0], graph, motion_columns)

    axes[0, 1].step([0, 10, 20, 30, 150], [0, 1, 0, 1, 0], where="post", color="#1f77b4", lw=2)
    axes[0, 1].set_title(
        f"Same two-column edge, reversed\n{motion_columns[0]['suffix']}→{motion_columns[1]['suffix']} vs reverse"
    )
    axes[0, 1].set_xlabel("time (ms)")
    axes[0, 1].set_ylabel("normalized ON contrast")
    axes[0, 1].set_xlim(0, 150)
    axes[0, 1].grid(alpha=0.2)

    vmax = float(max(np.max(np.abs(f_heat)), np.max(np.abs(r_heat)), 1e-6))
    for ax, values, title in (
        (axes[1, 0], f_heat, "Forward graded T4 activity"),
        (axes[1, 1], r_heat, "Reverse graded T4 activity"),
    ):
        image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis", vmin=-vmax, vmax=vmax, extent=[times[0], times[-1], len(order), 0])
        ax.set_title(title)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("T4 neurons (a/b/c/d, body ID order)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="graded activity proxy")

    diff_max = float(max(np.max(np.abs(difference)), 1e-6))
    image = axes[2, 0].imshow(difference, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-diff_max, vmax=diff_max, extent=[times[0], times[-1], len(order), 0])
    axes[2, 0].set_title("Forward − reverse T4 activity")
    axes[2, 0].set_xlabel("time (ms)")
    axes[2, 0].set_ylabel("T4 neurons (a/b/c/d, body ID order)")
    fig.colorbar(image, ax=axes[2, 0], fraction=0.046, pad=0.04, label="difference")

    def mean_max(trace: pd.DataFrame) -> pd.Series:
        per_neuron = trace[trace["stage"] == "t4_motion"].groupby(["bodyId", "type"])["activity_proxy"].max()
        return per_neuron.groupby(level="type").mean()

    types = ["T4a", "T4b", "T4c", "T4d"]
    x = np.arange(len(types))
    width = 0.25
    axes[2, 1].bar(x - width, mean_max(forward).reindex(types), width, label="forward")
    axes[2, 1].bar(x, mean_max(reverse).reindex(types), width, label="reverse")
    axes[2, 1].bar(x + width, mean_max(ablated).reindex(types), width, label="forward, inhibitory ablation")
    axes[2, 1].set_xticks(x, types)
    axes[2, 1].set_title("T4 subtype response and causal ablation")
    axes[2, 1].set_ylabel("mean peak graded activity")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(axis="y", alpha=0.2)
    fig.suptitle("EXP-002: MaleCNS-derived graded T4 motion circuit", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="data/malecns_exp002_local_circuit")
    parser.add_argument("--transmitters", default="data/body-neurotransmitters.feather")
    parser.add_argument("--output", default="results/exp002")
    parser.add_argument("--duration-ms", type=float, default=150.0)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    graph = _load_graph(bundle, Path(args.transmitters))
    motion_columns = manifest["motion_columns"]
    forward_events = motion_events(
        first_column=motion_columns[0]["suffix"],
        first_l1_ids=(int(motion_columns[0]["l1_body_id"]),),
        second_column=motion_columns[1]["suffix"],
        second_l1_ids=(int(motion_columns[1]["l1_body_id"]),),
        amplitude_mV=1.0,
    )
    reverse_events = motion_events(
        first_column=motion_columns[0]["suffix"],
        first_l1_ids=(int(motion_columns[0]["l1_body_id"]),),
        second_column=motion_columns[1]["suffix"],
        second_l1_ids=(int(motion_columns[1]["l1_body_id"]),),
        reverse=True,
        amplitude_mV=1.0,
    )
    params = GradedParameters()
    forward, forward_summary = run_graded(
        graph, forward_events, motion_columns=motion_columns, params=params, duration_ms=args.duration_ms
    )
    reverse, reverse_summary = run_graded(
        graph, reverse_events, motion_columns=motion_columns, params=params, duration_ms=args.duration_ms
    )
    ablated, ablated_summary = run_graded(
        graph,
        forward_events,
        motion_columns=motion_columns,
        params=params,
        duration_ms=args.duration_ms,
        ablate_inhibitory=True,
    )
    reverse_ablated, reverse_ablated_summary = run_graded(
        graph,
        reverse_events,
        motion_columns=motion_columns,
        params=params,
        duration_ms=args.duration_ms,
        ablate_inhibitory=True,
    )
    forward.to_csv(output / "graded_traces_forward.csv", index=False)
    reverse.to_csv(output / "graded_traces_reverse.csv", index=False)
    ablated.to_csv(output / "graded_traces_forward_inhibitory_ablation.csv", index=False)
    reverse_ablated.to_csv(output / "graded_traces_reverse_inhibitory_ablation.csv", index=False)
    comparison = t4_comparison(forward, reverse)
    comparison.to_csv(output / "t4_direction_comparison.csv", index=False)
    _plot_main(figure_dir / "exp002_direction_selectivity.png", graph, motion_columns, forward, reverse, ablated, reverse_ablated, comparison)

    ranked = comparison.copy()
    ranked["label"] = ranked["type"] + " " + ranked["bodyId"].astype(str)
    top = ranked.head(20).sort_values("max_abs_pointwise_activity_difference")
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.barh(top["label"], top["max_abs_pointwise_activity_difference"], color="#9467bd")
    ax.set_title("EXP-002 top T4 neurons by forward/reverse difference")
    ax.set_xlabel("max |forward − reverse|")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(figure_dir / "exp002_t4_ranked_difference.png", dpi=180)
    plt.close(fig)

    per_type = comparison.groupby("type", as_index=False).agg(
        neurons=("bodyId", "count"),
        mean_max_abs_difference=("max_abs_pointwise_activity_difference", "mean"),
        max_abs_difference=("max_abs_pointwise_activity_difference", "max"),
        mean_forward_area=("positive_area_forward", "mean"),
        mean_reverse_area=("positive_area_reverse", "mean"),
    )
    metrics = {
        "experiment_id": "EXP-002",
        "question": "Can a MaleCNS-derived T4 motion circuit distinguish opposite visual motion directions?",
        "model": forward_summary,
        "forward": forward_summary,
        "reverse": reverse_summary,
        "forward_inhibitory_ablation": ablated_summary,
        "reverse_inhibitory_ablation": reverse_ablated_summary,
        "t4_neurons": int(len(comparison)),
        "t4_neurons_with_nonzero_order_difference": int((comparison["max_abs_pointwise_activity_difference"] > 1e-7).sum()),
        "top_t4": comparison.head(10).to_dict(orient="records"),
        "by_type": per_type.to_dict(orient="records"),
        "outputs": {
            "figure": str(figure_dir / "exp002_direction_selectivity.png"),
            "comparison": str(output / "t4_direction_comparison.csv"),
            "ranked_figure": str(figure_dir / "exp002_t4_ranked_difference.png"),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_type.to_csv(output / "t4_direction_by_type.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
