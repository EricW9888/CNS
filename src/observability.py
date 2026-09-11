"""Plots and neuron-level diagnostics for the fixed MaleCNS motion run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .lif_model import LIFParameters
from .malecns_io import ConnectomeGraph


STAGE_ORDER = [
    "visual_input",
    "medulla",
    "t4_motion",
    "wide_field_projection",
    "descending_output",
]
STAGE_LABELS = {
    "visual_input": "L1",
    "medulla": "Mi1/Tm3",
    "t4_motion": "T4a-d",
    "wide_field_projection": "wide-field",
    "descending_output": "descending",
}
STAGE_COLORS = {
    "visual_input": "#2ca02c",
    "medulla": "#1f77b4",
    "t4_motion": "#9467bd",
    "wide_field_projection": "#ff7f0e",
    "descending_output": "#d62728",
}
STIMULUS_COLORS = ("#17becf", "#bcbd22")


def _ordered_nodes(graph: ConnectomeGraph) -> pd.DataFrame:
    nodes = graph.nodes.copy()
    rank = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    nodes["_stage_rank"] = nodes["stage"].map(rank).fillna(len(rank)).astype(int)
    return nodes.sort_values(["_stage_rank", "type", "bodyId"], na_position="last").drop(
        columns="_stage_rank"
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compare_t4_voltage(
    forward_trace: pd.DataFrame,
    reverse_trace: pd.DataFrame,
    forward_spikes: pd.DataFrame,
    reverse_spikes: pd.DataFrame,
) -> pd.DataFrame:
    """Compare forward/reverse voltage traces for every traced T4 neuron."""

    forward = forward_trace[forward_trace["stage"] == "t4_motion"].copy()
    reverse = reverse_trace[reverse_trace["stage"] == "t4_motion"].copy()
    if forward.empty or reverse.empty:
        raise ValueError("Both conditions must contain T4 voltage traces")
    f_columns = ["bodyId", "time_ms", "voltage_mV"]
    r_columns = ["bodyId", "time_ms", "voltage_mV"]
    merged = forward[f_columns].rename(columns={"voltage_mV": "voltage_forward_mV"}).merge(
        reverse[r_columns].rename(columns={"voltage_mV": "voltage_reverse_mV"}),
        on=["bodyId", "time_ms"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Forward and reverse traces have no shared T4 samples")

    metadata = (
        forward[["bodyId", "type", "instance", "somaSide"]]
        .drop_duplicates("bodyId")
        .set_index("bodyId")
    )
    forward_spike_counts = forward_spikes[forward_spikes["stage"] == "t4_motion"].groupby("bodyId").size()
    reverse_spike_counts = reverse_spikes[reverse_spikes["stage"] == "t4_motion"].groupby("bodyId").size()
    rows: list[dict] = []
    for body_id, group in merged.groupby("bodyId", sort=False):
        group = group.sort_values("time_ms")
        difference = group["voltage_forward_mV"].to_numpy() - group["voltage_reverse_mV"].to_numpy()
        abs_difference = np.abs(difference)
        max_index = int(np.argmax(abs_difference))
        info = metadata.loc[body_id]
        rows.append(
            {
                "bodyId": int(body_id),
                "type": info["type"],
                "instance": info["instance"],
                "somaSide": info["somaSide"],
                "max_voltage_forward_mV": float(group["voltage_forward_mV"].max()),
                "max_voltage_reverse_mV": float(group["voltage_reverse_mV"].max()),
                "max_voltage_difference_mV": float(
                    group["voltage_forward_mV"].max() - group["voltage_reverse_mV"].max()
                ),
                "max_abs_pointwise_voltage_difference_mV": float(abs_difference[max_index]),
                "time_of_max_abs_difference_ms": float(group.iloc[max_index]["time_ms"]),
                "spike_count_forward": int(forward_spike_counts.get(body_id, 0)),
                "spike_count_reverse": int(reverse_spike_counts.get(body_id, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_abs_pointwise_voltage_difference_mV", "bodyId"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _trace_matrix(trace: pd.DataFrame, body_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    selected = trace[trace["bodyId"].isin(list(body_ids))].copy()
    selected = selected.sort_values(["bodyId", "time_ms"])
    if selected.empty:
        return np.array([], dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    times = np.sort(selected["time_ms"].unique())
    ids = list(body_ids)
    values = np.full((len(ids), len(times)), np.nan, dtype=np.float32)
    time_index = {float(value): index for index, value in enumerate(times)}
    id_index = {int(value): index for index, value in enumerate(ids)}
    for row in selected.itertuples(index=False):
        values[id_index[int(row.bodyId)], time_index[float(row.time_ms)]] = float(row.voltage_mV)
    return times.astype(np.float32), values


def plot_pathway(
    graph: ConnectomeGraph,
    motion_columns: list[dict],
    path: Path,
) -> None:
    nodes = _ordered_nodes(graph)
    stage_x = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    positions: dict[int, tuple[float, float]] = {}
    for stage in STAGE_ORDER:
        stage_nodes = nodes[nodes["stage"] == stage]
        if stage_nodes.empty:
            continue
        y_values = np.linspace(0.96, 0.04, len(stage_nodes)) if len(stage_nodes) > 1 else [0.5]
        for y, row in zip(y_values, stage_nodes.itertuples(index=False)):
            positions[int(row.bodyId)] = (float(stage_x[stage]), float(y))

    synapses = graph.edges[graph.edges["pre_body"].isin(positions) & graph.edges["post_body"].isin(positions)]
    log_synapses = np.log1p(synapses["synapse_count"].to_numpy(dtype=float)) if len(synapses) else np.array([1.0])
    min_synapse = float(log_synapses.min())
    max_synapse = float(log_synapses.max())
    fig, ax = plt.subplots(figsize=(19, 14))
    for row in synapses.itertuples(index=False):
        x0, y0 = positions[int(row.pre_body)]
        x1, y1 = positions[int(row.post_body)]
        scaled = (np.log1p(float(row.synapse_count)) - min_synapse) / max(max_synapse - min_synapse, 1e-9)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-", "lw": 0.25 + 2.2 * scaled, "color": "#7f7f7f", "alpha": 0.22},
        )

    stimulus_by_body = {
        int(column["l1_body_id"]): str(column["suffix"])
        for column in motion_columns
    }
    for stage in STAGE_ORDER:
        stage_nodes = nodes[nodes["stage"] == stage]
        for row in stage_nodes.itertuples(index=False):
            body_id = int(row.bodyId)
            x, y = positions[body_id]
            color = STAGE_COLORS.get(stage, "#7f7f7f")
            edgecolor = "#333333"
            size = 44 if stage != "descending_output" else 24
            if body_id in stimulus_by_body:
                color = STIMULUS_COLORS[list(stimulus_by_body.values()).index(stimulus_by_body[body_id])]
                edgecolor = "#000000"
                size = 100
            ax.scatter([x], [y], s=size, c=[color], edgecolors=edgecolor, linewidths=0.7, zorder=3)
            if stage in {"visual_input", "t4_motion"} or body_id in stimulus_by_body:
                label = f"{row.type} {body_id}"
                if body_id in stimulus_by_body:
                    label = f"{stimulus_by_body[body_id]}: {label}"
                ax.text(x + 0.06, y, label, va="center", fontsize=5.5, zorder=4)

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_LABELS[stage] for stage in STAGE_ORDER], fontsize=11)
    ax.set_xlim(-0.4, len(STAGE_ORDER) - 0.6)
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("ordered neurons within stage")
    ax.set_title("MaleCNS v1.0 selected motion pathway\nedge width ∝ log(1 + synapse count)")
    ax.grid(axis="x", alpha=0.15)
    stage_handles = [Patch(facecolor=STAGE_COLORS[stage], label=STAGE_LABELS[stage]) for stage in STAGE_ORDER]
    stimulus_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="#000000",
               markersize=8, label=f"stimulated {column['suffix']} (L1 {column['l1_body_id']})")
        for color, column in zip(STIMULUS_COLORS, motion_columns)
    ]
    ax.legend(handles=stage_handles + stimulus_handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    _save_figure(fig, path)


def plot_spike_rasters(
    graph: ConnectomeGraph,
    forward_spikes: pd.DataFrame,
    reverse_spikes: pd.DataFrame,
    path: Path,
    duration_ms: float,
) -> None:
    nodes = _ordered_nodes(graph)
    body_ids = nodes["bodyId"].astype(int).tolist()
    body_to_row = {body_id: index for index, body_id in enumerate(body_ids)}
    labels = [f"{row.stage} | {row.type} | {int(row.bodyId)}" for row in nodes.itertuples(index=False)]
    fig, axes = plt.subplots(1, 2, figsize=(20, 16), sharey=True)
    for ax, spikes, title in zip(axes, (forward_spikes, reverse_spikes), ("forward order", "reverse order")):
        for stage in STAGE_ORDER:
            stage_ids = nodes.loc[nodes["stage"] == stage, "bodyId"].astype(int)
            if len(stage_ids):
                ymin, ymax = body_to_row[int(stage_ids.iloc[0])], body_to_row[int(stage_ids.iloc[-1])]
                ax.axhspan(ymin - 0.5, ymax + 0.5, color=STAGE_COLORS[stage], alpha=0.045, zorder=0)
        if len(spikes):
            rows = spikes["bodyId"].astype(int).map(body_to_row)
            ax.scatter(spikes["time_ms"], rows, s=13, color="#111111", marker="|", linewidths=0.8, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("time (ms)")
        ax.set_xlim(0, duration_ms)
        ax.set_yticks(range(len(body_ids)))
        ax.set_yticklabels(labels, fontsize=4)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_ylabel("stage | type | bodyId")
    fig.suptitle("Forward/reverse spike rasters grouped by pathway stage and neuron type", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save_figure(fig, path)


def plot_voltage_condition(
    trace: pd.DataFrame,
    condition: str,
    path: Path,
    threshold_mV: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    for ax, stage, title in zip(axes, ("medulla", "t4_motion"), ("Mi1/Tm3 membrane voltage", "T4 membrane voltage")):
        subset = trace[trace["stage"] == stage]
        types = sorted(subset["type"].dropna().astype(str).unique())
        type_colors = {cell_type: plt.get_cmap("tab10")(index % 10) for index, cell_type in enumerate(types)}
        for body_id, group in subset.groupby("bodyId", sort=True):
            group = group.sort_values("time_ms")
            cell_type = str(group["type"].iloc[0])
            ax.plot(
                group["time_ms"],
                group["voltage_mV"],
                color=type_colors[cell_type],
                alpha=0.72 if stage == "medulla" else 0.28,
                linewidth=1.1 if stage == "medulla" else 0.7,
                label=f"{cell_type} {int(body_id)}",
            )
        ax.axhline(
            threshold_mV,
            color="#555555",
            linestyle="--",
            linewidth=0.8,
            label=f"threshold {threshold_mV:g} mV",
        )
        ax.set_title(f"{title} ({len(subset['bodyId'].unique())} neurons)")
        ax.set_ylabel("voltage (mV)")
        ax.grid(alpha=0.2)
        if len(subset):
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6, ncol=2)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(f"{condition}: preserved subthreshold membrane traces", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.83, 0.985))
    _save_figure(fig, path)


def plot_t4_ranked(comparison: pd.DataFrame, path: Path) -> None:
    ranked = comparison.sort_values("max_abs_pointwise_voltage_difference_mV", ascending=True)
    labels = [f"{row.type} {int(row.bodyId)}" for row in ranked.itertuples(index=False)]
    values = ranked["max_abs_pointwise_voltage_difference_mV"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 13))
    type_colors = {
        cell_type: plt.get_cmap("tab10")(index % 10)
        for index, cell_type in enumerate(sorted(ranked["type"].astype(str).unique()))
    }
    ax.barh(labels, values, color=[type_colors[str(cell_type)] for cell_type in ranked["type"]])
    ax.set_xlabel("max |Vforward(t) − Vreverse(t)| (mV)")
    ax.set_ylabel("T4 neuron type and bodyId")
    ax.set_title("T4 neurons ranked by temporal-order voltage difference")
    ax.grid(axis="x", alpha=0.2)
    for index, row in enumerate(ranked.itertuples(index=False)):
        if index >= max(0, len(ranked) - 10):
            ax.text(float(row.max_abs_pointwise_voltage_difference_mV), index, f" {row.max_abs_pointwise_voltage_difference_mV:.4g}", va="center", fontsize=7)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_t4_heatmap(
    forward_trace: pd.DataFrame,
    reverse_trace: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    body_ids = comparison["bodyId"].astype(int).tolist()
    times, forward_values = _trace_matrix(forward_trace, body_ids)
    _, reverse_values = _trace_matrix(reverse_trace, body_ids)
    difference = forward_values - reverse_values
    fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
    panels = (
        (forward_values, "viridis", "forward voltage (mV)"),
        (reverse_values, "viridis", "reverse voltage (mV)"),
        (difference, "coolwarm", "forward − reverse voltage (mV)"),
    )
    labels = [f"{row.type} {int(row.bodyId)}" for row in comparison.itertuples(index=False)]
    for ax, (values, cmap, title) in zip(axes, panels):
        if values.size:
            if np.nanmax(values) == np.nanmin(values):
                half_range = 1e-6
                vmin, vmax = float(np.nanmin(values) - half_range), float(np.nanmax(values) + half_range)
            elif title.startswith("forward −"):
                limit = float(np.nanmax(np.abs(values)))
                vmin, vmax = -limit, limit
            else:
                vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
            image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
                              extent=[float(times[0]), float(times[-1]), len(body_ids) - 0.5, -0.5])
            fig.colorbar(image, ax=ax, pad=0.01, label="mV")
        ax.set_title(title)
        ax.set_ylabel("T4 neuron")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("T4 temporal-order heatmap", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save_figure(fig, path)


def write_observability_outputs(
    graph: ConnectomeGraph,
    motion_columns: list[dict],
    forward_name: str,
    reverse_name: str,
    forward_events: tuple,
    reverse_events: tuple,
    forward_spikes: pd.DataFrame,
    reverse_spikes: pd.DataFrame,
    forward_trace: pd.DataFrame,
    reverse_trace: pd.DataFrame,
    output_dir: str | Path,
    params: LIFParameters,
) -> dict:
    """Persist raw traces, comparisons, figures, and a JSON diagnosis summary."""

    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    forward_trace.to_csv(output_dir / "voltage_traces_forward.csv", index=False)
    reverse_trace.to_csv(output_dir / "voltage_traces_reverse.csv", index=False)

    nodes = _ordered_nodes(graph)
    node_metadata = nodes[
        ["bodyId", "type", "instance", "somaSide", "superclass", "stage"]
    ].copy()
    stimulus_by_body = {int(item["l1_body_id"]): item["suffix"] for item in motion_columns}
    node_metadata["stimulated_column"] = node_metadata["bodyId"].map(stimulus_by_body)
    node_metadata.to_csv(output_dir / "pathway_nodes.csv", index=False)

    comparison = compare_t4_voltage(forward_trace, reverse_trace, forward_spikes, reverse_spikes)
    comparison.to_csv(output_dir / "t4_forward_minus_reverse.csv", index=False)
    comparison.to_json(output_dir / "t4_forward_minus_reverse.json", orient="records", indent=2)

    plot_pathway(graph, motion_columns, figure_dir / "pathway_subgraph.png")
    plot_spike_rasters(
        graph,
        forward_spikes,
        reverse_spikes,
        figure_dir / "spike_rasters_forward_reverse.png",
        duration_ms=float(forward_trace["time_ms"].max()) if len(forward_trace) else 0.0,
    )
    plot_voltage_condition(
        forward_trace,
        forward_name,
        figure_dir / "voltage_forward_mitm_t4.png",
        params.v_threshold_mV,
    )
    plot_voltage_condition(
        reverse_trace,
        reverse_name,
        figure_dir / "voltage_reverse_mitm_t4.png",
        params.v_threshold_mV,
    )
    plot_t4_ranked(comparison, figure_dir / "t4_order_sensitivity_ranked.png")
    plot_t4_heatmap(
        forward_trace,
        reverse_trace,
        comparison,
        figure_dir / "t4_heatmap_forward_reverse_difference.png",
    )

    def excursion(trace: pd.DataFrame, stage: str) -> float:
        values = trace.loc[trace["stage"] == stage, "voltage_mV"].to_numpy(dtype=float)
        return float(np.max(np.abs(values - params.v_rest_mV))) if len(values) else 0.0

    def order_difference(stage: str) -> float:
        forward_stage = forward_trace[forward_trace["stage"] == stage][["bodyId", "time_ms", "voltage_mV"]]
        reverse_stage = reverse_trace[reverse_trace["stage"] == stage][["bodyId", "time_ms", "voltage_mV"]]
        merged = forward_stage.merge(
            reverse_stage,
            on=["bodyId", "time_ms"],
            suffixes=("_forward", "_reverse"),
            how="inner",
        )
        if merged.empty:
            return 0.0
        return float(
            np.max(np.abs(merged["voltage_mV_forward"] - merged["voltage_mV_reverse"]))
        )

    t4_threshold = 1e-7
    summary = {
        "forward_condition": forward_name,
        "reverse_condition": reverse_name,
        "motion_columns": motion_columns,
        "events_forward": [dict(event.__dict__) for event in forward_events],
        "events_reverse": [dict(event.__dict__) for event in reverse_events],
        "trace_node_counts": {
            "medulla_Mi1_Tm3": int(forward_trace.loc[forward_trace["stage"] == "medulla", "bodyId"].nunique()),
            "t4": int(forward_trace.loc[forward_trace["stage"] == "t4_motion", "bodyId"].nunique()),
        },
        "max_abs_voltage_excursion_from_rest_mV": {
            "medulla_Mi1_Tm3": excursion(forward_trace, "medulla"),
            "medulla_Mi1_Tm3_reverse": excursion(reverse_trace, "medulla"),
            "t4_forward": excursion(forward_trace, "t4_motion"),
            "t4_reverse": excursion(reverse_trace, "t4_motion"),
        },
        "max_abs_forward_reverse_voltage_difference_mV": {
            "medulla_Mi1_Tm3": order_difference("medulla"),
            "t4": order_difference("t4_motion"),
        },
        "t4_neurons_with_order_dependent_voltage_difference": int(
            (comparison["max_abs_pointwise_voltage_difference_mV"] > t4_threshold).sum()
        ),
        "max_t4_order_dependent_voltage_difference_mV": float(
            comparison["max_abs_pointwise_voltage_difference_mV"].max()
        ),
        "t4_neurons_with_forward_spikes": int((comparison["spike_count_forward"] > 0).sum()),
        "t4_neurons_with_reverse_spikes": int((comparison["spike_count_reverse"] > 0).sum()),
        "top_10_t4": comparison.head(10).to_dict(orient="records"),
        "outputs": {
            "figures_dir": str(figure_dir),
            "forward_trace_csv": str(output_dir / "voltage_traces_forward.csv"),
            "reverse_trace_csv": str(output_dir / "voltage_traces_reverse.csv"),
            "t4_comparison_csv": str(output_dir / "t4_forward_minus_reverse.csv"),
            "t4_comparison_json": str(output_dir / "t4_forward_minus_reverse.json"),
        },
    }
    (output_dir / "observability_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: value.item() if hasattr(value, "item") else value),
        encoding="utf-8",
    )
    return summary
