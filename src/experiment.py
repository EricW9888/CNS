"""Run the narrow adjacent-column motion-pathway experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .lif_model import LIFParameters, run_lif
from .malecns_io import ConnectomeGraph
from .observability import write_observability_outputs
from .visual_stimulus import motion_events


PROJECTION_TYPES = {"HSE", "HSN", "HSS", "HST", "VS", "VST1", "VST2", "VSm"}


def ablate_t4_projection(graph: ConnectomeGraph) -> ConnectomeGraph:
    """Remove T4->HS/VS chemical edges while preserving all other edges."""

    mask = ~(
        graph.edges["pre_type"].astype(str).str.startswith("T4")
        & graph.edges["post_type"].isin(PROJECTION_TYPES)
    )
    return graph.with_edges(graph.edges.loc[mask].copy())


def _condition_metrics(summary: dict, spikes: pd.DataFrame, condition: str, motion_order: str) -> dict:
    output = spikes[spikes["superclass"] == "descending_neuron"] if len(spikes) else spikes
    t4 = spikes[spikes["stage"] == "t4_motion"] if len(spikes) else spikes
    left = int((output["somaSide"] == "L").sum()) if len(output) else 0
    right = int((output["somaSide"] == "R").sum()) if len(output) else 0
    denominator = left + right
    return {
        "condition": condition,
        "motion_order": motion_order,
        "spike_count": summary["spike_count"],
        "active_nodes": summary["active_nodes"],
        "t4_spikes": int(len(t4)),
        "descending_spikes_left": left,
        "descending_spikes_right": right,
        "descending_lateralization": (left - right) / denominator if denominator else 0.0,
        "summary": summary,
    }


def run_experiment(
    graph: ConnectomeGraph,
    *,
    motion_columns: list[dict],
    output_dir: str | Path,
    duration_ms: float = 100.0,
    params: LIFParameters | None = None,
    amplitude_mV: float = 8.0,
) -> list[dict]:
    """Compare opposite temporal orders and one pathway ablation control."""

    if len(motion_columns) != 2:
        raise ValueError("The first experiment requires exactly two adjacent motion columns")
    first, second = motion_columns
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = params or LIFParameters()
    forward = motion_events(
        first_column=first["suffix"],
        first_l1_ids=(int(first["l1_body_id"]),),
        second_column=second["suffix"],
        second_l1_ids=(int(second["l1_body_id"]),),
        amplitude_mV=amplitude_mV,
    )
    reverse = motion_events(
        first_column=first["suffix"],
        first_l1_ids=(int(first["l1_body_id"]),),
        second_column=second["suffix"],
        second_l1_ids=(int(second["l1_body_id"]),),
        reverse=True,
        amplitude_mV=amplitude_mV,
    )
    conditions = [
        (f"motion_{first['suffix']}_to_{second['suffix']}", forward, graph, f"{first['suffix']}->{second['suffix']}"),
        (f"motion_{second['suffix']}_to_{first['suffix']}", reverse, graph, f"{second['suffix']}->{first['suffix']}"),
        (
            f"motion_{first['suffix']}_to_{second['suffix']}_t4_projection_ablation",
            forward,
            ablate_t4_projection(graph),
            f"{first['suffix']}->{second['suffix']}",
        ),
    ]
    results: list[dict] = []
    condition_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    trace_node_ids = tuple(
        int(value)
        for value in graph.nodes.loc[graph.nodes["stage"].isin(["medulla", "t4_motion"]), "bodyId"]
    )
    for index, (name, events, condition_graph, motion_order) in enumerate(conditions):
        spikes, summary, trace = run_lif(
            condition_graph,
            events,
            duration_ms=duration_ms,
            params=params,
            seed=index,
            trace_node_ids=trace_node_ids,
            return_trace=True,
        )
        spikes.to_csv(output_dir / f"spikes_{name}.csv", index=False)
        results.append(_condition_metrics(summary, spikes, name, motion_order))
        if name in {conditions[0][0], conditions[1][0]}:
            condition_outputs[name] = (spikes, trace)

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{key: value for key, value in result.items() if key != "summary"} for result in results]
    ).to_csv(output_dir / "metrics.csv", index=False)
    write_observability_outputs(
        graph=graph,
        motion_columns=motion_columns,
        forward_name=conditions[0][0],
        reverse_name=conditions[1][0],
        forward_events=forward,
        reverse_events=reverse,
        forward_spikes=condition_outputs[conditions[0][0]][0],
        reverse_spikes=condition_outputs[conditions[1][0]][0],
        forward_trace=condition_outputs[conditions[0][0]][1],
        reverse_trace=condition_outputs[conditions[1][0]][1],
        output_dir=output_dir,
        params=params,
    )
    return results
