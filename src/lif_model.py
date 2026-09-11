"""Transparent sparse LIF baseline for the narrow MaleCNS experiment.

The equations mirror the Brian2/Shiu baseline, but the runnable backend is a
small NumPy/SciPy integrator.  Brian2 2.9.0 currently fails during import with
NumPy 2.x before any model code runs; that failure is recorded in
``docs/diagnostics/brian2_numpy2.md`` rather than hidden by a NumPy downgrade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .malecns_io import ConnectomeGraph, effective_sparse_matrix
from .visual_stimulus import VisualEvent, VisualPulse, compile_events, event_dicts


@dataclass(frozen=True)
class LIFParameters:
    """Parameters taken from the published Shiu et al. baseline where possible."""

    v_rest_mV: float = -52.0
    v_threshold_mV: float = -45.0
    tau_membrane_ms: float = 20.0
    tau_synapse_ms: float = 5.0
    refractory_ms: float = 2.2
    synaptic_delay_ms: float = 1.8
    weight_per_synapse_mV: float = 0.275
    dt_ms: float = 0.1


def _empty_spikes() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "node_index", "bodyId", "time_ms", "type", "instance",
            "somaSide", "superclass", "stage",
        ]
    )


def run_lif(
    graph: ConnectomeGraph,
    events: tuple[VisualEvent, ...] | list[VisualEvent] | VisualPulse,
    *,
    duration_ms: float = 100.0,
    params: LIFParameters | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Run one deterministic event condition and summarize spikes.

    ``coupling`` is post-by-pre.  A spike schedules one delayed vector-matrix
    product, preserving the real queried edge weights and transmitter signs.
    """

    del seed  # deterministic baseline; retained in the public call signature
    params = params or LIFParameters()
    if params.dt_ms <= 0 or duration_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")

    coupling, used_edges = effective_sparse_matrix(
        graph, weight_per_synapse_mV=params.weight_per_synapse_mV
    )
    node_index = {int(body_id): i for i, body_id in enumerate(graph.node_ids)}
    event_tuple = (events,) if isinstance(events, VisualEvent) else tuple(events)
    drive = compile_events(
        event_tuple,
        node_index=node_index,
        n_nodes=graph.n_nodes,
        duration_ms=duration_ms,
        dt_ms=params.dt_ms,
    )

    n_steps = drive.shape[0]
    dt = float(params.dt_ms)
    delay_steps = max(0, int(round(params.synaptic_delay_ms / dt)))
    v = np.full(graph.n_nodes, params.v_rest_mV, dtype=np.float32)
    g = np.zeros(graph.n_nodes, dtype=np.float32)
    refractory = np.zeros(graph.n_nodes, dtype=np.float32)
    pending = [np.zeros(graph.n_nodes, dtype=np.float32) for _ in range(delay_steps + 1)]
    spike_rows: list[dict] = []
    max_voltage = np.full(graph.n_nodes, params.v_rest_mV, dtype=np.float32)
    spike_mask = np.zeros(graph.n_nodes, dtype=bool)

    for step in range(n_steps):
        slot = step % (delay_steps + 1)
        g += pending[slot]
        pending[slot].fill(0.0)
        g += (-g / params.tau_synapse_ms) * dt
        refractory = np.maximum(0.0, refractory - dt)
        active = refractory <= 0.0
        v[active] += (
            (params.v_rest_mV - v[active] + g[active] + drive[step, active])
            / params.tau_membrane_ms
        ) * dt
        v[~active] = params.v_rest_mV
        max_voltage = np.maximum(max_voltage, v)
        spike_mask = active & (v >= params.v_threshold_mV)
        if np.any(spike_mask):
            for node_index_value in np.flatnonzero(spike_mask):
                spike_rows.append(
                    {
                        "node_index": int(node_index_value),
                        "bodyId": int(graph.node_ids[node_index_value]),
                        "time_ms": float(step * dt),
                    }
                )
            v[spike_mask] = params.v_rest_mV
            g[spike_mask] = 0.0
            refractory[spike_mask] = params.refractory_ms
            arrival_slot = (step + delay_steps) % (delay_steps + 1)
            pending[arrival_slot] += coupling @ spike_mask.astype(np.float32)

    spikes = pd.DataFrame(spike_rows) if spike_rows else _empty_spikes()
    if len(spikes):
        annotations = graph.nodes.copy()
        annotations["node_index"] = np.arange(graph.n_nodes)
        spikes = spikes.merge(
            annotations[["node_index", "type", "instance", "somaSide", "superclass", "stage"]],
            on="node_index",
            how="left",
        )
    else:
        for column in ("type", "instance", "somaSide", "superclass", "stage"):
            spikes[column] = pd.Series(dtype="object")

    stage_max = (
        graph.nodes.assign(max_voltage_mV=max_voltage)
        .groupby("stage", dropna=False)["max_voltage_mV"]
        .max()
        .fillna(params.v_rest_mV)
        .to_dict()
    )
    counts = (
        spikes.groupby(["stage", "somaSide"], dropna=False)
        .size()
        .rename("spikes")
        .reset_index()
        if len(spikes)
        else pd.DataFrame(columns=["stage", "somaSide", "spikes"])
    )
    summary = {
        "backend": "numpy_sparse_lif",
        "parameters": asdict(params),
        "duration_ms": duration_ms,
        "events": event_dicts(event_tuple),
        "n_nodes": graph.n_nodes,
        "n_edges_structural": graph.n_edges,
        "n_edges_with_known_sign": len(used_edges),
        "spike_count": int(len(spikes)),
        "active_nodes": int(spikes["node_index"].nunique()) if len(spikes) else 0,
        "spikes_by_stage_side": counts.to_dict(orient="records"),
        "max_voltage_mV_by_stage": {str(k): float(v) for k, v in stage_max.items()},
    }
    return spikes, summary
