"""Transparent sparse LIF baseline for the narrow MaleCNS experiment.

The equations mirror the Brian2/Shiu baseline, but the runnable backend is a
small NumPy/SciPy integrator. A Brian2 2.9.0 import failure was reproduced with
NumPy 2.4.6 but not NumPy 2.0.2; that environment-specific diagnostic is recorded in
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


def _empty_trace() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "time_ms", "node_index", "bodyId", "voltage_mV", "type",
            "instance", "somaSide", "superclass", "stage",
        ]
    )


def _validate_parameters(params: LIFParameters, duration_ms: float) -> None:
    values = np.asarray(list(asdict(params).values()) + [duration_ms], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("LIF parameters and duration_ms must be finite")
    if params.dt_ms <= 0 or duration_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    if params.tau_membrane_ms <= 0 or params.tau_synapse_ms <= 0:
        raise ValueError("LIF time constants must be positive")
    if params.refractory_ms < 0 or params.synaptic_delay_ms < 0:
        raise ValueError("refractory period and synaptic delay cannot be negative")
    if params.weight_per_synapse_mV < 0:
        raise ValueError("weight_per_synapse_mV cannot be negative")
    if params.v_threshold_mV <= params.v_rest_mV:
        raise ValueError("v_threshold_mV must exceed v_rest_mV")


def run_lif(
    graph: ConnectomeGraph,
    events: tuple[VisualEvent, ...] | list[VisualEvent] | VisualPulse,
    *,
    duration_ms: float = 100.0,
    params: LIFParameters | None = None,
    seed: int = 0,
    trace_node_ids: tuple[int, ...] | list[int] | None = None,
    return_trace: bool = False,
) -> tuple[pd.DataFrame, dict] | tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run one deterministic event condition and summarize spikes.

    ``coupling`` is post-by-pre.  A spike schedules one delayed vector-matrix
    product, preserving the real queried edge weights and transmitter signs.
    """

    del seed  # deterministic baseline; retained in the public call signature
    params = params or LIFParameters()
    _validate_parameters(params, duration_ms)

    coupling, used_edges = effective_sparse_matrix(
        graph, weight_per_synapse_mV=params.weight_per_synapse_mV
    )
    node_index = {int(body_id): i for i, body_id in enumerate(graph.node_ids)}
    requested_trace_ids = tuple(dict.fromkeys(int(body_id) for body_id in (trace_node_ids or ())))
    missing_trace_ids = [body_id for body_id in requested_trace_ids if body_id not in node_index]
    if missing_trace_ids:
        raise ValueError(f"Trace nodes are outside the graph: {missing_trace_ids}")
    trace_indices = np.asarray([node_index[body_id] for body_id in requested_trace_ids], dtype=np.int64)
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
    voltage_trace = (
        np.empty((n_steps, len(trace_indices)), dtype=np.float32)
        if return_trace and len(trace_indices)
        else None
    )

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
        if voltage_trace is not None:
            # Capture the pre-reset membrane voltage so threshold crossings
            # remain visible in the diagnostic trace.
            voltage_trace[step] = v[trace_indices]
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
    if not return_trace:
        return spikes, summary

    if voltage_trace is None:
        trace = _empty_trace()
    else:
        trace = pd.DataFrame(
            {
                "time_ms": np.repeat(np.arange(n_steps, dtype=np.float32) * dt, len(trace_indices)),
                "node_index": np.tile(trace_indices, n_steps),
                "bodyId": np.tile(graph.node_ids[trace_indices], n_steps),
                "voltage_mV": voltage_trace.reshape(-1),
            }
        )
        annotations = graph.nodes.copy()
        annotations["node_index"] = np.arange(graph.n_nodes)
        trace = trace.merge(
            annotations[["node_index", "type", "instance", "somaSide", "superclass", "stage"]],
            on="node_index",
            how="left",
        )
    return spikes, summary, trace
