"""Literature-informed continuous local model for EXP-002.

This is deliberately a small signal model, not a replacement connectome or a
trained network. MaleCNS supplies the nodes, directed edges, and synapse
counts. The model supplies the missing physiology that EXP-001 did not have:
graded signals, receptor-mediated L1 sign inversion, and continuous T4 input
integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .malecns_io import ConnectomeGraph
from .visual_stimulus import VisualEvent, event_dicts


EXCITATORY_T4_INPUTS = {"Mi1", "Tm3"}
INHIBITORY_T4_INPUTS = {"Mi4", "Mi9", "C3", "CT1"}
T4_TYPES = {"T4a", "T4b", "T4c", "T4d"}
MEDULLA_TYPES = {"Mi1", "Tm3"}


@dataclass(frozen=True)
class GradedParameters:
    """Small fixed parameter set chosen from relative physiology, not fitted."""

    dt_ms: float = 0.1
    tau_l1_ms: float = 5.0
    tau_tm3_ms: float = 12.0
    tau_mi1_ms: float = 20.0
    tau_inhibitory_ms: float = 30.0
    tau_t4_ms: float = 10.0
    inhibitory_delay_ms: float = 15.0


def _lowpass(previous: float, target: float, tau_ms: float, dt_ms: float) -> float:
    return previous + ((target - previous) / tau_ms) * dt_ms


def _node_index(graph: ConnectomeGraph) -> dict[int, int]:
    return {int(body_id): index for index, body_id in enumerate(graph.node_ids)}


def _infer_column_coordinates(
    graph: ConnectomeGraph,
    motion_columns: list[dict],
) -> np.ndarray:
    """Infer a two-column coordinate from the retained MaleCNS edges.

    Mi1/Tm3 coordinates come from retained L1->Mi1/Tm3 synapses. T4
    coordinates come from their Mi1/Tm3 inputs. Other direct T4 inputs use
    the weighted coordinate of their T4 targets. Missing coordinates are
    assigned the midpoint and remain explicitly model-derived rather than
    treated as workbook facts.
    """

    index = _node_index(graph)
    coordinates = np.full(graph.n_nodes, np.nan, dtype=np.float64)
    seed_coordinates = {
        int(motion_columns[0]["l1_body_id"]): 0.0,
        int(motion_columns[1]["l1_body_id"]): 1.0,
    }
    for body_id, coordinate in seed_coordinates.items():
        if body_id in index:
            coordinates[index[body_id]] = coordinate

    nodes = graph.nodes.set_index("bodyId")
    edges = graph.edges

    for _ in range(3):
        for body_id in graph.node_ids:
            node = index[int(body_id)]
            node_type = str(nodes.loc[int(body_id), "type"])
            if node_type in MEDULLA_TYPES and not np.isfinite(coordinates[node]):
                incoming = edges[
                    (edges["post_body"] == int(body_id))
                    & edges["pre_body"].isin(seed_coordinates)
                ]
                if len(incoming):
                    source_coordinates = incoming["pre_body"].map(seed_coordinates)
                    coordinates[node] = float(
                        np.average(
                            source_coordinates.to_numpy(dtype=np.float64),
                            weights=incoming["synapse_count"].to_numpy(dtype=np.float64),
                        )
                    )

        for body_id in graph.node_ids:
            node = index[int(body_id)]
            node_type = str(nodes.loc[int(body_id), "type"])
            if node_type not in T4_TYPES or np.isfinite(coordinates[node]):
                continue
            incoming = edges[
                (edges["post_body"] == int(body_id))
                & edges["pre_type"].isin(MEDULLA_TYPES)
            ]
            incoming_indices = incoming["pre_body"].map(index).to_numpy(dtype=np.int64)
            valid = np.isfinite(coordinates[incoming_indices])
            incoming = incoming.loc[valid]
            if len(incoming):
                source_coordinates = coordinates[incoming_indices[valid]]
                coordinates[node] = float(
                    np.average(
                        source_coordinates,
                        weights=incoming["synapse_count"].to_numpy(dtype=np.float64),
                    )
                )

        for body_id in graph.node_ids:
            node = index[int(body_id)]
            node_type = str(nodes.loc[int(body_id), "type"])
            if node_type in MEDULLA_TYPES or node_type in T4_TYPES:
                continue
            outgoing = edges[
                (edges["pre_body"] == int(body_id))
                & edges["post_type"].isin(T4_TYPES)
            ]
            target_indices = outgoing["post_body"].map(index).to_numpy(dtype=np.int64)
            valid = np.isfinite(coordinates[target_indices])
            outgoing = outgoing.loc[valid]
            if len(outgoing):
                target_coordinates = coordinates[target_indices[valid]]
                coordinates[node] = float(
                    np.average(
                        target_coordinates,
                        weights=outgoing["synapse_count"].to_numpy(dtype=np.float64),
                    )
                )

    coordinates[~np.isfinite(coordinates)] = 0.5
    return coordinates


def _edge_sign(pre_type: str, post_type: str) -> float:
    if pre_type == "L1" and post_type in MEDULLA_TYPES:
        # L1's glutamatergic signal is represented as a release deviation for
        # an ON edge; the receptor-mediated inversion makes the Mi1/Tm3
        # postsynaptic signal positive.
        return -1.0
    if post_type in T4_TYPES:
        if pre_type in EXCITATORY_T4_INPUTS:
            return 1.0
        if pre_type in INHIBITORY_T4_INPUTS:
            return -1.0
    return 0.0


def _build_edges(graph: ConnectomeGraph, ablate_inhibitory: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = _node_index(graph)
    nodes = graph.nodes.set_index("bodyId")
    rows: list[tuple[int, int, float]] = []
    for row in graph.edges.itertuples(index=False):
        pre_type = str(row.pre_type)
        post_type = str(row.post_type)
        sign = _edge_sign(pre_type, post_type)
        if not sign or (ablate_inhibitory and post_type in T4_TYPES and pre_type in INHIBITORY_T4_INPUTS):
            continue
        rows.append((index[int(row.pre_body)], index[int(row.post_body)], sign * float(row.synapse_count)))

    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    edge_array = np.asarray(rows, dtype=np.float64)
    pre = edge_array[:, 0].astype(np.int64)
    post = edge_array[:, 1].astype(np.int64)
    weights = edge_array[:, 2]
    normalizer = np.zeros(graph.n_nodes, dtype=np.float64)
    np.add.at(normalizer, post, np.abs(weights))
    weights = weights / np.maximum(normalizer[post], 1.0)
    return pre, post, weights


def _event_fields(
    events: tuple[VisualEvent, ...] | list[VisualEvent],
    *,
    node_index: dict[int, int],
    n_nodes: int,
    n_steps: int,
    dt_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    l1_drive = np.zeros((n_steps, n_nodes), dtype=np.float64)
    column_drive = np.zeros((n_steps, 2), dtype=np.float64)
    for event in events:
        start = max(0, int(round(event.start_ms / dt_ms)))
        end = min(n_steps, int(round((event.start_ms + event.duration_ms) / dt_ms)))
        if start >= end:
            continue
        try:
            column_index = int(str(event.name).split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"EXP-002 event names must end in a column index: {event.name!r}") from exc
        # The experiment uses two explicit column suffixes; callers map event
        # names to 0/1 before simulation.
        if column_index not in (0, 1):
            raise ValueError("EXP-002 requires normalized event names 0 and 1")
        column_drive[start:end, column_index] = float(event.amplitude_mV)
        for body_id in event.node_ids:
            if int(body_id) not in node_index:
                raise ValueError(f"Visual event references node outside graph: {body_id}")
            # A positive visual contrast is a negative L1 release deviation;
            # the L1 receptor inversion below turns it into an ON response.
            l1_drive[start:end, node_index[int(body_id)]] = -float(event.amplitude_mV)
    return l1_drive, column_drive


def _normalize_events(
    events: tuple[VisualEvent, ...] | list[VisualEvent],
    motion_columns: list[dict],
) -> tuple[VisualEvent, ...]:
    suffix_to_index = {
        str(motion_columns[0]["suffix"]): "0",
        str(motion_columns[1]["suffix"]): "1",
    }
    normalized = []
    for event in events:
        if event.name not in suffix_to_index:
            raise ValueError(f"Event {event.name!r} is not one of the selected EXP-002 columns")
        normalized.append(
            VisualEvent(
                name=suffix_to_index[event.name],
                node_ids=event.node_ids,
                start_ms=event.start_ms,
                duration_ms=event.duration_ms,
                amplitude_mV=event.amplitude_mV,
            )
        )
    return tuple(normalized)


def run_graded(
    graph: ConnectomeGraph,
    events: tuple[VisualEvent, ...] | list[VisualEvent],
    *,
    motion_columns: list[dict],
    params: GradedParameters | None = None,
    duration_ms: float = 150.0,
    ablate_inhibitory: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run one continuous condition and return raw node activity plus summary."""

    params = params or GradedParameters()
    if params.dt_ms <= 0 or duration_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    normalized_events = _normalize_events(events, motion_columns)
    node_index = _node_index(graph)
    node_types = graph.nodes["type"].astype(str).to_numpy()
    coordinates = _infer_column_coordinates(graph, motion_columns)
    pre, post, weights = _build_edges(graph, ablate_inhibitory)
    n_steps = int(round(duration_ms / params.dt_ms)) + 1
    l1_drive, column_drive = _event_fields(
        normalized_events,
        node_index=node_index,
        n_nodes=graph.n_nodes,
        n_steps=n_steps,
        dt_ms=params.dt_ms,
    )
    inhibitory_indices = np.flatnonzero(np.isin(node_types, list(INHIBITORY_T4_INPUTS)))
    l1_indices = np.flatnonzero(node_types == "L1")
    tau = np.full(graph.n_nodes, params.tau_inhibitory_ms, dtype=np.float64)
    tau[node_types == "L1"] = params.tau_l1_ms
    tau[node_types == "Tm3"] = params.tau_tm3_ms
    tau[node_types == "Mi1"] = params.tau_mi1_ms
    tau[np.isin(node_types, list(T4_TYPES))] = params.tau_t4_ms

    inhibitory_drive = np.zeros((n_steps, len(inhibitory_indices)), dtype=np.float64)
    for local_index, node in enumerate(inhibitory_indices):
        spatial = np.maximum(0.0, 1.0 - np.abs(coordinates[node] - np.arange(2, dtype=np.float64)))
        inhibitory_drive[:, local_index] = column_drive @ spatial
        if node_types[node] == "Mi9":
            # Mi9 is an OFF-linked glutamatergic input in the literature; an
            # ON-only stimulus therefore does not directly drive its release.
            inhibitory_drive[:, local_index] = 0.0
    delay_steps = int(round(params.inhibitory_delay_ms / params.dt_ms))
    if delay_steps:
        inhibitory_drive[delay_steps:] = inhibitory_drive[:-delay_steps].copy()
        inhibitory_drive[:delay_steps] = 0.0

    state = np.zeros(graph.n_nodes, dtype=np.float64)
    output = np.zeros(graph.n_nodes, dtype=np.float64)
    trace = np.zeros((n_steps, graph.n_nodes), dtype=np.float32)
    incoming = np.zeros(graph.n_nodes, dtype=np.float64)
    for step in range(n_steps):
        incoming.fill(0.0)
        if len(pre):
            np.add.at(incoming, post, weights * output[pre])
        target = incoming
        target[l1_indices] = l1_drive[step, l1_indices]
        target[inhibitory_indices] = inhibitory_drive[step]
        state += ((target - state) / tau) * params.dt_ms
        output[:] = state
        output[node_types != "L1"] = np.maximum(output[node_types != "L1"], 0.0)
        trace[step] = state.astype(np.float32)

    rows: list[dict] = []
    for step in range(n_steps):
        for node, body_id in enumerate(graph.node_ids):
            rows.append(
                {
                    "time_ms": float(step * params.dt_ms),
                    "bodyId": int(body_id),
                    "activity_proxy": float(trace[step, node]),
                    "column_coordinate": float(coordinates[node]),
                }
            )
    result = pd.DataFrame(rows).merge(
        graph.nodes[["bodyId", "type", "instance", "somaSide", "stage"]],
        on="bodyId",
        how="left",
    )
    summary = {
        "backend": "graded_local_t4_model",
        "parameters": asdict(params),
        "duration_ms": duration_ms,
        "events": event_dicts(events),
        "ablate_inhibitory": ablate_inhibitory,
        "n_nodes": graph.n_nodes,
        "n_edges_structural": graph.n_edges,
        "n_edges_modeled": int(len(pre)),
        "coordinate_source": "weighted retained MaleCNS L1->Mi1/Tm3 and input->T4 synapse paths; midpoint fallback",
        "l1_release_polarity": "negative ON-edge release deviation",
        "t4_input_signs": {
            "Mi1/Tm3": "excitatory",
            "Mi4/C3/CT1": "inhibitory",
            "Mi9": "inhibitory receptor assumption, not driven by ON-only input",
        },
    }
    return result, summary


def t4_comparison(forward: pd.DataFrame, reverse: pd.DataFrame) -> pd.DataFrame:
    """Compare raw graded T4 activity neuron-by-neuron."""

    f = forward[forward["stage"] == "t4_motion"].copy()
    r = reverse[reverse["stage"] == "t4_motion"].copy()
    pivot_f = f.pivot(index="time_ms", columns="bodyId", values="activity_proxy")
    pivot_r = r.pivot(index="time_ms", columns="bodyId", values="activity_proxy")
    rows = []
    for body_id in sorted(set(pivot_f.columns) & set(pivot_r.columns)):
        difference = pivot_f[body_id].to_numpy() - pivot_r[body_id].to_numpy()
        f_values = pivot_f[body_id].to_numpy()
        r_values = pivot_r[body_id].to_numpy()
        peak_index = int(np.argmax(np.abs(difference)))
        metadata = f.loc[f["bodyId"] == body_id].iloc[0]
        rows.append(
            {
                "bodyId": int(body_id),
                "type": str(metadata["type"]),
                "max_activity_forward": float(np.max(f_values)),
                "max_activity_reverse": float(np.max(r_values)),
                "max_activity_difference": float(np.max(f_values) - np.max(r_values)),
                "max_abs_pointwise_activity_difference": float(np.max(np.abs(difference))),
                "time_of_max_abs_activity_difference_ms": float(pivot_f.index.to_numpy()[peak_index]),
                "positive_area_forward": float(np.trapezoid(np.maximum(f_values, 0.0), pivot_f.index)),
                "positive_area_reverse": float(np.trapezoid(np.maximum(r_values, 0.0), pivot_r.index)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_abs_pointwise_activity_difference", "bodyId"],
        ascending=[False, True],
    ).reset_index(drop=True)
