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

    if len(motion_columns) != 2:
        raise ValueError("Coordinate inference requires exactly two motion columns")
    index = _node_index(graph)
    coordinates = np.full(graph.n_nodes, np.nan, dtype=np.float64)
    seed_coordinates = {
        int(motion_columns[0]["l1_body_id"]): 0.0,
        int(motion_columns[1]["l1_body_id"]): 1.0,
    }
    if len(seed_coordinates) != 2:
        raise ValueError("Motion columns must reference two distinct L1 body IDs")
    for body_id, coordinate in seed_coordinates.items():
        if body_id in index:
            coordinates[index[body_id]] = coordinate

    node_types = graph.nodes["type"].astype(str).to_numpy()
    edge_pre_ids = graph.edges["pre_body"].to_numpy(dtype=np.int64)
    edge_post_ids = graph.edges["post_body"].to_numpy(dtype=np.int64)
    node_id_index = pd.Index(graph.node_ids)
    pre = node_id_index.get_indexer(edge_pre_ids)
    post = node_id_index.get_indexer(edge_post_ids)
    if np.any(pre < 0) or np.any(post < 0):
        raise ValueError("graph edges reference body IDs outside graph.nodes")
    weights = graph.edges["synapse_count"].to_numpy(dtype=np.float64)
    medulla_nodes = np.isin(node_types, list(MEDULLA_TYPES))
    t4_nodes = np.isin(node_types, list(T4_TYPES))
    seed_edges = np.isin(edge_pre_ids, list(seed_coordinates))

    def assign_weighted(
        targets: np.ndarray,
        sources: np.ndarray,
        edge_mask: np.ndarray,
        eligible_nodes: np.ndarray,
    ) -> None:
        valid = edge_mask & np.isfinite(coordinates[sources])
        if not np.any(valid):
            return
        denominator = np.bincount(
            targets[valid], weights=weights[valid], minlength=graph.n_nodes
        )
        numerator = np.bincount(
            targets[valid],
            weights=weights[valid] * coordinates[sources[valid]],
            minlength=graph.n_nodes,
        )
        assign = eligible_nodes & ~np.isfinite(coordinates) & (denominator > 0)
        coordinates[assign] = numerator[assign] / denominator[assign]

    # Three passes preserve the historical bounded propagation rule while each
    # pass uses contiguous edge arrays instead of N per-node DataFrame scans.
    for _ in range(3):
        assign_weighted(post, pre, seed_edges, medulla_nodes)
        assign_weighted(post, pre, medulla_nodes[pre] & t4_nodes[post], t4_nodes)
        assign_weighted(
            pre,
            post,
            t4_nodes[post],
            ~(medulla_nodes | t4_nodes),
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
    pre_types = graph.edges["pre_type"].astype(str).to_numpy()
    post_types = graph.edges["post_type"].astype(str).to_numpy()
    sign = np.zeros(len(graph.edges), dtype=np.float64)
    sign[(pre_types == "L1") & np.isin(post_types, list(MEDULLA_TYPES))] = -1.0
    t4_post = np.isin(post_types, list(T4_TYPES))
    sign[t4_post & np.isin(pre_types, list(EXCITATORY_T4_INPUTS))] = 1.0
    sign[t4_post & np.isin(pre_types, list(INHIBITORY_T4_INPUTS))] = -1.0
    if ablate_inhibitory:
        sign[t4_post & np.isin(pre_types, list(INHIBITORY_T4_INPUTS))] = 0.0
    selected = sign != 0.0
    if not np.any(selected):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    node_ids = pd.Index(graph.node_ids)
    pre = node_ids.get_indexer(
        graph.edges.loc[selected, "pre_body"].to_numpy(dtype=np.int64)
    )
    post = node_ids.get_indexer(
        graph.edges.loc[selected, "post_body"].to_numpy(dtype=np.int64)
    )
    if np.any(pre < 0) or np.any(post < 0):
        raise ValueError("graph edges reference body IDs outside graph.nodes")
    weights = (
        sign[selected]
        * graph.edges.loc[selected, "synapse_count"].to_numpy(dtype=np.float64)
    )
    normalizer = np.bincount(post, weights=np.abs(weights), minlength=graph.n_nodes)
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
        event_values = np.asarray(
            [event.start_ms, event.duration_ms, event.amplitude_mV],
            dtype=np.float64,
        )
        if not np.isfinite(event_values).all():
            raise ValueError(f"Visual event {event.name!r} contains non-finite values")
        if event.duration_ms <= 0:
            raise ValueError(f"Visual event {event.name!r} must have positive duration")
        start = min(n_steps, max(0, int(round(event.start_ms / dt_ms))))
        end = min(
            n_steps,
            max(0, int(round((event.start_ms + event.duration_ms) / dt_ms))),
        )
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
        column_drive[start:end, column_index] += float(event.amplitude_mV)
        for body_id in event.node_ids:
            if int(body_id) not in node_index:
                raise ValueError(f"Visual event references node outside graph: {body_id}")
            # A positive visual contrast is a negative L1 release deviation;
            # the L1 receptor inversion below turns it into an ON response.
            l1_drive[start:end, node_index[int(body_id)]] += -float(event.amplitude_mV)
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


def _validate_graded_parameters(params: GradedParameters, duration_ms: float | None = None) -> None:
    values = list(asdict(params).values())
    if duration_ms is not None:
        values.append(duration_ms)
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError("graded parameters and duration_ms must be finite")
    if params.dt_ms <= 0 or (duration_ms is not None and duration_ms <= 0):
        raise ValueError("duration_ms and dt_ms must be positive")
    if min(
        params.tau_l1_ms,
        params.tau_tm3_ms,
        params.tau_mi1_ms,
        params.tau_inhibitory_ms,
        params.tau_t4_ms,
    ) <= 0:
        raise ValueError("graded time constants must be positive")
    if params.inhibitory_delay_ms < 0:
        raise ValueError("inhibitory_delay_ms cannot be negative")


class GradedCircuitKernel:
    """Compiled one-step kernel shared by batch and online EXP-002 execution."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        *,
        motion_columns: list[dict],
        params: GradedParameters | None = None,
        ablate_inhibitory: bool = False,
    ) -> None:
        self.graph = graph
        self.motion_columns = motion_columns
        self.params = params or GradedParameters()
        _validate_graded_parameters(self.params)
        self.node_index = _node_index(graph)
        self.node_types = graph.nodes["type"].astype(str).to_numpy()
        self.coordinates = _infer_column_coordinates(graph, motion_columns)
        self.pre, self.post, self.weights = _build_edges(graph, ablate_inhibitory)
        self.l1_indices = np.flatnonzero(self.node_types == "L1")
        self.inhibitory_indices = np.flatnonzero(
            np.isin(self.node_types, list(INHIBITORY_T4_INPUTS))
        )
        self.t4_indices = np.flatnonzero(np.isin(self.node_types, list(T4_TYPES)))
        self.column_l1_indices = np.asarray(
            [self.node_index.get(int(column["l1_body_id"]), -1) for column in motion_columns],
            dtype=np.int64,
        )
        if self.column_l1_indices.shape != (2,) or np.any(self.column_l1_indices < 0):
            raise ValueError("Both motion-column L1 body IDs must be present in the graph")
        self.tau = np.full(graph.n_nodes, self.params.tau_inhibitory_ms, dtype=np.float64)
        self.tau[self.node_types == "L1"] = self.params.tau_l1_ms
        self.tau[self.node_types == "Tm3"] = self.params.tau_tm3_ms
        self.tau[self.node_types == "Mi1"] = self.params.tau_mi1_ms
        self.tau[np.isin(self.node_types, list(T4_TYPES))] = self.params.tau_t4_ms
        self.inhibitory_spatial = np.maximum(
            0.0,
            1.0
            - np.abs(
                self.coordinates[self.inhibitory_indices, None]
                - np.arange(2, dtype=np.float64)[None, :]
            ),
        )
        self.inhibitory_spatial[self.node_types[self.inhibitory_indices] == "Mi9"] = 0.0
        self.delay_steps = int(round(self.params.inhibitory_delay_ms / self.params.dt_ms))
        self._delay_buffer = np.zeros((self.delay_steps, 2), dtype=np.float64)
        self._step = 0
        self.state = np.zeros(graph.n_nodes, dtype=np.float64)
        self.output = np.zeros(graph.n_nodes, dtype=np.float64)
        self._incoming = np.zeros(graph.n_nodes, dtype=np.float64)

    def _delayed_columns(self, column_drive: np.ndarray) -> np.ndarray:
        if self.delay_steps == 0:
            return column_drive
        slot = self._step % self.delay_steps
        delayed = self._delay_buffer[slot].copy() if self._step >= self.delay_steps else np.zeros(2)
        self._delay_buffer[slot] = column_drive
        return delayed

    def step(
        self,
        column_drive: np.ndarray,
        *,
        l1_drive: np.ndarray | None = None,
    ) -> np.ndarray:
        columns = np.asarray(column_drive, dtype=np.float64)
        if columns.shape != (2,) or not np.isfinite(columns).all():
            raise ValueError("column_drive must contain two finite amplitudes")
        delayed_columns = self._delayed_columns(columns)
        self._incoming.fill(0.0)
        if len(self.pre):
            np.add.at(
                self._incoming,
                self.post,
                self.weights * self.output[self.pre],
            )
        if l1_drive is None:
            self._incoming[self.column_l1_indices] = -columns
        else:
            l1_values = np.asarray(l1_drive, dtype=np.float64)
            if l1_values.shape != (self.graph.n_nodes,) or not np.isfinite(l1_values).all():
                raise ValueError("l1_drive must be a finite full-node vector")
            self._incoming[self.l1_indices] = l1_values[self.l1_indices]
        self._incoming[self.inhibitory_indices] = (
            delayed_columns @ self.inhibitory_spatial.T
        )
        self.state += (
            (self._incoming - self.state) / self.tau
        ) * self.params.dt_ms
        self.output[:] = self.state
        self.output[self.node_types != "L1"] = np.maximum(
            self.output[self.node_types != "L1"], 0.0
        )
        self._step += 1
        return self.state


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
    _validate_graded_parameters(params, duration_ms)
    normalized_events = _normalize_events(events, motion_columns)
    kernel = GradedCircuitKernel(
        graph,
        motion_columns=motion_columns,
        params=params,
        ablate_inhibitory=ablate_inhibitory,
    )
    n_steps = int(round(duration_ms / params.dt_ms)) + 1
    l1_drive, column_drive = _event_fields(
        normalized_events,
        node_index=kernel.node_index,
        n_nodes=graph.n_nodes,
        n_steps=n_steps,
        dt_ms=params.dt_ms,
    )
    trace = np.zeros((n_steps, graph.n_nodes), dtype=np.float32)
    for step in range(n_steps):
        trace[step] = kernel.step(
            column_drive[step], l1_drive=l1_drive[step]
        ).astype(np.float32)

    result = pd.DataFrame(
        {
            "time_ms": np.repeat(
                np.arange(n_steps, dtype=np.float64) * params.dt_ms,
                graph.n_nodes,
            ),
            "bodyId": np.tile(graph.node_ids, n_steps),
            "activity_proxy": trace.reshape(-1).astype(np.float64),
            "column_coordinate": np.tile(kernel.coordinates, n_steps),
            "type": np.tile(graph.nodes["type"].to_numpy(), n_steps),
            "instance": np.tile(graph.nodes["instance"].to_numpy(), n_steps),
            "somaSide": np.tile(graph.nodes["somaSide"].to_numpy(), n_steps),
            "stage": np.tile(graph.nodes["stage"].to_numpy(), n_steps),
        }
    )
    summary = {
        "backend": "graded_local_t4_model",
        "parameters": asdict(params),
        "duration_ms": duration_ms,
        "events": event_dicts(events),
        "ablate_inhibitory": ablate_inhibitory,
        "n_nodes": graph.n_nodes,
        "n_edges_structural": graph.n_edges,
        "n_edges_modeled": int(len(kernel.pre)),
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
    if f.empty or r.empty:
        raise ValueError("Both conditions must contain T4 activity")
    pivot_f = f.pivot(index="time_ms", columns="bodyId", values="activity_proxy")
    pivot_r = r.pivot(index="time_ms", columns="bodyId", values="activity_proxy")
    if set(pivot_f.columns) != set(pivot_r.columns):
        raise ValueError("Forward and reverse conditions must contain the same T4 neurons")
    body_ids = sorted(int(value) for value in pivot_f.columns)
    pivot_f = pivot_f.reindex(columns=body_ids)
    pivot_r = pivot_r.reindex(columns=body_ids)
    if not pivot_f.index.equals(pivot_r.index):
        raise ValueError("Forward and reverse T4 traces must have identical time samples")
    rows = []
    for body_id in body_ids:
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
