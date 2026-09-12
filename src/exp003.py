"""EXP-003 orchestration primitives.

This module deliberately does not modify the EXP-002 model.  It reuses the
frozen model's private construction helpers so the online circuit uses the
same signs, synapse normalization, time constants, and inhibitory delay as
``run_graded``.  The new code starts at the already-computed T4 state and
adds a transparent, non-biological downstream readout for the embodied
demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .graded_model import (
    GradedCircuitKernel,
    GradedParameters,
)
from .malecns_io import ConnectomeGraph


DOWNSTREAM_STAGES = ("t4_motion", "wide_field_projection", "descending_output")


def _stage_ids(graph: ConnectomeGraph, stage: str) -> list[int]:
    return graph.nodes.loc[graph.nodes["stage"] == stage, "bodyId"].astype(int).tolist()


def _normalized_matrix(
    graph: ConnectomeGraph,
    pre_ids: Iterable[int],
    post_ids: Iterable[int],
) -> np.ndarray:
    """Return post-by-pre synapse-count weights normalized per post cell."""

    pre_ids = list(pre_ids)
    post_ids = list(post_ids)
    pre_index = {body_id: index for index, body_id in enumerate(pre_ids)}
    post_index = {body_id: index for index, body_id in enumerate(post_ids)}
    matrix = np.zeros((len(post_ids), len(pre_ids)), dtype=np.float64)
    selected = graph.edges[
        graph.edges["pre_body"].isin(pre_ids)
        & graph.edges["post_body"].isin(post_ids)
    ]
    for row in selected.itertuples(index=False):
        matrix[post_index[int(row.post_body)], pre_index[int(row.pre_body)]] += float(
            row.synapse_count
        )
    normalizer = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(normalizer, 1.0)


@dataclass(frozen=True)
class DownstreamMatrices:
    """MaleCNS-derived feed-forward T4 -> wide-field -> descending readout."""

    t4_ids: tuple[int, ...]
    wide_ids: tuple[int, ...]
    descending_ids: tuple[int, ...]
    t4_types: tuple[str, ...]
    wide_types: tuple[str, ...]
    descending_types: tuple[str, ...]
    t4_to_wide: np.ndarray
    wide_to_descending: np.ndarray
    t4a_path: np.ndarray
    t4d_path: np.ndarray
    bridge_weights: np.ndarray

    @classmethod
    def from_graph(cls, graph: ConnectomeGraph) -> "DownstreamMatrices":
        t4_ids = tuple(_stage_ids(graph, "t4_motion"))
        wide_ids = tuple(_stage_ids(graph, "wide_field_projection"))
        descending_ids = tuple(_stage_ids(graph, "descending_output"))
        node_types = graph.nodes.set_index("bodyId")["type"].astype(str).to_dict()
        t4_to_wide = _normalized_matrix(graph, t4_ids, wide_ids)
        wide_to_descending = _normalized_matrix(graph, wide_ids, descending_ids)

        t4a_indices = [index for index, body_id in enumerate(t4_ids) if node_types[body_id] == "T4a"]
        t4d_indices = [index for index, body_id in enumerate(t4_ids) if node_types[body_id] == "T4d"]
        t4a_path = wide_to_descending @ t4_to_wide[:, t4a_indices].sum(axis=1)
        t4d_path = wide_to_descending @ t4_to_wide[:, t4d_indices].sum(axis=1)
        raw_weights = t4a_path - t4d_path
        denominator = float(np.abs(raw_weights).sum())
        if denominator == 0.0:
            raise ValueError("The selected MaleCNS downstream graph has no T4a/T4d path contrast")
        bridge_weights = raw_weights / denominator
        return cls(
            t4_ids=t4_ids,
            wide_ids=wide_ids,
            descending_ids=descending_ids,
            t4_types=tuple(node_types[body_id] for body_id in t4_ids),
            wide_types=tuple(node_types[body_id] for body_id in wide_ids),
            descending_types=tuple(node_types[body_id] for body_id in descending_ids),
            t4_to_wide=t4_to_wide,
            wide_to_descending=wide_to_descending,
            t4a_path=t4a_path,
            t4d_path=t4d_path,
            bridge_weights=bridge_weights,
        )

    def propagate(self, t4_activity: np.ndarray, *, ablate_t4_projection: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Propagate a T4 activity vector through the materialized graph.

        The readout is intentionally linear and instantaneous.  It is an
        observability/readout assumption, not a claim about downstream
        membrane dynamics.
        """

        matrix = np.zeros_like(self.t4_to_wide) if ablate_t4_projection else self.t4_to_wide
        wide = matrix @ np.asarray(t4_activity, dtype=np.float64)
        descending = self.wide_to_descending @ wide
        return wide, descending

    def trace(
        self,
        t4_trace: pd.DataFrame,
        *,
        ablate_t4_projection: bool = False,
    ) -> pd.DataFrame:
        """Return long-form T4, wide-field, and descending activity traces."""

        pivot = t4_trace[t4_trace["bodyId"].isin(self.t4_ids)].pivot(
            index="time_ms", columns="bodyId", values="activity_proxy"
        )
        t4_values = pivot.reindex(columns=self.t4_ids).to_numpy(dtype=np.float64)
        wide_values = np.zeros((len(pivot), len(self.wide_ids)), dtype=np.float64)
        descending_values = np.zeros((len(pivot), len(self.descending_ids)), dtype=np.float64)
        for row_index, values in enumerate(t4_values):
            wide_values[row_index], descending_values[row_index] = self.propagate(
                values, ablate_t4_projection=ablate_t4_projection
            )

        rows: list[dict] = []
        for time_index, time_ms in enumerate(pivot.index.to_numpy(dtype=float)):
            for index, body_id in enumerate(self.wide_ids):
                rows.append(
                    {
                        "time_ms": float(time_ms),
                        "bodyId": int(body_id),
                        "activity_proxy": float(wide_values[time_index, index]),
                        "type": self.wide_types[index],
                        "stage": "wide_field_projection",
                    }
                )
            for index, body_id in enumerate(self.descending_ids):
                rows.append(
                    {
                        "time_ms": float(time_ms),
                        "bodyId": int(body_id),
                        "activity_proxy": float(descending_values[time_index, index]),
                        "type": self.descending_types[index],
                        "stage": "descending_output",
                    }
                )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class SteeringBridge:
    """Explicit temporary downstream-to-controller conversion.

    The two signs are assigned to connectivity-derived T4a-dominant and
    T4d-dominant descending pools.  This is the only artificial motor bridge
    in EXP-003.  It is not a direct stimulus-direction lookup and it is not a
    learned decoder.  ``activity_reference`` simply maps the dimensionless
    activity proxy into FlyGym's controller range; it is fixed before the
    conditions are run.
    """

    activity_reference: float = 0.01
    max_asymmetry: float = 0.35
    baseline: float = 1.0

    def raw_command(self, descending_activity: np.ndarray, matrices: DownstreamMatrices) -> float:
        values = np.asarray(descending_activity, dtype=np.float64)
        return float(values @ matrices.bridge_weights)

    def normalized_command(self, raw_command: float) -> float:
        return float(np.clip(raw_command / self.activity_reference, -1.0, 1.0))

    def action(self, raw_command: float) -> np.ndarray:
        normalized = self.normalized_command(raw_command)
        return np.asarray(
            [
                self.baseline + self.max_asymmetry * normalized,
                self.baseline - self.max_asymmetry * normalized,
            ],
            dtype=np.float64,
        )


class OnlineGradedCircuit:
    """One-step form of the frozen EXP-002 graded model for closed-loop use."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        *,
        motion_columns: list[dict],
        params: GradedParameters | None = None,
    ) -> None:
        self._kernel = GradedCircuitKernel(
            graph,
            motion_columns=motion_columns,
            params=params,
        )
        self.graph = self._kernel.graph
        self.motion_columns = self._kernel.motion_columns
        self.params = self._kernel.params
        self.node_index = self._kernel.node_index
        self.node_types = self._kernel.node_types
        self.coordinates = self._kernel.coordinates
        self.pre = self._kernel.pre
        self.post = self._kernel.post
        self.weights = self._kernel.weights
        self.state = self._kernel.state
        self.output = self._kernel.output
        self.l1_indices = self._kernel.l1_indices
        self.inhibitory_indices = self._kernel.inhibitory_indices
        self.t4_indices = self._kernel.t4_indices
        self.tau = self._kernel.tau
        self.delay_steps = self._kernel.delay_steps

    def step(self, column_drive: np.ndarray) -> np.ndarray:
        state = self._kernel.step(column_drive)
        return state[self.t4_indices].copy()


@dataclass(frozen=True)
class ClosedLoopMotionScene:
    """Two-column visual scene with a minimal yaw-dependent retinal shift."""

    reverse: bool
    start_ms: float = 10.0
    pulse_duration_ms: float = 10.0
    amplitude: float = 1.0
    yaw_feedback_ms_per_rad: float = 100.0

    def column_drive(self, time_ms: float, yaw_rad: float) -> np.ndarray:
        shift_ms = float(yaw_rad) * self.yaw_feedback_ms_per_rad
        first_column = 1 if self.reverse else 0
        second_column = 0 if self.reverse else 1
        drive = np.zeros(2, dtype=np.float64)
        for column_index, offset in ((first_column, 0.0), (second_column, self.pulse_duration_ms)):
            pulse_start = self.start_ms + offset + shift_ms
            if pulse_start <= time_ms < pulse_start + self.pulse_duration_ms:
                drive[column_index] = self.amplitude
        return drive
