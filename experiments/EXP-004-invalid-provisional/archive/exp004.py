"""INVALID provisional binocular H2/HS/DN network retained for audit.

MaleCNS supplies every chemical edge in the materialized graph.  The only
non-connectome interaction is the optional contralateral H2<->HS coupling,
which is explicitly derived from the binocular optic-flow literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .malecns_io import ConnectomeGraph


LPTC_TYPES = ("HSE", "HSN", "HSS", "H2")
DN_TYPES = ("DNp15", "DNa02")


@dataclass(frozen=True)
class BinocularParameters:
    """Free numerical parameters for the downstream graded proxy."""

    dt_ms: float = 0.1
    tau_intermediate_ms: float = 10.0
    tau_lptc_ms: float = 20.0
    tau_dn_ms: float = 20.0
    electrical_coupling_gain: float = 0.25


def _ids(graph: ConnectomeGraph, stage: str) -> tuple[int, ...]:
    return tuple(graph.nodes.loc[graph.nodes["stage"].eq(stage), "bodyId"].astype(int))


def _signed_matrix(
    graph: ConnectomeGraph,
    pre_ids: Iterable[int],
    post_ids: Iterable[int],
) -> np.ndarray:
    pre_ids = tuple(pre_ids)
    post_ids = tuple(post_ids)
    pre_index = {body_id: index for index, body_id in enumerate(pre_ids)}
    post_index = {body_id: index for index, body_id in enumerate(post_ids)}
    matrix = np.zeros((len(post_ids), len(pre_ids)), dtype=np.float64)
    selected = graph.edges[
        graph.edges["pre_body"].isin(pre_ids)
        & graph.edges["post_body"].isin(post_ids)
    ]
    for row in selected.itertuples(index=False):
        pre = int(row.pre_body)
        post = int(row.post_body)
        matrix[post_index[post], pre_index[pre]] += float(row.synapse_count) * float(row.pre_sign)
    normalizer = np.sum(np.abs(matrix), axis=1, keepdims=True)
    return matrix / np.maximum(normalizer, 1.0)


def _literature_gap_matrix(
    graph: ConnectomeGraph,
    lptc_ids: tuple[int, ...],
    lptc_types: tuple[str, ...],
) -> np.ndarray:
    """Contralateral H2<->HS coupling; not a MaleCNS chemical edge."""

    node_table = graph.nodes.set_index("bodyId")
    matrix = np.zeros((len(lptc_ids), len(lptc_ids)), dtype=np.float64)
    post_index = {body_id: index for index, body_id in enumerate(lptc_ids)}
    hs_types = {"HSE", "HSN", "HSS"}
    for pre_index, pre_body in enumerate(lptc_ids):
        pre_type = lptc_types[pre_index]
        pre_side = str(node_table.loc[pre_body, "somaSide"])
        for post_index_value, post_body in enumerate(lptc_ids):
            post_type = lptc_types[post_index_value]
            post_side = str(node_table.loc[post_body, "somaSide"])
            is_h2_hs = pre_type == "H2" and post_type in hs_types
            is_hs_h2 = pre_type in hs_types and post_type == "H2"
            if (is_h2_hs or is_hs_h2) and pre_side != post_side:
                matrix[post_index_value, pre_index] = 1.0
    normalizer = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(normalizer, 1.0)


@dataclass
class BinocularNetwork:
    """Chemical MaleCNS network plus optional literature-derived coupling."""

    graph: ConnectomeGraph
    params: BinocularParameters
    t4_ids: tuple[int, ...]
    intermediate_ids: tuple[int, ...]
    lptc_ids: tuple[int, ...]
    dn_ids: tuple[int, ...]
    t4_types: tuple[str, ...]
    intermediate_types: tuple[str, ...]
    lptc_types: tuple[str, ...]
    dn_types: tuple[str, ...]
    t4_to_intermediate: np.ndarray
    t4_to_lptc: np.ndarray
    intermediate_to_lptc: np.ndarray
    intermediate_to_dn: np.ndarray
    lptc_to_lptc: np.ndarray
    lptc_to_dn: np.ndarray
    gap_matrix: np.ndarray
    intermediate_state: np.ndarray
    lptc_state: np.ndarray
    dn_state: np.ndarray

    @classmethod
    def from_graph(
        cls,
        graph: ConnectomeGraph,
        *,
        params: BinocularParameters | None = None,
    ) -> "BinocularNetwork":
        params = params or BinocularParameters()
        t4_ids = _ids(graph, "t4_motion")
        intermediate_ids = _ids(graph, "motion_intermediate")
        lptc_ids = _ids(graph, "lptc_widefield")
        dn_ids = _ids(graph, "dn_readout")
        if not all((t4_ids, intermediate_ids, lptc_ids, dn_ids)):
            raise ValueError("EXP-004 graph is missing a required stage")
        types = graph.nodes.set_index("bodyId")["type"].astype(str).to_dict()
        t4_types = tuple(types[body_id] for body_id in t4_ids)
        intermediate_types = tuple(types[body_id] for body_id in intermediate_ids)
        lptc_types = tuple(types[body_id] for body_id in lptc_ids)
        dn_types = tuple(types[body_id] for body_id in dn_ids)
        return cls(
            graph=graph,
            params=params,
            t4_ids=t4_ids,
            intermediate_ids=intermediate_ids,
            lptc_ids=lptc_ids,
            dn_ids=dn_ids,
            t4_types=t4_types,
            intermediate_types=intermediate_types,
            lptc_types=lptc_types,
            dn_types=dn_types,
            t4_to_intermediate=_signed_matrix(graph, t4_ids, intermediate_ids),
            t4_to_lptc=_signed_matrix(graph, t4_ids, lptc_ids),
            intermediate_to_lptc=_signed_matrix(graph, intermediate_ids, lptc_ids),
            intermediate_to_dn=_signed_matrix(graph, intermediate_ids, dn_ids),
            lptc_to_lptc=_signed_matrix(graph, lptc_ids, lptc_ids),
            lptc_to_dn=_signed_matrix(graph, lptc_ids, dn_ids),
            gap_matrix=_literature_gap_matrix(graph, lptc_ids, lptc_types),
            intermediate_state=np.zeros(len(intermediate_ids), dtype=np.float64),
            lptc_state=np.zeros(len(lptc_ids), dtype=np.float64),
            dn_state=np.zeros(len(dn_ids), dtype=np.float64),
        )

    def reset(self) -> None:
        self.intermediate_state.fill(0.0)
        self.lptc_state.fill(0.0)
        self.dn_state.fill(0.0)

    def step(
        self,
        t4_activity: np.ndarray,
        *,
        electrical_coupling: bool = True,
        recurrent_chemical: bool = True,
        inhibitory_intermediates: bool = True,
    ) -> dict[str, np.ndarray]:
        t4_activity = np.asarray(t4_activity, dtype=np.float64)
        if t4_activity.shape != (len(self.t4_ids),):
            raise ValueError("t4_activity has the wrong combined bilateral shape")
        p = self.params
        intermediate_target = self.t4_to_intermediate @ t4_activity
        self.intermediate_state += (intermediate_target - self.intermediate_state) * p.dt_ms / p.tau_intermediate_ms
        if not inhibitory_intermediates:
            lpi_indices = [i for i, value in enumerate(self.intermediate_types) if value.startswith("LPi")]
            self.intermediate_state[lpi_indices] = 0.0

        lptc_target = self.t4_to_lptc @ t4_activity + self.intermediate_to_lptc @ self.intermediate_state
        if recurrent_chemical:
            lptc_target = lptc_target + self.lptc_to_lptc @ self.lptc_state
        if electrical_coupling:
            lptc_target = lptc_target + p.electrical_coupling_gain * (self.gap_matrix @ self.lptc_state)
        self.lptc_state += (lptc_target - self.lptc_state) * p.dt_ms / p.tau_lptc_ms

        dn_target = self.intermediate_to_dn @ self.intermediate_state + self.lptc_to_dn @ self.lptc_state
        self.dn_state += (dn_target - self.dn_state) * p.dt_ms / p.tau_dn_ms
        return {
            "intermediate": self.intermediate_state.copy(),
            "lptc": self.lptc_state.copy(),
            "dn": self.dn_state.copy(),
        }

    def embed_side_t4(
        self,
        left_ids: Iterable[int],
        left_activity: np.ndarray,
        right_ids: Iterable[int],
        right_activity: np.ndarray,
    ) -> np.ndarray:
        values = np.zeros(len(self.t4_ids), dtype=np.float64)
        index = {body_id: i for i, body_id in enumerate(self.t4_ids)}
        for body_id, activity in zip(left_ids, left_activity):
            values[index[int(body_id)]] = float(activity)
        for body_id, activity in zip(right_ids, right_activity):
            values[index[int(body_id)]] = float(activity)
        return values

    def population_means(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for cell_type in LPTC_TYPES + DN_TYPES:
            lptc_values = [self.lptc_state[i] for i, value in enumerate(self.lptc_types) if value == cell_type]
            dn_values = [self.dn_state[i] for i, value in enumerate(self.dn_types) if value == cell_type]
            values = lptc_values if lptc_values else dn_values
            result[cell_type] = float(np.mean(values)) if values else 0.0
        return result

    def side_population_means(self) -> dict[str, float]:
        node_table = self.graph.nodes.set_index("bodyId")
        result: dict[str, float] = {}
        for cell_type in LPTC_TYPES + DN_TYPES:
            for side in ("L", "R"):
                lptc_values = [
                    self.lptc_state[i]
                    for i, body_id in enumerate(self.lptc_ids)
                    if self.lptc_types[i] == cell_type and str(node_table.loc[body_id, "somaSide"]) == side
                ]
                dn_values = [
                    self.dn_state[i]
                    for i, body_id in enumerate(self.dn_ids)
                    if self.dn_types[i] == cell_type and str(node_table.loc[body_id, "somaSide"]) == side
                ]
                values = lptc_values if lptc_values else dn_values
                result[f"{cell_type}_{side}"] = float(np.mean(values)) if values else 0.0
        return result
