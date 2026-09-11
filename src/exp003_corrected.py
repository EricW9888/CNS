"""Corrected EXP-003 downstream readout built from the audited MaleCNS path.

This module intentionally keeps the frozen EXP-003 bridge and its normalized
linear readout unchanged.  The only difference is the materialized boundary:
T4 now feeds the relevant lobula-plate/visual target cells before those cells
feed descending neurons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exp003 import _normalized_matrix, _stage_ids
from .malecns_io import ConnectomeGraph


CORRECTED_STAGES = ("t4_motion", "lobula_plate_target", "descending_output")
T4_TYPES = ("T4a", "T4b", "T4c", "T4d")


@dataclass(frozen=True)
class CorrectedDownstreamMatrices:
    """Two-hop MaleCNS T4 -> lobula-plate target -> descending matrices."""

    t4_ids: tuple[int, ...]
    middle_ids: tuple[int, ...]
    descending_ids: tuple[int, ...]
    t4_types: tuple[str, ...]
    middle_types: tuple[str, ...]
    descending_types: tuple[str, ...]
    t4_to_middle: np.ndarray
    middle_to_descending: np.ndarray
    t4_type_paths: dict[str, np.ndarray]
    t4a_path: np.ndarray
    t4d_path: np.ndarray
    bridge_weights: np.ndarray

    @classmethod
    def from_graph(cls, graph: ConnectomeGraph) -> "CorrectedDownstreamMatrices":
        t4_ids = tuple(_stage_ids(graph, "t4_motion"))
        middle_ids = tuple(_stage_ids(graph, "lobula_plate_target"))
        descending_ids = tuple(_stage_ids(graph, "descending_output"))
        if not t4_ids or not middle_ids or not descending_ids:
            raise ValueError("Corrected downstream graph is missing one of its three stages")
        node_types = graph.nodes.set_index("bodyId")["type"].astype(str).to_dict()
        t4_to_middle = _normalized_matrix(graph, t4_ids, middle_ids)
        middle_to_descending = _normalized_matrix(graph, middle_ids, descending_ids)
        type_paths: dict[str, np.ndarray] = {}
        for t4_type in T4_TYPES:
            indices = [i for i, body_id in enumerate(t4_ids) if node_types[body_id] == t4_type]
            type_paths[t4_type] = (
                middle_to_descending @ t4_to_middle[:, indices].sum(axis=1)
                if indices
                else np.zeros(len(descending_ids), dtype=np.float64)
            )
        raw_weights = type_paths["T4a"] - type_paths["T4d"]
        denominator = float(np.abs(raw_weights).sum())
        if denominator == 0.0:
            raise ValueError("The corrected MaleCNS downstream graph has no frozen T4a/T4d path contrast")
        return cls(
            t4_ids=t4_ids,
            middle_ids=middle_ids,
            descending_ids=descending_ids,
            t4_types=tuple(node_types[body_id] for body_id in t4_ids),
            middle_types=tuple(node_types[body_id] for body_id in middle_ids),
            descending_types=tuple(node_types[body_id] for body_id in descending_ids),
            t4_to_middle=t4_to_middle,
            middle_to_descending=middle_to_descending,
            t4_type_paths=type_paths,
            t4a_path=type_paths["T4a"],
            t4d_path=type_paths["T4d"],
            bridge_weights=raw_weights / denominator,
        )

    def propagate(
        self,
        t4_activity: np.ndarray,
        *,
        ablate_t4_projection: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros_like(self.t4_to_middle) if ablate_t4_projection else self.t4_to_middle
        middle = matrix @ np.asarray(t4_activity, dtype=np.float64)
        descending = self.middle_to_descending @ middle
        return middle, descending

    def coverage(self) -> dict[str, dict[str, int]]:
        """Report direct and two-hop support for every T4 subtype."""

        result: dict[str, dict[str, int]] = {}
        for t4_type in T4_TYPES:
            indices = [i for i, value in enumerate(self.t4_types) if value == t4_type]
            direct = self.t4_to_middle[:, indices].sum(axis=0) > 0
            two_hop = self.t4_to_middle[:, indices].T @ self.middle_to_descending.T
            result[t4_type] = {
                "total_neurons": len(indices),
                "neurons_with_direct_target": int(np.count_nonzero(direct)),
                "neurons_with_target_to_descending_path": int(np.count_nonzero(np.any(two_hop > 0, axis=1))),
            }
        return result
