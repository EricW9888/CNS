"""Sparse, array-first building blocks for future connectome models.

Historical experiment modules keep their recorded model semantics. This module
provides the representation boundary for successor experiments: Pandas tables
are compiled once, numerical updates use sparse matrices and contiguous arrays,
and only explicitly selected states need to be recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from .malecns_io import ConnectomeGraph


Normalization = Literal["absolute", "sum"] | None


def _unique_ids(values: Iterable[int], *, name: str) -> np.ndarray:
    ids = np.fromiter((int(value) for value in values), dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} must contain unique body IDs")
    return ids


@dataclass(frozen=True)
class SparseProjection:
    """A target-by-source sparse operator with explicit body-ID axes."""

    source_ids: np.ndarray
    target_ids: np.ndarray
    matrix: sparse.csr_matrix
    structural_edge_rows: int

    def apply(self, source_activity: np.ndarray) -> np.ndarray:
        """Apply the projection to one vector or a time-by-source matrix."""

        values = np.asarray(source_activity, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("source_activity must be finite")
        if values.ndim == 1:
            if values.shape != (len(self.source_ids),):
                raise ValueError("source_activity length does not match source_ids")
            return np.asarray(self.matrix @ values).reshape(-1)
        if values.ndim == 2:
            if values.shape[1] != len(self.source_ids):
                raise ValueError("source_activity columns do not match source_ids")
            return np.asarray(self.matrix @ values.T).T
        raise ValueError("source_activity must be one- or two-dimensional")

    @property
    def storage_bytes(self) -> int:
        """Bytes used by CSR data and index arrays, excluding Python objects."""

        return int(
            self.matrix.data.nbytes
            + self.matrix.indices.nbytes
            + self.matrix.indptr.nbytes
        )


def compile_projection(
    graph: ConnectomeGraph,
    *,
    source_ids: Iterable[int],
    target_ids: Iterable[int],
    normalization: Normalization = "sum",
    sign_column: str | None = None,
) -> SparseProjection:
    """Compile selected graph edges into a CSR projection.

    Duplicate pre/post rows are summed by the sparse constructor. ``sum`` is
    appropriate for nonnegative synapse-count projections; ``absolute`` keeps
    signed operators bounded by total absolute input mass.
    """

    sources = _unique_ids(source_ids, name="source_ids")
    targets = _unique_ids(target_ids, name="target_ids")
    graph_ids = set(graph.node_ids.tolist())
    outside = (set(sources.tolist()) | set(targets.tolist())) - graph_ids
    if outside:
        raise ValueError(f"projection axes contain body IDs outside graph.nodes: {sorted(outside)[:5]}")
    if normalization not in (None, "sum", "absolute"):
        raise ValueError(f"unsupported normalization: {normalization!r}")
    if sign_column is not None and sign_column not in graph.edges.columns:
        raise ValueError(f"graph edges do not contain sign column {sign_column!r}")

    selected = graph.edges[
        graph.edges["pre_body"].isin(sources)
        & graph.edges["post_body"].isin(targets)
    ]
    source_index = pd.Index(sources)
    target_index = pd.Index(targets)
    columns = source_index.get_indexer(selected["pre_body"].to_numpy(dtype=np.int64))
    rows = target_index.get_indexer(selected["post_body"].to_numpy(dtype=np.int64))
    weights = selected["synapse_count"].to_numpy(dtype=np.float64)
    if sign_column is not None:
        weights = weights * selected[sign_column].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("projection weights must be finite")

    matrix = sparse.csr_matrix(
        (weights, (rows, columns)),
        shape=(len(targets), len(sources)),
        dtype=np.float64,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if normalization is not None and matrix.nnz:
        if normalization == "sum" and np.any(matrix.data < 0):
            raise ValueError("sum normalization requires nonnegative weights")
        mass_matrix = abs(matrix) if normalization == "absolute" else matrix
        row_mass = np.asarray(mass_matrix.sum(axis=1)).reshape(-1)
        inverse = np.divide(
            1.0,
            row_mass,
            out=np.zeros_like(row_mass),
            where=row_mass > 0,
        )
        matrix = sparse.diags(inverse, format="csr") @ matrix
    matrix.sort_indices()
    return SparseProjection(sources, targets, matrix, int(len(selected)))


@dataclass
class StateRecorder:
    """Preallocated selected-state recorder for array-first simulations."""

    node_ids: np.ndarray
    indices: np.ndarray
    values: np.ndarray

    @classmethod
    def create(
        cls,
        graph_node_ids: Iterable[int],
        recorded_body_ids: Iterable[int],
        *,
        n_steps: int,
        dtype: np.dtype | type = np.float32,
    ) -> "StateRecorder":
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        graph_ids = _unique_ids(graph_node_ids, name="graph_node_ids")
        recorded = _unique_ids(recorded_body_ids, name="recorded_body_ids")
        indices = pd.Index(graph_ids).get_indexer(recorded)
        if np.any(indices < 0):
            missing = recorded[indices < 0].tolist()
            raise ValueError(f"recorded body IDs are outside the graph: {missing[:5]}")
        return cls(
            node_ids=recorded,
            indices=indices.astype(np.int64, copy=False),
            values=np.empty((n_steps, len(recorded)), dtype=dtype),
        )

    def record(self, step: int, state: np.ndarray) -> None:
        if not 0 <= step < len(self.values):
            raise IndexError("recording step is outside the allocated trace")
        state_values = np.asarray(state)
        if state_values.ndim != 1 or (
            len(self.indices) and int(self.indices.max()) >= len(state_values)
        ):
            raise ValueError("state vector does not cover recorder indices")
        self.values[step] = state_values[self.indices]
