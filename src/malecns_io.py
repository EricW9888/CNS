"""MaleCNS file loading and connectome-to-sparse-matrix conversion.

The loader intentionally understands both the official bulk Feather export and
the small Parquet bundle materialized by ``scripts/query_malecns_subgraph.py``.
Biological annotations are kept alongside edges so that the effective sign of
an edge can be inspected instead of being hidden in a learned weight matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


POSITIVE_TRANSMITTERS = {"acetylcholine"}
NEGATIVE_TRANSMITTERS = {"gaba", "glutamate"}


@dataclass(frozen=True)
class ConnectomeGraph:
    """A directed graph with pre->post edges and inspectable metadata."""

    nodes: pd.DataFrame
    edges: pd.DataFrame

    @property
    def node_ids(self) -> np.ndarray:
        return self.nodes["bodyId"].to_numpy(dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def with_edges(self, edges: pd.DataFrame) -> "ConnectomeGraph":
        return ConnectomeGraph(self.nodes.copy(), edges.reset_index(drop=True))


def load_annotations(path: str | Path) -> pd.DataFrame:
    """Load the official body annotation Feather export."""

    frame = pd.read_feather(path)
    if "bodyId" not in frame.columns:
        raise ValueError(f"{path} does not contain the required bodyId column")
    frame = frame.copy()
    frame["bodyId"] = frame["bodyId"].astype("int64")
    return frame


def load_transmitters(path: str | Path) -> pd.DataFrame:
    """Load transmitter predictions and normalize the body identifier."""

    frame = pd.read_feather(path).copy()
    body_col = _first_existing(frame, ("body", "bodyId"))
    frame = frame.rename(columns={body_col: "bodyId"})
    frame["bodyId"] = frame["bodyId"].astype("int64")
    if frame["bodyId"].duplicated().any():
        examples = frame.loc[frame["bodyId"].duplicated(keep=False), "bodyId"].head(5).tolist()
        raise ValueError(f"Transmitter export contains duplicate body IDs: {examples}")
    return frame


def transmitter_label(row: pd.Series) -> str:
    for column in ("consensus_nt", "predicted_nt", "celltype_predicted_nt"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip().lower()
    return "unknown"


def transmitter_sign(label: str, unknown_sign: float = 0.0) -> float:
    """Map only defensible fast-transmitter signs.

    Acetylcholine is treated as excitatory and GABA/glutamate as inhibitory in
    this baseline.  Dopamine, serotonin, octopamine, histamine and unknown
    labels are left at ``unknown_sign`` because their net effect is
    receptor/circuit dependent and is not resolved by this export alone.
    """

    normalized = label.strip().lower()
    if normalized in POSITIVE_TRANSMITTERS:
        return 1.0
    if normalized in NEGATIVE_TRANSMITTERS:
        return -1.0
    return float(unknown_sign)


def _transmitter_labels(frame: pd.DataFrame) -> pd.Series:
    """Resolve transmitter labels for a frame without row-wise Python calls."""

    labels = np.full(len(frame), "unknown", dtype=object)
    # Iterate from lowest to highest priority so the highest-priority nonempty
    # value is the final assignment.
    for column in ("celltype_predicted_nt", "predicted_nt", "consensus_nt"):
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").str.strip().str.lower()
        valid = values.notna() & values.ne("")
        labels[valid.to_numpy()] = values.loc[valid].to_numpy(dtype=object)
    return pd.Series(labels, index=frame.index, dtype="object")


def _first_existing(frame: pd.DataFrame, names: Iterable[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"None of {tuple(names)} found in columns {tuple(frame.columns)}")


def load_edges(path: str | Path) -> pd.DataFrame:
    """Load official connectome weights or a normalized experiment edge file."""

    suffix = Path(path).suffix.lower()
    frame = pd.read_feather(path) if suffix == ".feather" else pd.read_parquet(path)
    pre = _first_existing(frame, ("pre_body", "body_pre", "bodyId_pre", "Presynaptic_Body"))
    post = _first_existing(frame, ("post_body", "body_post", "bodyId_post", "Postsynaptic_Body"))
    weight = _first_existing(frame, ("synapse_count", "weight", "synweight", "count", "bodyCount"))
    out = pd.DataFrame(
        {
            "pre_body": frame[pre].astype("int64"),
            "post_body": frame[post].astype("int64"),
            "synapse_count": frame[weight].astype("float32"),
        }
    )
    return out[out["synapse_count"] > 0].reset_index(drop=True)


def build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    transmitters: pd.DataFrame | None = None,
    *,
    unknown_sign: float = 0.0,
) -> ConnectomeGraph:
    """Join annotations/transmitters and retain only edges in the node set."""

    nodes = nodes.copy()
    required_node_columns = {
        "bodyId", "type", "instance", "somaSide", "superclass", "stage"
    }
    if missing := required_node_columns - set(nodes.columns):
        raise ValueError(f"nodes missing required columns: {sorted(missing)}")
    nodes["bodyId"] = nodes["bodyId"].astype("int64")
    if nodes["bodyId"].duplicated().any():
        examples = nodes.loc[nodes["bodyId"].duplicated(keep=False), "bodyId"].head(5).tolist()
        raise ValueError(f"nodes contain duplicate body IDs: {examples}")
    if transmitters is not None:
        tx = transmitters.copy()
        if "bodyId" not in tx.columns:
            raise ValueError("transmitters must contain a normalized bodyId column")
        tx["bodyId"] = tx["bodyId"].astype("int64")
        if tx["bodyId"].duplicated().any():
            examples = tx.loc[tx["bodyId"].duplicated(keep=False), "bodyId"].head(5).tolist()
            raise ValueError(f"transmitters contain duplicate body IDs: {examples}")
        # Avoid hashing a whole-brain transmitter table when materializing a
        # small circuit.
        tx = tx[tx["bodyId"].isin(nodes["bodyId"])].copy()
        nodes = nodes.merge(tx, on="bodyId", how="left", suffixes=("", "_tx"))
    nodes["nt_label"] = _transmitter_labels(nodes)
    signs = np.full(len(nodes), float(unknown_sign), dtype=np.float32)
    signs[nodes["nt_label"].isin(POSITIVE_TRANSMITTERS).to_numpy()] = 1.0
    signs[nodes["nt_label"].isin(NEGATIVE_TRANSMITTERS).to_numpy()] = -1.0
    nodes["nt_sign"] = signs

    node_set = set(nodes["bodyId"].tolist())
    edges = edges.copy()
    required_edge_columns = {"pre_body", "post_body", "synapse_count"}
    if missing := required_edge_columns - set(edges.columns):
        raise ValueError(f"edges missing required columns: {sorted(missing)}")
    edges["pre_body"] = pd.to_numeric(edges["pre_body"], errors="raise").astype("int64")
    edges["post_body"] = pd.to_numeric(edges["post_body"], errors="raise").astype("int64")
    edges["synapse_count"] = pd.to_numeric(edges["synapse_count"], errors="raise")
    if not np.isfinite(edges["synapse_count"].to_numpy(dtype=np.float64)).all():
        raise ValueError("edge synapse counts must be finite")
    if (edges["synapse_count"] <= 0).any():
        raise ValueError("edge synapse counts must be positive")
    edges = edges[edges["pre_body"].isin(node_set) & edges["post_body"].isin(node_set)].copy()
    edges = edges.merge(
        nodes[["bodyId", "type", "instance", "somaSide", "superclass", "nt_label", "nt_sign"]].rename(
            columns={
                "bodyId": "pre_body",
                "type": "pre_type",
                "instance": "pre_instance",
                "somaSide": "pre_side",
                "superclass": "pre_superclass",
                "nt_label": "pre_nt",
                "nt_sign": "pre_sign",
            }
        ),
        on="pre_body",
        how="left",
    )
    edges = edges.merge(
        nodes[["bodyId", "type", "instance", "somaSide", "superclass"]].rename(
            columns={
                "bodyId": "post_body",
                "type": "post_type",
                "instance": "post_instance",
                "somaSide": "post_side",
                "superclass": "post_superclass",
            }
        ),
        on="post_body",
        how="left",
    )
    edges["effective_sign"] = edges["pre_sign"].astype("float32")
    return ConnectomeGraph(nodes.reset_index(drop=True), edges.reset_index(drop=True))


def effective_sparse_matrix(
    graph: ConnectomeGraph,
    *,
    weight_per_synapse_mV: float,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Return post-by-pre coupling matrix and the rows used to construct it."""

    if not np.isfinite(weight_per_synapse_mV):
        raise ValueError("weight_per_synapse_mV must be finite")
    node_ids = pd.Index(graph.node_ids)
    usable = graph.edges[graph.edges["effective_sign"] != 0].copy()
    pre_index = node_ids.get_indexer(usable["pre_body"].to_numpy(dtype=np.int64))
    post_index = node_ids.get_indexer(usable["post_body"].to_numpy(dtype=np.int64))
    if np.any(pre_index < 0) or np.any(post_index < 0):
        raise ValueError("graph edges reference body IDs outside graph.nodes")
    values = (
        usable["synapse_count"].to_numpy(dtype=np.float32)
        * usable["effective_sign"].to_numpy(dtype=np.float32)
        * np.float32(weight_per_synapse_mV)
    )
    matrix = sparse.csr_matrix((values, (post_index, pre_index)), shape=(graph.n_nodes, graph.n_nodes))
    matrix.sort_indices()
    return matrix, usable
