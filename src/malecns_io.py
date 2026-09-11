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
    body_col = "body" if "body" in frame.columns else "bodyId"
    frame = frame.rename(columns={body_col: "bodyId"})
    frame["bodyId"] = frame["bodyId"].astype("int64")
    # The body-level export may contain repeated body/type records.  Keep the
    # first record, which is enough for the curated subgraph and deterministic.
    return frame.drop_duplicates("bodyId", keep="first")


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
    nodes["bodyId"] = nodes["bodyId"].astype("int64")
    if transmitters is not None:
        tx = transmitters.copy()
        nodes = nodes.merge(tx, on="bodyId", how="left", suffixes=("", "_tx"))
    nodes["nt_label"] = nodes.apply(transmitter_label, axis=1)
    nodes["nt_sign"] = nodes["nt_label"].map(lambda value: transmitter_sign(value, unknown_sign))

    node_set = set(nodes["bodyId"].tolist())
    edges = edges.copy()
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

    index = {int(body_id): i for i, body_id in enumerate(graph.node_ids)}
    usable = graph.edges[graph.edges["effective_sign"] != 0].copy()
    pre_index = usable["pre_body"].map(index).to_numpy(dtype=np.int64)
    post_index = usable["post_body"].map(index).to_numpy(dtype=np.int64)
    values = (
        usable["synapse_count"].to_numpy(dtype=np.float32)
        * usable["effective_sign"].to_numpy(dtype=np.float32)
        * np.float32(weight_per_synapse_mV)
    )
    matrix = sparse.csr_matrix((values, (post_index, pre_index)), shape=(graph.n_nodes, graph.n_nodes))
    return matrix, usable
