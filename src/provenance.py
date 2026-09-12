"""Validation helpers for local, ignored MaleCNS materializations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_materialization(
    manifest: Mapping[str, object],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    """Raise ``ValueError`` when a graph disagrees with its manifest."""

    required_manifest = {"dataset", "release", "node_count", "edge_count", "stage_counts"}
    missing_manifest = required_manifest - set(manifest)
    if missing_manifest:
        raise ValueError(f"manifest missing fields: {sorted(missing_manifest)}")
    required_nodes = {"bodyId", "stage"}
    required_edges = {"pre_body", "post_body", "synapse_count"}
    if missing := required_nodes - set(nodes.columns):
        raise ValueError(f"nodes missing fields: {sorted(missing)}")
    if missing := required_edges - set(edges.columns):
        raise ValueError(f"edges missing fields: {sorted(missing)}")
    if nodes["bodyId"].duplicated().any():
        raise ValueError("node body IDs must be unique")
    if int(manifest["node_count"]) != len(nodes):
        raise ValueError("manifest node_count does not match nodes")
    if int(manifest["edge_count"]) != len(edges):
        raise ValueError("manifest edge_count does not match edges")
    body_ids = set(nodes["bodyId"].astype("int64"))
    endpoints = set(edges["pre_body"].astype("int64")) | set(
        edges["post_body"].astype("int64")
    )
    if outside := endpoints - body_ids:
        raise ValueError(f"edge endpoints absent from nodes: {sorted(outside)[:5]}")
    if not pd.to_numeric(edges["synapse_count"], errors="coerce").notna().all():
        raise ValueError("all synapse counts must be numeric")
    if (pd.to_numeric(edges["synapse_count"]) < 0).any():
        raise ValueError("synapse counts cannot be negative")
    actual_stages = nodes["stage"].value_counts().to_dict()
    expected_stages = {
        str(key): int(value)
        for key, value in dict(manifest["stage_counts"]).items()
    }
    if actual_stages != expected_stages:
        raise ValueError(
            f"manifest stage_counts do not match nodes: {expected_stages} != {actual_stages}"
        )
