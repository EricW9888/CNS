"""Validated conversion from query rows to reproducible graph bundles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


NODE_COLUMNS = (
    "bodyId",
    "type",
    "instance",
    "somaSide",
    "superclass",
    "stage",
    "status",
    "consensus_nt",
    "predicted_nt",
    "predicted_nt_confidence",
)


def canonical_edge_frame(rows: Iterable[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    """Validate aggregate chemical edges without silently dropping duplicates."""

    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    required = {"pre_body", "post_body", "synapse_count"}
    if missing := required - set(frame.columns):
        raise ValueError(f"edge rows missing fields: {sorted(missing)}")
    output = frame[["pre_body", "post_body", "synapse_count"]].copy()
    output["pre_body"] = pd.to_numeric(output["pre_body"], errors="raise").astype("int64")
    output["post_body"] = pd.to_numeric(output["post_body"], errors="raise").astype("int64")
    output["synapse_count"] = pd.to_numeric(
        output["synapse_count"], errors="raise"
    ).astype("float32")
    if not np.isfinite(output["synapse_count"].to_numpy(dtype=np.float64)).all():
        raise ValueError("edge synapse counts must be finite")
    if (output["synapse_count"] <= 0).any():
        raise ValueError("edge synapse counts must be positive")
    duplicate = output.duplicated(["pre_body", "post_body"], keep=False)
    if duplicate.any():
        examples = output.loc[duplicate, ["pre_body", "post_body"]].head(5)
        raise ValueError(
            "query returned duplicate aggregate edge pairs: "
            f"{list(examples.itertuples(index=False, name=None))}"
        )
    return output.reset_index(drop=True)


def materialize_nodes(
    node_ids: Iterable[int],
    *,
    stages: Mapping[int, str],
    annotations: pd.DataFrame,
    transmitters: pd.DataFrame,
    default_stage: str = "intermediate",
) -> pd.DataFrame:
    """Select complete node metadata and fail on ambiguous or missing bodies."""

    requested = {int(value) for value in node_ids}
    annotation_frame = annotations.copy()
    if "bodyId" not in annotation_frame.columns:
        raise ValueError("annotations must contain bodyId")
    annotation_frame["bodyId"] = annotation_frame["bodyId"].astype("int64")
    if annotation_frame["bodyId"].duplicated().any():
        raise ValueError("annotations contain duplicate body IDs")

    transmitter_frame = transmitters.copy()
    body_column = "bodyId" if "bodyId" in transmitter_frame.columns else "body"
    if body_column not in transmitter_frame.columns:
        raise ValueError("transmitters must contain body or bodyId")
    transmitter_frame = transmitter_frame.rename(columns={body_column: "bodyId"})
    transmitter_frame["bodyId"] = transmitter_frame["bodyId"].astype("int64")
    if transmitter_frame["bodyId"].duplicated().any():
        raise ValueError("transmitters contain duplicate body IDs")

    nodes = annotation_frame[annotation_frame["bodyId"].isin(requested)].copy()
    found = set(nodes["bodyId"].tolist())
    if missing := requested - found:
        raise ValueError(f"annotations are missing requested body IDs: {sorted(missing)[:5]}")
    transmitter_columns = [
        "bodyId",
        "consensus_nt",
        "predicted_nt",
        "predicted_nt_confidence",
    ]
    if missing := set(transmitter_columns) - set(transmitter_frame.columns):
        raise ValueError(f"transmitters missing fields: {sorted(missing)}")
    collisions = (set(transmitter_columns) - {"bodyId"}) & set(nodes.columns)
    if collisions:
        raise ValueError(f"annotation/transmitter columns are ambiguous: {sorted(collisions)}")
    nodes = nodes.merge(
        transmitter_frame.loc[
            transmitter_frame["bodyId"].isin(requested), transmitter_columns
        ],
        on="bodyId",
        how="left",
        validate="one_to_one",
    )
    nodes["stage"] = nodes["bodyId"].map(
        {int(key): str(value) for key, value in stages.items()}
    ).fillna(default_stage)
    if missing := set(NODE_COLUMNS) - set(nodes.columns):
        raise ValueError(f"materialized nodes missing fields: {sorted(missing)}")
    return nodes[list(NODE_COLUMNS)].reset_index(drop=True)
