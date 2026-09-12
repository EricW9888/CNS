import pandas as pd
import pytest

from src.provenance import validate_materialization


def _frames():
    nodes = pd.DataFrame(
        {"bodyId": [1, 2], "stage": ["input", "output"], "type": ["L1", "T4a"]}
    )
    edges = pd.DataFrame(
        {"pre_body": [1], "post_body": [2], "synapse_count": [7]}
    )
    manifest = {
        "dataset": "male-cns:v1.0",
        "release": "MaleCNS v1.0",
        "node_count": 2,
        "edge_count": 1,
        "stage_counts": {"input": 1, "output": 1},
    }
    return manifest, nodes, edges


def test_materialization_contract_accepts_consistent_graph():
    validate_materialization(*_frames())


def test_materialization_contract_rejects_counts_and_external_endpoints():
    manifest, nodes, edges = _frames()
    manifest["edge_count"] = 2
    with pytest.raises(ValueError, match="edge_count"):
        validate_materialization(manifest, nodes, edges)

    manifest["edge_count"] = 1
    edges.loc[0, "post_body"] = 3
    with pytest.raises(ValueError, match="endpoints"):
        validate_materialization(manifest, nodes, edges)

    edges.loc[0, "post_body"] = 2
    edges.loc[0, "synapse_count"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_materialization(manifest, nodes, edges)

    duplicate_edges = pd.concat([_frames()[2], _frames()[2]], ignore_index=True)
    duplicate_manifest = _frames()[0]
    duplicate_manifest["edge_count"] = 2
    with pytest.raises(ValueError, match="unique"):
        validate_materialization(duplicate_manifest, _frames()[1], duplicate_edges)


def test_materialization_contract_rejects_nonfinite_weight():
    manifest, nodes, edges = _frames()
    edges["synapse_count"] = edges["synapse_count"].astype(float)
    edges.loc[0, "synapse_count"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_materialization(manifest, nodes, edges)
