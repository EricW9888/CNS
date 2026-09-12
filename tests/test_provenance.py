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
