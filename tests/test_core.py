from pathlib import Path

import pandas as pd
import pytest

from src.malecns_io import build_graph, effective_sparse_matrix, transmitter_sign


def test_transmitter_sign_is_conservative():
    assert transmitter_sign("acetylcholine") == 1
    assert transmitter_sign("GABA") == -1
    assert transmitter_sign("dopamine") == 0


def test_sparse_matrix_preserves_direction_and_weight():
    nodes = pd.DataFrame(
        {
            "bodyId": [1, 2],
            "type": ["L1", "Mi1"],
            "instance": ["L1_L", "Mi1_L"],
            "somaSide": ["L", "L"],
            "superclass": ["ol_intrinsic", "ol_intrinsic"],
            "stage": ["visual_input", "medulla"],
            "consensus_nt": ["acetylcholine", "acetylcholine"],
        }
    )
    edges = pd.DataFrame({"pre_body": [1], "post_body": [2], "synapse_count": [4]})
    graph = build_graph(nodes, edges)
    matrix, used = effective_sparse_matrix(graph, weight_per_synapse_mV=0.275)
    assert matrix.shape == (2, 2)
    assert matrix[1, 0] == 1.1
    assert used.iloc[0]["pre_body"] == 1


def test_build_graph_rejects_duplicate_nodes_and_nonpositive_edges():
    nodes = pd.DataFrame(
        {
            "bodyId": [1, 2],
            "type": ["L1", "Mi1"],
            "instance": ["L1_L", "Mi1_L"],
            "somaSide": ["L", "L"],
            "superclass": ["ol_intrinsic", "ol_intrinsic"],
            "stage": ["visual_input", "medulla"],
            "consensus_nt": ["acetylcholine", "acetylcholine"],
        }
    )
    edge = pd.DataFrame({"pre_body": [1], "post_body": [2], "synapse_count": [1]})
    with pytest.raises(ValueError, match="duplicate body IDs"):
        build_graph(pd.concat([nodes, nodes.iloc[[0]]], ignore_index=True), edge)
    edge.loc[0, "synapse_count"] = 0
    with pytest.raises(ValueError, match="positive"):
        build_graph(nodes, edge)


def test_sparse_matrix_sums_duplicate_edge_rows():
    nodes = pd.DataFrame(
        {
            "bodyId": [1, 2],
            "type": ["L1", "Mi1"],
            "instance": ["L1_L", "Mi1_L"],
            "somaSide": ["L", "L"],
            "superclass": ["ol_intrinsic", "ol_intrinsic"],
            "stage": ["visual_input", "medulla"],
            "consensus_nt": ["acetylcholine", "acetylcholine"],
        }
    )
    edges = pd.DataFrame(
        {"pre_body": [1, 1], "post_body": [2, 2], "synapse_count": [2, 3]}
    )
    graph = build_graph(nodes, edges)
    matrix, used = effective_sparse_matrix(graph, weight_per_synapse_mV=0.5)
    assert len(used) == 2
    assert matrix.nnz == 1
    assert matrix[1, 0] == pytest.approx(2.5)
