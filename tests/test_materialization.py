import pandas as pd
import pytest

from src.materialization import canonical_edge_frame, materialize_nodes


def test_canonical_edge_frame_rejects_ambiguous_duplicate_pairs():
    rows = [
        {"pre_body": 1, "post_body": 2, "synapse_count": 3},
        {"pre_body": 1, "post_body": 2, "synapse_count": 3},
    ]
    with pytest.raises(ValueError, match="duplicate aggregate edge"):
        canonical_edge_frame(rows)


def test_materialize_nodes_requires_complete_unique_metadata():
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2],
            "type": ["L1", "Mi1"],
            "instance": ["L1_R", "Mi1_R"],
            "somaSide": ["R", "R"],
            "superclass": ["optic", "optic"],
            "status": ["Traced", "Traced"],
        }
    )
    transmitters = pd.DataFrame(
        {
            "body": [1, 2],
            "consensus_nt": ["glutamate", "acetylcholine"],
            "predicted_nt": ["glutamate", "acetylcholine"],
            "predicted_nt_confidence": [0.9, 0.8],
        }
    )
    nodes = materialize_nodes(
        [2, 1],
        stages={1: "visual", 2: "medulla"},
        annotations=annotations,
        transmitters=transmitters,
    )
    assert nodes["bodyId"].tolist() == [1, 2]
    assert nodes["stage"].tolist() == ["visual", "medulla"]

    with pytest.raises(ValueError, match="missing requested"):
        materialize_nodes(
            [1, 3],
            stages={1: "visual", 3: "medulla"},
            annotations=annotations,
            transmitters=transmitters,
        )
