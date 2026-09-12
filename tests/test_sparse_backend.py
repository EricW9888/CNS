import numpy as np
import pandas as pd
import pytest

from src.malecns_io import ConnectomeGraph
from src.sparse_backend import StateRecorder, compile_projection


def _graph() -> ConnectomeGraph:
    nodes = pd.DataFrame(
        {
            "bodyId": [1, 2, 10, 11],
            "type": ["T4a", "T4b", "HS", "DNp15"],
            "stage": ["t4_motion", "t4_motion", "wide", "descending"],
        }
    )
    edges = pd.DataFrame(
        {
            "pre_body": [1, 1, 2],
            "post_body": [10, 10, 10],
            "synapse_count": [2.0, 3.0, 5.0],
            "effective_sign": [1.0, 1.0, -1.0],
        }
    )
    return ConnectomeGraph(nodes, edges)


def test_sparse_projection_sums_duplicates_normalizes_and_batches():
    projection = compile_projection(
        _graph(), source_ids=[1, 2], target_ids=[10, 11], normalization="sum"
    )
    assert projection.structural_edge_rows == 3
    assert projection.matrix.nnz == 2
    assert np.allclose(projection.matrix.toarray(), [[0.5, 0.5], [0.0, 0.0]])
    assert np.allclose(projection.apply([2.0, 4.0]), [3.0, 0.0])
    assert np.allclose(
        projection.apply([[2.0, 4.0], [4.0, 2.0]]),
        [[3.0, 0.0], [3.0, 0.0]],
    )


def test_signed_sparse_projection_uses_absolute_normalization():
    projection = compile_projection(
        _graph(),
        source_ids=[1, 2],
        target_ids=[10],
        normalization="absolute",
        sign_column="effective_sign",
    )
    assert np.allclose(projection.matrix.toarray(), [[0.5, -0.5]])
    with pytest.raises(ValueError, match="nonnegative"):
        compile_projection(
            _graph(),
            source_ids=[1, 2],
            target_ids=[10],
            normalization="sum",
            sign_column="effective_sign",
        )


def test_state_recorder_records_only_requested_nodes():
    recorder = StateRecorder.create([1, 2, 10], [10, 1], n_steps=2)
    recorder.record(0, np.asarray([1.0, 2.0, 3.0]))
    recorder.record(1, np.asarray([4.0, 5.0, 6.0]))
    assert recorder.node_ids.tolist() == [10, 1]
    assert np.allclose(recorder.values, [[3.0, 1.0], [6.0, 4.0]])
    assert recorder.values.dtype == np.float32
