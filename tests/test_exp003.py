import numpy as np
import pandas as pd

from src.exp003 import ClosedLoopMotionScene, DownstreamMatrices, SteeringBridge
from src.malecns_io import ConnectomeGraph


def _downstream_graph() -> ConnectomeGraph:
    nodes = pd.DataFrame(
        [
            {"bodyId": 1, "type": "T4a", "stage": "t4_motion"},
            {"bodyId": 2, "type": "T4d", "stage": "t4_motion"},
            {"bodyId": 3, "type": "T4b", "stage": "t4_motion"},
            {"bodyId": 10, "type": "HSS", "stage": "wide_field_projection"},
            {"bodyId": 11, "type": "VSm", "stage": "wide_field_projection"},
            {"bodyId": 20, "type": "DNa02", "stage": "descending_output"},
            {"bodyId": 21, "type": "DNg46", "stage": "descending_output"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"pre_body": 1, "post_body": 10, "synapse_count": 5},
            {"pre_body": 2, "post_body": 11, "synapse_count": 7},
            {"pre_body": 10, "post_body": 20, "synapse_count": 3},
            {"pre_body": 11, "post_body": 21, "synapse_count": 4},
        ]
    )
    return ConnectomeGraph(nodes, edges)


def test_downstream_readout_uses_connectivity_and_ablation_is_zero():
    matrices = DownstreamMatrices.from_graph(_downstream_graph())
    wide, descending = matrices.propagate(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(wide, [1.0, 0.0])
    assert np.allclose(descending, [1.0, 0.0])
    wide_ablated, descending_ablated = matrices.propagate(
        np.array([1.0, 0.0, 0.0]), ablate_t4_projection=True
    )
    assert np.allclose(wide_ablated, 0.0)
    assert np.allclose(descending_ablated, 0.0)
    assert matrices.t4_types == ("T4a", "T4d", "T4b")
    assert matrices.t4a_path[0] > 0
    assert matrices.t4d_path[1] > 0


def test_bridge_is_a_fixed_downstream_activity_readout():
    matrices = DownstreamMatrices.from_graph(_downstream_graph())
    bridge = SteeringBridge()
    a_command = bridge.raw_command(np.array([1.0, 0.0]), matrices)
    d_command = bridge.raw_command(np.array([0.0, 1.0]), matrices)
    assert a_command > 0
    assert d_command < 0
    assert np.allclose(bridge.action(0.0), [1.0, 1.0])
    assert bridge.action(a_command)[0] > bridge.action(a_command)[1]


def test_closed_loop_scene_reverses_column_order_and_applies_yaw_shift():
    forward = ClosedLoopMotionScene(reverse=False)
    reverse = ClosedLoopMotionScene(reverse=True)
    assert np.allclose(forward.column_drive(12.0, 0.0), [1.0, 0.0])
    assert np.allclose(reverse.column_drive(12.0, 0.0), [0.0, 1.0])
    assert np.allclose(forward.column_drive(12.0, 0.2), [0.0, 0.0])
    assert np.allclose(forward.column_drive(25.0, 0.0), [0.0, 1.0])
