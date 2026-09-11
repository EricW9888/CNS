import numpy as np

from src.exp003 import SteeringBridge
from src.exp003_bilateral import BilateralDownstreamReadout
from src.exp003_corrected import CorrectedDownstreamMatrices


def _matrices(side: str) -> CorrectedDownstreamMatrices:
    type_paths = {
        "T4a": np.asarray([1.0, 0.0]),
        "T4b": np.asarray([0.5, 0.0]),
        "T4c": np.asarray([0.0, 0.5]),
        "T4d": np.asarray([0.0, 1.0]),
    }
    return CorrectedDownstreamMatrices(
        t4_ids=(1, 2, 3, 4),
        middle_ids=(11,),
        descending_ids=(21, 22),
        t4_types=("T4a", "T4b", "T4c", "T4d"),
        middle_types=("HSS",),
        descending_types=("DNp15", "DNa02"),
        t4_to_middle=np.ones((1, 4)),
        middle_to_descending=np.ones((2, 1)),
        t4_type_paths=type_paths,
        t4a_path=type_paths["T4a"],
        t4d_path=type_paths["T4d"],
        bridge_weights=np.asarray([0.5, -0.5]),
    )


def test_bilateral_readout_uses_homologous_dn_difference():
    readout = BilateralDownstreamReadout.from_matrices(_matrices("left"), _matrices("right"))
    activities = readout.activities(np.zeros(4), np.ones(4))
    assert activities["left_DNp15"] == 0.0
    assert activities["right_DNp15"] > 0.0
    assert activities["right_minus_left_DNp15"] > 0.0
    raw, action = readout.steering_command(activities, SteeringBridge())
    assert raw > 0.0
    assert action[0] > action[1]


def test_bilateral_pathway_ablation_removes_both_side_difference():
    readout = BilateralDownstreamReadout.from_matrices(_matrices("left"), _matrices("right"))
    activities = readout.activities(np.zeros(4), np.ones(4), ablate=True)
    assert activities["right_minus_left_DNp15"] == 0.0
    assert activities["right_minus_left_DNa02"] == 0.0
