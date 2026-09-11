import numpy as np
import pandas as pd

from src.exp003_corrected import CorrectedDownstreamMatrices
from src.malecns_io import build_graph


def _synthetic_corrected_graph():
    rows = []
    for body_id, cell_type, stage, superclass in (
        (1, "T4a", "t4_motion", "optic_lobe_intrinsic"),
        (2, "T4b", "t4_motion", "optic_lobe_intrinsic"),
        (3, "T4c", "t4_motion", "optic_lobe_intrinsic"),
        (4, "T4d", "t4_motion", "optic_lobe_intrinsic"),
        (11, "LPi1", "lobula_plate_target", "ol_intrinsic"),
        (12, "LPLC1", "lobula_plate_target", "visual_projection"),
        (13, "LLPC1", "lobula_plate_target", "visual_projection"),
        (14, "VS", "lobula_plate_target", "visual_projection"),
        (21, "DN1", "descending_output", "descending_neuron"),
        (22, "DN2", "descending_output", "descending_neuron"),
    ):
        rows.append(
            {
                "bodyId": body_id,
                "type": cell_type,
                "instance": f"{cell_type}_{body_id}",
                "somaSide": "R",
                "superclass": superclass,
                "stage": stage,
            }
        )
    nodes = pd.DataFrame(rows)
    edges = pd.DataFrame(
        {
            "pre_body": [1, 2, 3, 4, 11, 12, 13, 14],
            "post_body": [11, 12, 13, 14, 21, 21, 22, 22],
            "synapse_count": [5, 4, 3, 2, 1, 1, 1, 1],
        }
    )
    return build_graph(nodes, edges)


def test_corrected_path_covers_all_t4_subtypes_and_propagates():
    matrices = CorrectedDownstreamMatrices.from_graph(_synthetic_corrected_graph())

    coverage = matrices.coverage()
    assert all(coverage[t]["neurons_with_direct_target"] == 1 for t in ("T4a", "T4b", "T4c", "T4d"))
    assert all(coverage[t]["neurons_with_target_to_descending_path"] == 1 for t in ("T4a", "T4b", "T4c", "T4d"))

    middle, descending = matrices.propagate(np.ones(4))
    assert middle.shape == (4,)
    assert descending.shape == (2,)
    assert np.all(descending > 0)


def test_corrected_projection_ablation_removes_downstream_activity_only():
    matrices = CorrectedDownstreamMatrices.from_graph(_synthetic_corrected_graph())
    middle, descending = matrices.propagate(np.ones(4), ablate_t4_projection=True)
    assert np.allclose(middle, 0.0)
    assert np.allclose(descending, 0.0)
    assert np.isfinite(matrices.bridge_weights).all()
