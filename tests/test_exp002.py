import pandas as pd

from src.graded_model import GradedParameters, run_graded, t4_comparison
from src.malecns_io import ConnectomeGraph
from src.visual_stimulus import motion_events


def _synthetic_graph() -> ConnectomeGraph:
    nodes = pd.DataFrame(
        [
            {"bodyId": 1, "type": "L1", "instance": "L1_0", "somaSide": "R", "superclass": "", "stage": "visual_input"},
            {"bodyId": 2, "type": "L1", "instance": "L1_1", "somaSide": "R", "superclass": "", "stage": "visual_input"},
            {"bodyId": 10, "type": "Mi1", "instance": "Mi1_0", "somaSide": "R", "superclass": "", "stage": "medulla_excitation"},
            {"bodyId": 11, "type": "Tm3", "instance": "Tm3_1", "somaSide": "R", "superclass": "", "stage": "medulla_excitation"},
            {"bodyId": 20, "type": "Mi4", "instance": "Mi4_0", "somaSide": "R", "superclass": "", "stage": "t4_input_inhibition"},
            {"bodyId": 30, "type": "T4a", "instance": "T4a_0", "somaSide": "R", "superclass": "", "stage": "t4_motion"},
            {"bodyId": 31, "type": "T4b", "instance": "T4b_1", "somaSide": "R", "superclass": "", "stage": "t4_motion"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"pre_body": 1, "post_body": 10, "synapse_count": 10, "pre_type": "L1", "post_type": "Mi1"},
            {"pre_body": 2, "post_body": 11, "synapse_count": 10, "pre_type": "L1", "post_type": "Tm3"},
            {"pre_body": 10, "post_body": 30, "synapse_count": 5, "pre_type": "Mi1", "post_type": "T4a"},
            {"pre_body": 11, "post_body": 30, "synapse_count": 2, "pre_type": "Tm3", "post_type": "T4a"},
            {"pre_body": 10, "post_body": 31, "synapse_count": 2, "pre_type": "Mi1", "post_type": "T4b"},
            {"pre_body": 11, "post_body": 31, "synapse_count": 5, "pre_type": "Tm3", "post_type": "T4b"},
            {"pre_body": 20, "post_body": 30, "synapse_count": 4, "pre_type": "Mi4", "post_type": "T4a"},
            {"pre_body": 20, "post_body": 31, "synapse_count": 1, "pre_type": "Mi4", "post_type": "T4b"},
        ]
    )
    return ConnectomeGraph(nodes, edges)


def _events(reverse: bool = False):
    return motion_events(
        first_column="01_07",
        first_l1_ids=(1,),
        second_column="01_08",
        second_l1_ids=(2,),
        reverse=reverse,
        amplitude_mV=1.0,
    )


def test_graded_model_propagates_l1_to_mi_and_t4_without_spikes():
    graph = _synthetic_graph()
    columns = [
        {"suffix": "01_07", "l1_body_id": 1},
        {"suffix": "01_08", "l1_body_id": 2},
    ]
    trace, summary = run_graded(
        graph,
        _events(),
        motion_columns=columns,
        params=GradedParameters(),
        duration_ms=80.0,
    )
    assert summary["backend"] == "graded_local_t4_model"
    assert trace.loc[trace["type"].isin(["Mi1", "Tm3"]), "activity_proxy"].max() > 0
    assert trace.loc[trace["stage"] == "t4_motion", "activity_proxy"].max() > 0


def test_graded_model_order_difference_is_reduced_by_inhibitory_ablation():
    graph = _synthetic_graph()
    columns = [
        {"suffix": "01_07", "l1_body_id": 1},
        {"suffix": "01_08", "l1_body_id": 2},
    ]
    forward, _ = run_graded(graph, _events(), motion_columns=columns, duration_ms=80.0)
    reverse, _ = run_graded(graph, _events(reverse=True), motion_columns=columns, duration_ms=80.0)
    forward_ablated, _ = run_graded(
        graph, _events(), motion_columns=columns, duration_ms=80.0, ablate_inhibitory=True
    )
    reverse_ablated, _ = run_graded(
        graph, _events(reverse=True), motion_columns=columns, duration_ms=80.0, ablate_inhibitory=True
    )
    full_difference = t4_comparison(forward, reverse)["max_abs_pointwise_activity_difference"].mean()
    ablated_difference = t4_comparison(forward_ablated, reverse_ablated)["max_abs_pointwise_activity_difference"].mean()
    assert full_difference > 0
    assert ablated_difference != full_difference


def test_t4_comparison_reports_body_type_and_timing():
    graph = _synthetic_graph()
    columns = [
        {"suffix": "01_07", "l1_body_id": 1},
        {"suffix": "01_08", "l1_body_id": 2},
    ]
    forward, _ = run_graded(graph, _events(), motion_columns=columns, duration_ms=80.0)
    reverse, _ = run_graded(graph, _events(reverse=True), motion_columns=columns, duration_ms=80.0)
    comparison = t4_comparison(forward, reverse)
    assert set(comparison["bodyId"]) == {30, 31}
    assert set(comparison["type"]) == {"T4a", "T4b"}
    assert (comparison["time_of_max_abs_activity_difference_ms"] >= 0).all()
