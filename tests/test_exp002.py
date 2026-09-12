import numpy as np
import pandas as pd

from src.graded_model import (
    GradedParameters,
    _infer_column_coordinates,
    run_graded,
    t4_comparison,
)
from src.exp003 import OnlineGradedCircuit
from src.malecns_io import ConnectomeGraph
from src.visual_stimulus import VisualEvent, motion_events


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


def test_graded_model_order_difference_changes_under_inhibitory_ablation():
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
    assert ablated_difference > 0
    assert not np.isclose(ablated_difference, full_difference)


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


def test_coordinate_inference_matches_weighted_path_definition():
    coordinates = _infer_column_coordinates(
        _synthetic_graph(),
        [
            {"suffix": "01_07", "l1_body_id": 1},
            {"suffix": "01_08", "l1_body_id": 2},
        ],
    )
    assert np.allclose(
        coordinates,
        [0.0, 1.0, 0.0, 1.0, 13.0 / 35.0, 2.0 / 7.0, 5.0 / 7.0],
    )


def test_graded_model_is_exactly_deterministic():
    graph = _synthetic_graph()
    columns = [
        {"suffix": "01_07", "l1_body_id": 1},
        {"suffix": "01_08", "l1_body_id": 2},
    ]
    first, first_summary = run_graded(
        graph, _events(), motion_columns=columns, duration_ms=80.0
    )
    second, second_summary = run_graded(
        graph, _events(), motion_columns=columns, duration_ms=80.0
    )
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first_summary == second_summary


def test_graded_model_rejects_invalid_parameter_domains():
    with np.testing.assert_raises_regex(ValueError, "time constants"):
        run_graded(
            _synthetic_graph(),
            _events(),
            motion_columns=[
                {"suffix": "01_07", "l1_body_id": 1},
                {"suffix": "01_08", "l1_body_id": 2},
            ],
            params=GradedParameters(tau_t4_ms=0.0),
            duration_ms=80.0,
        )


def test_online_and_batch_paths_share_identical_graded_updates():
    graph = _synthetic_graph()
    columns = [
        {"suffix": "01_07", "l1_body_id": 1},
        {"suffix": "01_08", "l1_body_id": 2},
    ]
    params = GradedParameters()
    events = _events()
    duration_ms = 80.0
    batch, _ = run_graded(
        graph,
        events,
        motion_columns=columns,
        params=params,
        duration_ms=duration_ms,
    )
    t4_ids = graph.nodes.loc[graph.nodes["stage"] == "t4_motion", "bodyId"].tolist()
    expected = (
        batch[batch["bodyId"].isin(t4_ids)]
        .pivot(index="time_ms", columns="bodyId", values="activity_proxy")
        .reindex(columns=t4_ids)
        .to_numpy(dtype=np.float32)
    )

    online = OnlineGradedCircuit(graph, motion_columns=columns, params=params)
    observed = []
    for time_ms in np.arange(len(expected)) * params.dt_ms:
        drive = np.zeros(2, dtype=np.float64)
        for event in events:
            column = 0 if event.name == columns[0]["suffix"] else 1
            if event.start_ms <= time_ms < event.start_ms + event.duration_ms:
                drive[column] += event.amplitude_mV
        observed.append(online.step(drive).astype(np.float32))
    assert np.array_equal(np.asarray(observed), expected)


def test_overlapping_graded_events_add_and_match_online_execution():
    graph = _synthetic_graph()
    columns = [
        {"suffix": "01_07", "l1_body_id": 1},
        {"suffix": "01_08", "l1_body_id": 2},
    ]
    params = GradedParameters()
    events = list(_events()) + [
        VisualEvent(
            "01_07",
            (1,),
            start_ms=15.0,
            duration_ms=5.0,
            amplitude_mV=0.5,
        )
    ]
    duration_ms = 40.0
    batch, _ = run_graded(
        graph,
        events,
        motion_columns=columns,
        params=params,
        duration_ms=duration_ms,
    )
    t4_ids = graph.nodes.loc[
        graph.nodes["stage"] == "t4_motion", "bodyId"
    ].tolist()
    expected = (
        batch[batch["bodyId"].isin(t4_ids)]
        .pivot(index="time_ms", columns="bodyId", values="activity_proxy")
        .reindex(columns=t4_ids)
        .to_numpy(dtype=np.float32)
    )
    online = OnlineGradedCircuit(graph, motion_columns=columns, params=params)
    observed = []
    for time_ms in np.arange(len(expected)) * params.dt_ms:
        drive = np.zeros(2, dtype=np.float64)
        for event in events:
            column = 0 if event.name == columns[0]["suffix"] else 1
            if event.start_ms <= time_ms < event.start_ms + event.duration_ms:
                drive[column] += event.amplitude_mV
        observed.append(online.step(drive).astype(np.float32))
    assert np.array_equal(np.asarray(observed), expected)


def test_t4_comparison_rejects_incomplete_condition_axes():
    graph = _synthetic_graph()
    columns = [
        {"suffix": "01_07", "l1_body_id": 1},
        {"suffix": "01_08", "l1_body_id": 2},
    ]
    forward, _ = run_graded(
        graph, _events(), motion_columns=columns, duration_ms=80.0
    )
    reverse, _ = run_graded(
        graph, _events(reverse=True), motion_columns=columns, duration_ms=80.0
    )
    missing_neuron = reverse[reverse["bodyId"] != 31]
    with np.testing.assert_raises_regex(ValueError, "same T4 neurons"):
        t4_comparison(forward, missing_neuron)
    missing_time = reverse[
        ~((reverse["stage"] == "t4_motion") & (reverse["time_ms"] == 0.0))
    ]
    with np.testing.assert_raises_regex(ValueError, "identical time"):
        t4_comparison(forward, missing_time)
