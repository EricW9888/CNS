import numpy as np
import pytest

from src.visual_stimulus import VisualEvent, compile_events, motion_events


def test_motion_orders_reverse_adjacent_events():
    forward = motion_events(
        first_column="01_07",
        first_l1_ids=(101,),
        second_column="01_08",
        second_l1_ids=(202,),
    )
    reverse = motion_events(
        first_column="01_07",
        first_l1_ids=(101,),
        second_column="01_08",
        second_l1_ids=(202,),
        reverse=True,
    )
    assert [event.name for event in forward] == ["01_07", "01_08"]
    assert [event.name for event in reverse] == ["01_08", "01_07"]
    assert [event.start_ms for event in forward] == [10.0, 20.0]
    assert [event.start_ms for event in reverse] == [10.0, 20.0]


def test_compile_events_writes_only_selected_nodes():
    events = motion_events(
        first_column="01_07",
        first_l1_ids=(101,),
        second_column="01_08",
        second_l1_ids=(202,),
    )
    values = compile_events(
        events,
        node_index={101: 0, 202: 1, 303: 2},
        n_nodes=3,
        duration_ms=40.0,
        dt_ms=0.1,
    )
    assert np.max(values[:, 2]) == 0.0
    assert np.max(values[:, 0]) == 8.0
    assert np.max(values[:, 1]) == 8.0


def test_compile_events_adds_overlaps_and_clips_out_of_window_events():
    overlapping = [
        VisualEvent("a", (1,), start_ms=1.0, duration_ms=2.0, amplitude_mV=2.0),
        VisualEvent("b", (1,), start_ms=2.0, duration_ms=2.0, amplitude_mV=3.0),
        VisualEvent("past", (1,), start_ms=-5.0, duration_ms=1.0, amplitude_mV=9.0),
    ]
    values = compile_events(
        overlapping,
        node_index={1: 0},
        n_nodes=1,
        duration_ms=5.0,
        dt_ms=1.0,
    )
    assert np.allclose(values[:, 0], [0.0, 2.0, 5.0, 3.0, 0.0, 0.0])


def test_compile_events_rejects_invalid_time_values():
    with pytest.raises(ValueError, match="positive duration"):
        compile_events(
            [VisualEvent("bad", (1,), duration_ms=0.0)],
            node_index={1: 0},
            n_nodes=1,
            duration_ms=5.0,
            dt_ms=1.0,
        )
    with pytest.raises(ValueError, match="non-finite"):
        compile_events(
            [VisualEvent("bad", (1,), amplitude_mV=float("nan"))],
            node_index={1: 0},
            n_nodes=1,
            duration_ms=5.0,
            dt_ms=1.0,
        )
