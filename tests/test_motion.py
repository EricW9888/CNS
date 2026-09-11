import numpy as np

from src.visual_stimulus import compile_events, motion_events


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
