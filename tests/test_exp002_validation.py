import pandas as pd

from src.exp002_validation import (
    adjacent_pairs,
    evenly_spaced_pairs,
    parse_grid_suffix,
    summarize_type_preferences,
)


def test_grid_suffix_and_axis_adjacency_are_explicit():
    assert parse_grid_suffix("01_07") == (1, 7)
    pairs = adjacent_pairs(["01_07", "01_08", "02_07", "02_08", "03_08"])
    assert ("01_07", "02_07") in pairs["first_index"]
    assert ("01_07", "01_08") in pairs["second_index"]
    assert ("02_08", "03_08") in pairs["first_index"]


def test_pair_selection_does_not_depend_on_model_outputs():
    pairs = [(f"01_{index:02d}", f"01_{index + 1:02d}") for index in range(1, 9)]
    selected = evenly_spaced_pairs(pairs, count=3, exclude={pairs[0]})
    assert len(selected) == 3
    assert pairs[0] not in selected
    assert selected == evenly_spaced_pairs(pairs, count=3, exclude={pairs[0]})


def test_type_summary_counts_both_order_preferences():
    comparison = pd.DataFrame(
        [
            {
                "bodyId": 1,
                "type": "T4a",
                "max_activity_forward": 2.0,
                "max_activity_reverse": 1.0,
                "max_activity_difference": 1.0,
                "max_abs_pointwise_activity_difference": 1.0,
            },
            {
                "bodyId": 2,
                "type": "T4a",
                "max_activity_forward": 1.0,
                "max_activity_reverse": 2.0,
                "max_activity_difference": -1.0,
                "max_abs_pointwise_activity_difference": 1.0,
            },
        ]
    )
    summary = summarize_type_preferences(comparison)
    assert summary.loc[0, "forward_gt_reverse"] == 1
    assert summary.loc[0, "reverse_gt_forward"] == 1
