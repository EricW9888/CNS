import pandas as pd

from src.observability import compare_t4_voltage


def _trace(body_id, cell_type, voltages):
    return pd.DataFrame(
        {
            "time_ms": [0.0, 1.0, 2.0],
            "node_index": [0, 0, 0],
            "bodyId": [body_id, body_id, body_id],
            "voltage_mV": voltages,
            "type": [cell_type] * 3,
            "instance": [f"{cell_type}_R"] * 3,
            "somaSide": ["R"] * 3,
            "superclass": ["ol_intrinsic"] * 3,
            "stage": ["t4_motion"] * 3,
        }
    )


def test_compare_t4_voltage_reports_pointwise_order_difference():
    forward = pd.concat([_trace(101, "T4a", [-52.0, -50.0, -51.0]), _trace(202, "T4b", [-52.0] * 3)])
    reverse = pd.concat([_trace(101, "T4a", [-52.0, -51.0, -51.0]), _trace(202, "T4b", [-52.0] * 3)])
    spikes = pd.DataFrame(columns=["bodyId", "stage"])

    comparison = compare_t4_voltage(forward, reverse, spikes, spikes)

    row = comparison.loc[comparison["bodyId"] == 101].iloc[0]
    assert row["max_voltage_forward_mV"] == -50.0
    assert row["max_voltage_reverse_mV"] == -51.0
    assert row["max_voltage_difference_mV"] == 1.0
    assert row["max_abs_pointwise_voltage_difference_mV"] == 1.0
    assert row["time_of_max_abs_difference_ms"] == 1.0
    assert comparison.iloc[1]["max_abs_pointwise_voltage_difference_mV"] == 0.0
