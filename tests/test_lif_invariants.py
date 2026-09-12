import numpy as np
import pandas as pd
import pytest

from src.lif_model import LIFParameters, run_lif
from src.malecns_io import build_graph
from src.visual_stimulus import VisualEvent


def _single_neuron_graph():
    nodes = pd.DataFrame(
        {
            "bodyId": [1],
            "type": ["L1"],
            "instance": ["L1_R"],
            "somaSide": ["R"],
            "superclass": ["optic"],
            "stage": ["visual_input"],
            "consensus_nt": ["acetylcholine"],
        }
    )
    edges = pd.DataFrame(columns=["pre_body", "post_body", "synapse_count"])
    return build_graph(nodes, edges)


def test_passive_lif_matches_discrete_euler_solution():
    graph = _single_neuron_graph()
    params = LIFParameters(dt_ms=0.1, v_threshold_mV=100.0)
    _, _, trace = run_lif(
        graph,
        VisualEvent("constant", (1,), start_ms=0.0, duration_ms=1.0, amplitude_mV=20.0),
        duration_ms=1.0,
        params=params,
        trace_node_ids=[1],
        return_trace=True,
    )
    alpha = 1.0 - params.dt_ms / params.tau_membrane_ms
    expected_after_ten_updates = params.v_rest_mV + 20.0 * (1.0 - alpha**10)
    assert trace.iloc[9]["voltage_mV"] == pytest.approx(
        expected_after_ten_updates, abs=2e-6
    )
    assert trace["voltage_mV"].iloc[-1] < trace["voltage_mV"].iloc[-2]


@pytest.mark.parametrize(
    "params, message",
    [
        (LIFParameters(tau_membrane_ms=0.0), "time constants"),
        (LIFParameters(synaptic_delay_ms=-0.1), "cannot be negative"),
        (LIFParameters(v_threshold_mV=-53.0), "must exceed"),
        (LIFParameters(dt_ms=float("nan")), "finite"),
    ],
)
def test_lif_rejects_invalid_parameter_domains(params, message):
    with pytest.raises(ValueError, match=message):
        run_lif(_single_neuron_graph(), [], duration_ms=1.0, params=params)
