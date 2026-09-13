from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.binocular_physiology import (
    BinocularNetwork, PhysiologyParameters, controlled_drive,
    discrimination_index, response_table,
)
from src.malecns_io import ConnectomeGraph


def example_graph():
    nodes = pd.DataFrame({
        "bodyId": [1,2,3,4,5], "type": ["HSE","H2","PS321","PS072","DNp15"],
        "stage": ["HS","H2","bIPS","uLPTCrn","DNp15"],
        "somaSide": ["R","L","L","R","R"],
        "consensus_nt": ["acetylcholine","acetylcholine","gaba","gaba","acetylcholine"],
    })
    edges = pd.DataFrame({"pre_body": [1,2,1,3,4,1,3,4],
        "post_body": [3,3,4,4,3,5,5,1], "synapse_count": [10,5,8,4,6,20,15,3]})
    return ConnectomeGraph(nodes,edges)


def pair():
    return {"first": 1, "second": 2, "provenance": "https://doi.org/10.1038/s41467-024-53173-w", "evidence": "HSE-contralateral H2"}


def network(control="full", parameters=None):
    return BinocularNetwork.compile(example_graph(),[pair()],parameters or PhysiologyParameters(),control)


def test_gap_semantics_in_the_executed_network():
    model = network()
    eq = np.ones(5)
    assert np.array_equal(model.electrical.current(eq),np.zeros(5))
    first = eq.copy(); first[:2] = [1,4]
    second = eq.copy(); second[:2] = [4,1]
    current = model.electrical.current(first)
    assert np.allclose(current,-model.electrical.current(second))
    assert current.sum() == pytest.approx(0)
    assert first @ current < 0
    after = first + .1*current
    assert abs(after[0]-after[1]) < abs(first[0]-first[1])


def test_nonlinear_zero_input_decay_and_effective_stability():
    preflight = network().preflight()
    assert preflight["passed"]
    assert preflight["global_euler_lipschitz_inf_bound"] < 1
    assert preflight["baseline_one_sided_effective_transition_rho_max"] < 1
    assert max(preflight["zero_input_2000ms_final_initial_inf_ratios"]) < 1e-8


def test_preflight_rejects_a_numerically_unstable_euler_step():
    assert not network(parameters=PhysiologyParameters(dt_ms=100.)).preflight()["passed"]


def test_execution_matches_independent_dense_reference_and_is_deterministic():
    model = network()
    drive = np.random.default_rng(17).normal(size=(200,5))
    actual = model.simulate(drive)
    reference = np.empty_like(actual); state = np.zeros(5)
    chemical = model.chemical.toarray(); gap = model.electrical.operator.toarray()
    h = model.parameters.dt_ms/model.parameters.tau_ms
    for k in range(len(drive)):
        reference[k] = state
        # Deliberately use per-cell/pair summation, not the sparse execution.
        target = np.asarray([sum(chemical[i,j]*max(state[j],0) for j in range(5)) for i in range(5)])
        current = np.asarray([sum(gap[i,j]*state[j] for j in range(5)) for i in range(5)])
        state = state + h*(-state+target+current+drive[k])
    assert np.allclose(actual,reference,atol=1e-14,rtol=1e-14)
    assert np.array_equal(actual,model.simulate(drive))


def test_body_and_edge_order_do_not_change_output():
    graph = example_graph()
    shuffled = ConnectomeGraph(graph.nodes.sample(frac=1,random_state=3),graph.edges.sample(frac=1,random_state=4))
    model = BinocularNetwork.compile(shuffled,[pair()],PhysiologyParameters())
    drive = np.random.default_rng(11).normal(size=(50,5))
    assert np.array_equal(model.simulate(drive),network().simulate(drive))


@pytest.mark.parametrize("control", ["no_bIPS_chemical_output","no_bIPS_GABA_input","no_DNp15_GABA_input","no_chemical_feedback"])
def test_ablation_removes_active_edges_without_rescaling_survivors(control):
    original = network().chemical.toarray()
    ablated = network(control).chemical.toarray()
    survivors = ablated != 0
    assert np.array_equal(ablated[survivors],original[survivors])
    assert np.count_nonzero(ablated) < np.count_nonzero(original)
    assert network(control).electrical.pair_count == 1
    assert network(control).preflight()["passed"]


def test_electrical_ablation_leaves_chemical_anatomy_and_weights_unchanged():
    assert np.array_equal(network().chemical.toarray(),network("no_electrical").chemical.toarray())
    assert network("no_electrical").electrical.operator.nnz == 0


def test_feedforward_control_is_stage_dag_with_unchanged_surviving_weights():
    model=network("feedforward_chemical_only")
    assert model.electrical.pair_count==0
    assert model.preflight()["passed"]
    assert np.count_nonzero(model.chemical.toarray())<np.count_nonzero(network().chemical.toarray())
    assert (model.chemical @ model.chemical @ model.chemical).nnz==0
    actual=model.chemical.toarray(); original=network().chemical.toarray()
    assert np.array_equal(actual[actual!=0],original[actual!=0])


def test_global_contraction_bound_covers_all_active_masks_of_fixture():
    import itertools
    model=network(); bound=model.preflight()["global_euler_lipschitz_inf_bound"]
    for active in itertools.product([False,True],repeat=5):
        transition=model.transition(np.asarray(active))
        assert np.linalg.norm(transition,ord=np.inf)<=bound+1e-15
        assert np.max(abs(np.linalg.eigvals(transition)))<1


def test_external_drive_only_enters_identified_upstream_cells_and_reverses():
    model = network()
    time,drive = controlled_drive(model,[1,-1])
    assert np.array_equal(drive,-controlled_drive(model,[-1,1])[1])
    assert np.count_nonzero(drive[:,2:]) == 0
    assert np.count_nonzero(drive[time<2000]) == 0
    assert np.count_nonzero(drive[time>=4000]) == 0
    trace = model.simulate(drive)
    assert trace[0].sum() == 0
    assert trace[time==2000].sum() == 0  # no hidden one-step timing offset


def test_effective_jacobian_matches_finite_difference_of_actual_update():
    model = network(); state = np.asarray([.2,-.4,.1,-.3,.5])
    # simulate's sample 1 is precisely one update from the supplied state.
    drive = np.zeros((2,5)); baseline = model.simulate(drive,initial_state=state)[1]
    columns = []
    for k in range(5):
        perturbed = state.copy(); perturbed[k] += 1e-7
        columns.append((model.simulate(drive,initial_state=perturbed)[1]-baseline)/1e-7)
    assert np.allclose(np.column_stack(columns),model.transition(state>0),atol=1e-9)


def test_dt_halving_converges_without_model_parameter_tuning():
    parameters = PhysiologyParameters(onset_ms=0,offset_ms=40,duration_ms=100)
    coarse = network(parameters=parameters)
    fine = network(parameters=replace(parameters,dt_ms=.25))
    a = coarse.simulate(controlled_drive(coarse,[1,-1])[1])
    b = fine.simulate(controlled_drive(fine,[1,-1])[1])[::2]
    assert np.max(abs(a-b)) < .006


def test_di_zero_denominator_and_sign_and_scaling_contracts():
    assert np.isnan(discrimination_index(0,0))
    assert discrimination_index(1,3) == pytest.approx(-.5)
    assert discrimination_index(10,30) == pytest.approx(-.5)
    assert discrimination_index(3,1) == pytest.approx(.5)
    with pytest.raises(ValueError,match="finite"):
        discrimination_index(np.inf,1)


def test_raw_negative_states_are_preserved_in_response_analysis():
    model = network(); time = np.arange(10.)
    traces = {c: np.full((10,5),v) for c,v in {"yaw_L_F":1.,"yaw_R_F":-1.,"translation_F":.5,"translation_B":-.5}.items()}
    table = response_table(model,time,traces,[0,10])
    assert np.array_equal(table.yaw_R_F,np.full(5,-1.))
    assert np.array_equal(table.yaw_R_F_emission,np.zeros(5))
    assert table.iloc[0].of_yaw == -2.  # known right-HS orientation, not post-hoc max
    assert table.iloc[1].of_yaw == -2.  # left H2 uses B, opposite HS convention


def test_unproven_or_generic_electrical_pairings_are_rejected():
    bad = pair(); bad.pop("provenance")
    with pytest.raises(ValueError,match="provenance"):
        BinocularNetwork.compile(example_graph(),[bad],PhysiologyParameters())
    bad = pair(); bad["second"] = 5
    with pytest.raises(ValueError,match="specific"):
        BinocularNetwork.compile(example_graph(),[bad],PhysiologyParameters())


@pytest.mark.parametrize("change", [{"dt_ms":0},{"tau_ms":-1},{"chemical_gain":np.nan},{"gap_conductance":-1},{"offset_ms":7000}])
def test_invalid_parameters_are_rejected(change):
    with pytest.raises(ValueError):
        PhysiologyParameters(**change)


def test_nonfinite_drive_and_missing_anatomy_are_rejected():
    with pytest.raises(ValueError,match="finite"):
        network().simulate(np.full((1,5),np.nan))
    graph = example_graph(); graph.edges.loc[0,"post_body"] = 99
    with pytest.raises(ValueError,match="outside"):
        BinocularNetwork.compile(graph,[pair()],PhysiologyParameters())
