from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.malecns_io import ConnectomeGraph
from src.neck_motor import MotorParameters, NeckMotorPathway, head_torque


def example():
    nodes=pd.DataFrame({"bodyId":[11215,12069,801678,804343,813696,903152],
        "somaSide":["R","L","R","L","R","L"],
        "stage":["DNp15"]*2+["neck_motor"]*4,
        "consensus_nt":["acetylcholine"]*2+["unclear"]*4})
    edges=pd.DataFrame([(11215,804343,40),(11215,903152,82),
        (12069,801678,150),(12069,813696,76),(801678,813696,1),(804343,903152,1)],
        columns=["pre_body","post_body","synapse_count"])
    totals=[{"bodyId":i,"incoming_count":t} for i,t in [(801678,7598),(804343,1921),(813696,2204),(903152,6446)]]
    identity=[{"bodyId":body,"paper_cell":"CvNA2","muscle":"TH2","axon_side":side,
        "identity_source":"Gorko Supplementary Table 2","laterality_source":"MANC CvN instance"}
        for body,side in [(804343,"R"),(813696,"L")]]
    return ConnectomeGraph(nodes,edges),totals,identity


def test_sparse_direct_weights_match_independent_counts_not_selected_normalization():
    graph,totals,_=example()
    path=NeckMotorPathway.compile(graph,totals)
    expected=np.array([[0,150/7598],[40/1921,0],[0,76/2204],[82/6446,0]])*.5
    np.testing.assert_array_equal(path.source_ids,[11215,12069])
    np.testing.assert_allclose(path.weights.toarray(),expected,atol=1e-18)
    assert path.weights.nnz==4 and graph.n_edges==6
    assert path.weights.sum(axis=1).max()<.02


def test_exact_held_input_solution_causality_and_rectification():
    graph,totals,_=example()
    path=NeckMotorPathway.compile(graph,totals)
    drive=np.zeros((400,2));drive[100:,0]=1
    output=path.simulate(drive)
    assert np.all(output[:101]==0)  # input k affects k+1, never k
    target=path.weights.toarray()[:,0]
    expected=(1-np.exp(-np.arange(300)*.5/20))[:,None]*target
    np.testing.assert_allclose(output[100:],expected,rtol=1e-13,atol=1e-18)
    assert not np.any(path.simulate(-np.ones((400,2))))


def test_randomized_dense_reference_determinism_and_node_order_invariance():
    graph,totals,_=example()
    path=NeckMotorPathway.compile(graph,totals)
    drive=np.random.default_rng(6).normal(size=(113,2))
    output=path.simulate(drive)
    reference=np.zeros_like(output)
    alpha=1-np.exp(-.5/20)
    dense=path.weights.toarray()
    for k in range(len(drive)-1):
        reference[k+1]=(1-alpha)*reference[k]+alpha*dense@np.maximum(drive[k],0)
    np.testing.assert_allclose(output,reference,atol=1e-17)
    np.testing.assert_array_equal(output,path.simulate(drive))
    shuffled=ConnectomeGraph(graph.nodes.sample(frac=1,random_state=4),graph.edges.sample(frac=1,random_state=8))
    np.testing.assert_array_equal(output,NeckMotorPathway.compile(shuffled,totals).simulate(drive))


def test_zero_input_transition_and_exact_timestep_subdivision():
    graph,totals,_=example()
    coarse=NeckMotorPathway.compile(graph,totals)
    fine=NeckMotorPathway.compile(graph,totals,replace(coarse.parameters,dt_ms=.25))
    drive=np.random.default_rng(5).normal(size=(220,2))
    np.testing.assert_allclose(coarse.simulate(drive),fine.simulate(np.repeat(drive,2,axis=0))[::2],atol=1e-16)
    for path in [coarse,fine]:
        assert path.preflight()["passed"]
        assert path.preflight()["motor_transition_spectral_radius_and_contraction_bound"]<1
        initial=np.array([-2,1,4,-3.])
        output=path.simulate(np.zeros((100,2)),initial_state=initial)
        expected=np.exp(-np.arange(100)*path.parameters.dt_ms/20)[:,None]*initial
        np.testing.assert_allclose(output,expected,rtol=1e-13)


@pytest.mark.parametrize("control,removed",[("full",[]),("DN_disconnected",[0,1]),
    ("DN_L_disconnected",[1]),("DN_R_disconnected",[0]),("motor_disconnected",[])])
def test_all_controls_preserve_surviving_weights_and_disconnection(control,removed):
    graph,totals,identity=example()
    full=NeckMotorPathway.compile(graph,totals)
    path=NeckMotorPathway.compile(graph,totals,control=control)
    expected=full.weights.toarray();expected[:,removed]=0
    np.testing.assert_array_equal(path.weights.toarray(),expected)
    assert path.preflight()["passed"]
    activity=path.simulate(np.ones((400,2)))
    torque=head_torque(activity,path.motor_ids,identity,disconnected=control=="motor_disconnected")
    if control=="DN_disconnected":
        assert not np.any(activity) and not np.any(torque)
    if control=="motor_disconnected":
        assert np.any(activity) and not np.any(torque)


def test_motor_torque_depends_on_axon_not_soma_or_condition_and_is_not_peak_normalized():
    _,_,identity=example()
    ids=[804343,813696]
    states=np.array([[1,0],[0,1],[1,1],[.25,0],[-1,-2.]])
    values=head_torque(states,ids,identity)
    np.testing.assert_allclose(values,np.cos(np.deg2rad(40))*np.array([1,-1,0,.25,0]))
    # Arbitrary additional soma/condition metadata cannot affect command.
    rows=[dict(r,somaSide="R",condition="ignored") for r in identity]
    np.testing.assert_array_equal(values,head_torque(states,ids,rows))
    np.testing.assert_array_equal(values,head_torque(states[:,::-1],ids[::-1],identity))
    np.testing.assert_allclose(-values,head_torque(states[:,::-1],ids,identity))
    np.testing.assert_allclose(2*values,head_torque(2*states,ids,identity))


@pytest.mark.parametrize("change",["duplicate_total","missing_total","small_total","nan_total","wrong_nt","wrong_side"])
def test_invalid_anatomical_or_physiological_contracts_are_rejected(change):
    graph,totals,_=example()
    if change=="duplicate_total":totals.append(dict(totals[0]))
    elif change=="missing_total":totals.pop()
    elif change=="small_total":totals[0]["incoming_count"]=1
    elif change=="nan_total":totals[0]["incoming_count"]=np.nan
    elif change=="wrong_nt":graph.nodes.loc[0,"consensus_nt"]="gaba"
    elif change=="wrong_side":graph.nodes.loc[0,"somaSide"]="L"
    with pytest.raises(ValueError):NeckMotorPathway.compile(graph,totals)


def test_unsupported_muscle_provenance_nonfinite_states_and_parameters_rejected():
    graph,totals,identity=example()
    identity[0]["muscle"]="unknown"
    with pytest.raises(ValueError):head_torque(np.ones((2,2)),[804343,813696],identity)
    path=NeckMotorPathway.compile(graph,totals)
    with pytest.raises(ValueError):path.simulate(np.full((2,2),np.nan))
    with pytest.raises(ValueError):path.simulate(np.zeros((2,3)))
    with pytest.raises(ValueError):MotorParameters(tau_ms=0)
    with pytest.raises(ValueError):MotorParameters(torque_native_per_activity=-1)
    with pytest.raises(ValueError):MotorParameters(paper_body_pitch_degrees=100)


def test_requery_cannot_overwrite_frozen_anatomy_on_a_mismatch(tmp_path):
    from scripts.prepare_exp006_neck import store_or_verify_bundle
    graph,_,_=example()
    documents=[("manifest.json",{"dataset":"male-cns:v1.0"})]
    store_or_verify_bundle(tmp_path,graph.nodes,graph.edges,documents)
    original={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    store_or_verify_bundle(tmp_path,graph.nodes,graph.edges,documents,check=True)
    changed=graph.edges.copy();changed.loc[0,"synapse_count"]+=1
    with pytest.raises(AssertionError):
        store_or_verify_bundle(tmp_path,graph.nodes,changed,documents,check=True)
    with pytest.raises(ValueError):
        store_or_verify_bundle(tmp_path,graph.nodes,graph.edges,[("manifest.json",{"dataset":"different"})],check=True)
    assert original=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
