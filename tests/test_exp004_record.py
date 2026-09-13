import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.binocular_physiology import BinocularNetwork, CONTROLS, PhysiologyParameters

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT / "experiments/EXP-004-binocular-physiology"


def test_frozen_specification_bytes_survive_git_checkout_without_normalization():
    relative="experiments/EXP-004-binocular-physiology/specification.json"
    frozen_sha="1dc992eb7ed695544365dfe5dcee14de2dbbc34e5b0a2c18bd4e754ac628aafb"
    staged=subprocess.check_output(["git","cat-file","blob",":"+relative],cwd=ROOT)
    assert hashlib.sha256(staged).hexdigest()==frozen_sha
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()==frozen_sha


def test_successor_preserves_frozen_specification_and_reports_partial_and_negative_results():
    spec=json.loads((FOLDER / "specification.json").read_text())
    record=json.loads((FOLDER / "record.json").read_text())
    assert record["status"]=="completed_stable_negative"
    assert record["metrics"]["specification_sha256"]==hashlib.sha256((FOLDER / "specification.json").read_bytes()).hexdigest()
    assert record["metrics"]["parameters"]==spec["parameters"]==PhysiologyParameters().__dict__
    assert record["anatomy"]["node_count"]==37
    assert record["anatomy"]["edge_count"]==360
    assert record["metrics"]["qualitative_target"]["limited_neural_transformation_gate"]
    assert not record["metrics"]["qualitative_target"]["recurrent_enhancement_contract_passed"]
    assert record["metrics"]["deterministic_repeat_all_conditions"]
    assert not any(record["model_scope"][key] for key in ["T4_T5_reconnected","body_simulation","parameter_search","DNa02_included"])


def test_release_aware_identity_and_bips_contracts_are_complete():
    spec=json.loads((FOLDER / "specification.json").read_text())
    mapping={r["root_id_630"]:r for r in spec["identity_provenance"]["H2rn"]["released_crosswalk"]}
    assert len(mapping)==5
    assert mapping["720575940621865991"]=={"root_id_630":"720575940621865991","supervoxel_id":"78183935993544196","root_id_783":"720575940620118430","cell_type":"CB3740"}
    assert mapping["720575940622082451"]=={"root_id_630":"720575940622082451","supervoxel_id":"78254098645935549","root_id_783":"720575940622812287","cell_type":"CB3740"}
    assert {n["bodyId"] for n in spec["nodes"] if n["stage"]=="bIPS"}=={11919,12072}
    assert spec["bips_connectivity_verification"]["unrestricted_neighborhood_edges"]==898
    assert spec["bips_connectivity_verification"]["retained_induced_edges_checked"]==80
    assert len(spec["electrical_pairs"])==4
    assert all(p["provenance"] and p["evidence"] for p in spec["electrical_pairs"])
    assert "class-level" in spec["identity_provenance"]["uLPTCrn"]


def test_final_graph_all_controls_pass_numerical_and_weight_contracts():
    record=json.loads((FOLDER / "record.json").read_text())
    assert tuple(record["metrics"]["preflight"])==CONTROLS
    for control,preflight in record["metrics"]["preflight"].items():
        assert preflight["passed"] and preflight["half_dt_preflight"]["passed"]
        assert preflight["surviving_chemical_weights_identical"]
        assert preflight["global_euler_lipschitz_inf_bound"]<1
        assert preflight["encountered_effective_transition_rho_max"]<1
        assert max(preflight["zero_input_2000ms_final_initial_inf_ratios"])<1e-8
        assert preflight["timestep_convergence"]["passed"]
        assert max(preflight["timestep_convergence"]["max_abs_trace_difference_by_condition"].values())<=.01
        assert preflight["timestep_convergence"]["max_abs_late_response_or_index_difference"]<=1e-6
        assert record["metrics"]["responses"][control]["zero_condition_max_abs_state"]==0


def test_historical_records_remain_byte_identical():
    expected={
        "EXP-001-adjacent-column-motion":"4cd383c125a055605c73681faf3bfad3078f78b691267742e942c5c61fa2a8b2",
        "EXP-002-graded-t4-motion":"bb784b2bfca8fc0839218347f08a124561f317651ed4110efb30fb7ed2043e57",
        "EXP-003-bilateral":"0c3894c9e8aadf098cd6562f6458285a4cce07abb3850223d7d0bccb84cfbb74",
        "EXP-003-downstream-correction":"44364d7a1d82b96b32f6be7d0177d97d72de62c219a573d40bb10369a07e8fa5",
        "EXP-003-embodied-steering":"83b3e2508ade01a19aa212167044b2ac1a81e3dedb73fec804142b13cee08ae4",
        "EXP-004-preflight-failure":"0d2ec2e2d3a5c1657003795bbbccd14503caa7fcaa0436b9decf3a481cb9724c",
    }
    for name,digest in expected.items():
        assert hashlib.sha256((ROOT / "experiments" / name / "record.json").read_bytes()).hexdigest()==digest


def test_resolved_chemical_operator_and_update_match_independent_dense_reference():
    if not (ROOT / "data/malecns_exp004_physiology/nodes.parquet").exists():
        pytest.skip("local MaleCNS materialization is intentionally not tracked")
    from scripts.run_exp004_physiology import load_frozen
    spec,graph=load_frozen()
    model=BinocularNetwork.compile(graph,spec["electrical_pairs"],PhysiologyParameters())
    ids=model.nodes.bodyId.astype(int).tolist(); index={body:k for k,body in enumerate(ids)}
    chemical=np.zeros((len(ids),len(ids)))
    denominator=np.zeros(len(ids))
    for edge in graph.edges.itertuples(index=False):
        denominator[index[edge.post_body]]+=edge.synapse_count
    metadata=model.nodes.set_index("bodyId")
    for edge in graph.edges.itertuples(index=False):
        i,j=index[edge.post_body],index[edge.pre_body]
        sign=-1 if metadata.loc[edge.pre_body,"consensus_nt"]=="gaba" else 1
        chemical[i,j]=.5*sign*edge.synapse_count/denominator[i]
    assert np.allclose(model.chemical.toarray(),chemical,rtol=0,atol=1e-16)
    gap=np.zeros_like(chemical)
    for pair in spec["electrical_pairs"]:
        i,j=index[pair["first"]],index[pair["second"]]
        gap[i,j]+=.1; gap[j,i]+=.1; gap[i,i]-=.1; gap[j,j]-=.1
    assert np.array_equal(gap,model.electrical.operator.toarray())
    drive=np.random.default_rng(29).normal(size=(200,len(ids)))
    state=np.zeros(len(ids)); reference=np.empty_like(drive)
    for k in range(len(drive)):
        reference[k]=state
        state=state+.025*(-state+chemical@np.maximum(state,0)+gap@state+drive[k])
    assert np.allclose(model.simulate(drive),reference,rtol=0,atol=1e-14)
    counts=graph.edges.set_index(["pre_body","post_body"]).synapse_count
    assert counts.loc[(11919,11215)]==158
    assert counts.loc[(12072,12069)]==127
