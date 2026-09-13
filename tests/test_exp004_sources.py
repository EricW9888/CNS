import numpy as np
import pandas as pd
import pytest

from scripts.prepare_exp004_physiology import H2RN_ROOTS, resolve_h2rn_roots, source_statistics
from src.binocular_physiology import discrimination_index, evaluate_transformation


def identity_fixture():
    old = pd.DataFrame({"root_id": H2RN_ROOTS, "supervoxel_id": ["sv1","sv2","sv3","sv4","sv5"]})
    current = pd.DataFrame({"root_id": ["new1","new2",*H2RN_ROOTS[2:]],
                            "supervoxel_id": ["sv1","sv2","sv3","sv4","sv5"],
                            "cell_type": ["CB3740"]*4+["CB3748"]})
    return old,current


def test_release_crosswalk_joins_exact_anchors_not_reused_or_missing_roots():
    old,current = identity_fixture()
    resolved = resolve_h2rn_roots(old,current)
    assert len(resolved)==5
    mapping = {r["root_id_630"]:r for r in resolved}
    assert mapping[H2RN_ROOTS[0]]["root_id_783"]=="new1"
    assert mapping[H2RN_ROOTS[1]]["root_id_783"]=="new2"
    assert {r["cell_type"] for r in resolved}=={"CB3740","CB3748"}
    assert resolved==resolve_h2rn_roots(old.sample(frac=1,random_state=2),current.sample(frac=1,random_state=3))


def test_missing_root_or_anchor_stops_instead_of_guessing():
    old,current = identity_fixture()
    with pytest.raises(ValueError,match="release 630"):
        resolve_h2rn_roots(old.iloc[1:],current)
    with pytest.raises(ValueError,match="anchor"):
        resolve_h2rn_roots(old,current.iloc[1:])
    with pytest.raises(pd.errors.MergeError):
        resolve_h2rn_roots(old,pd.concat([current,current.iloc[[0]]]))


def test_external_source_parser_excludes_pvalues_and_second_comparison(monkeypatch):
    frame = pd.DataFrame([["bFtoB-Yaw Discrimination index",None,None],
                          ["HS",-.2,-.4],["DNp15",-.6,-.8],
                          ["p=0.001",None,None],[None,None,None],
                          ["Progressive-Yaw Discrimination index",None,None],["HS",-1,-1]])
    monkeypatch.setattr(pd,"read_excel",lambda *args,**kwargs: frame)
    result = source_statistics("unused.xlsx","1h")
    assert set(result)=={"HS","DNp15"}
    assert result["HS"]["n_rois"]==2
    assert result["HS"]["mean_di"]==pytest.approx(-.3)
    assert result["DNp15"]["source_excel_row"]==3
    assert result["HS"]["descriptive_roi_t95_interval"][0]<-.3


def response_fixture():
    rows=[]
    for stage,n in [("HS",3),("H2",1),("DNp15",1)]:
        for side in ["L","R"]:
            for _ in range(n):
                left_preferred = (side=="L") ^ (stage=="H2")
                row={"bodyId":len(rows)+1,"stage":stage,"somaSide":side,
                     "yaw_L_F":1. if left_preferred else -1.,
                     "yaw_R_F":-1. if left_preferred else 1.,
                     "translation_F":1. if stage!="H2" else -1.,
                     "translation_B":-1. if stage!="H2" else 1.}
                if stage=="DNp15":
                    row["yaw_L_F"] = 1. if side=="L" else -.2
                    row["yaw_R_F"] = -.2 if side=="L" else 1.
                    row["translation_F"] = .2
                    row["translation_B"] = -.1
                row["of_yaw"]=abs(row["yaw_L_F"]-row["yaw_R_F"])
                row["of_translation"]=row["translation_F"]-row["translation_B"]
                row["di"]=float(discrimination_index(row["of_translation"],row["of_yaw"]))
                row["magnitude_di"]=float(discrimination_index(abs(row["of_translation"]),abs(row["of_yaw"])))
                rows.append(row)
    return pd.DataFrame(rows)


def test_acceptance_requires_magnitude_reduction_with_matched_uncancelled_upstream():
    table=response_fixture()
    result=evaluate_transformation(table,epsilon=1e-10)
    assert result["limited_neural_transformation_gate"]
    assert all(r["upstream_translation_yaw_ratio"]==pytest.approx(1) for r in result["per_side"])
    assert all(r["DNp15_translation_yaw_ratio"]==pytest.approx(.25) for r in result["per_side"])
    assert all(len(r["matched_upstream_body_ids"])==4 for r in result["per_side"])


def test_large_signed_di_from_inverted_translation_cannot_pass():
    table=response_fixture(); dn=table.stage.eq("DNp15")
    table.loc[dn,"translation_F"]=-2.
    table.loc[dn,"translation_B"]=0.
    table.loc[dn,"of_translation"]=-2.
    table.loc[dn,"di"]=-1.
    table.loc[dn,"magnitude_di"]=float(discrimination_index(2,1.2))
    assert not evaluate_transformation(table,epsilon=1e-10)["limited_neural_transformation_gate"]


def test_loss_or_reversal_of_yaw_response_cannot_pass_even_if_translation_is_zero():
    table=response_fixture(); dn=table.stage.eq("DNp15")
    table.loc[dn,["yaw_L_F","yaw_R_F"]]=0.
    table.loc[dn,"of_yaw"]=0.
    table.loc[dn,"of_translation"]=0.
    table.loc[dn,"magnitude_di"]=np.nan
    assert not evaluate_transformation(table,epsilon=1e-10)["limited_neural_transformation_gate"]


def test_matched_upstream_contract_rejects_incomplete_identity():
    with pytest.raises(ValueError,match="three ipsilateral HS"):
        evaluate_transformation(response_fixture().iloc[1:],epsilon=1e-10)
