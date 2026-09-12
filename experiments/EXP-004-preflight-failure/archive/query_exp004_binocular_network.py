"""Provisional EXP-004 materializer retained after failed preflight.

The visual T4 bundles are inherited from the bilateral EXP-003 query.  This
export follows their corrected direct target rule into HSE/HSN/HSS/H2 and the
identified DNp15/DNa02 outputs.  Gap junctions are intentionally absent from
the exported graph because they are not represented as chemical edges here;
the model adds those interactions as an explicit literature-derived control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_exp003_corrected_downstream import TARGET_RULE, _edge_query  # noqa: E402
from scripts.query_malecns_subgraph import DATASET, NEUPRINT_URL  # noqa: E402


LPTC_TYPES = ("HSE", "HSN", "HSS", "H2")
DN_TYPES = ("DNp15", "DNa02")
LPTC_OR_DN_RULE = (
    'b.type IN ["HSE","HSN","HSS","H2","DNp15","DNa02"]'
)


def _write_nodes(
    node_ids: set[int],
    stages: dict[int, str],
    annotations: pd.DataFrame,
    transmitters: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    nodes = annotations[annotations["bodyId"].isin(node_ids)].copy()
    nodes = nodes.merge(
        transmitters[["bodyId", "consensus_nt", "predicted_nt", "predicted_nt_confidence"]],
        on="bodyId",
        how="left",
    )
    nodes["stage"] = nodes["bodyId"].map(stages).fillna("intermediate")
    keep = [
        "bodyId", "type", "instance", "somaSide", "superclass", "stage",
        "status", "consensus_nt", "predicted_nt", "predicted_nt_confidence",
    ]
    nodes[keep].to_parquet(output / "nodes.parquet", index=False)
    return nodes[keep]


def _write_edges(rows: list[dict], output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows).drop_duplicates(["pre_body", "post_body"])
    frame["pre_body"] = frame["pre_body"].astype("int64")
    frame["post_body"] = frame["post_body"].astype("int64")
    frame["synapse_count"] = frame["synapse_count"].astype("float32")
    frame[["pre_body", "post_body", "synapse_count"]].to_parquet(output / "edges.parquet", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bilateral-bundle", type=Path, default=ROOT / "data" / "malecns_exp003_bilateral")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data" / "body-annotations.feather")
    parser.add_argument("--transmitters", type=Path, default=ROOT / "data" / "body-neurotransmitters.feather")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "malecns_exp004_binocular_network")
    args = parser.parse_args()

    left_nodes = pd.read_parquet(args.bilateral_bundle / "left_visual" / "nodes.parquet")
    right_nodes = pd.read_parquet(args.bilateral_bundle / "right_visual" / "nodes.parquet")
    t4_ids = sorted(
        set(left_nodes.loc[left_nodes["stage"].eq("t4_motion"), "bodyId"].astype(int))
        | set(right_nodes.loc[right_nodes["stage"].eq("t4_motion"), "bodyId"].astype(int))
    )
    direct_t4_rows = _edge_query(t4_ids, TARGET_RULE)
    candidate_ids = sorted({int(row["post_body"]) for row in direct_t4_rows})
    candidate_rows = _edge_query(candidate_ids, LPTC_OR_DN_RULE)
    lptc_ids = sorted(
        {int(row["post_body"]) for row in candidate_rows if row["post_type"] in LPTC_TYPES}
    )
    lptc_rows = _edge_query(lptc_ids, LPTC_OR_DN_RULE)
    if not direct_t4_rows or not candidate_rows or not lptc_rows:
        raise RuntimeError("EXP-004 binocular materialization returned an incomplete chemical path")

    used_candidates = {int(row["pre_body"]) for row in candidate_rows}
    used_t4_rows = [row for row in direct_t4_rows if int(row["post_body"]) in used_candidates]
    dn_ids = sorted(
        {int(row["post_body"]) for row in candidate_rows + lptc_rows if row["post_type"] in DN_TYPES}
    )
    all_lptc_ids = sorted(
        set(lptc_ids)
        | {int(row["post_body"]) for row in lptc_rows if row["post_type"] in LPTC_TYPES}
    )
    intermediate_ids = sorted(used_candidates - set(all_lptc_ids) - set(t4_ids))
    stages = {body_id: "t4_motion" for body_id in t4_ids}
    stages.update({body_id: "motion_intermediate" for body_id in intermediate_ids})
    stages.update({body_id: "lptc_widefield" for body_id in all_lptc_ids})
    stages.update({body_id: "dn_readout" for body_id in dn_ids})
    node_ids = set(stages)
    annotations = pd.read_feather(args.annotations)
    annotations["bodyId"] = annotations["bodyId"].astype("int64")
    transmitters = pd.read_feather(args.transmitters).rename(columns={"body": "bodyId"})
    transmitters["bodyId"] = transmitters["bodyId"].astype("int64")
    transmitters = transmitters.drop_duplicates("bodyId", keep="first")

    args.output.mkdir(parents=True, exist_ok=True)
    nodes = _write_nodes(node_ids, stages, annotations, transmitters, args.output)
    edge_rows = used_t4_rows + candidate_rows + lptc_rows
    edges = _write_edges(edge_rows, args.output)
    edge_counts = (
        edges.merge(nodes[["bodyId", "stage"]].rename(columns={"bodyId": "pre_body", "stage": "pre_stage"}), on="pre_body")
        .merge(nodes[["bodyId", "stage"]].rename(columns={"bodyId": "post_body", "stage": "post_stage"}), on="post_body")
        .groupby(["pre_stage", "post_stage"]).size().astype(int).to_dict()
    )
    manifest = {
        "experiment": "EXP-004",
        "dataset": DATASET,
        "neuprint_endpoint": NEUPRINT_URL,
        "release": "MaleCNS v1.0",
        "parent_experiment": "EXP-003-bilateral",
        "t4_body_count": len(t4_ids),
        "query_rules": {
            "t4_to_intermediate": TARGET_RULE,
            "intermediate_to_lptc_or_dn": LPTC_OR_DN_RULE,
            "lptc_recurrent_or_dn": LPTC_OR_DN_RULE,
        },
        "chemical_graph_only": True,
        "literature_derived_electrical_edges": "none in graph; added only as a model term",
        "lptc_types": list(LPTC_TYPES),
        "dn_types": list(DN_TYPES),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "stage_counts": nodes["stage"].value_counts().to_dict(),
        "edge_counts_by_stage": {"|".join(key): value for key, value in edge_counts.items()},
        "direct_t4_edge_rows_retained": int(len(used_t4_rows)),
        "candidate_intermediate_count": int(len(intermediate_ids)),
        "lptc_count": int(len(all_lptc_ids)),
        "dn_count": int(len(dn_ids)),
        "selection_note": "Only T4-reachable chemical paths that reach HSE/HSN/HSS/H2 or DNp15/DNa02 are retained; no undocumented chemical or electrical edge is added.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
