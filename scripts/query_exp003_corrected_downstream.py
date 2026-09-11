"""Materialize the audited MaleCNS T4 downstream path for EXP-003.

The original EXP-003 bundle selected a small named H/VS population.  This
export keeps the visual computation fixed and expands only the direct T4
target boundary to the lobula-plate/visual populations supported by MaleCNS:
visual projection/centrifugal cells and LPi intrinsic cells.  A second query
keeps only their direct descending-neuron outputs, making the corrected graph
an explicit two-hop T4 -> lobula-plate target -> descending path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_malecns_subgraph import DATASET, cypher_list, query  # noqa: E402


TARGET_RULE = (
    '(b.superclass IN ["visual_projection","visual_centrifugal"] '
    'OR b.type STARTS WITH "LPi")'
)


def _edge_query(pre_ids: list[int], target_rule: str) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(pre_ids), 500):
        chunk = pre_ids[start : start + 500]
        cypher = f"""
        MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
        WHERE a.bodyId IN {cypher_list(chunk)} AND {target_rule}
        RETURN a.bodyId as pre_body, a.type as pre_type,
               b.bodyId as post_body, b.type as post_type,
               b.superclass as post_superclass, b.somaSide as post_soma_side,
               c.weight as synapse_count
        """
        rows.extend(query(" ".join(cypher.split())))
    return rows


def _load_t4_ids(exp002_bundle: Path) -> list[int]:
    nodes = pd.read_parquet(exp002_bundle / "nodes.parquet")
    return (
        nodes.loc[nodes["stage"].eq("t4_motion"), "bodyId"]
        .astype("int64")
        .tolist()
    )


def _coverage(t4_nodes: pd.DataFrame, t4_edges: pd.DataFrame, dn_edges: pd.DataFrame) -> dict:
    rows: dict[str, dict] = {}
    for cell_type, group in t4_nodes.groupby("type", sort=True):
        ids = set(group["bodyId"].astype(int))
        direct = t4_edges[t4_edges["pre_body"].isin(ids)]
        supported_targets = set(direct["post_body"].astype(int))
        paths = dn_edges[dn_edges["pre_body"].isin(supported_targets)]
        path_t4 = set(direct[direct["post_body"].isin(set(paths["pre_body"].astype(int)))]["pre_body"].astype(int))
        rows[str(cell_type)] = {
            "total_neurons": int(len(ids)),
            "neurons_with_direct_target": int(len(set(direct["pre_body"].astype(int)))),
            "neurons_with_target_to_descending_path": int(len(path_t4)),
            "direct_target_cells": int(len(supported_targets)),
            "descending_cells": int(len(set(paths["post_body"].astype(int)))),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp002-bundle", type=Path, default=ROOT / "data" / "malecns_exp002_local_circuit")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data" / "body-annotations.feather")
    parser.add_argument("--transmitters", type=Path, default=ROOT / "data" / "body-neurotransmitters.feather")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "malecns_exp003_corrected_downstream")
    args = parser.parse_args()

    t4_ids = _load_t4_ids(args.exp002_bundle)
    if len(t4_ids) != 43:
        raise RuntimeError(f"Expected the frozen EXP-002 bundle to contain 43 T4 neurons, found {len(t4_ids)}")

    t4_rows = _edge_query(t4_ids, TARGET_RULE)
    target_ids = sorted({int(row["post_body"]) for row in t4_rows})
    dn_rows = _edge_query(target_ids, 'b.superclass = "descending_neuron"')

    t4_edges = pd.DataFrame(t4_rows)
    dn_edges = pd.DataFrame(dn_rows)
    if t4_edges.empty or dn_edges.empty:
        raise RuntimeError("Corrected downstream queries returned no edges")
    t4_edges = t4_edges.rename(columns={"post_superclass": "post_superclass"})
    dn_edges = dn_edges.rename(columns={"post_superclass": "post_superclass"})
    edge_frame = pd.concat(
        [
            t4_edges[["pre_body", "post_body", "synapse_count"]],
            dn_edges[["pre_body", "post_body", "synapse_count"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["pre_body", "post_body"])
    edge_frame["pre_body"] = edge_frame["pre_body"].astype("int64")
    edge_frame["post_body"] = edge_frame["post_body"].astype("int64")
    edge_frame["synapse_count"] = edge_frame["synapse_count"].astype("float32")

    annotations = pd.read_feather(args.annotations)
    annotations["bodyId"] = annotations["bodyId"].astype("int64")
    tx = pd.read_feather(args.transmitters).rename(columns={"body": "bodyId"})
    tx["bodyId"] = tx["bodyId"].astype("int64")
    tx = tx.drop_duplicates("bodyId", keep="first")
    node_ids = sorted(set(t4_ids) | set(target_ids) | set(dn_edges["post_body"].astype(int)))
    nodes = annotations[annotations["bodyId"].isin(node_ids)].copy()
    nodes = nodes.merge(
        tx[["bodyId", "consensus_nt", "predicted_nt", "predicted_nt_confidence"]],
        on="bodyId",
        how="left",
    )
    t4_set = set(t4_ids)
    target_set = set(target_ids)
    dn_set = set(dn_edges["post_body"].astype(int))
    nodes["stage"] = nodes["bodyId"].map(
        lambda body_id: "t4_motion" if int(body_id) in t4_set else (
            "lobula_plate_target" if int(body_id) in target_set else (
                "descending_output" if int(body_id) in dn_set else "intermediate"
            )
        )
    )
    keep = [
        "bodyId", "type", "instance", "somaSide", "superclass", "stage",
        "status", "consensus_nt", "predicted_nt", "predicted_nt_confidence",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    nodes[keep].to_parquet(args.output / "nodes.parquet", index=False)
    edge_frame.to_parquet(args.output / "edges.parquet", index=False)

    coverage = _coverage(nodes[nodes["bodyId"].isin(t4_set)], t4_edges, dn_edges)
    target_type_counts = (
        t4_edges.groupby("post_type")["post_body"].nunique().sort_values(ascending=False).astype(int).to_dict()
    )
    manifest = {
        "dataset": DATASET,
        "release": "MaleCNS v1.0",
        "parent_bundle": "data/malecns_exp002_local_circuit",
        "target_rule": TARGET_RULE,
        "path_depth_edges": 2,
        "query_stages": [
            "frozen EXP-002 T4a/T4b/T4c/T4d -> visual_projection/visual_centrifugal/LPi",
            "selected lobula-plate targets -> descending_neuron",
        ],
        "node_count": int(len(nodes)),
        "edge_count": int(len(edge_frame)),
        "stage_counts": nodes["stage"].value_counts().to_dict(),
        "t4_direct_edge_rows": int(len(t4_edges)),
        "target_to_descending_edge_rows": int(len(dn_edges)),
        "target_type_unique_cell_counts": target_type_counts,
        "subtype_coverage": coverage,
        "selection_note": "The original H/VS-only target clause was incomplete for T4b/T4c and omitted the direct LPi/LPLC/LLPC/LPC/LPT/vCal/Nod/OLVC pathways present in MaleCNS.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
