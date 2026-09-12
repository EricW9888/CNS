"""Materialize the smallest MaleCNS local circuit needed for EXP-002.

The circuit keeps the two EXP-001 L1 seeds and the same reachable T4
population, but adds the direct T4 input classes identified in the published
T4 connectome analysis.  The exported topology remains MaleCNS-derived; the
graded model decides how those signals are transformed in time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_malecns_subgraph import DATASET, NEUPRINT_URL, choose_l1_ids, query  # noqa: E402
from src.materialization import canonical_edge_frame, materialize_nodes  # noqa: E402


T4_TYPES = ["T4a", "T4b", "T4c", "T4d"]
T4_INPUT_TYPES = ["Mi1", "Tm3", "Mi4", "Mi9", "C3", "CT1"]


def cypher_list(values: list[int]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def outgoing_edges(pre_ids: list[int], target_types: list[str]) -> list[dict]:
    if not pre_ids:
        return []
    rows: list[dict] = []
    type_list = "[" + ",".join(json.dumps(value) for value in target_types) + "]"
    for start in range(0, len(pre_ids), 500):
        chunk = pre_ids[start : start + 500]
        cypher = f"""
        MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
        WHERE a.bodyId IN {cypher_list(chunk)} AND b.type IN {type_list}
        RETURN a.bodyId as pre_body, a.type as pre_type,
               c.weight as synapse_count, b.bodyId as post_body,
               b.type as post_type
        """
        rows.extend(query(" ".join(cypher.split())))
    return rows


def incoming_edges(post_ids: list[int], source_types: list[str]) -> list[dict]:
    if not post_ids:
        return []
    rows: list[dict] = []
    type_list = "[" + ",".join(json.dumps(value) for value in source_types) + "]"
    for start in range(0, len(post_ids), 500):
        chunk = post_ids[start : start + 500]
        cypher = f"""
        MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
        WHERE b.bodyId IN {cypher_list(chunk)} AND a.type IN {type_list}
        RETURN a.bodyId as pre_body, a.type as pre_type,
               c.weight as synapse_count, b.bodyId as post_body,
               b.type as post_type
        """
        rows.extend(query(" ".join(cypher.split())))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/body-annotations.feather")
    parser.add_argument("--transmitters", default="data/body-neurotransmitters.feather")
    parser.add_argument("--workbook", default="data/optic-column-type-assignments-v1.0.xlsx")
    parser.add_argument("--output", default="data/malecns_exp002_local_circuit")
    parser.add_argument("--eye", choices=("left", "right"), default="right")
    parser.add_argument("--suffixes", nargs="*", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected_ids, left_ids, right_ids, motion_columns = choose_l1_ids(
        Path(args.workbook), args.suffixes, args.eye
    )
    annotations = pd.read_feather(args.annotations)
    annotations["bodyId"] = annotations["bodyId"].astype("int64")

    stage_edges = outgoing_edges(selected_ids, ["Mi1", "Tm3"])
    medulla_ids = sorted({int(row["post_body"]) for row in stage_edges})
    t4_edges = outgoing_edges(medulla_ids, T4_TYPES)
    t4_ids = sorted({int(row["post_body"]) for row in t4_edges})
    direct_t4_inputs = incoming_edges(t4_ids, T4_INPUT_TYPES)
    input_ids = sorted({int(row["pre_body"]) for row in direct_t4_inputs})

    edges = stage_edges + direct_t4_inputs
    edge_frame = canonical_edge_frame(edges)

    stages: dict[int, str] = {body_id: "visual_input" for body_id in selected_ids}
    stages.update({body_id: "medulla_excitation" for body_id in input_ids if body_id in medulla_ids})
    stages.update(
        {
            body_id: "t4_input_inhibition"
            for body_id in input_ids
            if body_id not in medulla_ids
        }
    )
    stages.update({body_id: "t4_motion" for body_id in t4_ids})
    node_ids = sorted(set(stages) | set(edge_frame["pre_body"]) | set(edge_frame["post_body"]))

    tx = pd.read_feather(args.transmitters).rename(columns={"body": "bodyId"})
    tx["bodyId"] = tx["bodyId"].astype("int64")
    nodes = materialize_nodes(
        node_ids,
        stages=stages,
        annotations=annotations,
        transmitters=tx,
    )
    nodes.to_parquet(output / "nodes.parquet", index=False)
    edge_frame.to_parquet(output / "edges.parquet", index=False)
    manifest = {
        "experiment": "EXP-002",
        "dataset": DATASET,
        "neuprint_endpoint": NEUPRINT_URL,
        "release": "MaleCNS v1.0",
        "source_workbook": "flyconnectome/2025malecns supplemental_data/optic-column-type-assignments-v1.0.xlsx",
        "selected_column_suffixes": [entry["suffix"] for entry in motion_columns],
        "eye": args.eye,
        "left_l1_body_ids": left_ids,
        "right_l1_body_ids": right_ids,
        "selected_l1_body_ids": selected_ids,
        "motion_columns": motion_columns,
        "direct_t4_input_types": T4_INPUT_TYPES,
        "stage_counts": nodes["stage"].value_counts().to_dict(),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edge_frame)),
        "query_stages": [
            "selected L1 -> Mi1/Tm3",
            "reachable Mi1/Tm3 -> T4a/T4b/T4c/T4d",
            "all selected-T4 inputs of types Mi1/Tm3/Mi4/Mi9/C3/CT1",
            "L1 -> Mi1/Tm3 edges retained for the graded receptor-inversion model",
        ],
        "anatomical_boundary": "two adjacent right-eye L1 seeds and the direct T4 input circuit reachable around their EXP-001 T4 set",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
