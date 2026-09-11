"""Materialize a small, official MaleCNS visual-to-DN path via neuPrint.

This is intentionally a query/export helper rather than a hidden replacement
for the MaleCNS graph.  Every exported edge is returned by neuPrint from the
``male-cns:v1.0`` dataset and retains its synapse count.  The static loader can
later consume the official 1.1 GB connectome-weights Feather file directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


NEUPRINT_URL = "https://neuprint.janelia.org/api/custom/custom"
DATASET = "male-cns:v1.0"


def query(cypher: str) -> list[list]:
    payload = json.dumps({"cypher": cypher, "dataset": DATASET}).encode("utf-8")
    request = urllib.request.Request(
        NEUPRINT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    if "error" in body:
        raise RuntimeError(body["error"])
    columns = body.get("columns", [])
    return [dict(zip(columns, row)) for row in body.get("data", [])]


def cypher_list(values: list[int]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def edge_rows(pre_ids: list[int], target_clause: str) -> list[dict]:
    if not pre_ids:
        return []
    rows = []
    for start in range(0, len(pre_ids), 500):
        chunk = pre_ids[start : start + 500]
        cypher = f"""
        MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
        WHERE a.bodyId IN {cypher_list(chunk)} AND {target_clause}
        RETURN a.bodyId as pre_body, b.bodyId as post_body,
               c.weight as synapse_count
        """
        rows.extend(query(" ".join(cypher.split())))
    return rows


def _validate_adjacent(suffixes: list[str]) -> None:
    if len(suffixes) != 2:
        raise ValueError("Exactly two optic-column suffixes are required for the motion sweep")
    try:
        first = tuple(int(value) for value in suffixes[0].split("_"))
        second = tuple(int(value) for value in suffixes[1].split("_"))
    except ValueError as exc:
        raise ValueError("Column suffixes must look like ROW_COL, for example 01_07") from exc
    if len(first) != 2 or len(second) != 2 or sum(abs(a - b) for a, b in zip(first, second)) != 1:
        raise ValueError(f"Column suffixes are not adjacent in the workbook grid: {suffixes}")


def choose_l1_ids(
    workbook: Path, suffixes: list[str], eye: str
) -> tuple[list[int], list[int], list[dict], list[str]]:
    right = pd.read_excel(workbook, sheet_name="Right OL")
    left = pd.read_excel(workbook, sheet_name="Left OL")
    right["suffix"] = right["column"].astype(str).str.split("col_").str[-1]
    left["suffix"] = left["column"].astype(str).str.split("col_").str[-1]
    available = sorted(set(right["suffix"]) if eye == "right" else set(left["suffix"]))
    selected = suffixes or ["01_07", "01_08"]
    _validate_adjacent(selected)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"No {eye}-eye optic-column match for suffixes: {missing}")

    left_rows = left[left["suffix"].isin(selected)]
    right_rows = right[right["suffix"].isin(selected)]
    left_ids = [int(value) for value in left_rows["L1"] if int(value) > 0]
    right_ids = [int(value) for value in right_rows["L1"] if int(value) > 0]
    selected_rows = right_rows if eye == "right" else left_rows
    selected_ids = [int(value) for value in selected_rows["L1"] if int(value) > 0]
    if len(selected_ids) != len(selected) or not selected_ids:
        raise ValueError(f"Selected {eye}-eye optic columns did not contain one valid L1 body ID each")
    motion_columns = [
        {"suffix": suffix, "column": str(row["column"]), "l1_body_id": int(row["L1"])}
        for suffix in selected
        for _, row in selected_rows[selected_rows["suffix"] == suffix].iterrows()
    ]
    if len(motion_columns) != 2:
        raise ValueError("Expected exactly two selected motion columns")
    return selected_ids, left_ids, right_ids, motion_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/body-annotations.feather")
    parser.add_argument("--transmitters", default="data/body-neurotransmitters.feather")
    parser.add_argument("--workbook", default="data/optic-column-type-assignments-v1.0.xlsx")
    parser.add_argument("--output", default="data/malecns_visual_subgraph")
    parser.add_argument("--eye", choices=("left", "right"), default="right")
    parser.add_argument("--suffixes", nargs="*", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected_ids, left_ids, right_ids, motion_columns = choose_l1_ids(
        Path(args.workbook), args.suffixes, args.eye
    )
    selected = [entry["suffix"] for entry in motion_columns]
    seed_ids = selected_ids
    annotations = pd.read_feather(args.annotations)
    annotations["bodyId"] = annotations["bodyId"].astype("int64")

    edges: list[dict] = []
    stages: dict[int, str] = {body_id: "visual_input" for body_id in seed_ids}

    stage1 = edge_rows(seed_ids, 'b.type IN ["Mi1","Tm3"]')
    edges.extend(stage1)
    for row in stage1:
        stages[int(row["post_body"])] = "medulla"

    medulla_ids = [body_id for body_id, stage in stages.items() if stage == "medulla"]
    stage2 = edge_rows(medulla_ids, 'b.type IN ["T4a","T4b","T4c","T4d"]')
    edges.extend(stage2)
    for row in stage2:
        stages[int(row["post_body"])] = "t4_motion"

    # Keep only T4a-d reached by the selected adjacent-column L1 seeds.  This
    # is intentionally a small pathway experiment, not a whole-T4 broadcast.
    t4_ids = [body_id for body_id, stage in stages.items() if stage == "t4_motion"]
    projection_clause = 'b.type IN ["HSE","HSN","HSS","HST","VS","VST1","VST2","VSm"]'
    stage3 = edge_rows(t4_ids, projection_clause)
    edges.extend(stage3)
    for row in stage3:
        stages[int(row["post_body"])] = "wide_field_projection"

    projection_ids = [body_id for body_id, stage in stages.items() if stage == "wide_field_projection"]
    stage4 = edge_rows(projection_ids, 'b.superclass = "descending_neuron"')
    edges.extend(stage4)
    for row in stage4:
        stages[int(row["post_body"])] = "descending_output"

    edge_frame = pd.DataFrame(edges).drop_duplicates(["pre_body", "post_body"])
    edge_frame["pre_body"] = edge_frame["pre_body"].astype("int64")
    edge_frame["post_body"] = edge_frame["post_body"].astype("int64")
    edge_frame["synapse_count"] = edge_frame["synapse_count"].astype("float32")

    tx = pd.read_feather(args.transmitters).rename(columns={"body": "bodyId"})
    tx["bodyId"] = tx["bodyId"].astype("int64")
    node_ids = sorted(set(stages) | set(edge_frame["pre_body"]) | set(edge_frame["post_body"]))
    nodes = annotations[annotations["bodyId"].isin(node_ids)].copy()
    nodes = nodes.merge(
        tx[["bodyId", "consensus_nt", "predicted_nt", "predicted_nt_confidence"]],
        on="bodyId",
        how="left",
    )
    nodes["stage"] = nodes["bodyId"].map(stages).fillna("intermediate")
    keep = [
        "bodyId", "type", "instance", "somaSide", "superclass", "stage",
        "status", "consensus_nt", "predicted_nt", "predicted_nt_confidence",
    ]
    nodes[keep].to_parquet(output / "nodes.parquet", index=False)
    edge_frame.to_parquet(output / "edges.parquet", index=False)
    manifest = {
        "dataset": DATASET,
        "neuprint_endpoint": NEUPRINT_URL,
        "release": "MaleCNS v1.0",
        "source_workbook": "flyconnectome/2025malecns supplemental_data/optic-column-type-assignments-v1.0.xlsx",
        "selected_column_suffixes": selected,
        "eye": args.eye,
        "left_l1_body_ids": left_ids,
        "right_l1_body_ids": right_ids,
        "selected_l1_body_ids": selected_ids,
        "motion_columns": motion_columns,
        "stage_counts": pd.Series(list(stages.values())).value_counts().to_dict(),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edge_frame)),
        "query_stages": [
            "L1 -> Mi1/Tm3",
            "Mi1/Tm3 -> T4a/T4b/T4c/T4d",
            "T4 -> HSE/HSN/HSS/HST/VS/VST1/VST2/VSm",
            "wide-field visual projection -> descending_neuron",
        ],
        "visual_input_boundary": "selected adjacent optic-column L1 IDs",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
