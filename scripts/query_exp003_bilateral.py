"""Materialize homologous left/right EXP-003 visual-to-DN pathways.

The two eyes are queried independently from the same MaleCNS v1.0 workbook
columns. Each side uses the frozen EXP-002 local query and the corrected
two-hop T4 -> lobula-plate/visual target -> descending selection.  No
left/right steering sign is encoded in this export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_exp002_circuit import T4_INPUT_TYPES, T4_TYPES, incoming_edges, outgoing_edges  # noqa: E402
from scripts.query_malecns_subgraph import DATASET, NEUPRINT_URL, choose_l1_ids  # noqa: E402
from scripts.query_exp003_corrected_downstream import TARGET_RULE, _coverage, _edge_query  # noqa: E402
from src.materialization import canonical_edge_frame, materialize_nodes  # noqa: E402


SELECTED_DN_TYPES = ("DNp15", "DNa02")


def _write_nodes(
    node_ids: set[int],
    stages: dict[int, str],
    annotations: pd.DataFrame,
    transmitters: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    nodes = materialize_nodes(
        node_ids,
        stages=stages,
        annotations=annotations,
        transmitters=transmitters,
    )
    nodes.to_parquet(output / "nodes.parquet", index=False)
    return nodes


def _write_edges(rows: list[dict], output: Path) -> pd.DataFrame:
    frame = canonical_edge_frame(rows)
    frame.to_parquet(output / "edges.parquet", index=False)
    return frame


def materialize_local(
    *,
    eye: str,
    workbook: Path,
    annotations: pd.DataFrame,
    transmitters: pd.DataFrame,
    output: Path,
) -> dict:
    selected_ids, left_ids, right_ids, motion_columns = choose_l1_ids(workbook, ["01_07", "01_08"], eye)
    stage_edges = outgoing_edges(selected_ids, ["Mi1", "Tm3"])
    medulla_ids = sorted({int(row["post_body"]) for row in stage_edges})
    t4_edges = outgoing_edges(medulla_ids, T4_TYPES)
    t4_ids = sorted({int(row["post_body"]) for row in t4_edges})
    direct_t4_inputs = incoming_edges(t4_ids, T4_INPUT_TYPES)
    input_ids = sorted({int(row["pre_body"]) for row in direct_t4_inputs})
    rows = stage_edges + direct_t4_inputs
    stages = {body_id: "visual_input" for body_id in selected_ids}
    stages.update({body_id: "medulla_excitation" for body_id in input_ids if body_id in medulla_ids})
    stages.update({body_id: "t4_input_inhibition" for body_id in input_ids if body_id not in medulla_ids})
    stages.update({body_id: "t4_motion" for body_id in t4_ids})
    node_ids = set(stages) | {int(row["pre_body"]) for row in rows} | {int(row["post_body"]) for row in rows}
    output.mkdir(parents=True, exist_ok=True)
    nodes = _write_nodes(node_ids, stages, annotations, transmitters, output)
    edges = _write_edges(rows, output)
    manifest = {
        "experiment": "EXP-003-bilateral",
        "dataset": DATASET,
        "neuprint_endpoint": NEUPRINT_URL,
        "release": "MaleCNS v1.0",
        "eye": eye,
        "selected_column_suffixes": ["01_07", "01_08"],
        "left_l1_body_ids": left_ids,
        "right_l1_body_ids": right_ids,
        "selected_l1_body_ids": selected_ids,
        "motion_columns": motion_columns,
        "direct_t4_input_types": T4_INPUT_TYPES,
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "t4_count": int(len(t4_ids)),
        "stage_counts": nodes["stage"].value_counts().to_dict(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def materialize_downstream(
    *,
    local_bundle: Path,
    annotations: pd.DataFrame,
    transmitters: pd.DataFrame,
    output: Path,
) -> dict:
    local_nodes = pd.read_parquet(local_bundle / "nodes.parquet")
    t4_ids = local_nodes.loc[local_nodes["stage"].eq("t4_motion"), "bodyId"].astype(int).tolist()
    t4_rows = _edge_query(t4_ids, TARGET_RULE)
    target_ids = sorted({int(row["post_body"]) for row in t4_rows})
    dn_rows = _edge_query(target_ids, 'b.superclass = "descending_neuron"')
    if not t4_rows or not dn_rows:
        raise RuntimeError("Bilateral downstream query returned no T4 or descending edges")
    rows = t4_rows + dn_rows
    stages = {body_id: "t4_motion" for body_id in t4_ids}
    stages.update({body_id: "lobula_plate_target" for body_id in target_ids})
    stages.update({int(row["post_body"]): "descending_output" for row in dn_rows})
    node_ids = set(stages)
    output.mkdir(parents=True, exist_ok=True)
    nodes = _write_nodes(node_ids, stages, annotations, transmitters, output)
    edges = _write_edges(rows, output)
    t4_frame = pd.DataFrame(t4_rows)
    dn_frame = pd.DataFrame(dn_rows)
    manifest = {
        "experiment": "EXP-003-bilateral",
        "dataset": DATASET,
        "release": "MaleCNS v1.0",
        "parent_local_bundle": str(local_bundle),
        "target_rule": TARGET_RULE,
        "continuation_rule": "selected target -> descending_neuron",
        "path_depth_edges": 2,
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "stage_counts": nodes["stage"].value_counts().to_dict(),
        "t4_direct_edge_rows": int(len(t4_frame)),
        "target_to_descending_edge_rows": int(len(dn_frame)),
        "subtype_coverage": _coverage(
            nodes[nodes["stage"].eq("t4_motion")], t4_frame, dn_frame
        ),
        "selected_dn_types_present": sorted(
            set(nodes.loc[nodes["stage"].eq("descending_output"), "type"]) & set(SELECTED_DN_TYPES)
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=ROOT / "data" / "optic-column-type-assignments-v1.0.xlsx")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data" / "body-annotations.feather")
    parser.add_argument("--transmitters", type=Path, default=ROOT / "data" / "body-neurotransmitters.feather")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "malecns_exp003_bilateral")
    args = parser.parse_args()

    annotations = pd.read_feather(args.annotations)
    annotations["bodyId"] = annotations["bodyId"].astype("int64")
    transmitters = pd.read_feather(args.transmitters).rename(columns={"body": "bodyId"})
    transmitters["bodyId"] = transmitters["bodyId"].astype("int64")

    manifests = {}
    for eye in ("left", "right"):
        local_output = args.output / f"{eye}_visual"
        downstream_output = args.output / f"{eye}_downstream"
        local_manifest = materialize_local(
            eye=eye,
            workbook=args.workbook,
            annotations=annotations,
            transmitters=transmitters,
            output=local_output,
        )
        downstream_manifest = materialize_downstream(
            local_bundle=local_output,
            annotations=annotations,
            transmitters=transmitters,
            output=downstream_output,
        )
        manifests[eye] = {"visual": local_manifest, "downstream": downstream_manifest}

    bilateral_manifest = {
        "experiment": "EXP-003-bilateral",
        "dataset": DATASET,
        "release": "MaleCNS v1.0",
        "selected_column_suffixes": ["01_07", "01_08"],
        "side_pairing": "same workbook suffix on left and right eye; each side queried independently",
        "homology_rule": "same T4 subtype/type, same selected downstream target rule, and selected DN type with opposite somaSide",
        "selected_dn_types": list(SELECTED_DN_TYPES),
        "dn_selection_justification": "DNp15/DNHS1 is a bilateral HS-linked optic-flow DN, and DNa02 is identified as steering-related in recent binocular optic-flow circuit work; only cells of these exact MaleCNS types are eligible for the bilateral readout.",
        "sides": manifests,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(bilateral_manifest, indent=2), encoding="utf-8")
    print(json.dumps(bilateral_manifest, indent=2))


if __name__ == "__main__":
    main()
