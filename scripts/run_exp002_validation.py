"""Validate frozen EXP-002 across held-out grid pairs and a structural null.

This script is deliberately an analysis/orchestration layer.  It calls the
existing circuit query and frozen ``run_exp002.py`` runner; it does not alter
the graded model, its parameters, its signs, or its stimulus.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_exp002_circuit import T4_TYPES  # noqa: E402
from scripts.query_malecns_subgraph import cypher_list, query  # noqa: E402
from src.exp002_validation import (  # noqa: E402
    adjacent_pairs,
    evenly_spaced_pairs,
    mean_order_contrast,
    summarize_type_preferences,
)
from src.graded_model import INHIBITORY_T4_INPUTS, run_graded, t4_comparison  # noqa: E402
from src.malecns_io import build_graph, load_edges, load_transmitters  # noqa: E402
from src.visual_stimulus import motion_events  # noqa: E402


PRIMARY_PAIR = ("01_07", "01_08")


def _load_graph(bundle: Path, transmitters: Path):
    nodes = pd.read_parquet(bundle / "nodes.parquet")
    edges = load_edges(bundle / "edges.parquet")
    tx = load_transmitters(transmitters) if transmitters.exists() else None
    return build_graph(nodes, edges, tx)


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + completed.stdout[-4000:]
            + "\nstderr:\n"
            + completed.stderr[-4000:]
        )


def _run_pair(
    pair: tuple[str, str],
    *,
    axis: str,
    index: int,
    args: argparse.Namespace,
) -> dict:
    safe_name = f"{pair[0]}_to_{pair[1]}"
    bundle = args.data_root / f"malecns_exp002_validation_{safe_name}"
    output = args.results_root / safe_name
    _run_command(
        [
            sys.executable,
            "scripts/query_exp002_circuit.py",
            "--annotations",
            str(args.annotations),
            "--transmitters",
            str(args.transmitters),
            "--workbook",
            str(args.workbook),
            "--eye",
            "right",
            "--suffixes",
            pair[0],
            pair[1],
            "--output",
            str(bundle),
        ]
    )
    _run_command(
        [
            sys.executable,
            "run_exp002.py",
            "--bundle",
            str(bundle),
            "--transmitters",
            str(args.transmitters),
            "--output",
            str(output),
        ]
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(output / "t4_direction_comparison.csv")
    by_type = summarize_type_preferences(comparison)
    row = {
        "pair": f"{pair[0]}->{pair[1]}",
        "axis": axis,
        "selection_index": index,
        "node_count": int(manifest["node_count"]),
        "edge_count": int(manifest["edge_count"]),
        "t4_count": int(metrics["t4_neurons"]),
        "mean_order_contrast": mean_order_contrast(comparison),
        "mean_order_contrast_inhibitory_ablation": mean_order_contrast(
            pd.read_csv(output / "t4_direction_comparison.csv")
        ),
        "by_type": by_type.to_dict(orient="records"),
        "output": str(output),
    }
    # The runner's metrics include ablation traces, but its comparison CSV is
    # intentionally the full circuit comparison.  Recompute the ablated
    # comparison here so the validation summary cannot confuse the two.
    full_graph = _load_graph(bundle, args.transmitters)
    motion_columns = manifest["motion_columns"]
    forward_events = motion_events(
        first_column=motion_columns[0]["suffix"],
        first_l1_ids=(int(motion_columns[0]["l1_body_id"]),),
        second_column=motion_columns[1]["suffix"],
        second_l1_ids=(int(motion_columns[1]["l1_body_id"]),),
        amplitude_mV=1.0,
    )
    reverse_events = motion_events(
        first_column=motion_columns[0]["suffix"],
        first_l1_ids=(int(motion_columns[0]["l1_body_id"]),),
        second_column=motion_columns[1]["suffix"],
        second_l1_ids=(int(motion_columns[1]["l1_body_id"]),),
        reverse=True,
        amplitude_mV=1.0,
    )
    forward_ablated, _ = run_graded(
        full_graph,
        forward_events,
        motion_columns=motion_columns,
        duration_ms=150.0,
        ablate_inhibitory=True,
    )
    reverse_ablated, _ = run_graded(
        full_graph,
        reverse_events,
        motion_columns=motion_columns,
        duration_ms=150.0,
        ablate_inhibitory=True,
    )
    ablated_comparison = t4_comparison(forward_ablated, reverse_ablated)
    row["mean_order_contrast_inhibitory_ablation"] = mean_order_contrast(ablated_comparison)
    row["inhibitory_reduction_fraction"] = 1.0 - (
        row["mean_order_contrast_inhibitory_ablation"] / row["mean_order_contrast"]
    )
    row["ablated_by_type"] = summarize_type_preferences(ablated_comparison).to_dict(orient="records")
    return row


def _query_t4_downstream_edges(t4_ids: list[int]) -> pd.DataFrame:
    if not t4_ids:
        return pd.DataFrame(columns=["pre_body", "pre_type", "synapse_count", "post_body", "post_type"])
    cypher = f"""
    MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
    WHERE a.bodyId IN {cypher_list(t4_ids)}
    RETURN a.bodyId as pre_body, a.type as pre_type,
           c.weight as synapse_count, b.bodyId as post_body,
           b.type as post_type
    """
    rows = query(" ".join(cypher.split()))
    return pd.DataFrame(rows)


def _matched_neutral_control(
    *,
    bundle: Path,
    annotations: Path,
    transmitters: Path,
    output: Path,
) -> dict:
    """Compare inhibitory ablation with a matched downstream structural null.

    T4 outgoing edges are downstream of the first-pass T4 readout and have no
    sign in the frozen feed-forward model.  We remove the same number of these
    structural edges as modeled inhibitory edges.  This is a matched negative
    control for the current readout, not a claim that downstream T4 targets
    are biologically unimportant in a larger model.
    """

    graph = _load_graph(bundle, transmitters)
    t4_ids = graph.nodes.loc[graph.nodes["stage"] == "t4_motion", "bodyId"].astype(int).tolist()
    neutral_edges = _query_t4_downstream_edges(t4_ids)
    if neutral_edges.empty:
        raise RuntimeError("MaleCNS query returned no T4 downstream neutral-control edges")
    neutral_edges["pre_body"] = neutral_edges["pre_body"].astype("int64")
    neutral_edges["post_body"] = neutral_edges["post_body"].astype("int64")
    neutral_edges["synapse_count"] = neutral_edges["synapse_count"].astype("float32")
    duplicate_edges = neutral_edges.duplicated(["pre_body", "post_body"], keep=False)
    if duplicate_edges.any():
        examples = neutral_edges.loc[
            duplicate_edges, ["pre_body", "post_body"]
        ].head(5)
        raise RuntimeError(
            "MaleCNS query returned duplicate aggregate T4 downstream pairs: "
            f"{list(examples.itertuples(index=False, name=None))}"
        )
    neutral_edges = (
        neutral_edges[
            (~neutral_edges["post_type"].isin(T4_TYPES))
            & (~neutral_edges["post_body"].isin(t4_ids))
        ]
        .sort_values(["post_body", "pre_body"])
        .reset_index(drop=True)
    )
    output.mkdir(parents=True, exist_ok=True)
    neutral_edges.to_csv(output / "t4_downstream_neutral_edges.csv", index=False)

    inhibitory_edges = graph.edges[graph.edges["pre_type"].isin(INHIBITORY_T4_INPUTS)]
    if len(neutral_edges) < len(inhibitory_edges):
        raise RuntimeError(
            f"Only {len(neutral_edges)} downstream neutral edges are available; "
            f"need {len(inhibitory_edges)} for the matched control"
        )
    neutral_edges = neutral_edges.head(len(inhibitory_edges)).copy()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    columns = manifest["motion_columns"]
    first_column, second_column = columns
    forward_events = motion_events(
        first_column=first_column["suffix"],
        first_l1_ids=(int(first_column["l1_body_id"]),),
        second_column=second_column["suffix"],
        second_l1_ids=(int(second_column["l1_body_id"]),),
        amplitude_mV=1.0,
    )
    reverse_events = motion_events(
        first_column=first_column["suffix"],
        first_l1_ids=(int(first_column["l1_body_id"]),),
        second_column=second_column["suffix"],
        second_l1_ids=(int(second_column["l1_body_id"]),),
        reverse=True,
        amplitude_mV=1.0,
    )
    base_nodes = pd.read_parquet(bundle / "nodes.parquet")
    annotation_frame = pd.read_feather(annotations)
    if base_nodes["bodyId"].duplicated().any():
        raise ValueError("base bundle contains duplicate body IDs")
    if annotation_frame["bodyId"].duplicated().any():
        raise ValueError("annotation export contains duplicate body IDs")
    extra_ids = sorted(set(neutral_edges["post_body"].astype(int)) - set(base_nodes["bodyId"].astype(int)))
    extra_nodes = annotation_frame[annotation_frame["bodyId"].astype(int).isin(extra_ids)].copy()
    extra_nodes["bodyId"] = extra_nodes["bodyId"].astype("int64")
    missing_extra_ids = set(extra_ids) - set(extra_nodes["bodyId"].astype(int))
    if missing_extra_ids:
        raise ValueError(
            "annotations are missing neutral-control target IDs: "
            f"{sorted(missing_extra_ids)[:5]}"
        )
    extra_nodes["stage"] = "neutral_downstream"
    for column in base_nodes.columns:
        if column not in extra_nodes.columns:
            extra_nodes[column] = pd.NA
    neutral_nodes = pd.concat(
        [base_nodes, extra_nodes[base_nodes.columns]], ignore_index=True
    )
    base_edges = load_edges(bundle / "edges.parquet")
    neutral_graph = build_graph(
        neutral_nodes,
        pd.concat(
            [
                base_edges,
                neutral_edges[["pre_body", "post_body", "synapse_count"]],
            ],
            ignore_index=True,
        ),
        load_transmitters(transmitters),
    )
    full_forward, _ = run_graded(
        graph, forward_events, motion_columns=columns, duration_ms=150.0
    )
    full_reverse, _ = run_graded(
        graph, reverse_events, motion_columns=columns, duration_ms=150.0
    )
    full_contrast = mean_order_contrast(t4_comparison(full_forward, full_reverse))
    forward_neutral, _ = run_graded(
        neutral_graph, forward_events, motion_columns=columns, duration_ms=150.0
    )
    reverse_neutral, _ = run_graded(
        neutral_graph, reverse_events, motion_columns=columns, duration_ms=150.0
    )
    neutral_comparison = t4_comparison(forward_neutral, reverse_neutral)
    metrics = {
        "control": "matched T4->downstream structural null",
        "full_circuit_mean_order_contrast": full_contrast,
        "inhibitory_edge_count": int(len(inhibitory_edges)),
        "inhibitory_synapse_count": float(inhibitory_edges["synapse_count"].sum()),
        "neutral_edge_count": int(len(neutral_edges)),
        "neutral_synapse_count": float(neutral_edges["synapse_count"].sum()),
        "neutral_edge_count_fraction_of_inhibitory": float(len(neutral_edges) / len(inhibitory_edges)),
        "neutral_mean_order_contrast": mean_order_contrast(neutral_comparison),
        "neutral_minus_full": float(
            mean_order_contrast(neutral_comparison)
            - full_contrast
        ),
        "neutral_by_type": summarize_type_preferences(neutral_comparison).to_dict(orient="records"),
    }
    (output / "matched_control_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=Path("data/optic-column-type-assignments-v1.0.xlsx"))
    parser.add_argument("--annotations", type=Path, default=Path("data/body-annotations.feather"))
    parser.add_argument("--transmitters", type=Path, default=Path("data/body-neurotransmitters.feather"))
    parser.add_argument("--primary-bundle", type=Path, default=Path("data/malecns_exp002_local_circuit"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results/exp002_validation"))
    parser.add_argument("--pairs-per-axis", type=int, default=4)
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="Reuse the previously written held-out sweep and rerun only the matched control.",
    )
    args = parser.parse_args()

    if args.control_only:
        prior_path = args.results_root / "validation_summary.json"
        if not prior_path.exists():
            raise FileNotFoundError("--control-only requires an existing validation_summary.json")
        held_out = json.loads(prior_path.read_text(encoding="utf-8"))["held_out_pairs"]
    else:
        workbook = pd.read_excel(args.workbook, sheet_name="Right OL")
        workbook["suffix"] = workbook["column"].astype(str).str.split("col_").str[-1]
        pairs_by_axis = adjacent_pairs(workbook["suffix"].tolist())
        selected_pairs: list[dict] = []
        for axis, pairs in pairs_by_axis.items():
            selected = evenly_spaced_pairs(
                pairs,
                count=args.pairs_per_axis,
                exclude={PRIMARY_PAIR},
            )
            selected_pairs.extend(
                {"axis": axis, "pair": pair, "index": index}
                for index, pair in enumerate(selected)
            )

        held_out = [
            _run_pair(item["pair"], axis=item["axis"], index=item["index"], args=args)
            for item in selected_pairs
        ]
    control = _matched_neutral_control(
        bundle=args.primary_bundle,
        transmitters=args.transmitters,
        annotations=args.annotations,
        output=args.results_root / "matched_control",
    )
    summary = {
        "experiment_id": "EXP-002",
        "implementation_commit": "5dc96b4",
        "parameter_set": {
            "dt_ms": 0.1,
            "tau_l1_ms": 5.0,
            "tau_tm3_ms": 12.0,
            "tau_mi1_ms": 20.0,
            "tau_inhibitory_ms": 30.0,
            "tau_t4_ms": 10.0,
            "inhibitory_delay_ms": 15.0,
            "stimulus_amplitude": 1.0,
            "pulse_duration_ms": 10.0,
            "event_starts_ms": [10.0, 20.0],
        },
        "geometry": {
            "primary_pair": "01_07->01_08",
            "eye": "right",
            "workbook_interpretation": "positive step in the second ROW_COL grid index",
            "canonical_body_axis_mapping": None,
            "reason_unresolved": "MaleCNS v1.0 workbook names and orders grid columns but does not publish a mapping from these suffix indices to front/back, dorsal/ventral, or T4 cardinal directions.",
        },
        "held_out_pairs": held_out,
        "matched_control": control,
        "outputs": {
            "summary_json": str(args.results_root / "validation_summary.json"),
            "held_out_csv": str(args.results_root / "held_out_pair_summary.csv"),
            "matched_control_metrics": str(args.results_root / "matched_control" / "matched_control_metrics.json"),
        },
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"by_type", "ablated_by_type"}
            }
            for row in held_out
        ]
    ).to_csv(args.results_root / "held_out_pair_summary.csv", index=False)
    (args.results_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
