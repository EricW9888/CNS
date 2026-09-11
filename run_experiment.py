"""Command-line entry point for the first MaleCNS experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.experiment import run_experiment
from src.lif_model import LIFParameters
from src.malecns_io import build_graph, load_annotations, load_edges, load_transmitters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="data/malecns_visual_subgraph")
    parser.add_argument("--annotations", default="data/body-annotations.feather")
    parser.add_argument("--transmitters", default="data/body-neurotransmitters.feather")
    parser.add_argument("--output", default="results/first_experiment")
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--amplitude-mv", type=float, default=20.0)
    parser.add_argument("--weight-per-synapse-mv", type=float, default=0.275)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    nodes = pd.read_parquet(bundle / "nodes.parquet")
    edges = load_edges(bundle / "edges.parquet")
    # The bundle already contains transmitter columns; loading the official
    # export again keeps this entrypoint compatible with custom exports that do
    # not include those columns.
    transmitters = load_transmitters(args.transmitters) if Path(args.transmitters).exists() else None
    graph = build_graph(nodes, edges, transmitters)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    params = LIFParameters(
        dt_ms=args.dt_ms,
        weight_per_synapse_mV=args.weight_per_synapse_mv,
    )
    motion_columns = manifest.get("motion_columns", [])
    if len(motion_columns) != 2:
        raise RuntimeError("The materialized subgraph does not describe two adjacent motion columns")
    results = run_experiment(
        graph,
        motion_columns=motion_columns,
        output_dir=args.output,
        duration_ms=args.duration_ms,
        params=params,
        amplitude_mV=args.amplitude_mv,
    )
    print(json.dumps({"graph": {"nodes": graph.n_nodes, "edges": graph.n_edges}, "results": results}, indent=2))


if __name__ == "__main__":
    main()
