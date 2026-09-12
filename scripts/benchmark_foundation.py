"""Benchmark numerical representations relevant to future CNS experiments.

The benchmark is deterministic, performs no network access, and does not write
results unless stdout is redirected. It compares edge-list scatter accumulation
with CSR matrix multiplication and, when the local corrected EXP-003 bundle is
available, dense with sparse downstream propagation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp003_corrected import CorrectedDownstreamMatrices  # noqa: E402
from src.malecns_io import ConnectomeGraph, load_edges  # noqa: E402
from src.sparse_backend import compile_projection  # noqa: E402


def _median_seconds(function: Callable[[], np.ndarray], repeats: int) -> float:
    function()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _edge_accumulation_case(
    *, n_nodes: int, n_edges: int, steps: int, repeats: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n_nodes, n_edges, dtype=np.int64)
    post = rng.integers(0, n_nodes, n_edges, dtype=np.int64)
    weights = rng.normal(size=n_edges)
    state = rng.normal(size=n_nodes)
    matrix = sparse.csr_matrix((weights, (post, pre)), shape=(n_nodes, n_nodes))
    matrix.sum_duplicates()
    matrix.sort_indices()

    def scatter() -> np.ndarray:
        output = np.zeros(n_nodes, dtype=np.float64)
        for _ in range(steps):
            output.fill(0.0)
            np.add.at(output, post, weights * state[pre])
        return output

    def csr() -> np.ndarray:
        output = np.zeros(n_nodes, dtype=np.float64)
        for _ in range(steps):
            output = matrix @ state
        return output

    scatter_result = scatter()
    csr_result = csr()
    scatter_seconds = _median_seconds(scatter, repeats)
    csr_seconds = _median_seconds(csr, repeats)
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "steps": steps,
        "scatter_median_seconds": scatter_seconds,
        "csr_median_seconds": csr_seconds,
        "csr_speedup": scatter_seconds / csr_seconds,
        "max_abs_difference": float(np.max(np.abs(scatter_result - csr_result))),
        "csr_storage_bytes": int(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        ),
        "dense_storage_bytes_estimate": int(n_nodes * n_nodes * 8),
    }


def _downstream_case(bundle: Path, repeats: int) -> dict | None:
    if not (bundle / "nodes.parquet").exists():
        return None
    graph = ConnectomeGraph(
        pd.read_parquet(bundle / "nodes.parquet"),
        load_edges(bundle / "edges.parquet"),
    )
    dense = CorrectedDownstreamMatrices.from_graph(graph)
    first = compile_projection(
        graph,
        source_ids=dense.t4_ids,
        target_ids=dense.middle_ids,
        normalization="sum",
    )
    second = compile_projection(
        graph,
        source_ids=dense.middle_ids,
        target_ids=dense.descending_ids,
        normalization="sum",
    )
    rng = np.random.default_rng(42)
    activity = rng.random((1501, len(dense.t4_ids)))

    def dense_run() -> np.ndarray:
        return (activity @ dense.t4_to_middle.T) @ dense.middle_to_descending.T

    def sparse_run() -> np.ndarray:
        return second.apply(first.apply(activity))

    dense_result = dense_run()
    sparse_result = sparse_run()
    dense_seconds = _median_seconds(dense_run, repeats)
    sparse_seconds = _median_seconds(sparse_run, repeats)
    dense_bytes = dense.t4_to_middle.nbytes + dense.middle_to_descending.nbytes
    sparse_bytes = first.storage_bytes + second.storage_bytes
    return {
        "bundle": str(bundle),
        "time_steps": len(activity),
        "dense_median_seconds": dense_seconds,
        "sparse_median_seconds": sparse_seconds,
        "sparse_speedup": dense_seconds / sparse_seconds,
        "max_abs_difference": float(np.max(np.abs(dense_result - sparse_result))),
        "dense_storage_bytes": int(dense_bytes),
        "sparse_storage_bytes": int(sparse_bytes),
        "dense_to_sparse_storage_ratio": dense_bytes / sparse_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--downstream-bundle",
        type=Path,
        default=ROOT / "data" / "malecns_exp003_corrected_downstream",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    output = {
        "edge_accumulation": [
            _edge_accumulation_case(
                n_nodes=187,
                n_edges=958,
                steps=1501,
                repeats=args.repeats,
                seed=1,
            ),
            _edge_accumulation_case(
                n_nodes=50_000,
                n_edges=500_000,
                steps=20,
                repeats=args.repeats,
                seed=2,
            ),
        ],
        "corrected_downstream": _downstream_case(
            args.downstream_bundle, args.repeats
        ),
        "trace_storage_estimate": {
            "whole_cns_nodes": 166_691,
            "time_steps": 1501,
            "all_nodes_float32_bytes": 166_691 * 1501 * 4,
            "five_hundred_probes_float32_bytes": 500 * 1501 * 4,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
