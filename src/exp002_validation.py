"""Pure analysis helpers for the frozen EXP-002 validation pass.

These helpers do not alter the graded model.  They define deterministic grid
pair selection and summary statistics so the validation sweep can be rerun
without choosing pairs or direction labels by looking at the result.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def parse_grid_suffix(suffix: str) -> tuple[int, int]:
    """Return the two integer coordinates encoded by a workbook suffix."""

    parts = str(suffix).split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected ROW_COL suffix, got {suffix!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Expected numeric ROW_COL suffix, got {suffix!r}") from exc


def format_grid_suffix(coordinate: tuple[int, int]) -> str:
    return f"{coordinate[0]:02d}_{coordinate[1]:02d}"


def adjacent_pairs(suffixes: Iterable[str]) -> dict[str, list[tuple[str, str]]]:
    """Enumerate directed positive steps on each workbook grid index."""

    available = {str(suffix) for suffix in suffixes}
    pairs: dict[str, list[tuple[str, str]]] = {
        "first_index": [],
        "second_index": [],
    }
    for suffix in sorted(available):
        coordinate = parse_grid_suffix(suffix)
        for axis, name in ((0, "first_index"), (1, "second_index")):
            next_coordinate = list(coordinate)
            next_coordinate[axis] += 1
            next_suffix = format_grid_suffix(tuple(next_coordinate))
            if next_suffix in available:
                pairs[name].append((suffix, next_suffix))
    return pairs


def evenly_spaced_pairs(
    pairs: list[tuple[str, str]],
    *,
    count: int,
    exclude: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Select pairs by grid position, independent of model output."""

    if count <= 0:
        return []
    excluded = exclude or set()
    candidates = [pair for pair in pairs if pair not in excluded]
    if len(candidates) <= count:
        return candidates
    indices = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int)
    return [candidates[index] for index in np.unique(indices)]


def summarize_type_preferences(
    comparison: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Count forward/reverse peak preferences by T4 subtype."""

    frame = comparison.copy()
    delta = frame["max_activity_difference"].to_numpy(dtype=float)
    frame["forward_gt_reverse"] = delta > tolerance
    frame["reverse_gt_forward"] = delta < -tolerance
    frame["approximately_equal"] = np.abs(delta) <= tolerance
    return (
        frame.groupby("type", as_index=False)
        .agg(
            neurons=("bodyId", "count"),
            forward_gt_reverse=("forward_gt_reverse", "sum"),
            reverse_gt_forward=("reverse_gt_forward", "sum"),
            approximately_equal=("approximately_equal", "sum"),
            mean_forward_peak=("max_activity_forward", "mean"),
            mean_reverse_peak=("max_activity_reverse", "mean"),
            mean_order_contrast=("max_abs_pointwise_activity_difference", "mean"),
        )
        .sort_values("type")
        .reset_index(drop=True)
    )


def mean_order_contrast(comparison: pd.DataFrame) -> float:
    """Return mean neuron-level ``|peak_forward - peak_reverse|`` contrast."""

    return float(comparison["max_activity_difference"].abs().mean())
