"""Small, explicit visual events for the first MaleCNS experiment.

The primary stimulus is a two-event sweep over adjacent optic columns.  A
single-column pulse remains available as a smoke-test primitive, but it is not
used as the motion readout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VisualEvent:
    """A rectangular drive into explicitly named visual body IDs."""

    name: str
    node_ids: tuple[int, ...]
    start_ms: float = 10.0
    duration_ms: float = 10.0
    amplitude_mV: float = 8.0


VisualPulse = VisualEvent


def motion_events(
    *,
    first_column: str,
    first_l1_ids: tuple[int, ...],
    second_column: str,
    second_l1_ids: tuple[int, ...],
    reverse: bool = False,
    start_ms: float = 10.0,
    pulse_duration_ms: float = 10.0,
    gap_ms: float = 0.0,
    amplitude_mV: float = 8.0,
) -> tuple[VisualEvent, VisualEvent]:
    """Create one of two opposite-order adjacent-column sweeps.

    The names deliberately describe workbook column order rather than
    claiming a calibrated leftward/rightward retinal direction.
    """

    first_event = VisualEvent(
        name=first_column,
        node_ids=tuple(first_l1_ids),
        start_ms=start_ms,
        duration_ms=pulse_duration_ms,
        amplitude_mV=amplitude_mV,
    )
    second_event = VisualEvent(
        name=second_column,
        node_ids=tuple(second_l1_ids),
        start_ms=start_ms + pulse_duration_ms + gap_ms,
        duration_ms=pulse_duration_ms,
        amplitude_mV=amplitude_mV,
    )
    if reverse:
        return (
            VisualEvent(**{**asdict(second_event), "start_ms": start_ms}),
            VisualEvent(**{**asdict(first_event), "start_ms": second_event.start_ms}),
        )
    return first_event, second_event


def compile_events(
    events: tuple[VisualEvent, ...] | list[VisualEvent],
    *,
    node_index: dict[int, int],
    n_nodes: int,
    duration_ms: float,
    dt_ms: float,
) -> np.ndarray:
    """Compile additive visual events into a time-by-node current matrix."""

    if not np.isfinite(duration_ms) or not np.isfinite(dt_ms):
        raise ValueError("duration_ms and dt_ms must be finite")
    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    if n_nodes < 0:
        raise ValueError("n_nodes cannot be negative")
    n_steps = int(round(duration_ms / dt_ms)) + 1
    values = np.zeros((n_steps, n_nodes), dtype=np.float32)
    for event in events:
        event_values = np.asarray(
            [event.start_ms, event.duration_ms, event.amplitude_mV], dtype=np.float64
        )
        if not np.isfinite(event_values).all():
            raise ValueError(f"Visual event {event.name!r} contains non-finite values")
        if event.duration_ms <= 0:
            raise ValueError(f"Visual event {event.name!r} must have positive duration")
        start = min(n_steps, max(0, int(round(event.start_ms / dt_ms))))
        end = min(
            n_steps,
            max(0, int(round((event.start_ms + event.duration_ms) / dt_ms))),
        )
        missing = [body_id for body_id in event.node_ids if int(body_id) not in node_index]
        if missing:
            raise ValueError(f"Visual event {event.name!r} references nodes outside the graph: {missing}")
        if start >= end:
            continue
        indices = [node_index[int(body_id)] for body_id in event.node_ids]
        values[start:end, indices] += event.amplitude_mV
    return values


def compile_pulse(
    pulse: VisualPulse,
    *,
    node_index: dict[int, int],
    n_nodes: int,
    duration_ms: float,
    dt_ms: float,
) -> np.ndarray:
    """Backward-compatible single-event smoke-test compiler."""

    return compile_events(
        (pulse,),
        node_index=node_index,
        n_nodes=n_nodes,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
    )


def event_dicts(events: tuple[VisualEvent, ...] | list[VisualEvent]) -> list[dict]:
    return [asdict(event) for event in events]
