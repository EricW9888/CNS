"""Small numerical invariants for recurrent scientific models.

Historical experiment modules remain unchanged. These helpers define the
preflight semantics required before a successor recurrent model can be treated
as evidence.
"""

from __future__ import annotations

import numpy as np


def normalize_rows_by_absolute_sum(matrix: np.ndarray) -> np.ndarray:
    """Normalize nonzero rows by absolute mass while preserving signs."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("matrix must be a finite two-dimensional array")
    denominator = np.sum(np.abs(values), axis=1, keepdims=True)
    return np.divide(
        values,
        denominator,
        out=np.zeros_like(values),
        where=denominator > 0,
    )


def diffusive_coupling_current(
    state: np.ndarray,
    conductance: np.ndarray,
) -> np.ndarray:
    """Return electrical current ``sum_j g_ij * (x_j - x_i)``.

    ``conductance`` is required to be finite, symmetric, nonnegative, and to
    have a zero diagonal. Those are deliberately strict preflight conditions
    for an undirected gap-junction conductance matrix.
    """

    values = np.asarray(state, dtype=np.float64)
    coupling = np.asarray(conductance, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("state must be one-dimensional")
    if coupling.shape != (len(values), len(values)):
        raise ValueError("conductance shape must match the state vector")
    if not np.isfinite(values).all() or not np.isfinite(coupling).all():
        raise ValueError("state and conductance must be finite")
    if np.any(coupling < 0):
        raise ValueError("electrical conductances cannot be negative")
    if not np.allclose(coupling, coupling.T):
        raise ValueError("gap-junction conductance must be symmetric")
    if not np.allclose(np.diag(coupling), 0.0):
        raise ValueError("gap-junction conductance must have a zero diagonal")
    return coupling @ values - coupling.sum(axis=1) * values


def effective_euler_transition(
    feedback: np.ndarray,
    *,
    dt: float,
    tau: float | np.ndarray,
) -> np.ndarray:
    """Return the zero-input Euler transition for ``dx/dt=(W x-x)/tau``."""

    weights = np.asarray(feedback, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("feedback must be a square matrix")
    if not np.isfinite(weights).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("feedback and positive dt must be finite")
    time_constants = np.broadcast_to(np.asarray(tau, dtype=np.float64), (len(weights),))
    if not np.isfinite(time_constants).all() or np.any(time_constants <= 0):
        raise ValueError("tau must contain finite positive values")
    rates = dt / time_constants
    return np.eye(len(weights)) + rates[:, None] * (
        weights - np.eye(len(weights))
    )


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the largest absolute eigenvalue of a finite square matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.isfinite(values).all():
        raise ValueError("matrix must be finite")
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(values))))


def zero_input_norms(
    transition: np.ndarray,
    initial_state: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    """Apply a fixed transition and return the state norm after each step."""

    matrix = np.asarray(transition, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    if matrix.shape != (len(state), len(state)):
        raise ValueError("transition shape must match initial_state")
    if steps < 1:
        raise ValueError("steps must be positive")
    norms = np.empty(steps, dtype=np.float64)
    for index in range(steps):
        state = matrix @ state
        norms[index] = np.linalg.norm(state)
    return norms
