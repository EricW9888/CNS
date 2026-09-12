import numpy as np
import pytest

from src.numerics import (
    diffusive_coupling_current,
    effective_euler_transition,
    normalize_rows_by_absolute_sum,
    spectral_radius,
    zero_input_norms,
)


def test_absolute_row_normalization_preserves_signs_and_zero_rows():
    normalized = normalize_rows_by_absolute_sum(
        np.asarray([[2.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    )
    assert np.allclose(normalized[0], [0.5, -0.25, 0.25])
    assert np.sum(np.abs(normalized[0])) == pytest.approx(1.0)
    assert np.allclose(normalized[1], 0.0)


def test_diffusive_coupling_is_zero_at_equality_and_reverses_on_swap():
    conductance = np.asarray([[0.0, 2.0], [2.0, 0.0]])
    assert np.allclose(diffusive_coupling_current([3.0, 3.0], conductance), 0.0)
    first = diffusive_coupling_current([1.0, 4.0], conductance)
    swapped = diffusive_coupling_current([4.0, 1.0], conductance)
    assert np.allclose(first, [6.0, -6.0])
    assert np.allclose(swapped, -first)
    assert np.dot(np.asarray([1.0, 4.0]), first) < 0.0


def test_diffusive_coupling_rejects_directed_or_negative_conductance():
    with pytest.raises(ValueError, match="symmetric"):
        diffusive_coupling_current([0.0, 1.0], [[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="negative"):
        diffusive_coupling_current([0.0, 1.0], [[0.0, -1.0], [-1.0, 0.0]])


def test_effective_transition_not_raw_weight_is_stability_gate():
    stable = effective_euler_transition(np.zeros((2, 2)), dt=0.1, tau=20.0)
    unstable = effective_euler_transition(1.25 * np.eye(2), dt=0.1, tau=20.0)
    assert spectral_radius(stable) == pytest.approx(0.995)
    assert spectral_radius(unstable) == pytest.approx(1.00125)
    assert zero_input_norms(stable, [1.0, -1.0], steps=100)[-1] < 1.0
    assert zero_input_norms(unstable, [1.0, -1.0], steps=100)[-1] > 1.0
