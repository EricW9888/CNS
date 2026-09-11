"""Bilateral readout primitives for the next EXP-003 hypothesis test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exp003 import SteeringBridge
from .exp003_corrected import CorrectedDownstreamMatrices


PRIMARY_DN_TYPE = "DNp15"
SECONDARY_DN_TYPE = "DNa02"


@dataclass(frozen=True)
class BilateralDownstreamReadout:
    """A homologous left/right DNp15 difference with transparent diagnostics."""

    left: CorrectedDownstreamMatrices
    right: CorrectedDownstreamMatrices
    left_primary_indices: tuple[int, ...]
    right_primary_indices: tuple[int, ...]
    left_secondary_indices: tuple[int, ...]
    right_secondary_indices: tuple[int, ...]

    @classmethod
    def from_matrices(
        cls,
        left: CorrectedDownstreamMatrices,
        right: CorrectedDownstreamMatrices,
    ) -> "BilateralDownstreamReadout":
        left_primary = tuple(i for i, value in enumerate(left.descending_types) if value == PRIMARY_DN_TYPE)
        right_primary = tuple(i for i, value in enumerate(right.descending_types) if value == PRIMARY_DN_TYPE)
        left_secondary = tuple(i for i, value in enumerate(left.descending_types) if value == SECONDARY_DN_TYPE)
        right_secondary = tuple(i for i, value in enumerate(right.descending_types) if value == SECONDARY_DN_TYPE)
        if not left_primary or not right_primary:
            raise ValueError("Both sides must contain the homologous DNp15/DNHS1 type")
        return cls(left, right, left_primary, right_primary, left_secondary, right_secondary)

    @staticmethod
    def _pool(values: np.ndarray, indices: tuple[int, ...]) -> float:
        return float(np.mean(values[list(indices)])) if indices else 0.0

    def activities(
        self,
        left_t4: np.ndarray,
        right_t4: np.ndarray,
        *,
        ablate: bool = False,
    ) -> dict[str, float]:
        _, left_descending = self.left.propagate(left_t4, ablate_t4_projection=ablate)
        _, right_descending = self.right.propagate(right_t4, ablate_t4_projection=ablate)
        left_primary = self._pool(left_descending, self.left_primary_indices)
        right_primary = self._pool(right_descending, self.right_primary_indices)
        left_secondary = self._pool(left_descending, self.left_secondary_indices)
        right_secondary = self._pool(right_descending, self.right_secondary_indices)
        return {
            "left_DNp15": left_primary,
            "right_DNp15": right_primary,
            "right_minus_left_DNp15": right_primary - left_primary,
            "left_DNa02": left_secondary,
            "right_DNa02": right_secondary,
            "right_minus_left_DNa02": right_secondary - left_secondary,
        }

    def steering_command(
        self,
        activities: dict[str, float],
        bridge: SteeringBridge,
    ) -> tuple[float, np.ndarray]:
        """Map only the bilateral DNp15 difference into the frozen bridge."""

        raw = float(activities["right_minus_left_DNp15"])
        return raw, bridge.action(raw)
