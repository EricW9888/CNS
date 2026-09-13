"""Direct, anatomy-weighted DN drive and a provenance-gated neck torque proxy.

This is not muscle physiology or a controller. Its uncalibrated transduction
is isolated from both frozen sensory dynamics and the optional body solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .malecns_io import ConnectomeGraph
from .sparse_backend import compile_projection


@dataclass(frozen=True)
class MotorParameters:
    dt_ms: float = .5
    tau_ms: float = 20.
    chemical_gain: float = .5
    torque_native_per_activity: float = 1.
    paper_body_pitch_degrees: float = 40.

    def __post_init__(self):
        values = np.array([self.dt_ms, self.tau_ms, self.chemical_gain,
                           self.torque_native_per_activity, self.paper_body_pitch_degrees])
        if not np.isfinite(values).all() or min(values[:2]) <= 0 or min(values[2:4]) < 0:
            raise ValueError("invalid motor parameters")
        if not 0 <= self.paper_body_pitch_degrees < 90:
            raise ValueError("unsupported experimental frame")


@dataclass(frozen=True)
class NeckMotorPathway:
    source_ids: np.ndarray
    motor_ids: np.ndarray
    weights: sparse.csr_matrix
    parameters: MotorParameters

    @classmethod
    def compile(cls, graph: ConnectomeGraph, incoming_totals, parameters=MotorParameters(),
                control="full"):
        controls = {"full", "DN_disconnected", "DN_L_disconnected", "DN_R_disconnected", "motor_disconnected"}
        if control not in controls:
            raise ValueError("unknown motor control")
        nodes = graph.nodes.set_index("bodyId")
        if nodes.index.duplicated().any():
            raise ValueError("duplicate motor nodes")
        sources = np.sort(nodes[nodes.stage.eq("DNp15")].index.to_numpy(dtype=np.int64))
        targets = np.sort(nodes[nodes.stage.eq("neck_motor")].index.to_numpy(dtype=np.int64))
        if set(sources) != {11215, 12069} or len(targets) == 0:
            raise ValueError("both identified DNp15 required")
        if nodes.loc[11215, "somaSide"] != "R" or nodes.loc[12069, "somaSide"] != "L":
            raise ValueError("DN hemisphere correspondence changed")
        if not nodes.loc[sources, "consensus_nt"].eq("acetylcholine").all():
            raise ValueError("DN excitation requires verified presynaptic ACh")
        if not graph.edges.pre_body.isin(np.concatenate([sources, targets])).all() or not graph.edges.post_body.isin(targets).all():
            raise ValueError("unmodeled induced recurrence or intermediate")
        recurrent = graph.edges[graph.edges.pre_body.isin(targets)]
        if len(recurrent) and not nodes.loc[recurrent.pre_body, "consensus_nt"].eq("unclear").all():
            raise ValueError("motor-source edges require explicit physiological specification")
        totals = {int(r["bodyId"]): float(r["incoming_count"]) for r in incoming_totals}
        if len(totals) != len(incoming_totals) or set(totals) != set(targets):
            raise ValueError("motor input denominators missing or duplicated")
        denominator = np.array([totals[int(i)] for i in targets])
        counts = compile_projection(graph, source_ids=sources, target_ids=targets, normalization=None).matrix
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError("invalid unrestricted input count")
        if np.any(counts.data <= 0) or np.any(np.asarray(counts.sum(axis=1)).ravel() > denominator):
            raise ValueError("selected count exceeds unrestricted total")
        weights = sparse.diags(parameters.chemical_gain / denominator, format="csr") @ counts
        # Ablate AFTER all-source normalization. Do not compensate remaining input.
        mask = np.ones(len(sources))
        if control == "DN_disconnected":
            mask[:] = 0
        elif control in {"DN_L_disconnected", "DN_R_disconnected"}:
            side = "L" if control == "DN_L_disconnected" else "R"
            mask[nodes.loc[sources, "somaSide"].eq(side).to_numpy()] = 0
        weights = (weights @ sparse.diags(mask, format="csr")).tocsr()
        weights.eliminate_zeros()
        return cls(sources, targets, weights, parameters)

    def simulate(self, descending_activity, *, initial_state=None):
        """Pre-update motor state; exact stable held-input exponential filter."""
        drive = np.asarray(descending_activity, dtype=np.float64)
        if drive.ndim != 2 or drive.shape[1] != len(self.source_ids) or len(drive) == 0 or not np.isfinite(drive).all():
            raise ValueError("invalid descending trace")
        state = np.zeros(len(self.motor_ids)) if initial_state is None else np.asarray(initial_state, dtype=float).copy()
        if state.shape != (len(self.motor_ids),) or not np.isfinite(state).all():
            raise ValueError("invalid initial motor state")
        target = np.asarray(self.weights @ np.maximum(drive, 0).T).T
        alpha = -np.expm1(-self.parameters.dt_ms / self.parameters.tau_ms)
        output = np.empty_like(target)
        for k, value in enumerate(target):
            output[k] = state
            state += alpha * (value-state)
        if not np.isfinite(output).all():
            raise FloatingPointError("nonfinite motor activity")
        return output

    def preflight(self):
        decay = float(np.exp(-self.parameters.dt_ms/self.parameters.tau_ms))
        drive = np.zeros((int(np.ceil(2000/self.parameters.dt_ms))+1, len(self.source_ids)))
        perturbed = self.simulate(drive, initial_state=np.linspace(-1,1,len(self.motor_ids)))
        ratio = float(np.max(abs(perturbed[-1])))
        return {"passed": bool(decay < 1 and ratio < 1e-8),
                "motor_transition_spectral_radius_and_contraction_bound": decay,
                "zero_input_2000ms_final_initial_inf_ratio": ratio,
                "chemical_edges": self.weights.nnz,
                "incoming_weight_mass": np.asarray(self.weights.sum(axis=1)).ravel().tolist()}


def head_torque(motor_states, motor_ids, identity, parameters=MotorParameters(), *, disconnected=False):
    """CvNA2 yaw-component proxy. Never consumes condition labels or DN laterality."""
    values = np.asarray(motor_states, dtype=float)
    ids = np.asarray(motor_ids, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(ids) or len(np.unique(ids)) != len(ids) or not np.isfinite(values).all():
        raise ValueError("invalid motor-state axes")
    selected = [r for r in identity if r["paper_cell"] == "CvNA2"]
    if (len(selected) != 2 or {r["axon_side"] for r in selected} != {"L", "R"}
        or len({r["bodyId"] for r in selected}) != 2):
        raise ValueError("two independently crosswalked CvNA2 axon sides required")
    locations = {}
    for row in selected:
        if row["muscle"] != "TH2" or not row.get("identity_source") or not row.get("laterality_source"):
            raise ValueError("unsupported muscle/laterality provenance")
        found = np.flatnonzero(ids == int(row["bodyId"]))
        if len(found) != 1:
            raise ValueError("CvNA2 absent from motor trace")
        locations[row["axon_side"]] = int(found[0])
    delta = np.maximum(values[:,locations["R"]], 0)-np.maximum(values[:,locations["L"]], 0)
    if disconnected:
        return np.zeros(len(values))
    return parameters.torque_native_per_activity * np.cos(np.deg2rad(parameters.paper_body_pitch_degrees)) * delta
