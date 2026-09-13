"""Explicit luminance -> provisional T4/T5 channels -> MaleCNS visual projection.

No condition label or requested downstream answer enters either numerical kernel.
Direction/type assignments belong to the independently sourced sensory interface.
The frozen central network is simulated separately with the resulting HS/H2 drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .malecns_io import ConnectomeGraph
from .sparse_backend import compile_projection

CHANNELS = tuple(f"{side}_T{number}{subtype}" for side in "LR" for number in [4, 5] for subtype in "abcd")


@dataclass(frozen=True)
class SensoryParameters:
    dt_ms: float = .5
    pixel_degrees: float = 2.25
    side_pixels: int = 16
    wavelength_degrees: float = 18.
    speed_degrees_s: float = 45.
    mean_luminance: float = .5
    contrast: float = .5
    highpass_tau_ms: float = 50.
    delay_tau_ms: float = 15.
    emission_tau_ms: float = 10.
    LPi_tau_ms: float = 20.
    chemical_gain: float = .5
    onset_ms: float = 2000.
    offset_ms: float = 4000.
    duration_ms: float = 6000.

    def __post_init__(self):
        if not np.isfinite(list(self.__dict__.values())).all():
            raise ValueError("nonfinite sensory parameters")
        times = [self.highpass_tau_ms, self.delay_tau_ms, self.emission_tau_ms, self.LPi_tau_ms]
        if self.dt_ms <= 0 or self.dt_ms > min(times):
            raise ValueError("filter update must be positive and non-amplifying")
        if self.side_pixels < 2 or int(self.side_pixels) != self.side_pixels:
            raise ValueError("invalid image shape")
        if min(self.pixel_degrees, self.wavelength_degrees, self.duration_ms) <= 0 or self.speed_degrees_s < 0:
            raise ValueError("invalid stimulus units")
        if not 0 <= self.contrast <= min(self.mean_luminance, 1-self.mean_luminance):
            raise ValueError("luminance must stay in [0,1]")
        if not 0 <= self.onset_ms < self.offset_ms <= self.duration_ms:
            raise ValueError("invalid motion interval")
        if not np.isclose(self.duration_ms/self.dt_ms, round(self.duration_ms/self.dt_ms)):
            raise ValueError("duration must be an integer number of samples")
        if not 0 <= self.chemical_gain < 1:
            raise ValueError("visual recurrent gain must be below leak")


def luminance_movie(p: SensoryParameters, eye_orders, *, axis: str = "x"):
    """Return time and actual two-eye movie, with fixed phase after motion ends.

    x/y are published stimulus-space directions, never workbook coordinates.
    Input orders affect only image translation, not the motion-detection kernel.
    """
    orders = np.asarray(eye_orders, dtype=float)
    if orders.shape != (2,) or not np.isfinite(orders).all() or axis not in {"x", "y"}:
        raise ValueError("expected finite two-eye motion and x/y axis")
    time = np.arange(round(p.duration_ms/p.dt_ms))*p.dt_ms
    elapsed = np.clip(time-p.onset_ms, 0, p.offset_ms-p.onset_ms)/1000
    positions = np.arange(p.side_pixels)*p.pixel_degrees
    displacement = elapsed[:, None]*orders[None, :]*p.speed_degrees_s
    phase = 2*np.pi*(positions[None, None, :]-displacement[:, :, None])/p.wavelength_degrees
    line = p.mean_luminance + p.contrast*np.cos(phase)
    movie = np.broadcast_to(line[:, :, None, :] if axis == "x" else line[:, :, :, None],
                            (len(time), 2, p.side_pixels, p.side_pixels))
    return time, movie


def motion_channels(movie: np.ndarray, p: SensoryParameters) -> np.ndarray:
    """Causal two-quadrant opponent products; time x 16 channel emissions.

    Filter and emission samples precede their updates. Adaptation starts at the
    first static image, giving exactly zero response to an unchanged image.
    Cyclic adjacent sampling is explicit; production gratings span two periods.
    """
    images = np.asarray(movie, dtype=np.float64)
    if images.ndim != 4 or images.shape[1:] != (2, p.side_pixels, p.side_pixels) or not len(images):
        raise ValueError("expected time x two eyes x y x x movie")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("invalid luminance")
    adaptation = images[0].copy()
    delayed = np.zeros((2, 2, p.side_pixels, p.side_pixels))  # eye, ON/OFF, y, x
    state = np.zeros(16)
    out = np.empty((len(images), 16))
    h_adapt, h_delay, h_emit = p.dt_ms/np.asarray([p.highpass_tau_ms, p.delay_tau_ms, p.emission_tau_ms])
    for k, image in enumerate(images):
        out[k] = state
        hp = image-adaptation
        current = np.stack([np.maximum(hp, 0), np.maximum(-hp, 0)], axis=1)
        target = np.empty((2, 2, 4))
        for axis, first in [(-1, 0), (-2, 2)]:
            # Delayed neighbor at x-1/y-1 times current at x/y, minus swapped arm.
            opponent = (np.roll(delayed, 1, axis=axis)*current - np.roll(current, 1, axis=axis)*delayed).mean(axis=(-2, -1))
            target[:, :, first] = np.maximum(opponent, 0)
            target[:, :, first+1] = np.maximum(-opponent, 0)
        state += h_emit*(target.reshape(16)-state)
        adaptation += h_adapt*(image-adaptation)
        delayed += h_delay*(current-delayed)
    return out


@dataclass(frozen=True)
class VisualProjection:
    """Exact sparse reduction under the declared shared-channel assignment."""

    nodes: pd.DataFrame
    input_ids: np.ndarray
    LPi_ids: np.ndarray
    motion_ids: np.ndarray
    motion_channel_indices: np.ndarray
    motion_to_targets: sparse.csr_matrix
    LPi_to_targets: sparse.csr_matrix
    target_ids: np.ndarray
    parameters: SensoryParameters

    @classmethod
    def compile(cls, graph: ConnectomeGraph, p: SensoryParameters):
        nodes = graph.nodes.sort_values("bodyId").reset_index(drop=True)
        motion = nodes[nodes.stage.isin(["T4", "T5"])]
        LPi = nodes[nodes.stage.eq("LPi")]
        inputs = nodes[nodes.stage.isin(["HS", "H2"])]
        if len(LPi) != 6 or len(inputs) != 8 or set(motion.type) != {f"T{n}{s}" for n in [4, 5] for s in "abcd"}:
            raise ValueError("unexpected selected visual populations")
        expected = np.where(nodes.stage.eq("LPi"), "gaba", "acetylcholine")
        if not nodes.consensus_nt.eq(expected).all() or not nodes.somaSide.isin(["L", "R"]).all():
            raise ValueError("unresolved visual sign/side")
        indices = np.asarray([CHANNELS.index(f"{r.somaSide}_{r.type}") for r in motion.itertuples()], dtype=np.int64)
        sources = np.concatenate([motion.bodyId.to_numpy(), LPi.bodyId.to_numpy()])
        targets = np.concatenate([LPi.bodyId.to_numpy(), inputs.bodyId.to_numpy()])
        if not graph.edges.pre_body.isin(sources).all() or not graph.edges.post_body.isin(targets).all():
            raise ValueError("edge outside the declared visual interface")
        edges = graph.edges.copy()
        edges["visual_sign"] = np.where(edges.pre_body.isin(LPi.bodyId), -1., 1.)
        full = compile_projection(ConnectomeGraph(nodes, edges), source_ids=sources, target_ids=targets,
                                  normalization="absolute", sign_column="visual_sign").matrix*p.chemical_gain
        assignment = sparse.csr_matrix((np.ones(len(motion)), (np.arange(len(motion)), indices)), shape=(len(motion), 16))
        return cls(nodes, inputs.bodyId.to_numpy(), LPi.bodyId.to_numpy(), motion.bodyId.to_numpy(), indices,
                   (full[:, :len(motion)] @ assignment).tocsr(), full[:, len(motion):].tocsr(), targets, p)

    def simulate(self, channels: np.ndarray, *, no_LPi_output=False, initial_LPi=None):
        rates = np.asarray(channels, dtype=float)
        if rates.ndim != 2 or rates.shape[1] != 16 or not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("expected nonnegative time x 16 motion emissions")
        n = len(self.LPi_ids)
        state = np.zeros(n) if initial_LPi is None else np.asarray(initial_LPi, dtype=float).copy()
        if state.shape != (n,) or not np.isfinite(state).all():
            raise ValueError("invalid LPi initial state")
        projected = np.asarray(self.motion_to_targets @ rates.T).T
        traces = np.empty((len(rates), n))
        drive = np.empty((len(rates), len(self.input_ids)))
        recurrent = self.LPi_to_targets if not no_LPi_output else sparse.csr_matrix(self.LPi_to_targets.shape)
        step = self.parameters.dt_ms/self.parameters.LPi_tau_ms
        for k in range(len(rates)):
            traces[k] = state
            total = projected[k] + recurrent @ np.maximum(state, 0)
            drive[k] = total[n:]
            state += step*(-state+total[:n])
        if not np.isfinite(traces).all() or not np.isfinite(drive).all():
            raise FloatingPointError("nonfinite visual propagation")
        return traces, drive

    def preflight(self, *, no_LPi_output=False):
        n = len(self.LPi_ids)
        h = self.parameters.dt_ms/self.parameters.LPi_tau_ms
        feedback = self.LPi_to_targets[:n].toarray() if not no_LPi_output else np.zeros((n, n))
        bound = float(max(abs(1-h)+h*np.abs(feedback).sum(axis=1)))
        rho = max(float(max(abs(np.linalg.eigvals((1-h)*np.eye(n)+h*feedback*mask)))) for mask in [0., 1.])
        steps = round(2000/self.parameters.dt_ms)+1
        zero = np.zeros((steps, 16))
        ratios = [float(np.max(abs(self.simulate(zero, no_LPi_output=no_LPi_output, initial_LPi=x)[0][-1]))/max(abs(x)))
                  for x in [np.ones(n), -np.ones(n), np.linspace(-1, 1, n)]]
        return {"passed": bool(bound < 1 and rho < 1 and max(ratios) < 1e-8),
                "global_euler_lipschitz_inf_bound": bound, "effective_transition_rho_max": rho,
                "zero_input_2000ms_final_initial_inf_ratios": ratios}


def central_drive(input_ids, projected: np.ndarray, central_ids) -> np.ndarray:
    """Insert only actual projected HS/H2 drive, without touching central weights."""
    source_ids, target_ids = list(map(int, input_ids)), list(map(int, central_ids))
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("duplicate drive axis")
    values = np.asarray(projected, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(source_ids) or not np.isfinite(values).all():
        raise ValueError("invalid projected drive")
    columns = [target_ids.index(i) for i in source_ids]
    out = np.zeros((len(values), len(target_ids)))
    out[:, columns] = values
    return out
