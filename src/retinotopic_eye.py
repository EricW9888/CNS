"""Measured-neighbor motion representation, separate from anatomical body identity.

Inputs are scalar facet light only. No world pose, camera, flow or condition label
enters this kernel. Local ON/OFF emissions remain spatially explicit until the
lossy, explicitly provisional adapter to the frozen shared-type projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compound_eye import EyeGeometry
from .sensory_chain import SensoryParameters


def tangent_unit(vector, axes):
    v, u = np.asarray(vector, float), np.asarray(axes, float)
    v = v - np.sum(v*u, axis=-1, keepdims=True)*u
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.isfinite(v).all() or np.any(norm < 1e-12):
        raise ValueError("degenerate local visual basis")
    return v/norm


@dataclass(frozen=True)
class RetinotopicReadout:
    centers: np.ndarray
    neighbors: np.ndarray
    left: np.ndarray
    preferred_directions: np.ndarray  # center x a,b,c,d x XYZ; unit tangents
    facet_count: int

    @classmethod
    def compile(cls, geometry: EyeGeometry):
        ids = np.flatnonzero((geometry.neighbors >= 0).all(axis=1))
        if not len(ids) or not all(np.any(geometry.left[ids] == side) for side in (True, False)):
            raise ValueError("both measured eyes need complete neighborhoods")
        n, u = geometry.neighbors[ids], geometry.directions[ids]
        # Zhao Fig.4C / ED Fig.7: local +h predicts b; local -v predicts d.
        # Keep the measured shearing and nonorthogonality, not azimuth/elevation
        # cardinal axes. Slot addresses are checked against the primary grid.
        b = tangent_unit(geometry.directions[n[:, [2, 3]]].mean(axis=1)
                         - geometry.directions[n[:, [0, 5]]].mean(axis=1), u)
        d = tangent_unit(geometry.directions[n[:, 4]]-geometry.directions[n[:, 1]], u)
        pd = np.stack([-b, b, -d, d], axis=1)
        arrays = [ids, n.copy(), geometry.left[ids].copy(), pd]
        for a in arrays:
            a.setflags(write=False)
        return cls(*arrays, len(geometry.directions))

    def pool(self, local):
        """Lossy adapter ONLY: mean local emissions within eye/polarity/subtype.

        This does not assign facets to MaleCNS bodies or reconstruct wide-field
        receptive fields. Positive/negative split precedes pooling; opposing
        local signals are not canceled before they are recorded or emitted.
        """
        values = np.asarray(local, float)
        if values.shape[-3:] != (len(self.centers), 2, 4):
            raise ValueError("expected local center x ON/OFF x a/b/c/d axes")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("invalid local emissions")
        return np.concatenate([values[..., self.left == side, :, :].mean(axis=-3).reshape(*values.shape[:-3], 8)
                               for side in (True, False)], axis=-1)

    def responses(self, light, p: SensoryParameters, *, save_stride=1):
        """Causal adaptation, six adjacent products, local rectification/emission.

        All numerical state is float64. Local storage can be decimated at an
        explicitly recorded stride; channel means are retained at every step.
        Outputs precede updates, matching the frozen pathway time convention.
        """
        intensity = np.asarray(light, float)
        if (intensity.ndim != 2 or intensity.shape[1] != self.facet_count or not len(intensity)
                or not np.isfinite(intensity).all() or np.any((intensity < 0) | (intensity > 1))
                or not isinstance(save_stride, int) or save_stride < 1):
            raise ValueError("expected bounded time x measured-facet light and positive storage stride")
        adaptation = intensity[0].copy()
        delayed = np.zeros((self.facet_count, 2))
        state = np.zeros((len(self.centers), 2, 4))
        channels = np.empty((len(intensity), 16))
        local = np.empty(((len(intensity)+save_stride-1)//save_stride, *state.shape))
        ha, hd, he = p.dt_ms/np.asarray([p.highpass_tau_ms, p.delay_tau_ms, p.emission_tau_ms])
        for k, sample in enumerate(intensity):
            # Avoid per-step validation/copies; shape, positivity and convex
            # updates establish the same contract checked by public pool().
            channels[k] = np.concatenate([state[self.left == side].mean(axis=0).reshape(8)
                                           for side in (True, False)])
            if k % save_stride == 0:
                local[k//save_stride] = state
            hp = sample-adaptation
            q = np.stack([np.maximum(hp, 0), np.maximum(-hp, 0)], axis=-1)
            c = (delayed[self.neighbors]*q[self.centers, None, :]
                 - q[self.neighbors]*delayed[self.centers, None, :])
            # Each term is an adjacent neighbor->center correlator. The
            # symmetric stencil follows measured grid axes; no periodic wrap.
            rb = (c[:, 0]+c[:, 5]-c[:, 2]-c[:, 3])/4
            rd = (c[:, 1]-c[:, 4])/2
            signed = np.stack([-rb, rb, -rd, rd], axis=-1)
            state += he*(np.maximum(signed, 0)-state)
            adaptation += ha*(sample-adaptation)
            delayed += hd*(q-delayed)
        if not np.isfinite(local).all() or not np.isfinite(channels).all():
            raise FloatingPointError("nonfinite retinotopic response")
        return channels, local


def source_field_comparison(readout, geometry, reference):
    """Compare measured-grid hypothesis to released right-eye T4 predictions.

    Literal source eye-map indices, not nearest-cell matching. Reference vectors
    are anatomy-derived kernel-regression estimates, not T4 calcium recordings.
    """
    ids = np.asarray(reference["facet_ids"])
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids >= len(geometry.left)))
            or len(np.unique(ids)) != len(ids) or np.any(geometry.left[ids])):
        raise ValueError("invalid source right-eye registration")
    if not np.array_equal(geometry.directions[ids], reference["viewing_axes"]):
        raise ValueError("source registration axes differ from measured specimen")
    common, ri, si = np.intersect1d(readout.centers, ids, return_indices=True)
    if not len(common):
        raise ValueError("no source-registered complete neighborhoods")
    axes = geometry.directions[common, None, :]
    ref = tangent_unit(np.asarray(reference["pd_endpoints"])[si]-axes, axes)
    cosines = np.sum(readout.preferred_directions[ri]*ref, axis=-1)
    angles = np.rad2deg(np.arccos(np.clip(cosines, -1, 1)))
    return {"registered_right_facets": int(len(ids)), "compared_complete_right_facets": int(len(common)),
            "angles_degrees_p50_p95_max": {s: np.percentile(angles[:, i], [50, 95, 100]).tolist()
                                            for i, s in enumerate("abcd")},
            "all_local_orientations_same_hemiplane": bool(np.all(cosines > 0)),
            "scope": "right-eye anatomy-predicted T4 field; a/c antiparallel extensions; not individual MaleCNS or left-eye/T5 physiology"}
