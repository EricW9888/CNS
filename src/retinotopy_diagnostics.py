"""Read-only numerical diagnostics; no replacement for the frozen motion model."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def error_field(a, b):
    """Common-time max norm and spatial energy concentration, without rescaling."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim < 2 or not all(np.isfinite(x).all() for x in (a, b)):
        raise ValueError("finite coincident time x entity arrays required")
    delta = abs(a - b)
    axes = (0, *range(2, a.ndim))
    peaks = delta.max(axis=axes)
    energy = np.square(delta).sum(axis=axes)
    denominator = max(float(abs(a).max()), float(abs(b).max()), 1e-12)
    order = np.argsort(-energy, kind="stable")
    total = float(energy.sum())
    result = {"relative_error": float(delta.max()) / denominator,
              "max_abs_difference": float(delta.max()), "global_normalizer": denominator,
              "entities_above_1pct_global": int(np.count_nonzero(peaks > .01 * denominator)),
              "entities_above_5pct_global": int(np.count_nonzero(peaks > .05 * denominator)),
              "top_10_energy_fraction": float(energy[order[:10]].sum() / total) if total else 0.,
              "top_1pct_energy_fraction": float(energy[order[:max(1, int(np.ceil(len(energy) * .01)))]].sum() / total) if total else 0.}
    return result, peaks, energy


def filter_states(light, p):
    """Independent transfer-function reconstruction of BEFORE-update states.

    A unit delay in each numerator implements the frozen output-before-update
    convention. This is diagnostic-only, tested against scalar Euler recurrences.
    """
    intensity = np.asarray(light, float)
    if intensity.ndim != 2 or not len(intensity) or not np.isfinite(intensity).all():
        raise ValueError("finite time x facet light required")
    ha, hd = p.dt_ms / np.array([p.highpass_tau_ms, p.delay_tau_ms])
    if not 0 < min(ha, hd) <= max(ha, hd) <= 1:
        raise ValueError("non-convex filter coefficients")
    adaptation = lfilter([0, ha], [1, -(1-ha)], intensity-intensity[0], axis=0) + intensity[0]
    hp = intensity-adaptation
    q = np.stack([np.maximum(hp, 0), np.maximum(-hp, 0)], axis=-1)
    delayed = lfilter([0, hd], [1, -(1-hd)], q, axis=0)
    return adaptation, q, delayed


def correlators(light, readout, p, *, stride=4):
    """Independent raw neighbor products and signed b/d stencils at common times."""
    adaptation, q, delayed = filter_states(light, p)
    q, delayed = q[::stride], delayed[::stride]
    c = delayed[:, readout.neighbors] * q[:, readout.centers, None] - q[:, readout.neighbors] * delayed[:, readout.centers, None]
    stencil = np.stack([(c[:, :, 0]+c[:, :, 5]-c[:, :, 2]-c[:, :, 3])/4,
                        (c[:, :, 1]-c[:, :, 4])/2], axis=-1)
    return {"adaptation": adaptation[::stride], "rectified_highpass": q,
            "delayed": delayed, "neighbor_correlators": c, "signed_bd": stencil}


def weighted_occlusion(sampler, scene, translations):
    """Observer diagnostic only: visible acceptance weight hitting the blocker."""
    from .compound_eye import pose_rays, sphere_distance
    origins = np.broadcast_to(sampler.geometry.lens_mm[:, None], sampler.local_rays.shape)
    result = []
    for translation in translations:
        o, d = pose_rays(origins, sampler.local_rays, np.eye(3), translation)
        wall = sphere_distance(o, d, [0, 0, 0], scene.radius_mm)
        blocker = sphere_distance(o, d, scene.occluder_center, scene.occluder_radius_mm)
        result.append((blocker < wall) @ sampler.weights)
    return np.asarray(result)


def possible_occlusion(geometry, scene, speed, *, fwhm_degrees=5.):
    """Analytic support-cone overlap over the motion interval, not visibility weights.

    This conservatively localizes facets that can see a spherical occluder;
    it is never delivered to a sensory or neural model.
    """
    sigma = np.deg2rad(fwhm_degrees) / np.sqrt(8*np.log(2))
    possible = np.zeros(len(geometry.left), bool)
    for seconds in np.linspace(0., 2., 2001):
        delta = np.asarray(scene.occluder_center) - geometry.lens_mm - [speed*seconds, 0., 0.]
        distance = np.linalg.norm(delta, axis=-1)
        cosine = np.sum(delta * geometry.directions, axis=-1) / distance
        angle = np.arccos(np.clip(cosine, -1, 1))
        radius = np.arcsin(np.minimum(scene.occluder_radius_mm/distance, 1.))
        possible |= angle <= radius + 3*sigma
    return possible
