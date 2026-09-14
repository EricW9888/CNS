"""Visibility-split evaluation of the frozen spherical Gaussian optical integral.

Numerical geometry only: no observer camera, motion label, neural state or
preferred direction is available here. This solver is specific to the enclosing
painted sphere and fully enclosed opaque spherical blocker used by EXP-008.
It is not a generic renderer or a replacement acceptance function.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .compound_eye import EyeGeometry, Scene, pose_rays


def acceptance_density(theta, sigma):
    return np.sin(theta)*np.exp(-theta*theta/(2*sigma*sigma))


def blocker_arc(theta, beta, alpha):
    """Half-width of blocked azimuthal arc; 0 = clear, pi = fully blocked."""
    denominator = np.sin(theta)*np.sin(beta)
    numerator = np.cos(alpha)-np.cos(theta)*np.cos(beta)
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=abs(denominator) > 1e-15)
    arc = np.arccos(np.clip(ratio, -1., 1.))
    return np.where(abs(denominator) > 1e-15, arc, np.where(numerator <= 0, np.pi, 0.))


@dataclass(frozen=True)
class AcceptanceIntegral:
    geometry: EyeGeometry
    sigma: float
    radial: int
    azimuthal: int
    normalization: float
    nodes: np.ndarray
    weights: np.ndarray
    phi_nodes: np.ndarray
    phi_weights: np.ndarray
    reference_normalization: float

    @classmethod
    def compile(cls, geometry, *, fwhm_degrees=5., radial=16, azimuthal=32):
        if (not np.isfinite(fwhm_degrees) or not 0 < fwhm_degrees < 30
                or not isinstance(radial, int) or not isinstance(azimuthal, int)
                or radial < 2 or azimuthal < 4):
            raise ValueError("invalid acceptance integration order")
        sigma = np.deg2rad(fwhm_degrees)/np.sqrt(8*np.log(2))
        x, w = leggauss(radial)
        limit = 3*sigma
        theta = (x+1)*limit/2
        z = float(limit/2*np.dot(w, acceptance_density(theta, sigma)))
        xn, wn = leggauss(256)
        reference = float(limit/2*np.dot(wn, acceptance_density((xn+1)*limit/2, sigma)))
        px, pw = leggauss(azimuthal)
        return cls(geometry, sigma, radial, azimuthal, z, x, w, px, pw, reference)

    def sample(self, scene: Scene, rotation=None, translation=None, *, diagnostics=False, smooth_light=None):
        r = np.eye(3) if rotation is None else rotation
        t = np.zeros(3) if translation is None else translation
        origins, axes = pose_rays(self.geometry.lens_mm, self.geometry.directions, r, t)
        if (np.any(np.linalg.norm(origins, axis=1) >= scene.radius_mm)
                or np.linalg.norm(scene.occluder_center)+scene.occluder_radius_mm >= scene.radius_mm):
            raise ValueError("solver requires enclosing wall and fully enclosed blocker")
        delta = np.asarray(scene.occluder_center)-origins
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= scene.occluder_radius_mm):
            raise ValueError("solver requires lens origins outside blocker")
        v = delta/distance[:, None]
        beta = np.arccos(np.clip(np.sum(v*axes, axis=1), -1, 1))
        alpha = np.arcsin(scene.occluder_radius_mm/distance)
        reference = np.where((abs(self.geometry.directions[:, 2]) < .9)[:, None], [0., 0., 1.], [1., 0., 0.])
        first = np.cross(reference, self.geometry.directions)
        first /= np.linalg.norm(first, axis=1)[:, None]
        first = first @ np.asarray(r).T
        second = np.cross(axes, first)
        gamma = np.arctan2(np.sum(v*second, axis=1), np.sum(v*first, axis=1))
        limit = 3*self.sigma
        mixed = (beta+alpha > 0) & (abs(beta-alpha) < limit) & (scene.occluder_radius_mm > 0)
        fully_blocked = (alpha >= beta+limit) & (scene.occluder_radius_mm > 0)
        light = np.full(len(axes), scene.occluder_light)
        blocked_mass = fully_blocked.astype(float)
        clear = ~mixed & ~fully_blocked
        # Clear cones have smooth radiance: the frozen Gauss/theta and periodic
        # phi quadrature (without a visibility discontinuity) remains sufficient.
        if smooth_light is not None:
            smooth_light = np.asarray(smooth_light)
            if smooth_light.shape != light.shape or not np.isfinite(smooth_light).all():
                raise ValueError("invalid smooth-wall integral")
            light[clear] = smooth_light[clear]
        elif clear.any():
            theta = (self.nodes+1)*limit/2
            weights = self.weights*limit/2*acceptance_density(theta, self.sigma)/self.normalization
            phi = np.arange(self.azimuthal)*2*np.pi/self.azimuthal
            rays = (axes[clear, None, None]*np.cos(theta)[None, :, None, None]
                    + first[clear, None, None]*(np.sin(theta)[:, None]*np.cos(phi))[None, :, :, None]
                    + second[clear, None, None]*(np.sin(theta)[:, None]*np.sin(phi))[None, :, :, None])
            light[clear] = np.sum(scene.radiance(origins[clear, None, None], rays).mean(axis=-1)*weights, axis=-1)
        ids = np.flatnonzero(mixed)
        if len(ids):
            cuts = np.column_stack([np.zeros(len(ids)), np.clip(abs(beta[ids]-alpha[ids]), 0, limit),
                                    np.clip(beta[ids]+alpha[ids], 0, limit), np.full(len(ids), limit)])
            # The sin^2 map handles square-root visibility at both tangencies.
            s = (self.nodes+1)*np.pi/4
            length = np.diff(cuts, axis=1)
            theta = cuts[:, :-1, None]+length[:, :, None]*np.sin(s)**2
            weight = length[:, :, None]*(np.pi/2*np.sin(s)*np.cos(s)*self.weights)
            weight *= acceptance_density(theta, self.sigma)/self.normalization
            half = blocker_arc(theta, beta[ids, None, None], alpha[ids, None, None])
            visible = 1-half/np.pi
            phi = gamma[ids, None, None, None]+half[..., None]+(self.phi_nodes+1)*(np.pi-half)[..., None]
            rays = (axes[ids, None, None, None]*np.cos(theta)[..., None, None]
                    + first[ids, None, None, None]*(np.sin(theta)[..., None]*np.cos(phi))[..., None]
                    + second[ids, None, None, None]*(np.sin(theta)[..., None]*np.sin(phi))[..., None])
            # Analytic arc classification already resolves nearest visibility.
            # The wall expression is identical to Scene.radiance before its
            # blocker mask; full-block rings have zero visible weight.
            values = wall_radiance(scene, origins[ids, None, None, None], rays)
            means = np.sum(values*self.phi_weights/2, axis=-1)
            light[ids] = np.sum(weight*(visible*means+(1-visible)*scene.occluder_light), axis=(1, 2))
            blocked_mass[ids] = np.sum(weight*(1-visible), axis=(1, 2))
        if not np.isfinite(light).all() or np.any((light < -1e-12) | (light > 1+1e-12)):
            raise FloatingPointError("invalid integrated light")
        return (light, {"mixed": mixed, "blocked_mass": blocked_mass,
                        "numerator": light*self.normalization,
                        "full_cone_mass": self.normalization,
                        "reference_full_cone_mass": self.reference_normalization,
                        "full_cone_mass_error": self.normalization-self.reference_normalization}) if diagnostics else light


def wall_radiance(scene, origins, rays):
    """Enclosing-wall expression independently compared with Scene.radiance."""
    a = np.sum(rays*rays, axis=-1)
    b = np.sum(origins*rays, axis=-1)
    distance = (-b+np.sqrt(b*b+a*(scene.radius_mm**2-np.sum(origins*origins, axis=-1))))/a
    xy = origins[..., :2]+distance[..., None]*rays[..., :2]
    return scene.mean+scene.amplitude*np.linalg.norm(xy, axis=-1)/scene.radius_mm*np.cos(
        scene.stripe_cycles*np.arctan2(xy[..., 1], xy[..., 0]))


class SmoothWallTable:
    """Fixed-tolerance static-world integral cache, never a visibility cache."""

    def __init__(self, sampler, scene, lo, hi, *, tolerance=1e-10, max_depth=12):
        from dataclasses import replace
        from numpy.polynomial.chebyshev import chebfit, chebval
        self.sampler, self.scene = sampler, replace(scene, occluder_radius_mm=0.)
        self.pieces, self.max_checked_error = [], 0.
        nodes = np.cos(np.arange(17)*np.pi/16)
        checks = np.cos((np.arange(16)+.5)*np.pi/16)
        cache = {}

        def value(x):
            if x not in cache:
                cache[x] = sampler.sample(self.scene, translation=[x, 0., 0.])
            return cache[x]

        def fit(a, b, depth):
            values = np.array([value((a+b)/2+(b-a)/2*x) for x in nodes])
            coefficients = chebfit(nodes, values, 16)
            direct = np.array([value((a+b)/2+(b-a)/2*x) for x in checks])
            error = float(abs(chebval(checks, coefficients).T-direct).max())
            if error <= tolerance:
                self.pieces.append((a, b, coefficients))
                self.max_checked_error = max(self.max_checked_error, error)
            elif depth == max_depth:
                raise RuntimeError("smooth-wall cache cannot meet predeclared tolerance")
            else:
                fit(a, (a+b)/2, depth+1)
                fit((a+b)/2, b, depth+1)
        fit(lo, hi, 0)
        self.direct_evaluations = len(cache)

    def validate(self, positions):
        """Return the maximum full-eye error at explicit unoccluded poses.

        Construction checks Chebyshev off-nodes, but callers also need an
        independent physical-pose check over the interval actually used by a
        run.  This method intentionally evaluates the underlying sampler at
        every requested position; it never validates a visibility cache.
        """
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 1 or not len(positions) or not np.isfinite(positions).all():
            raise ValueError("finite one-dimensional validation positions required")
        cached = np.asarray([self.evaluate(float(x)) for x in positions])
        direct = np.asarray([
            self.sampler.sample(self.scene, translation=[float(x), 0., 0.])
            for x in positions
        ])
        return float(abs(cached-direct).max())

    def evaluate(self, x):
        from numpy.polynomial.chebyshev import chebval
        for lo, hi, coefficients in self.pieces:
            if lo-1e-14 <= x <= hi+1e-14:
                return chebval((2*x-lo-hi)/(hi-lo), coefficients)
        raise ValueError("pose outside smooth-wall cache")
