"""Measured eye geometry, scalar ray sampling, and a replaceable local readout.

Only per-facet light enters the motion transform. World coordinates, visibility,
pose labels and observer-camera images are not arguments to that transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .sensory_chain import SensoryParameters


@dataclass(frozen=True)
class EyeGeometry:
    lens_mm: np.ndarray
    directions: np.ndarray
    left: np.ndarray
    neighbors: np.ndarray  # lens-order global indices; six slots, -1 = absent
    grid: np.ndarray       # author's per-eye p,q addresses in lens order

    def __post_init__(self):
        n = len(self.directions)
        if (self.lens_mm.shape != (n, 3) or self.directions.shape != (n, 3)
                or self.left.shape != (n,) or self.left.dtype != bool
                or self.neighbors.shape != (n, 6) or self.grid.shape != (n, 2)):
            raise ValueError("inconsistent eye axes")
        if not all(np.isfinite(x).all() for x in (self.lens_mm, self.directions, self.grid)):
            raise ValueError("nonfinite eye geometry")
        if not np.allclose(np.linalg.norm(self.directions, axis=1), 1, atol=1e-10):
            raise ValueError("viewing axes must be unit vectors")
        if self.neighbors.dtype.kind not in "iu" or np.any((self.neighbors < -1) | (self.neighbors >= n)):
            raise ValueError("invalid neighbor indices")
        if not self.left.any() or self.left.all():
            raise ValueError("both measured eyes required")
        rows, slots = np.nonzero(self.neighbors >= 0)
        if np.any(self.left[rows] != self.left[self.neighbors[rows, slots]]):
            raise ValueError("neighbor crosses eyes")
        for i, j in zip(rows, self.neighbors[rows, slots]):
            if i == j or i not in self.neighbors[j]:
                raise ValueError("neighbors must be reciprocal and non-self")
        if any(len(np.unique(row[row >= 0])) != np.count_nonzero(row >= 0) for row in self.neighbors):
            raise ValueError("duplicate neighbors")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{k: data[k].copy() for k in
                          ("lens_mm", "directions", "left", "neighbors", "grid")})


def rotation_z(angle_radians: float) -> np.ndarray:
    c, s = np.cos(angle_radians), np.sin(angle_radians)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def pose_rays(origins, directions, rotation, translation):
    """Proper head-to-world rigid transform; X front, Y left, Z up, millimetres."""
    r, t = np.asarray(rotation, float), np.asarray(translation, float)
    o, d = np.asarray(origins, float), np.asarray(directions, float)
    if (r.shape != (3, 3) or t.shape != (3,) or o.shape != d.shape or o.shape[-1] != 3
            or not all(np.isfinite(x).all() for x in (r, t, o, d))
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-12)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-12)):
        raise ValueError("invalid rigid pose/ray axes")
    return o @ r.T + t, d @ r.T


def sphere_distance(origins, directions, center, radius):
    """Nearest positive intersection, including rays beginning inside a sphere."""
    o, d, c = np.asarray(origins), np.asarray(directions), np.asarray(center)
    delta = o-c
    b = np.sum(delta*d, axis=-1)
    a = np.sum(d*d, axis=-1)
    discriminant = b*b-a*(np.sum(delta*delta, axis=-1)-radius*radius)
    root = np.sqrt(np.maximum(discriminant, 0.))
    near, far = (-b-root)/a, (-b+root)/a
    return np.where((discriminant >= 0) & (far > 1e-12),
                    np.where(near > 1e-12, near, far), np.inf)


@dataclass(frozen=True)
class Scene:
    """Finite painted spherical wall and nearer opaque sphere; achromatic radiance."""
    radius_mm: float = 10.
    stripe_cycles: int = 20
    mean: float = .5
    amplitude: float = .4
    occluder_center: tuple = (4., 0., 0.)
    occluder_radius_mm: float = .8
    occluder_light: float = .1

    def __post_init__(self):
        if (not np.isfinite([self.radius_mm, self.stripe_cycles, self.mean, self.amplitude,
                             *self.occluder_center, self.occluder_radius_mm, self.occluder_light]).all()
                or self.radius_mm <= 0 or self.occluder_radius_mm < 0
                or self.stripe_cycles < 1 or int(self.stripe_cycles) != self.stripe_cycles
                or not 0 <= self.amplitude <= min(self.mean, 1-self.mean)
                or not 0 <= self.occluder_light <= 1):
            raise ValueError("invalid scalar scene")

    def radiance(self, origins, directions):
        wall = sphere_distance(origins, directions, np.zeros(3), self.radius_mm)
        if not np.isfinite(wall).all():
            raise ValueError("rays must see enclosing wall")
        point = origins + wall[..., None]*directions
        az = np.arctan2(point[..., 1], point[..., 0])
        # Painted longitude stripes taper at the poles; no optic-flow field is supplied.
        taper = np.hypot(point[..., 0], point[..., 1])/self.radius_mm
        light = self.mean + self.amplitude*taper*np.cos(self.stripe_cycles*az)
        if self.occluder_radius_mm > 0:
            blocker = sphere_distance(origins, directions, self.occluder_center, self.occluder_radius_mm)
            light = np.where(blocker < wall, self.occluder_light, light)
        return light


@dataclass(frozen=True)
class EyeSampler:
    geometry: EyeGeometry
    local_rays: np.ndarray
    weights: np.ndarray

    @classmethod
    def compile(cls, geometry, *, fwhm_degrees=5., radial=3, azimuthal=12):
        if not np.isfinite(fwhm_degrees) or not 0 < fwhm_degrees < 30 or radial < 1 or azimuthal < 4:
            raise ValueError("invalid acceptance quadrature")
        sigma = np.deg2rad(fwhm_degrees)/np.sqrt(8*np.log(2))
        # Independently implemented spherical Gaussian integral; truncate at 3 sigma.
        nodes, weights = leggauss(radial)
        theta = (nodes+1)*1.5*sigma
        weights = weights*np.sin(theta)*np.exp(-theta*theta/(2*sigma*sigma))
        weights = np.repeat(weights, azimuthal)
        weights /= weights.sum()
        theta = np.repeat(theta, azimuthal)
        phi = np.tile(np.arange(azimuthal)*2*np.pi/azimuthal, radial)
        axes = geometry.directions
        reference = np.where((abs(axes[:, 2]) < .9)[:, None], [0., 0., 1.], [1., 0., 0.])
        tangent = np.cross(reference, axes)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        second = np.cross(axes, tangent)
        rays = (axes[:, None, :]*np.cos(theta)[None, :, None]
                + tangent[:, None, :]*(np.sin(theta)*np.cos(phi))[None, :, None]
                + second[:, None, :]*(np.sin(theta)*np.sin(phi))[None, :, None])
        return cls(geometry, rays, weights)

    def sample(self, scene, rotation=None, translation=None):
        r = np.eye(3) if rotation is None else rotation
        t = np.zeros(3) if translation is None else translation
        origins = np.broadcast_to(self.geometry.lens_mm[:, None, :], self.local_rays.shape)
        o, d = pose_rays(origins, self.local_rays, r, t)
        return scene.radiance(o, d) @ self.weights


@dataclass(frozen=True)
class LocalReadout:
    """Central lateral eye only, not a globally cardinal peripheral T4 field.

    Two adjacent oblique neighbors supply horizontal pairs with the center;
    an adjacent ventral neighbor supplies the vertical pair with the center.
    Directions orient each axis, never the stimulus label.
    Approximate central a/b = rearward/frontward; c/d = dorsal/ventral. All other
    measured facets are sensed and saved but not assigned to MaleCNS bodies.
    """
    centers: tuple
    endpoints: tuple  # per eye: ROI x four endpoints x two neighbor indices

    @classmethod
    def compile(cls, geometry, *, azimuth_range=(60., 100.), elevation_limit=15.):
        u = geometry.directions
        az = np.rad2deg(np.arctan2(u[:, 1], u[:, 0]))
        el = np.rad2deg(np.arcsin(u[:, 2]))
        centers, endpoints = [], []
        for left in (True, False):
            ids = np.flatnonzero((geometry.left == left) & (abs(az) >= azimuth_range[0])
                                 & (abs(az) <= azimuth_range[1]) & (abs(el) <= elevation_limit)
                                 & (geometry.neighbors >= 0).all(axis=1))
            if not len(ids):
                raise ValueError("no complete central eye neighborhood")
            n = geometry.neighbors[ids]
            # Author's six slots: two opposing diagonal pairs, dorsal/ventral singletons.
            rear, front = n[:, [2, 3]], n[:, [0, 5]]
            flip = abs(az[rear]).mean(axis=1) < abs(az[front]).mean(axis=1)
            rear, front = np.where(flip[:, None], front, rear), np.where(flip[:, None], rear, front)
            up, down = np.repeat(n[:, [1]], 2, axis=1), np.repeat(n[:, [4]], 2, axis=1)
            flip = el[up[:, 0]] < el[down[:, 0]]
            up, down = np.where(flip[:, None], down, up), np.where(flip[:, None], up, down)
            centers.append(ids)
            center = np.repeat(ids[:, None], 2, axis=1)
            endpoints.append(np.stack([front, center, down, center], axis=1))
        return cls(tuple(centers), tuple(endpoints))

    def channels(self, light, p: SensoryParameters):
        intensity = np.asarray(light, dtype=float)
        if intensity.ndim != 2 or not len(intensity) or not np.isfinite(intensity).all() or np.any((intensity < 0) | (intensity > 1)):
            raise ValueError("expected bounded time x facet intensities")
        if max(int(x.max()) for x in self.endpoints) >= intensity.shape[1]:
            raise ValueError("incomplete facet axis")
        # Spatial reduction gathers real neighbors only; no periodic image wrapping.
        # Adapt and rectify each sampled facet BEFORE spatial pooling.
        signals = [intensity[:, endpoint] for endpoint in self.endpoints]
        adaptation = [s[0].copy() for s in signals]
        delayed = [np.zeros((2, len(self.centers[side]), 4, 2)) for side in range(2)]
        state = np.zeros((2, 2, 4))
        output = np.empty((len(intensity), 16))
        ha, hd, he = p.dt_ms/np.array([p.highpass_tau_ms, p.delay_tau_ms, p.emission_tau_ms])
        for k in range(len(intensity)):
            output[k] = state.reshape(16)
            for side in range(2):
                hp = signals[side][k]-adaptation[side]
                current = np.stack([np.maximum(hp, 0), np.maximum(-hp, 0)])
                for axis in range(2):
                    a, b = 2*axis, 2*axis+1
                    opponent = (delayed[side][:, :, a]*current[:, :, b]
                                - current[:, :, a]*delayed[side][:, :, b]).mean(axis=(1, 2))
                    target = np.stack([np.maximum(opponent, 0), np.maximum(-opponent, 0)], axis=1)
                    state[side, :, a:b+1] += he*(target-state[side, :, a:b+1])
                adaptation[side] += ha*(signals[side][k]-adaptation[side])
                delayed[side] += hd*(current-delayed[side])
        return output


def observer_camera(scene, *, pixels=96):
    """Observer-only scalar pinhole view. Never used by EyeSampler or LocalReadout."""
    y, z = np.meshgrid(np.linspace(-1, 1, pixels), np.linspace(1, -1, pixels))
    d = np.stack([np.ones_like(y), y, z], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return scene.radiance(np.zeros_like(d), d)
