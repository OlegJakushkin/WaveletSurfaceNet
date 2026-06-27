"""Analytic SDF primitives and point-cloud sampling.

These shapes give *exact* ground-truth signed distance and surface normals, which
makes them ideal for (a) unit tests of the whole PAT pipeline and (b) generating
training data for the notebook without needing a mesh/ray-tracer.  We also provide
mesh-based sampling (via :mod:`trimesh`) for the optional mesh demos.

All SDFs follow Inigo Quilez's standard formulas and return signed distance with
the usual convention: negative inside, positive outside.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Analytic primitives
# --------------------------------------------------------------------------- #
class Shape:
    """Base class: an analytic surface with exact SDF, normals and surface sampling."""

    def sdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def normal(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Outward unit normal via central differences of the SDF (gradient)."""
        x = np.asarray(x, dtype=np.float64)
        g = np.empty_like(x)
        for k in range(3):
            d = np.zeros(3)
            d[k] = eps
            g[..., k] = (self.sdf(x + d) - self.sdf(x - d)) / (2 * eps)
        return g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)

    def sample_surface(self, n: int, rng: np.random.Generator):
        """Return ``(points (n,3), normals (n,3))`` sampled on the surface."""
        raise NotImplementedError

    def sample_by_projection(self, n, rng, bound=1.3, band=None, batch=20000,
                             newton=2, max_iter=200):
        """Generic surface sampler for arbitrary SDFs (rejection + Newton projection).

        Draws bulk points in ``[-bound, bound]^3``, keeps those within a thin band of
        the zero level set, then projects each onto the surface by a few Newton steps
        ``x <- x - sdf(x) * grad(x)``.  Works for any CSG / lattice SDF without a
        bespoke parameterization.  Returns ``(points (n,3), normals (n,3))``.
        """
        pts = []
        it = 0
        while len(pts) < n and it < max_iter:
            it += 1
            x = rng.uniform(-bound, bound, size=(batch, 3))
            d = self.sdf(x)
            b = band if band is not None else 1.5 * (2 * bound) / batch ** (1 / 3)
            keep = x[np.abs(d) < b]
            for _ in range(newton):
                g = self.normal(keep)
                keep = keep - self.sdf(keep)[:, None] * g
            keep = keep[np.abs(self.sdf(keep)) < 1e-3]
            pts.extend(keep)
        pts = np.asarray(pts[:n])
        if len(pts) < n:                       # pad by resampling if under-filled
            extra = pts[rng.integers(0, max(len(pts), 1), size=n - len(pts))]
            pts = np.concatenate([pts, extra], axis=0) if len(pts) else extra
        return pts, self.normal(pts)


class Sphere(Shape):
    def __init__(self, radius=1.0, center=(0, 0, 0)):
        self.R = float(radius)
        self.c = np.asarray(center, dtype=np.float64)

    def sdf(self, x):
        return np.linalg.norm(np.asarray(x) - self.c, axis=-1) - self.R

    def sample_surface(self, n, rng):
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return self.c + self.R * d, d


class Plane(Shape):
    def __init__(self, point=(0, 0, 0), normal=(0, 0, 1), extent=1.0):
        self.p = np.asarray(point, dtype=np.float64)
        self.n = np.asarray(normal, dtype=np.float64)
        self.n /= np.linalg.norm(self.n)
        self.extent = float(extent)

    def sdf(self, x):
        return (np.asarray(x) - self.p) @ self.n

    def sample_surface(self, n, rng):
        # build tangent basis
        a = np.array([1.0, 0, 0]) if abs(self.n[0]) < 0.9 else np.array([0, 1.0, 0])
        s = np.cross(self.n, a); s /= np.linalg.norm(s)
        t = np.cross(self.n, s)
        uv = rng.uniform(-self.extent, self.extent, size=(n, 2))
        pts = self.p + uv[:, :1] * s + uv[:, 1:] * t
        nrm = np.tile(self.n, (n, 1))
        return pts, nrm


class Torus(Shape):
    def __init__(self, R=0.6, r=0.25, center=(0, 0, 0), axis=(0, 0, 1)):
        self.R, self.r = float(R), float(r)
        self.c = np.asarray(center, dtype=np.float64)
        u = np.asarray(axis, dtype=np.float64)
        self.u = u / np.linalg.norm(u)

    def sdf(self, x):
        rel = np.asarray(x) - self.c
        axial = rel @ self.u
        perp = rel - axial[..., None] * self.u
        q = np.linalg.norm(perp, axis=-1) - self.R
        return np.sqrt(q * q + axial * axial) - self.r

    def sample_surface(self, n, rng):
        a = np.array([1.0, 0, 0]) if abs(self.u[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(self.u, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.u, e1)
        theta = rng.uniform(0, 2 * np.pi, n)   # around the ring
        phi = rng.uniform(0, 2 * np.pi, n)     # around the tube
        ring = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
        pts = self.c + (self.R + self.r * np.cos(phi))[:, None] * ring \
            + (self.r * np.sin(phi))[:, None] * self.u
        nrm = np.cos(phi)[:, None] * ring + np.sin(phi)[:, None] * self.u
        return pts, nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def _superpow(c, e):
    """Signed power ``sign(c) |c|^e`` used by super-ellipse parameterizations."""
    return np.sign(c) * np.abs(c) ** e


class SuperToroid(Shape):
    """A supertoroid: a torus whose ring/tube cross-sections are L^p super-ellipses.

    ``p_tube = p_ring = 2`` is an ordinary torus.  Larger exponents give boxier
    (rounded-square) cross-sections that an ordinary torus cannot represent -- this
    is the shape used to demonstrate the supertoroid's extra expressiveness.

    The surface sampler is exact (parametric).  The :meth:`sdf` is the same radial
    L^p approximation used by :func:`pat.core.supertoroid_sdf` (exact only at
    ``p = 2``); it serves as a consistent reference for the field comparison.
    """

    def __init__(self, R=0.6, r=0.25, p_tube=4.0, p_ring=2.0, center=(0, 0, 0),
                 axis=(0, 0, 1)):
        self.R, self.r = float(R), float(r)
        self.pt, self.pr = float(p_tube), float(p_ring)
        self.c = np.asarray(center, dtype=np.float64)
        u = np.asarray(axis, dtype=np.float64)
        self.u = u / np.linalg.norm(u)
        a = np.array([1.0, 0, 0]) if abs(self.u[0]) < 0.9 else np.array([0, 1.0, 0])
        self.e1 = np.cross(self.u, a); self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.u, self.e1)

    def sdf(self, x):
        rel = np.asarray(x) - self.c
        axial = rel @ self.u
        a = rel @ self.e1
        b = rel @ self.e2
        er = 2.0 / self.pr
        et = 2.0 / self.pt
        ring = (np.abs(a) ** (self.pr) + np.abs(b) ** (self.pr)) ** (1.0 / self.pr)
        dx = ring - self.R
        return (np.abs(dx) ** self.pt + np.abs(axial) ** self.pt) ** (1.0 / self.pt) - self.r

    def sample_surface(self, n, rng):
        # superellipse exponents map p -> shape exponent e = 2/p
        et, er = 2.0 / self.pt, 2.0 / self.pr
        theta = rng.uniform(0, 2 * np.pi, n)     # around the ring
        phi = rng.uniform(0, 2 * np.pi, n)       # around the tube
        ca = _superpow(np.cos(theta), er)[:, None]
        sa = _superpow(np.sin(theta), er)[:, None]
        cp = _superpow(np.cos(phi), et)
        sp = _superpow(np.sin(phi), et)
        ring_dir = ca * self.e1 + sa * self.e2
        pts = self.c + (self.R + self.r * cp)[:, None] * ring_dir \
            + (self.r * sp)[:, None] * self.u
        nrm = self.normal(pts)
        return pts, nrm


class RoundedBox(Shape):
    """Box with rounded edges -- has large flat regions that challenge plain tori."""

    def __init__(self, half=(0.5, 0.5, 0.5), radius=0.08, center=(0, 0, 0)):
        self.b = np.asarray(half, dtype=np.float64)
        self.rad = float(radius)
        self.c = np.asarray(center, dtype=np.float64)

    def sdf(self, x):
        q = np.abs(np.asarray(x) - self.c) - self.b
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside - self.rad

    def sample_surface(self, n, rng):
        # rejection: sample a slightly inflated box shell, project onto level set
        pts, nrm = [], []
        ext = self.b + self.rad + 0.05
        while len(pts) < n:
            x = self.c + rng.uniform(-ext, ext, size=(n, 3))
            d = self.sdf(x)
            keep = np.abs(d) < 0.25 * self.rad + 0.02
            for xi in x[keep]:
                g = self.normal(xi)
                xi = xi - self.sdf(xi) * g          # project to surface
                pts.append(xi)
                nrm.append(self.normal(xi))
                if len(pts) >= n:
                    break
        return np.asarray(pts[:n]), np.asarray(nrm[:n])


# --------------------------------------------------------------------------- #
#  Mesh sampling (optional, needs trimesh)
# --------------------------------------------------------------------------- #
def sample_mesh(mesh, n: int, rng: np.random.Generator):
    """Uniformly sample ``n`` points + face normals from a ``trimesh.Trimesh``."""
    pts, fidx = mesh.sample(n, return_index=True)
    nrm = mesh.face_normals[fidx]
    return np.asarray(pts), np.asarray(nrm)


def normalize_to_unit_cube(mesh):
    """Center a mesh at the origin and scale it to fit in ``[-1, 1]^3`` (paper Sec. 5)."""
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    scale = np.max(np.abs(mesh.bounds))
    mesh.apply_scale(1.0 / scale)
    return mesh
