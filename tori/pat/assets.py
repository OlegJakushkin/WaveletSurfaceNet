"""Composite analytic SDF assets (built from Quilez primitives + CSG).

These are richer ground-truth shapes than the single primitives in
:mod:`pat.shapes`: a near-sharp cube, a fullerene (C60 "buckyball") wireframe
cage, and a bored/bossed rounded box.  They exist to stress-test the PAT
pipeline on geometry with sharp features, thin tubes, high genus and CSG holes.

Every asset subclasses :class:`pat.shapes.Shape`, returns an exact signed
distance (negative inside, positive outside) vectorized over the leading dims,
is centered near the origin and scaled to fit roughly within ``[-1, 1]^3`` so it
matches the SDF grid and renderer.  All SDFs follow Inigo Quilez's standard
formulas.  Surface sampling uses the base-class generic Newton projector.
"""

from __future__ import annotations

import numpy as np

from .shapes import Shape


# --------------------------------------------------------------------------- #
#  Quilez SDF helpers (vectorized over the leading dims of ``p``)
# --------------------------------------------------------------------------- #
def _sd_box(p: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed distance to an axis-aligned box of half-extents ``b``."""
    q = np.abs(p) - b
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def _sd_capsule(p: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Signed distance to a capsule (line segment ``a``--``b`` thickened by ``r``)."""
    pa = p - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def _sd_capped_cylinder_x(p: np.ndarray, h: float, ra: float) -> np.ndarray:
    """Signed distance to a cylinder along the x axis, half-height ``h``, radius ``ra``."""
    # distance in the (radial, axial) 2D profile, mirrored back to 3D
    dr = np.linalg.norm(p[..., 1:3], axis=-1) - ra
    da = np.abs(p[..., 0]) - h
    inside = np.minimum(np.maximum(dr, da), 0.0)
    outside = np.linalg.norm(np.maximum(np.stack([dr, da], axis=-1), 0.0), axis=-1)
    return inside + outside


def _sd_capped_cylinder_z(p: np.ndarray, zc: float, h: float, ra: float) -> np.ndarray:
    """Signed distance to a cylinder along z, centered at height ``zc``, half-height ``h``."""
    dr = np.linalg.norm(p[..., 0:2], axis=-1) - ra
    da = np.abs(p[..., 2] - zc) - h
    inside = np.minimum(np.maximum(dr, da), 0.0)
    outside = np.linalg.norm(np.maximum(np.stack([dr, da], axis=-1), 0.0), axis=-1)
    return inside + outside


# --------------------------------------------------------------------------- #
#  1) Cube -- a near-sharp box
# --------------------------------------------------------------------------- #
class Cube(Shape):
    """A near-sharp cube (box SDF with a tiny rounding radius for numerical niceness)."""

    def __init__(self, half=0.6, rounding=0.02, center=(0, 0, 0)):
        self.b = np.full(3, float(half))
        self.rad = float(rounding)
        self.c = np.asarray(center, dtype=np.float64)

    def sdf(self, x):
        return _sd_box(np.asarray(x, dtype=np.float64) - self.c, self.b) - self.rad

    def sample_surface(self, n, rng):
        return self.sample_by_projection(n, rng, bound=1.3)


# --------------------------------------------------------------------------- #
#  2) Buckyball -- a truncated-icosahedron (C60) wireframe cage
# --------------------------------------------------------------------------- #
def _truncated_icosahedron_vertices() -> np.ndarray:
    """The 60 vertices of a truncated icosahedron (soccer ball / fullerene C60).

    Vertices are all EVEN permutations of the coordinate triples
    ``(0, +-1, +-3*phi)``, ``(+-2, +-(1+2*phi), +-phi)`` and
    ``(+-1, +-(2+phi), +-2*phi)`` with ``phi = (1 + sqrt5) / 2``.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # the three base magnitude-triples (signs are applied below)
    bases = [
        (0.0, 1.0, 3.0 * phi),
        (2.0, 1.0 + 2.0 * phi, phi),
        (1.0, 2.0 + phi, 2.0 * phi),
    ]
    # even permutations of indices (0,1,2): the 3 cyclic rotations
    even_perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    verts = set()
    for base in bases:
        for perm in even_perms:
            permuted = (base[perm[0]], base[perm[1]], base[perm[2]])
            # all sign combinations; a zero coordinate has no independent sign
            for sx in (1.0, -1.0):
                for sy in (1.0, -1.0):
                    for sz in (1.0, -1.0):
                        v = (sx * permuted[0], sy * permuted[1], sz * permuted[2])
                        # round to collapse +0.0/-0.0 and floating dupes
                        verts.add(tuple(np.round(v, 9)))
    return np.asarray(sorted(verts), dtype=np.float64)


def _edges_from_vertices(verts: np.ndarray) -> np.ndarray:
    """Edges = all vertex pairs whose distance equals the minimum pairwise distance."""
    diff = verts[:, None, :] - verts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu, ju = np.triu_indices(len(verts), k=1)
    d = dist[iu, ju]
    dmin = d.min()
    sel = d < dmin * 1.001            # tolerance for floating equality
    return np.stack([iu[sel], ju[sel]], axis=1)


class Buckyball(Shape):
    """A truncated-icosahedron wireframe cage (C60 / soccer ball / fullerene).

    The surface is the union (``min``) of rounded tubes (capsule SDFs) along the
    90 edges of a truncated icosahedron, unioned with little spheres at the 60
    vertex joints -- a hollow lattice like a fullerene cage.  Vertices are
    rescaled to fit ``[-1, 1]^3`` (divide by max abs coord, then * ~0.9).
    """

    def __init__(self, tube_radius=0.045, joint_radius=0.06):
        self.tube_r = float(tube_radius)
        self.joint_r = float(joint_radius)

        verts = _truncated_icosahedron_vertices()
        self.edges = _edges_from_vertices(verts)
        # rescale to fit nicely inside [-1, 1]^3
        verts = verts / np.max(np.abs(verts)) * 0.9
        self.verts = verts
        # precompute edge endpoints for fast vectorized sdf
        self._a = verts[self.edges[:, 0]]      # (90, 3)
        self._b = verts[self.edges[:, 1]]      # (90, 3)

    def sdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        d = np.full(x.shape[:-1], np.inf)
        # union over the 90 tube edges (accumulate with np.minimum)
        for a, b in zip(self._a, self._b):
            d = np.minimum(d, _sd_capsule(x, a, b, self.tube_r))
        # union the 60 vertex joint spheres
        for v in self.verts:
            d = np.minimum(d, np.linalg.norm(x - v, axis=-1) - self.joint_r)
        return d

    def sample_surface(self, n, rng):
        return self.sample_by_projection(n, rng, bound=1.1)


# --------------------------------------------------------------------------- #
#  3) BoxWithCylinders -- rounded box + boss, with a bored-through tunnel
# --------------------------------------------------------------------------- #
class BoxWithCylinders(Shape):
    """A rounded box with a cylindrical tunnel bored through it and a protruding boss.

    Reproduces the composite from the paper's noisy figure:

    * ``base``   -- a rounded box,
    * ``boss``   -- a solid cylinder protruding from the ``+x`` face,
    * ``tunnel`` -- a long cylinder bored straight along the x axis.

    ``sdf = max(min(base, boss), -tunnel)``: union the boss onto the box, then
    subtract the through-hole.  Centered near the origin, fits ``[-1, 1]^3``.
    """

    def __init__(self, half=(0.45, 0.45, 0.45), rounding=0.06,
                 tunnel_radius=0.18, boss_radius=0.22, center=(0, 0, 0)):
        self.b = np.asarray(half, dtype=np.float64)
        self.rad = float(rounding)
        self.tunnel_r = float(tunnel_radius)
        self.boss_r = float(boss_radius)
        self.c = np.asarray(center, dtype=np.float64)
        # boss spans x in [0.45, 0.75] -> center 0.60, half-length 0.15
        self.boss_center_x = 0.60
        self.boss_half = 0.15

    def sdf(self, x):
        p = np.asarray(x, dtype=np.float64) - self.c
        base = _sd_box(p, self.b) - self.rad
        # long through-tunnel along x (half-length large so it punches both faces)
        tunnel = _sd_capped_cylinder_x(p, h=1.5, ra=self.tunnel_r)
        # boss: cylinder along x centered at boss_center_x
        pb = p.copy()
        pb[..., 0] = pb[..., 0] - self.boss_center_x
        boss = _sd_capped_cylinder_x(pb, h=self.boss_half, ra=self.boss_r)
        # union boss onto box, then subtract the tunnel
        return np.maximum(np.minimum(base, boss), -tunnel)

    def sample_surface(self, n, rng):
        return self.sample_by_projection(n, rng, bound=1.3)


# --------------------------------------------------------------------------- #
#  4) TexturedCylinder -- a diamond-knurled handle (the barbell-knurl pattern)
# --------------------------------------------------------------------------- #
class TexturedCylinder(Shape):
    """A cylinder with a diamond cross-hatch *knurl*, like a barbell/tool handle.

    The knurl is two crossing families of helical ridges; their overlap raises a
    grid of sharp diamond pyramids on the surface.  This is a hard "texture +
    sharp corners" test: a round torus tube must round off every pyramid ridge,
    while a supertoroid's boxier cross-section can hug the faceted texture better.

    The side SDF is the radial inside-outside approximation ``rho - (R + disp)``
    (exact distance only for a smooth cylinder); the diamond displacement makes it
    a tight bumpy proxy, which is all the reconstruction comparison needs.
    """

    def __init__(self, radius=0.34, half_length=0.74, amp=0.05, n_around=26,
                 n_axial=20, center=(0, 0, 0)):
        self.R = float(radius)
        self.L = float(half_length)
        self.amp = float(amp)
        self.n_around = int(n_around)        # diamonds around the circumference
        self.n_axial = int(n_axial)          # diamond pitch along the axis
        self.c = np.asarray(center, dtype=np.float64)

    def _disp(self, theta, z):
        """Diamond knurl height in [0, amp] as a function of angle and axial coord."""
        u = self.n_around * (theta % (2 * np.pi)) / (2 * np.pi)   # seamless wrap
        v = self.n_axial * (z + self.L) / (2 * self.L)
        tri = lambda s: 1.0 - 2.0 * np.abs((s % 1.0) - 0.5)       # triangle wave 0..1
        return self.amp * np.minimum(tri(u + v), tri(u - v))       # crossing helices

    def sdf(self, x):
        p = np.asarray(x, dtype=np.float64) - self.c
        rho = np.linalg.norm(p[..., 0:2], axis=-1)
        theta = np.arctan2(p[..., 1], p[..., 0])
        z = p[..., 2]
        dr = rho - (self.R + self._disp(theta, z))
        da = np.abs(z) - self.L
        inside = np.minimum(np.maximum(dr, da), 0.0)
        outside = np.linalg.norm(np.maximum(np.stack([dr, da], axis=-1), 0.0), axis=-1)
        return inside + outside

    def sample_surface(self, n, rng):
        # sample the knurled side parametrically (exact zero set), plus a few cap points
        n_cap = n // 12
        n_side = n - 2 * n_cap
        theta = rng.uniform(0, 2 * np.pi, n_side)
        z = rng.uniform(-self.L, self.L, n_side)
        r = self.R + self._disp(theta, z)
        side = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
        pts = [side]
        for sgn in (-1.0, 1.0):
            rad = np.sqrt(rng.uniform(0, 1, n_cap)) * self.R
            ang = rng.uniform(0, 2 * np.pi, n_cap)
            cap = np.stack([rad * np.cos(ang), rad * np.sin(ang),
                            np.full(n_cap, sgn * self.L)], axis=1)
            pts.append(cap)
        pts = np.concatenate(pts, axis=0)[:n] + self.c
        return pts, self.normal(pts)


# --------------------------------------------------------------------------- #
#  5) BoltPlate -- a rounded plate with through-holes and inserted bolts
# --------------------------------------------------------------------------- #
class BoltPlate(Shape):
    """A flat rounded plate with a ring of through-holes, half of them holding bolts.

    Each bolt is a thin stud (narrower than its hole) capped by a wider cylindrical
    head resting on the plate -- the "hole + bolts" mechanical part.  Mixes flat
    faces, round holes and small round studs/heads with sharp rims.
    """

    def __init__(self, plate_half=(0.62, 0.62, 0.10), rounding=0.03, hole_r=0.085,
                 bolt_r=0.055, head_r=0.11, n_holes=6, ring_r=0.40, center=(0, 0, 0)):
        self.b = np.asarray(plate_half, dtype=np.float64)
        self.rad = float(rounding)
        self.hole_r, self.bolt_r, self.head_r = float(hole_r), float(bolt_r), float(head_r)
        self.ring_r = float(ring_r)
        self.c = np.asarray(center, dtype=np.float64)
        ang = np.linspace(0, 2 * np.pi, n_holes, endpoint=False)
        self.holes = np.stack([ring_r * np.cos(ang), ring_r * np.sin(ang)], axis=1)
        self.has_bolt = (np.arange(n_holes) % 2 == 0)   # every other hole holds a bolt
        self.top = float(plate_half[2])

    def sdf(self, x):
        p = np.asarray(x, dtype=np.float64) - self.c
        d = _sd_box(p, self.b) - self.rad
        for (hx, hy) in self.holes:                      # punch through-holes
            ph = p.copy(); ph[..., 0] -= hx; ph[..., 1] -= hy
            hole = _sd_capped_cylinder_z(ph, zc=0.0, h=1.0, ra=self.hole_r)
            d = np.maximum(d, -hole)
        for (hx, hy), bolt in zip(self.holes, self.has_bolt):   # add bolts
            if not bolt:
                continue
            ph = p.copy(); ph[..., 0] -= hx; ph[..., 1] -= hy
            stud = _sd_capped_cylinder_z(ph, zc=self.top - 0.02, h=self.top + 0.06,
                                         ra=self.bolt_r)
            head = _sd_capped_cylinder_z(ph, zc=self.top + 0.07, h=0.05, ra=self.head_r)
            d = np.minimum(d, np.minimum(stud, head))
        return d

    def sample_surface(self, n, rng):
        return self.sample_by_projection(n, rng, bound=1.2)
