"""Tests for the new analytic assets (cube, buckyball lattice, composite box+cylinders)."""

import numpy as np
import pytest

from pat import PAT
from pat.assets import BoltPlate, BoxWithCylinders, Buckyball, Cube, TexturedCylinder


def _eikonal(shape, n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 3))
    eps = 1e-3
    g = np.zeros((n, 3))
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        g[:, k] = (shape.sdf(x + d) - shape.sdf(x - d)) / (2 * eps)
    return np.abs(np.linalg.norm(g, axis=1) - 1.0).mean()


@pytest.mark.parametrize("shape", [Cube(), Buckyball(), BoxWithCylinders()])
def test_asset_surface_and_eikonal(shape):
    rng = np.random.default_rng(0)
    pts, nrm = shape.sample_surface(800, rng)
    assert np.abs(shape.sdf(pts)).max() < 3e-3       # on the zero level set
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-3)
    assert _eikonal(shape) < 0.1                       # approximately a distance function
    assert np.isin(np.sign(shape.sdf(np.array([[5.0, 5, 5]]))), [1.0]).all()  # far = outside


def test_buckyball_is_truncated_icosahedron():
    b = Buckyball()
    assert len(b.verts) == 60
    assert len(b.edges) == 90


def test_buckyball_is_hollow_lattice():
    # the lattice has holes: a marching-cubes reconstruction has many components /
    # high vertex count, unlike a solid blob.
    from skimage import measure
    b = Buckyball()
    res = 96
    lin = np.linspace(-1.2, 1.2, res)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    vol = b.sdf(np.stack([gx, gy, gz], -1).reshape(-1, 3)).reshape(res, res, res)
    assert vol.min() < 0 < vol.max()
    v, f, _, _ = measure.marching_cubes(vol, level=0.0)
    assert len(v) > 5000


@pytest.mark.parametrize("shape", [Cube(), Buckyball(), BoxWithCylinders()])
def test_pat_runs_on_asset(shape):
    rng = np.random.default_rng(0)
    pts, nrm = shape.sample_surface(1024, rng)
    pat = PAT(pts, nrm, k=24)
    grid = rng.uniform(-1, 1, (500, 3))
    assert np.isfinite(pat.sdf(grid, neighbors=64)).all()


def test_composite_has_tunnel_and_boss():
    # a point on the central axis inside the bore should be OUTSIDE (positive);
    # a point in the solid box wall should be inside (negative).
    s = BoxWithCylinders()
    assert s.sdf(np.array([[0.0, 0, 0]]))[0] > 0          # inside the bored tunnel = empty
    assert s.sdf(np.array([[0.0, 0.35, 0.35]]))[0] < 0    # solid box corner region


def test_textured_cylinder_has_knurl():
    # the knurl modulates the surface radius (a real texture, not a plain cylinder)
    s = TexturedCylinder()
    rng = np.random.default_rng(0)
    pts, nrm = s.sample_surface(4000, rng)
    assert np.abs(s.sdf(pts)).max() < 3e-3               # samples on the zero set
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-3)
    side = pts[np.abs(pts[:, 2]) < 0.6]                  # exclude flat end caps
    radius = np.linalg.norm(side[:, :2], axis=1)
    assert radius.max() - radius.min() > 0.6 * s.amp     # the diamonds raise the radius
    # marching cubes captures many facets (the diamond grid)
    from skimage import measure
    res = 96; lin = np.linspace(-1.2, 1.2, res)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    vol = s.sdf(np.stack([gx, gy, gz], -1).reshape(-1, 3)).reshape(res, res, res)
    assert measure.marching_cubes(vol, 0.0)[0].shape[0] > 8000


def test_bolt_plate_holes_and_bolts():
    s = BoltPlate()
    assert np.abs(s.sdf(s.sample_surface(800, np.random.default_rng(0))[0])).max() < 3e-3
    # center of an empty (no-bolt) hole is outside; a bolt stud location is inside
    empty = s.holes[1]                                   # has_bolt is True on evens
    assert s.sdf(np.array([[empty[0], empty[1], 0.0]]))[0] > 0
    bolt = s.holes[0]
    assert s.sdf(np.array([[bolt[0], bolt[1], s.top + 0.05]]))[0] < 0   # inside bolt head
