"""Tests for the Stanford bunny MeshShape wrapper."""

import os

import numpy as np
import pytest

from pat import PAT

BUNNY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "stanford-bunny.obj")
pytestmark = pytest.mark.skipif(not os.path.exists(BUNNY), reason="bunny asset not downloaded")


def test_bunny_loads_normalized():
    from pat.bunny import load_bunny
    m = load_bunny(normalize=True)
    assert len(m.vertices) > 1000
    assert np.abs(m.vertices).max() <= 1.05            # fits in the unit cube


def test_bunny_shape_signed_distance():
    from pat.bunny import bunny_shape
    s = bunny_shape()
    rng = np.random.default_rng(0)
    pts, nrm = s.sample_surface(1500, rng)
    assert np.abs(s.sdf(pts)).max() < 0.03             # surface samples ~ zero set
    assert s.sdf(np.array([[2.0, 2, 2]]))[0] > 0       # far outside is positive
    # ground-truth SDF must handle arbitrary leading dims
    q = rng.uniform(-1, 1, (4, 5, 3))
    assert s.sdf(q).shape == (4, 5)


def test_pat_reconstructs_bunny():
    from pat.bunny import bunny_shape
    s = bunny_shape()
    rng = np.random.default_rng(0)
    pts, nrm = s.sample_surface(1024, rng)
    pat = PAT(pts, nrm, k=24)
    grid = rng.uniform(-1, 1, (300, 3))
    assert np.isfinite(pat.sdf(grid, neighbors=64)).all()
    verts, faces = pat.reconstruct(res=48, bound=1.1, neighbors=64)
    assert len(verts) > 2000
