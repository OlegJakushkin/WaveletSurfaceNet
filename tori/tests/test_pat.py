"""End-to-end tests for the high-level PAT pipeline (Algorithm 1)."""

import numpy as np
import pytest

from pat import PAT
from pat.shapes import Plane, Sphere, Torus


def test_pat_reconstructs_sphere_sdf():
    rng = np.random.default_rng(0)
    sph = Sphere(0.6)
    pts, nrm = sph.sample_surface(1500, rng)
    pat = PAT(pts, nrm, k=24)
    grid = rng.uniform(-1, 1, (3000, 3))
    err = np.mean(np.abs(pat.sdf(grid, neighbors=64) - sph.sdf(grid)))
    assert err < 0.01


def test_pat_reconstructs_plane_sdf():
    rng = np.random.default_rng(0)
    pl = Plane(extent=1.0)
    pts, nrm = pl.sample_surface(1200, rng)
    pat = PAT(pts, nrm, k=24)
    grid = rng.uniform(-0.8, 0.8, (3000, 3))
    err = np.mean(np.abs(pat.sdf(grid, neighbors=64) - pl.sdf(grid)))
    assert err < 0.02


def test_exact_and_knn_evaluation_agree_near_surface():
    rng = np.random.default_rng(0)
    sph = Sphere(0.6)
    pts, nrm = sph.sample_surface(800, rng)
    pat = PAT(pts, nrm, k=24)
    grid = rng.uniform(-0.9, 0.9, (500, 3))
    exact = pat.sdf(grid, neighbors=None)
    knn = pat.sdf(grid, neighbors=128)
    assert np.mean(np.abs(exact - knn)) < 0.02


def test_supertoroid_pat_at_p2_matches_torus_pat():
    rng = np.random.default_rng(0)
    sph = Sphere(0.6)
    pts, nrm = sph.sample_surface(600, rng)
    coeffs = None
    pt = PAT(pts, nrm, k=24)
    ps = PAT(pts, nrm, k=24, supertoroid=True, p_tube=2.0, p_ring=2.0,
             coeffs=pt.coeffs.numpy())
    grid = rng.uniform(-1, 1, (800, 3))
    a = pt.sdf(grid, neighbors=64)
    b = ps.sdf(grid, neighbors=64)
    assert np.max(np.abs(a - b)) < 5e-3


def test_reconstruct_returns_mesh_near_sphere():
    rng = np.random.default_rng(0)
    sph = Sphere(0.6)
    pts, nrm = sph.sample_surface(1500, rng)
    pat = PAT(pts, nrm, k=24)
    verts, faces = pat.reconstruct(res=48, bound=1.0, neighbors=64)
    assert len(verts) > 100 and len(faces) > 100
    # reconstructed vertices should be ~ on the sphere of radius 0.6
    radii = np.linalg.norm(verts, axis=1)
    assert abs(np.median(radii) - 0.6) < 0.05
