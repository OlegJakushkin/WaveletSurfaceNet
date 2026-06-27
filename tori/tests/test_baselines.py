"""Tests for the SSPD / signed Hopf-Cole baselines (Sec. 5.2)."""

import numpy as np

from pat import baselines
from pat.shapes import Plane, Sphere


def test_sspd_exact_on_plane():
    rng = np.random.default_rng(0)
    pl = Plane(extent=1.5)
    pts, nrm = pl.sample_surface(1500, rng)
    grid = rng.uniform(-0.8, 0.8, (1000, 3))
    err = np.mean(np.abs(baselines.sspd(grid, pts, nrm) - pl.sdf(grid)))
    # planar function is the exact local model for a plane
    assert err < 0.02


def test_sspd_runs_on_sphere():
    rng = np.random.default_rng(0)
    sph = Sphere(0.6)
    pts, nrm = sph.sample_surface(1000, rng)
    grid = rng.uniform(-0.9, 0.9, (500, 3))
    phi = baselines.sspd(grid, pts, nrm)
    assert np.isfinite(phi).all()
    # sign should mostly agree with truth near the surface
    true = sph.sdf(grid)
    band = np.abs(true) < 0.3
    assert np.mean(np.sign(phi[band]) == np.sign(true[band])) > 0.8


def test_signed_hopf_cole_finite_and_signs():
    rng = np.random.default_rng(0)
    sph = Sphere(0.6)
    pts, nrm = sph.sample_surface(800, rng)
    grid = rng.uniform(-0.9, 0.9, (400, 3))
    phi = baselines.signed_hopf_cole(grid, pts, nrm)
    assert np.isfinite(phi).all()
    true = sph.sdf(grid)
    band = np.abs(true) < 0.3
    assert np.mean(np.sign(phi[band]) == np.sign(true[band])) > 0.7


def test_grid_error_helper():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 0.0, 0.0])
    assert abs(baselines.grid_error(a, b) - 1.0) < 1e-9
