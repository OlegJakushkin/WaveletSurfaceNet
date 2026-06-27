"""Tests for the torus-vs-supertoroid optimization and comparison (item 3).

Kept small so it runs in CI; the demo/notebook use larger settings.
"""

import numpy as np

from pat.optimize import compare_torus_vs_supertoroid, optimize_cloud
from pat.shapes import RoundedBox, Sphere, SuperToroid


def test_optimization_reduces_loss():
    rng = np.random.default_rng(0)
    sph = Sphere(0.6)
    pts, nrm = sph.sample_surface(120, rng)
    _, history = optimize_cloud(pts, nrm, sph, supertoroid=False, steps=40,
                                n_query=600, lr=8e-3, seed=0)
    assert np.mean(history[-5:]) < np.mean(history[:5])


def test_supertoroid_warm_start_is_no_worse_than_torus():
    # the supertoroid is warm-started from the torus's optimized coefficients and
    # shares the query stream, so it begins at the torus optimum and can only
    # match or improve (a small slack covers held-out-grid / optimizer noise).
    res = compare_torus_vs_supertoroid(RoundedBox(half=(0.5, 0.5, 0.5), radius=0.1),
                                       n_points=150, steps=50, grid_res=16,
                                       n_query=700, lr=8e-3, seed=0)
    assert np.isfinite(res["err_torus"]) and np.isfinite(res["err_supertoroid"])
    assert res["err_supertoroid"] <= res["err_torus"] * 1.1


def test_supertoroid_wins_on_boxy_target():
    # on a genuinely boxy supertoroid target the squareness should help on the grid.
    res = compare_torus_vs_supertoroid(SuperToroid(R=0.6, r=0.28, p_tube=4.0, p_ring=2.0),
                                       n_points=260, steps=120, grid_res=22,
                                       n_query=1100, lr=8e-3, square_reg=0.005, seed=0)
    assert res["err_supertoroid"] <= res["err_torus"]
    assert res["p_tube_mean"] > 2.0          # tube genuinely became boxier


def test_compare_returns_fitted_pats():
    res = compare_torus_vs_supertoroid(Sphere(0.6), n_points=120, steps=30,
                                       grid_res=14, n_query=600, seed=0)
    for key in ("pat_torus", "pat_supertoroid"):
        grid = np.random.default_rng(1).uniform(-1, 1, (50, 3))
        assert np.isfinite(res[key].sdf(grid, neighbors=32)).all()
