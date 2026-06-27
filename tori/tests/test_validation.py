"""Validation that a default torus reconstructs cleanly -- including its inner ring.

This is the "torus inner ring" acceptance check for BOTH models (plain torus and
supertoroid). It runs against the trained checkpoints in ``assets/`` when present,
and always runs against a per-cloud optimized fit so the suite has a standing
torus-quality guard even without a checkpoint.
"""

import os

import numpy as np
import pytest

from pat import PAT
from pat.shapes import Torus

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
R0, r0 = 0.6, 0.24


def _torus_errors(pat, seed=123):
    """Return (overall, inner-ring) mean abs SDF error for a default torus."""
    sh = Torus(R0, r0)
    rng = np.random.default_rng(seed)
    grid = rng.uniform(-1.2, 1.2, (6000, 3))
    overall = float(np.mean(np.abs(pat.sdf(grid, neighbors=64) - sh.sdf(grid))))
    # inner ring = the donut-hole rim: an annulus near rho = R-r, |z| small
    th = rng.uniform(0, 2 * np.pi, 3000)
    rho = (R0 - r0) + rng.uniform(-0.08, 0.08, 3000)
    z = rng.uniform(-0.08, 0.08, 3000)
    ring = np.stack([rho * np.cos(th), rho * np.sin(th), z], axis=1)
    inner = float(np.mean(np.abs(pat.sdf(ring, neighbors=64) - sh.sdf(ring))))
    return overall, inner


def _fit(pat_points, model=None, supertoroid=False):
    rng = np.random.default_rng(0)
    pts, nrm = Torus(R0, r0).sample_surface(pat_points, rng)
    if model is not None:
        return PAT(pts, nrm, model=model, k=16, C=16)
    if supertoroid:
        from pat.optimize import optimize_cloud
        ps, _ = optimize_cloud(pts, nrm, Torus(R0, r0), supertoroid=True, steps=120,
                               n_query=1000, seed=0)
        return PAT(pts, nrm, coeffs=ps.coeffs.numpy(), supertoroid=True,
                   p_tube=ps.p_tube.numpy(), p_ring=ps.p_ring.numpy(), C=16)
    from pat.optimize import optimize_cloud
    pt, _ = optimize_cloud(pts, nrm, Torus(R0, r0), supertoroid=False, steps=120,
                           n_query=1000, seed=0)
    return PAT(pts, nrm, coeffs=pt.coeffs.numpy(), C=16)


@pytest.mark.parametrize("supertoroid", [False, True])
def test_optimized_fit_reconstructs_torus_inner_ring(supertoroid):
    pat = _fit(1024, supertoroid=supertoroid)
    overall, inner = _torus_errors(pat)
    # both errors are small enough to be invisible by eye on a unit-scale torus
    assert overall < 0.02, f"overall torus err {overall:.4f}"
    assert inner < 0.03, f"inner-ring err {inner:.4f}"


@pytest.mark.parametrize("name", ["pat_torus.pt", "pat_supertoroid.pt"])
def test_trained_model_reconstructs_torus_inner_ring(name):
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not trained yet (run train_gpu.py)")
    import torch
    from pat.model import CoeffNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = CoeffNet(**ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()
    pat = _fit(1024, model=model)
    overall, inner = _torus_errors(pat)
    assert overall < 0.015, f"{name} overall torus err {overall:.4f}"
    assert inner < 0.025, f"{name} inner-ring err {inner:.4f}"
