"""Tests specific to the supertoroid extension."""

import numpy as np
import torch

from pat import PAT, core
from pat.shapes import SuperToroid, Torus


def test_supertoroid_shape_reduces_to_torus_at_p2():
    st = SuperToroid(R=0.6, r=0.25, p_tube=2.0, p_ring=2.0)
    to = Torus(0.6, 0.25)
    x = np.random.default_rng(0).uniform(-1.2, 1.2, (500, 3))
    assert np.abs(st.sdf(x) - to.sdf(x)).max() < 1e-6


def test_supertoroid_surface_samples_on_zero_level_set():
    st = SuperToroid(R=0.6, r=0.28, p_tube=4.0, p_ring=2.0)
    rng = np.random.default_rng(0)
    pts, _ = st.sample_surface(800, rng)
    # the approximate radial SDF is ~0 on the parametric surface
    assert np.abs(st.sdf(pts)).max() < 0.02


def test_boxy_tube_extends_past_round_tube():
    # for a boxy tube (p>2) the cross-section reaches farther toward its corners.
    c = torch.zeros(3); u = torch.tensor([0., 0., 1.]); ea = torch.tensor([1., 0., 0.])
    R, r = torch.tensor(0.6), torch.tensor(0.2)
    corner = torch.tensor([[0.6 + 0.14, 0.0, 0.14]])     # 45 deg in tube plane
    sround = core.supertoroid_sdf(corner, c, u, ea, R, r, torch.tensor(2.0), torch.tensor(2.0))
    sbox = core.supertoroid_sdf(corner, c, u, ea, R, r, torch.tensor(6.0), torch.tensor(2.0))
    assert sbox.item() < sround.item()           # corner is "more inside" the boxy tube


def test_supertoroid_pat_runs_and_reconstructs():
    rng = np.random.default_rng(0)
    st = SuperToroid(R=0.6, r=0.28, p_tube=4.0, p_ring=2.0)
    pts, nrm = st.sample_surface(1200, rng)
    pat = PAT(pts, nrm, k=24, supertoroid=True, p_tube=4.0, p_ring=2.0)
    grid = rng.uniform(-1, 1, (1500, 3))
    phi = pat.sdf(grid, neighbors=64)
    assert np.isfinite(phi).all()
    verts, faces = pat.reconstruct(res=40, bound=1.1, neighbors=64)
    assert len(verts) > 100 and len(faces) > 100


def test_supertoroid_more_expressive_than_torus_with_exact_base():
    # Expressiveness, isolated from curvature-estimation error: give BOTH models the
    # *exact* osculating base torus of a boxy supertoroid (same center/axis/R/r for
    # every point), and only vary the cross-section squareness.  The supertoroid
    # (correct p) reproduces the target almost exactly; the torus (p=2) cannot.
    rng = np.random.default_rng(0)
    R, r, pt = 0.6, 0.28, 4.0
    st = SuperToroid(R=R, r=r, p_tube=pt, p_ring=2.0)   # axis z, center 0
    pts, _ = st.sample_surface(1500, rng)
    P = torch.as_tensor(pts, dtype=torch.float32)
    N = len(pts)
    base = {
        "center": torch.zeros(N, 3),
        "axis": torch.tensor([0., 0., 1.]).repeat(N, 1),
        "ea": torch.tensor([1., 0., 0.]).repeat(N, 1),
        "R": torch.full((N,), R), "r": torch.full((N,), r),
        "sign": torch.ones(N),
    }
    grid = torch.as_tensor(rng.uniform(-1, 1, (3000, 3)), dtype=torch.float32)
    gt = torch.as_tensor(st.sdf(grid.numpy()), dtype=torch.float32)
    sel = {k: v.unsqueeze(0) for k, v in base.items()}
    g_t = core.g_torus(grid.unsqueeze(1), sel)
    g_s = core.g_supertoroid(grid.unsqueeze(1), sel, torch.tensor(pt), torch.tensor(2.0))
    err_t = (core.blend(grid, P, g_t) - gt).abs().mean().item()
    err_s = (core.blend(grid, P, g_s) - gt).abs().mean().item()
    assert err_s < 0.3 * err_t          # supertoroid is much closer to the boxy target
