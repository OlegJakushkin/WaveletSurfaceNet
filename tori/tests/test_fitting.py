"""Tests for the polynomial-coefficient -> torus-parameter fitting (Sec. 4.1)."""

import numpy as np
import torch

from pat import core
from pat.lstsq import fit_coeffs_lstsq
from pat.shapes import Plane, Sphere


def test_sphere_coeffs_give_sphere_torus():
    # sphere init coeffs a02=a20=-0.5 -> torus with R=0 (a sphere) of radius 1.
    N = 100
    p = torch.randn(N, 3)
    p = p / p.norm(dim=1, keepdim=True)
    n = p.clone()
    a = torch.zeros(N, 6)
    a[:, core.A02] = -0.5
    a[:, core.A20] = -0.5
    P = core.coeffs_to_torus(p, n, a)
    # R ~ 0 (a tiny residual comes from the disc gradient-floor); r ~ radius 1.
    assert P["R"].abs().max() < 5e-3                 # degenerate torus == sphere
    assert (P["r"] - 1.0).abs().max() < 5e-3
    assert P["center"].norm(dim=1).max() < 5e-3      # centered at origin
    assert (P["sign"] == 1.0).all()


def test_plane_falls_back_to_tangent_sphere():
    # exactly-flat coeffs (all zero) must use the planar fallback, not a giant torus.
    N = 50
    p = torch.randn(N, 3)
    n = torch.tensor([0.0, 0.0, 1.0]).repeat(N, 1)
    a = torch.zeros(N, 6)
    P = core.coeffs_to_torus(p, n, a, kappa_floor=0.05)
    assert (P["R"] == 0).all()                       # sphere, not torus
    assert torch.allclose(P["r"], torch.full((N,), 20.0), atol=1e-3)
    assert (P["sign"] == 1.0).all()
    # the per-point function is ~ the planar SDF near the point
    x = p + 0.1 * n
    sel = {k: P[k] for k in ("center", "axis", "R", "r", "sign")}
    g = core.g_torus(x, sel)
    assert (g - 0.1).abs().max() < 1e-2


def test_lstsq_recovers_sphere_curvature():
    rng = np.random.default_rng(0)
    sph = Sphere(0.5)
    pts, nrm = sph.sample_surface(1500, rng)
    co = fit_coeffs_lstsq(pts, nrm, k=24)
    P = core.coeffs_to_torus(torch.tensor(pts, dtype=torch.float32),
                             torch.tensor(nrm, dtype=torch.float32), co)
    # every fitted torus should be ~ the same sphere of radius 0.5 at the origin
    assert (P["r"] - 0.5).abs().median() < 0.03
    assert P["R"].abs().median() < 0.03
    assert P["center"].norm(dim=1).median() < 0.03


def test_lstsq_plane_is_flat():
    rng = np.random.default_rng(0)
    pl = Plane(extent=1.0)
    pts, nrm = pl.sample_surface(800, rng)
    co = fit_coeffs_lstsq(pts, nrm, k=24)
    # second-order coefficients should be ~0 for a plane
    assert co[:, [core.A11, core.A02, core.A20]].abs().max() < 1e-3
