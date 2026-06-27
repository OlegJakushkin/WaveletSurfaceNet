"""Unit tests for the differentiable core: torus/supertoroid SDFs and blending."""

import numpy as np
import torch

from pat import core
from pat.shapes import Torus


def test_local_basis_is_orthonormal():
    n = torch.randn(200, 3)
    n = n / n.norm(dim=1, keepdim=True)
    s, t = core.local_basis(n)
    assert torch.allclose(s.norm(dim=1), torch.ones(200), atol=1e-5)
    assert torch.allclose(t.norm(dim=1), torch.ones(200), atol=1e-5)
    assert (s * t).sum(1).abs().max() < 1e-5
    assert (s * n).sum(1).abs().max() < 1e-5
    assert (t * n).sum(1).abs().max() < 1e-5
    # right-handed: s x t == n
    assert torch.allclose(torch.cross(s, t, dim=1), n, atol=1e-5)


def test_lp_norm_special_cases():
    x = torch.tensor([3.0, 1.0, -2.0])
    y = torch.tensor([4.0, 0.0, 2.0])
    mx = torch.maximum(x.abs(), y.abs())
    # p = 2 -> Euclidean
    assert torch.allclose(core.lp_norm2(x, y, torch.tensor(2.0)),
                          torch.sqrt(x * x + y * y), atol=1e-5)
    # for any p in [1, inf): max <= ||.||_p <= max * 2^(1/p), and -> max as p grows.
    for p in (3.0, 10.0, 200.0):
        v = core.lp_norm2(x, y, torch.tensor(p))
        upper = mx * 2 ** (1.0 / p)
        assert (v >= mx - 1e-4).all() and (v <= upper + 1e-4).all()
    assert (core.lp_norm2(x, y, torch.tensor(200.0)) - mx).abs().max() < 1e-2


def test_torus_sdf_matches_analytic_shape():
    R, r = 0.7, 0.2
    shape = Torus(R, r, center=(0.1, -0.2, 0.05), axis=(0.0, 1.0, 0.3))
    x = torch.tensor(np.random.default_rng(1).uniform(-1.5, 1.5, (500, 3)),
                     dtype=torch.float32)
    c = torch.tensor(shape.c, dtype=torch.float32)
    u = torch.tensor(shape.u, dtype=torch.float32)
    sdf = core.torus_sdf(x, c, u, torch.tensor(R), torch.tensor(r))
    ref = torch.tensor(shape.sdf(x.numpy()), dtype=torch.float32)
    assert (sdf - ref).abs().max() < 1e-4


def test_supertoroid_reduces_to_torus_at_p2():
    R, r = 0.6, 0.25
    c = torch.zeros(3)
    u = torch.tensor([0.0, 0.0, 1.0])
    ea = torch.tensor([1.0, 0.0, 0.0])
    x = torch.tensor(np.random.default_rng(2).uniform(-1.2, 1.2, (400, 3)),
                     dtype=torch.float32)
    t = core.torus_sdf(x, c, u, torch.tensor(R), torch.tensor(r))
    s = core.supertoroid_sdf(x, c, u, ea, torch.tensor(R), torch.tensor(r),
                             torch.tensor(2.0), torch.tensor(2.0))
    assert (t - s).abs().max() < 1e-3


def test_supertoroid_squareness_makes_tube_boxier():
    # A large tube exponent should let points at the "corner" of the square tube
    # sit *outside* the round tube (boxier cross-section reaches farther).
    c = torch.zeros(3)
    u = torch.tensor([0.0, 0.0, 1.0])
    ea = torch.tensor([1.0, 0.0, 0.0])
    R, r = torch.tensor(0.6), torch.tensor(0.2)
    # point at 45 deg in the tube cross-section, at radius r along both axes
    corner = torch.tensor([[0.6 + 0.2, 0.0, 0.2]])  # (ring_radius-R, axial)=(0.2,0.2)
    round_sdf = core.supertoroid_sdf(corner, c, u, ea, R, r,
                                     torch.tensor(2.0), torch.tensor(2.0))
    box_sdf = core.supertoroid_sdf(corner, c, u, ea, R, r,
                                   torch.tensor(8.0), torch.tensor(2.0))
    # round tube: this point is outside (dist sqrt(.08)-.2>0); box tube: still inside
    assert round_sdf.item() > 0
    assert box_sdf.item() < round_sdf.item()


def test_blend_recovers_constant_field():
    # if every g_i equals the same value v at x, the blend must return v.
    pts = torch.randn(50, 3)
    x = torch.randn(20, 3)
    g = torch.full((20, 50), 3.14)
    phi = core.blend(x, pts, g)
    assert torch.allclose(phi, torch.full((20,), 3.14), atol=1e-4)


def test_blend_is_robust_for_far_queries():
    # far queries used to underflow the denominator; ensure they don't.
    pts = torch.randn(30, 3)
    x = torch.randn(10, 3) * 50.0          # very far away
    g = (x.norm(dim=1, keepdim=True) - 1.0).repeat(1, 30)
    phi = core.blend(x, pts, g)
    assert torch.isfinite(phi).all()
    assert (phi - (x.norm(dim=1) - 1.0)).abs().max() < 1e-2


def test_raw_to_p_maps_to_torus_at_default():
    assert abs(core.raw_to_p(torch.tensor(core.P2_RAW)).item() - 2.0) < 1e-5
    assert (core.raw_to_p(torch.randn(100)) >= 1.0).all()
