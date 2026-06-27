"""Tests for the learned coefficient predictor (Sec. 4.3) and the training loss."""

import numpy as np
import torch

from pat import core
from pat.model import CoeffNet
from pat.neighbors import knn_neighborhoods
from pat.shapes import Sphere
from pat.train import pat_loss, sample_queries


def _neighborhoods(pts, nrm, k=16):
    idx = knn_neighborhoods(pts, k)
    nb = torch.as_tensor(idx, dtype=torch.long)
    P = torch.as_tensor(pts, dtype=torch.float32)
    Nn = torch.as_tensor(nrm, dtype=torch.float32)
    return P[nb], Nn[nb]


def test_coeffnet_output_shapes():
    rng = np.random.default_rng(0)
    pts, nrm = Sphere(0.6).sample_surface(64, rng)
    nbr_pos, nbr_nrm = _neighborhoods(pts, nrm, k=16)
    net = CoeffNet(d_embed=32, n_layers=2, n_heads=4, d_ff=64)
    coeffs, sigma, sq = net(nbr_pos, nbr_nrm)
    assert coeffs.shape == (64, 6)
    assert sigma.shape == (64,)
    assert sq is None


def test_coeffnet_init_is_tangent_sphere():
    # at init the network output must be the paper's sphere coefficients,
    # independent of the input neighborhood (zero weights, sphere bias).
    rng = np.random.default_rng(0)
    pts, nrm = Sphere(0.6).sample_surface(32, rng)
    nbr_pos, nbr_nrm = _neighborhoods(pts, nrm, k=16)
    net = CoeffNet(d_embed=32, n_layers=2)
    coeffs, sigma, _ = net(nbr_pos, nbr_nrm)
    # rescaled: a02,a20 = -0.5/sigma ; a00,a01,a10,a11 ~ 0
    assert coeffs[:, core.A00].abs().max() < 1e-5
    assert coeffs[:, core.A11].abs().max() < 1e-5
    assert torch.allclose(coeffs[:, core.A02], -0.5 / sigma, atol=1e-4)


def test_supertoroid_model_inits_to_torus():
    rng = np.random.default_rng(0)
    pts, nrm = Sphere(0.6).sample_surface(32, rng)
    nbr_pos, nbr_nrm = _neighborhoods(pts, nrm, k=16)
    net = CoeffNet(d_embed=32, n_layers=2, supertoroid=True)
    _, _, sq = net(nbr_pos, nbr_nrm)
    assert sq.shape == (32, 2)
    assert torch.allclose(sq, torch.full_like(sq, 2.0), atol=1e-4)


def test_loss_decreases_with_training():
    # train the tiny network for a few steps on one sphere; loss must drop.
    rng = np.random.default_rng(0)
    sph = Sphere(0.6)
    pts, nrm = sph.sample_surface(128, rng)
    nbr_pos, nbr_nrm = _neighborhoods(pts, nrm, k=16)
    P = torch.as_tensor(pts, dtype=torch.float32)
    Nn = torch.as_tensor(nrm, dtype=torch.float32)
    net = CoeffNet(d_embed=32, n_layers=2, n_heads=4, d_ff=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def step():
        q, phi, _ = sample_queries(sph, 64, 64, 1.0, rng)
        coeffs, _, _ = net(nbr_pos, nbr_nrm)
        loss, _, _ = pat_loss(P, Nn, coeffs, torch.as_tensor(q),
                              torch.as_tensor(phi), eikonal_weight=0.1)
        return loss

    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
