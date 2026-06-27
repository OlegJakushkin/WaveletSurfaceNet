"""Tests for the noisy real-dataset training pipeline."""

import os

import numpy as np
import pytest
import torch

BUNNY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "stanford-bunny.obj")
pytestmark = pytest.mark.skipif(not os.path.exists(BUNNY), reason="bunny asset not downloaded")


def _bunny_mesh():
    from pat.bunny import load_bunny
    return load_bunny(normalize=True)


def test_modelnet_index_no_crash():
    from pat.datasets import modelnet_index
    idx = modelnet_index()
    assert isinstance(idx, list)                       # [] if not downloaded, else paths


def test_noisy_point_cloud_actually_adds_noise():
    from pat.datasets import noisy_point_cloud
    from pat.shapes import sample_mesh
    mesh = _bunny_mesh()
    rng = np.random.default_rng(0)
    clean, _ = sample_mesh(mesh, 2000, rng)
    noisy, _ = noisy_point_cloud(mesh, 2000, np.random.default_rng(0), noise_std=0.02)
    # the noisy points should sit ~noise_std off the clean surface, not on it
    from scipy.spatial import cKDTree
    d, _ = cKDTree(clean).query(noisy)
    assert d.mean() > 0.005                            # genuinely displaced
    assert d.mean() < 0.1                              # but not wildly


def test_training_example_matches_model_and_loss():
    from pat.datasets import make_training_example
    from pat.model import CoeffNet
    from pat.train import pat_loss
    rng = np.random.default_rng(0)
    ex = make_training_example(_bunny_mesh(), rng, n_points=200, k=16, n_query=120,
                               noise_std=0.01)
    for key in ("P", "Nn", "nbr_pos", "nbr_nrm", "q", "phi"):
        assert torch.isfinite(ex[key]).all()
    assert (ex["phi"] > 0).any() and (ex["phi"] < 0).any()   # both sides of the surface
    net = CoeffNet(d_embed=32, n_layers=2, n_heads=4, d_ff=64, supertoroid=True)
    coeffs, _, sq = net(ex["nbr_pos"], ex["nbr_nrm"])
    loss, _, _ = pat_loss(ex["P"], ex["Nn"], coeffs, ex["q"], ex["phi"],
                          supertoroid=(sq[:, 0], sq[:, 1]))
    assert torch.isfinite(loss)
