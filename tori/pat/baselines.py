"""Baseline point-cloud distance estimators used in the comparison (Sec. 5.2).

* **SSPD** -- smoothed signed planar distance (Eq. 29), the self-normalized blend
  of the naive tangent-plane function ``g(x, p_i) = <x - p_i, n_i>``.  This is the
  ablation that PAT improves on: replace the planar ``g`` with a torus/supertoroid.
* **SHC** -- signed Hopf-Cole / signed LogSumExp distance (Eq. 28), an instance of
  the non-self-normalized convolutional formula.

Both use the same per-query shift/screening machinery as :func:`pat.core.blend`.
"""

from __future__ import annotations

import numpy as np
import torch

from . import core


def _setup(x, points, C):
    d = torch.cdist(x, points)
    sigma = 0.5 * d.max(dim=1, keepdim=True).values
    lam = C / (sigma + core.EPS)
    dmin = d.min(dim=1, keepdim=True).values
    return d, sigma, lam, dmin


def sspd(x, points, normals, C=64.0):
    """Smoothed signed planar distance (Eq. 29). ``x (Q,3)`` -> ``(Q,)``."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    p = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    n = torch.as_tensor(np.asarray(normals), dtype=torch.float32)
    d, sigma, lam, dmin = _setup(x, p, C)
    planar = ((x.unsqueeze(1) - p.unsqueeze(0)) * n.unsqueeze(0)).sum(-1)   # (Q,N)
    w = torch.exp(-lam * (d - dmin))
    phi = (w * planar).sum(1) / w.sum(1).clamp_min(core.EPS)
    return phi.numpy()


def signed_hopf_cole(x, points, normals, C=64.0):
    """Signed Hopf-Cole / signed LogSumExp distance (Eq. 28). ``x (Q,3)`` -> ``(Q,)``."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    p = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    n = torch.as_tensor(np.asarray(normals), dtype=torch.float32)
    d, sigma, lam, dmin = _setup(x, p, C)
    diff = x.unsqueeze(1) - p.unsqueeze(0)                  # (Q,N,3)
    dot = (diff * n.unsqueeze(0)).sum(-1)                   # <x-p, n>
    # Yukawa double-layer kernel (Eq. 13/28), evaluated with a per-query exponent
    # shift for numerical stability.
    kern = (lam * d + 1.0) * dot / (2 * np.pi * d.clamp_min(core.EPS) ** 3)
    w_shift = torch.exp(-lam * (d - dmin))                 # = exp(lam*dmin) * raw_w
    wsum = (kern * w_shift).sum(1)
    # In Eq. 28 the +sigma_x outside the log exactly cancels the -sigma_x inside the
    # exponent, leaving phi = -sign(w)/lam * log|raw_w|.  With the dmin shift,
    # log|raw_w| = log|wsum| - lam*dmin, so:
    lam0 = lam.squeeze(1)
    log_raw = torch.log(wsum.abs().clamp_min(core.EPS)) - lam0 * dmin.squeeze(1)
    phi = torch.sign(wsum) * (-1.0 / lam0 * log_raw)
    return phi.numpy()


def grid_error(sdf_values, true_values):
    """Mean absolute error over a grid (the paper's ``epsilon(phi)``, Sec. 5.2)."""
    return float(np.mean(np.abs(np.asarray(sdf_values) - np.asarray(true_values))))
