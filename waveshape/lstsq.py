"""Analytic least-squares fit of the six polynomial coefficients (the classic baseline).

The paper *learns* these coefficients because weighted least squares is brittle on
real, noisy point clouds (Fig. 3).  But on clean data with good normals it is an
excellent, training-free coefficient provider -- so we use it both as the default
back-end (the whole PAT pipeline runs with no checkpoint) and as the "naive"
baseline in the comparison against the learned / supertoroid models.

For each neighborhood we express neighbors in the central point's local frame
``(s, t, n)``, then fit the height function ``h(s, t) = sum a_{n,m} s^n t^m`` (degree
2, six terms) by weighted least squares with Gaussian weights.
"""

from __future__ import annotations

import numpy as np
import torch

from . import core
from .neighbors import knn_neighborhoods


def fit_coeffs_lstsq(points: np.ndarray, normals: np.ndarray, k: int = 24,
                     sigma_scale: float = 1.0) -> torch.Tensor:
    """Return the ``(N, 6)`` polynomial coefficients (PAT order) for every point.

    Args:
        points:  ``(N, 3)`` positions.
        normals: ``(N, 3)`` unit normals.
        k:       neighborhood size (number of neighbors, excluding the point).
        sigma_scale: width of the Gaussian neighbor weights, in units of the
                     median neighbor distance.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    N = len(points)
    idx = knn_neighborhoods(points, k)
    nt = torch.from_numpy(normals)
    s_all, t_all = core.local_basis(nt)
    s_all = s_all.numpy()
    t_all = t_all.numpy()

    out = np.zeros((N, 6))
    for i in range(N):
        nb = idx[i]
        rel = points[nb] - points[i]
        s, t, n = s_all[i], t_all[i], normals[i]
        sc = rel @ s
        tc = rel @ t
        hc = rel @ n
        d2 = sc * sc + tc * tc
        med = np.median(np.sqrt(d2[1:])) if len(nb) > 1 else 1.0
        med = max(med, 1e-9) * sigma_scale
        w = np.exp(-d2 / (med * med))
        # design matrix columns in PAT order [1, t, s, s*t, t^2, s^2]
        Mcol = np.stack([np.ones_like(sc), tc, sc, sc * tc, tc * tc, sc * sc], axis=1)
        W = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(Mcol * W, hc * W[:, 0], rcond=None)
        out[i] = coef
    return torch.from_numpy(out).float()
