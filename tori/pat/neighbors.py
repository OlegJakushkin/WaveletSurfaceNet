"""k-nearest-neighbor neighborhoods and the per-point input features (Sec. 4.3).

The network and the analytic least-squares estimator both consume the same
local, rigid-motion-invariant encoding of a point's neighborhood.  Building it
once here keeps train-time and inference-time features identical.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import core


def knn_neighborhoods(points: np.ndarray, k: int) -> np.ndarray:
    """Return, for every point, the indices of itself + its ``k`` nearest neighbors.

    Shape ``(N, k+1)``; column 0 is the point itself.
    """
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    return np.atleast_2d(idx)


def neighborhood_features(nbr_pos: torch.Tensor, nbr_nrm: torch.Tensor):
    """Encode neighborhoods into the ``R^6`` per-point features of Sec. 4.3.

    Args:
        nbr_pos: ``(B, M, 3)`` neighbor positions; column 0 is the central point.
        nbr_nrm: ``(B, M, 3)`` neighbor unit normals.

    Returns:
        feats: ``(B, M, 6)`` features = local-frame coordinates (scaled by the
               median neighbor distance ``sigma_i``) concatenated with local-frame
               normal components.
        sigma: ``(B,)`` per-neighborhood median distance used for scaling.
        s, t:  ``(B, 3)`` tangent basis vectors of the central point.
    """
    center = nbr_pos[:, 0, :]                       # (B,3)
    cnrm = nbr_nrm[:, 0, :]
    s, t = core.local_basis(cnrm)                   # (B,3) each

    rel = nbr_pos - center[:, None, :]              # (B,M,3)
    dist = rel.norm(dim=-1)                          # (B,M)
    # median distance to neighbors (exclude self at column 0)
    sigma = dist[:, 1:].median(dim=1).values.clamp_min(core.EPS)   # (B,)

    inv = (1.0 / sigma)[:, None]                     # (B,1)
    cs = (rel * s[:, None, :]).sum(-1) * inv          # (B,M)
    ct = (rel * t[:, None, :]).sum(-1) * inv
    cn = (rel * cnrm[:, None, :]).sum(-1) * inv
    ns = (nbr_nrm * s[:, None, :]).sum(-1)
    nt = (nbr_nrm * t[:, None, :]).sum(-1)
    nn = (nbr_nrm * cnrm[:, None, :]).sum(-1)
    feats = torch.stack([cs, ct, cn, ns, nt, nn], dim=-1)   # (B,M,6)
    return feats, sigma, s, t


def rescale_coeffs(a_raw: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Undo the neighborhood scaling on predicted coefficients (Sec. 4.3).

    The network sees a neighborhood scaled by ``1/sigma``; its raw outputs
    ``a'_{n,m}`` are mapped back to physical coordinates by::

        a00 <- sigma * a'00
        a01, a10 <- unchanged
        a11, a02, a20 <- sigma^{-1} * a'

    ``a_raw`` is ``(..., 6)`` in PAT order; ``sigma`` is ``(...,)`` (one scale per
    coefficient vector).
    """
    s = sigma
    out = a_raw.clone()
    out[..., core.A00] = a_raw[..., core.A00] * s
    out[..., core.A01] = a_raw[..., core.A01]
    out[..., core.A10] = a_raw[..., core.A10]
    out[..., core.A11] = a_raw[..., core.A11] / s
    out[..., core.A02] = a_raw[..., core.A02] / s
    out[..., core.A20] = a_raw[..., core.A20] / s
    return out
