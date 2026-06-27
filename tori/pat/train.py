"""Training utilities: query sampling, the L1 + eikonal loss (Eq. 27), and a step.

These are imported by both the unit tests (a few-step "does the loss go down?"
check) and the self-contained Colab notebook (full training).  Keeping them here
guarantees the notebook trains exactly the model the rest of the code consumes.
"""

from __future__ import annotations

import numpy as np
import torch

from . import core
from .neighbors import knn_neighborhoods


def sample_queries(shape, n_band, n_cube, bound, rng, band=0.2):
    """Sample query points around a shape and their ground-truth signed distance.

    Mirrors the three-way split of Sec. 4.3 (surface / narrow band / bulk).  Here
    we use an analytic :class:`pat.shapes.Shape`, so ``phi_true`` is exact.

    Returns ``(q (Q,3), phi_true (Q,), grad_true (Q,3))``.
    """
    surf, _ = shape.sample_surface(n_band, rng)
    band_q = surf + rng.normal(scale=band, size=surf.shape)
    cube_q = rng.uniform(-bound, bound, size=(n_cube, 3))
    q = np.concatenate([band_q, cube_q], axis=0)
    phi = shape.sdf(q)
    grad = shape.normal(q)                  # |grad phi| = 1 for a true SDF
    return (q.astype(np.float32), phi.astype(np.float32), grad.astype(np.float32))


def pat_sdf_and_grad(points, normals, coeffs, queries, C=64.0, supertoroid=None):
    """Differentiable blended SDF and its spatial gradient at ``queries``.

    Args:
        points, normals, coeffs: ``(N,3),(N,3),(N,6)`` tensors (require grad via
            ``coeffs`` for training).
        queries: ``(Q,3)`` tensor.
        supertoroid: optional ``(p_tube (N,), p_ring (N,))`` exponent tensors.

    Returns ``(phi (Q,), grad (Q,3))`` where ``grad`` is ``d phi / d x`` obtained by
    autograd (used for the eikonal term).
    """
    q = queries.detach().clone().requires_grad_(True)
    params = core.coeffs_to_torus(points, normals, coeffs)
    sel = {k: params[k].unsqueeze(0) for k in ("center", "axis", "ea", "R", "r", "sign")}
    x = q.unsqueeze(1)
    if supertoroid is not None:
        p_tube, p_ring = supertoroid
        g = core.g_supertoroid(x, sel, p_tube[None], p_ring[None])
    else:
        g = core.g_torus(x, sel)
    phi = core.blend(q, points, g, C=C)
    grad, = torch.autograd.grad(phi.sum(), q, create_graph=True)
    return phi, grad


def pat_loss(points, normals, coeffs, queries, phi_true, C=64.0,
             eikonal_weight=1.0, supertoroid=None):
    """L1 distance + eikonal loss of Eq. 27 (per-neighborhood, aggregated)."""
    phi, grad = pat_sdf_and_grad(points, normals, coeffs, queries, C=C,
                                 supertoroid=supertoroid)
    l_dist = (phi - phi_true).abs().mean()
    l_eik = (1.0 - grad.norm(dim=-1)).abs().mean()
    return l_dist + eikonal_weight * l_eik, l_dist.detach(), l_eik.detach()
