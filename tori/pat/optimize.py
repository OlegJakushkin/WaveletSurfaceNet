"""Per-cloud optimization and the torus-vs-supertoroid comparison (item 3).

The paper *learns* coefficients once, offline, for plain tori.  Here we instead
directly *optimize* the per-point parameters for a single cloud against
ground-truth distance, for two models:

* **torus**       -- optimize only the six polynomial coefficients (squareness
                     fixed at ``p = 2``); this is the paper's primitive.
* **supertoroid** -- additionally optimize two per-point squareness exponents,
                     so each tube/ring cross-section can become a rounded square.

Because the supertoroid contains the torus at ``p = 2``, it can only match or beat
it.  The payoff shows up on shapes with flat or boxy regions (e.g. a rounded box),
where a circular tube cannot sit flush against a face.  This module is the
single-cloud, gradient-based analogue of the network training in the notebook.
"""

from __future__ import annotations

import numpy as np
import torch

from . import core
from .lstsq import fit_coeffs_lstsq
from .pat import PAT
from .train import pat_loss, sample_queries


def optimize_cloud(points, normals, shape, *, supertoroid=False, steps=300,
                   lr=5e-3, k=24, C=64.0, eikonal_weight=0.1, n_query=2000,
                   bound=1.2, seed=0, square_reg=0.02, warm_coeffs=None,
                   freeze_coeffs=False, verbose=False):
    """Optimize PAT parameters for one cloud against a shape's exact SDF.

    ``square_reg`` penalizes squareness exponents that deviate from ``p = 2`` (a
    torus), so the tube/ring only becomes boxy when the data genuinely benefits.
    This keeps the supertoroid a safe generalization of the torus -- it falls back
    to a torus on round shapes instead of over-fitting the extra freedom.

    Returns ``(pat, history)`` where ``pat`` is a fitted :class:`PAT` and
    ``history`` is the list of total-loss values.
    """
    rng = np.random.default_rng(seed)
    pts = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    nrm = torch.as_tensor(np.asarray(normals), dtype=torch.float32)
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(core.EPS)
    N = pts.shape[0]

    if warm_coeffs is not None:
        coeffs = torch.as_tensor(np.asarray(warm_coeffs), dtype=torch.float32).clone()
    else:
        coeffs = fit_coeffs_lstsq(points, normals, k=k).clone()
    coeffs.requires_grad_(not freeze_coeffs)
    paramset = []
    groups = []
    if not freeze_coeffs:
        paramset.append(coeffs)
        groups.append({"params": [coeffs], "lr": lr})
    raw_p = None
    if supertoroid:
        raw_p = torch.full((N, 2), core.P2_RAW, requires_grad=True)
        paramset.append(raw_p)
        # The squareness lives in softplus space and needs a larger step than the
        # coefficients to actually specialize away from the torus (p = 2).
        groups.append({"params": [raw_p], "lr": lr * 10.0})

    opt = torch.optim.Adam(groups)
    history = []
    for step in range(steps):
        q, phi_true, _ = sample_queries(shape, n_query // 2, n_query - n_query // 2,
                                        bound, rng)
        qt = torch.as_tensor(q)
        pt = torch.as_tensor(phi_true)
        st = None
        if supertoroid:
            p = core.raw_to_p(raw_p)
            st = (p[:, 0], p[:, 1])
        loss, ld, le = pat_loss(pts, nrm, coeffs, qt, pt, C=C,
                                eikonal_weight=eikonal_weight, supertoroid=st)
        if supertoroid and square_reg > 0:
            loss = loss + square_reg * ((raw_p - core.P2_RAW) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(paramset, 1.0)
        opt.step()
        history.append(float(loss.detach()))
        if verbose and step % max(1, steps // 10) == 0:
            print(f"  step {step:4d}  loss {float(loss):.4f}  "
                  f"dist {float(ld):.4f}  eik {float(le):.4f}")

    p_tube = p_ring = 2.0
    if supertoroid:
        with torch.no_grad():
            p = core.raw_to_p(raw_p)
        p_tube, p_ring = p[:, 0].numpy(), p[:, 1].numpy()
    pat = PAT(points, normals, coeffs=coeffs.detach().numpy(),
              supertoroid=supertoroid, p_tube=p_tube, p_ring=p_ring, C=C)
    return pat, history


def grid_eval(shape, bound=1.0, res=48):
    """Regular grid points and exact SDF, for the paper's mean-abs-error metric."""
    lin = np.linspace(-bound, bound, res)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    return grid, shape.sdf(grid)


def fit_pair(points, normals, shape, *, steps=300, seed=0, verbose=False, **kw):
    """Fit a torus and a supertoroid PAT to the *same* given cloud.

    The torus is optimized first; the supertoroid is warm-started from the torus's
    optimized coefficients with the coefficients **frozen**, so it begins exactly at
    the torus optimum and only specializes the cross-section squareness.  Returns
    ``(pat_torus, pat_supertoroid)``.
    """
    pat_t, _ = optimize_cloud(points, normals, shape, supertoroid=False, steps=steps,
                              seed=seed, verbose=verbose, **kw)
    pat_s, _ = optimize_cloud(points, normals, shape, supertoroid=True, steps=steps,
                              seed=seed, warm_coeffs=pat_t.coeffs.numpy(),
                              freeze_coeffs=True, verbose=verbose, **kw)
    return pat_t, pat_s


def compare_torus_vs_supertoroid(shape, *, n_points=512, steps=300, grid_res=40,
                                 grid_bound=1.0, seed=0, verbose=False, **kw):
    """Fit both models to a sampled cloud and report grid SDF error for each.

    Returns a dict with ``err_torus``, ``err_supertoroid``, the two fitted ``PAT``
    objects, and the squareness statistics of the supertoroid fit.
    """
    rng = np.random.default_rng(seed)
    pts, nrm = shape.sample_surface(n_points, rng)
    pat_t, pat_s = fit_pair(pts, nrm, shape, steps=steps, seed=seed,
                            verbose=verbose, **kw)

    grid, gt = grid_eval(shape, bound=grid_bound, res=grid_res)
    m = min(64, n_points)
    err_t = float(np.mean(np.abs(pat_t.sdf(grid, neighbors=m) - gt)))
    err_s = float(np.mean(np.abs(pat_s.sdf(grid, neighbors=m) - gt)))
    return {
        "err_torus": err_t,
        "err_supertoroid": err_s,
        "improvement": (err_t - err_s) / err_t if err_t > 0 else 0.0,
        "pat_torus": pat_t,
        "pat_supertoroid": pat_s,
        "p_tube_mean": float(np.mean(pat_s.p_tube.numpy())),
        "p_ring_mean": float(np.mean(pat_s.p_ring.numpy())),
        "points": pts, "normals": nrm,
    }
