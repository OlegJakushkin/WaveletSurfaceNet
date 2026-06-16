"""High-level *Points as Tori* pipeline (numpy-friendly, wraps the torch core).

``PAT`` turns a point cloud with normals into a callable signed-distance function,
exactly as in Algorithm 1 of the paper:

    1. **Precompute** -- fit one torus (or, in our extension, one supertoroid) per
       point, from polynomial coefficients produced either by least squares
       (training-free) or by the learned network.
    2. **Inference** -- evaluate the self-normalized blend of Eq. 25 at any query.

The same object reconstructs the zero level set with marching cubes.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import core
from .lstsq import fit_coeffs_lstsq
from .neighbors import knn_neighborhoods, neighborhood_features, rescale_coeffs


def _to_t(x, device="cpu"):
    if torch.is_tensor(x):                       # e.g. coeffs straight off a GPU model
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class PAT:
    """A fitted SDF over a point cloud.

    Args:
        points, normals: ``(N, 3)`` arrays.
        coeffs:  optional precomputed ``(N, 6)`` coefficients (overrides provider).
        model:   optional trained :class:`pat.model.CoeffNet`; if given, used to
                 predict coefficients (and squareness, if it is a supertoroid model).
        k:       neighborhood size for the coefficient provider.
        supertoroid: if True, fit supertoroids instead of tori.
        p_tube, p_ring: super-ellipse exponents (scalars or ``(N,)``); only used
                 when ``supertoroid`` is True and no model squareness is available.
                 ``2.0`` reproduces an ordinary torus.
        C:       blending precision constant (Eq. 26).
    """

    def __init__(self, points, normals, *, coeffs=None, model=None, k=24,
                 supertoroid=False, p_tube=2.0, p_ring=2.0, C=64.0, device="cpu"):
        self.device = device
        self.points = _to_t(points, device)
        self.normals = _to_t(normals, device)
        self.normals = self.normals / self.normals.norm(dim=1, keepdim=True).clamp_min(core.EPS)
        self.N = self.points.shape[0]
        self.C = float(C)
        self.supertoroid = supertoroid
        self._tree = cKDTree(self.points.cpu().numpy())

        p_tube_t = torch.full((self.N,), float(p_tube)) if np.isscalar(p_tube) else _to_t(p_tube)
        p_ring_t = torch.full((self.N,), float(p_ring)) if np.isscalar(p_ring) else _to_t(p_ring)

        if model is not None:
            coeffs, sq = self._run_model(model, k)
            if sq is not None:                       # model predicts squareness too
                p_tube_t, p_ring_t = sq[:, 0], sq[:, 1]
                self.supertoroid = True
        if coeffs is None:
            coeffs = fit_coeffs_lstsq(points, normals, k=k)
        self.coeffs = _to_t(coeffs, device)

        self.params = core.coeffs_to_torus(self.points, self.normals, self.coeffs)
        self.p_tube = p_tube_t.to(device)
        self.p_ring = p_ring_t.to(device)

    # ------------------------------------------------------------------ #
    def _run_model(self, model, k):
        idx = knn_neighborhoods(self.points.cpu().numpy(), k)
        nb = torch.as_tensor(idx, dtype=torch.long, device=self.device)   # index on-device
        nbr_pos = self.points[nb]
        nbr_nrm = self.normals[nb]
        model.eval()
        with torch.no_grad():
            coeffs, _sigma, sq = model(nbr_pos, nbr_nrm)
        return coeffs, sq

    # ------------------------------------------------------------------ #
    def _g_vals(self, xq, pidx=None):
        """Per-point signed values ``g_i(x_q)`` as ``(Q, M)``.

        ``pidx`` optionally selects, per query, the column indices of the points
        to sum over (``(Q, M)`` long tensor for kNN-accelerated evaluation);
        otherwise all ``N`` points are used.
        """
        if pidx is None:
            sel = {k: v.unsqueeze(0) for k, v in self.params.items()
                   if k in ("center", "axis", "ea", "R", "r", "sign")}
            x = xq.unsqueeze(1)                      # (Q,1,3)
            if self.supertoroid:
                return core.g_supertoroid(x, sel, self.p_tube[None], self.p_ring[None])
            return core.g_torus(x, sel)
        sel = {k: self.params[k][pidx] for k in ("center", "axis", "ea", "R", "r", "sign")}
        x = xq.unsqueeze(1)                          # (Q,1,3)
        if self.supertoroid:
            return core.g_supertoroid(x, sel, self.p_tube[pidx], self.p_ring[pidx])
        return core.g_torus(x, sel)

    def _points_for(self, pidx):
        return self.points if pidx is None else self.points[pidx]

    # ------------------------------------------------------------------ #
    def sdf(self, x, *, neighbors=None, chunk=4096):
        """Evaluate the blended SDF at query points ``x`` ``(Q, 3)`` -> ``(Q,)``.

        Args:
            neighbors: if an int ``m``, restrict each query's blend to its ``m``
                       nearest cloud points (the paper's kNN acceleration);
                       if None, sum over all points (exact, used for small clouds).
            chunk:     query batch size to bound memory.
        """
        x = np.asarray(x, dtype=np.float32)
        out = np.empty(len(x), dtype=np.float32)
        for a in range(0, len(x), chunk):
            xb = _to_t(x[a:a + chunk], self.device)
            if neighbors is None:
                g = self._g_vals(xb)
                phi = core.blend(xb, self.points, g, C=self.C)
            else:
                m = min(neighbors, self.N)
                _, pidx = self._tree.query(x[a:a + chunk], k=m)
                pidx = torch.as_tensor(np.atleast_2d(pidx), dtype=torch.long, device=self.device)
                g = self._g_vals(xb, pidx)
                pts = self.points[pidx]              # (Q,m,3)
                d = (xb.unsqueeze(1) - pts).norm(dim=-1)
                sigma = 0.5 * d.max(dim=1, keepdim=True).values
                lam = self.C / (sigma + core.EPS)
                dmin = d.min(dim=1, keepdim=True).values
                w = torch.exp(-lam * (d - dmin))
                phi = (w * g).sum(1) / w.sum(1).clamp_min(core.EPS)
            out[a:a + chunk] = phi.detach().cpu().numpy()
        return out

    # ------------------------------------------------------------------ #
    def reconstruct(self, res=64, bound=1.2, neighbors=64, level=0.0):
        """Extract the zero level set with marching cubes -> ``(verts, faces)``.

        Returns vertices in world coordinates and triangle faces.  Requires
        :mod:`scikit-image`.
        """
        from skimage import measure

        lin = np.linspace(-bound, bound, res)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        vol = self.sdf(grid, neighbors=neighbors).reshape(res, res, res)
        verts, faces, _, _ = measure.marching_cubes(vol, level=level)
        verts = verts / (res - 1) * (2 * bound) - bound   # grid -> world
        return verts, faces
