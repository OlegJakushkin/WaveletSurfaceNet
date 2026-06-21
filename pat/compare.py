"""Head-to-head harness: the original tori network vs. the wavelet denoiser.

This module makes the two reconstruction models trainable and comparable **on the
same data, in the same notebook**, so the comparison is fair:

* :func:`train_tori_cache` trains the paper's :class:`pat.model.CoeffNet` (the
  "original tori network") on a dense ``{P, N, Q, PHI}`` mesh cache with the
  paper's L1 + eikonal blend loss (Eq. 27) — a compact, **device-agnostic** copy
  of ``train_gpu.py``'s batched trainer (that script hard-requires CUDA on import;
  this one runs on CPU for tests and on GPU in Colab).
* :class:`pat.wavelet.WaveletDenoiser` trains via :func:`pat.wavelet.train_wavelet`.
* :func:`head_to_head` reconstructs a held-out mesh with **both** models from the
  *same* noisy cloud and reports voxel-free IoU\\* (:mod:`pat.eval3d`), symmetric
  Chamfer distance, and a 3-panel ground-truth / tori / wavelet render.

The whole point is an apples-to-apples comparison: identical meshes, identical
noisy input clouds, identical metrics.
"""

from __future__ import annotations

import numpy as np
import torch

from . import core
from .model import CoeffNet
from .pat import PAT


# --------------------------------------------------------------------------- #
#  Device-agnostic batched tori trainer over a {P, N, Q, PHI} cache
# --------------------------------------------------------------------------- #
def _gpu_knn(pts, k):
    """k-NN indices (incl. self) for a batch of clouds ``pts (B,N,3)`` -> ``(B,N,k+1)``."""
    d = torch.cdist(pts, pts)
    return d.topk(k + 1, dim=2, largest=False).indices


def _batched_coeffs(net, pts, nrm, k, chunk=2048):
    """Run ``net`` on every neighborhood of a batch of clouds -> coeffs ``(B,N,6)``, sq."""
    B, N, _ = pts.shape
    idx = _gpu_knn(pts, k)
    bi = torch.arange(B, device=pts.device)[:, None, None]
    nbr_pos = pts[bi, idx].reshape(B * N, k + 1, 3)
    nbr_nrm = nrm[bi, idx].reshape(B * N, k + 1, 3)
    cs, ss = [], []
    for s in range(0, B * N, chunk):
        c, _, sq = net(nbr_pos[s:s + chunk], nbr_nrm[s:s + chunk])
        cs.append(c); ss.append(sq)
    coeffs = torch.cat(cs, 0).reshape(B, N, 6)
    sq = torch.cat(ss, 0).reshape(B, N, 2) if ss[0] is not None else None
    return coeffs, sq


def tori_blend_loss(net, pts, nrm, q, phi_true, k, C=64.0, eik=0.1, chunk=2048,
                    square_reg=0.0):
    """L1 + eikonal blend loss over a batch of clouds (Eq. 27), all on ``pts.device``.

    A device-agnostic copy of ``train_gpu.batched_loss``: supports the plain torus
    and (when ``net.supertoroid``) the supertoroid, with an optional ``square_reg``
    pull toward ``p = 2``.  Returns ``(loss, l_dist, l_eik)`` (last two detached).
    """
    coeffs, sq = _batched_coeffs(net, pts, nrm, k, chunk=chunk)
    params = core.coeffs_to_torus(pts, nrm, coeffs)
    q = q.detach().clone().requires_grad_(True)
    x = q.unsqueeze(2)                                         # (B, Q, 1, 3)
    c = params["center"].unsqueeze(1); u = params["axis"].unsqueeze(1)
    R = params["R"].unsqueeze(1); r = params["r"].unsqueeze(1)
    sign = params["sign"].unsqueeze(1)
    if sq is not None:
        ea = params["ea"].unsqueeze(1)
        sdf = core.supertoroid_sdf(x, c, u, ea, R, r,
                                   sq[..., 0].unsqueeze(1), sq[..., 1].unsqueeze(1))
    else:
        sdf = core.torus_sdf(x, c, u, R, r)
    phi = core.blend_batched(q, pts, sign * sdf, C=C)
    grad, = torch.autograd.grad(phi.sum(), q, create_graph=True)
    grad = torch.nan_to_num(grad)
    l_dist = (phi - phi_true).abs().mean()
    l_eik = (1.0 - grad.norm(dim=-1)).abs().mean()
    loss = l_dist + eik * l_eik
    if sq is not None and square_reg > 0.0:
        loss = loss + square_reg * ((sq - 2.0) ** 2).mean()
    return loss, l_dist.detach(), l_eik.detach()


def train_tori_cache(cache, *, k=16, epochs=4, batch=8, n_points=512, noise_std=0.015,
                     frac_noisy=1.0, lr=8e-4, C=64.0, eik=0.1, device="cpu",
                     subset=None, supertoroid=False, d_embed=128, n_layers=8,
                     dropout=0.0, log_every=50, seed=0, net=None):
    """Train the paper's :class:`CoeffNet` on a dense ``{P, N, Q, PHI}`` cache.

    Each step draws a random point subset of each cached cloud, adds fresh Gaussian
    noise to a fraction ``frac_noisy`` of its points (the noise-robustness regime of
    Sec. 5; the GT signed distance ``PHI`` is always to the *clean* surface), and
    minimizes the L1 + eikonal blend loss.  Returns ``(net, history)``.

    Mirrors ``train_gpu.py``'s regime (so it is genuinely "the original tori
    network") but is device-agnostic and trains from the cache only — no analytic
    assets — so it sees exactly the ModelNet40 meshes the wavelet net sees.
    """
    P, Nn, Q, PHI = cache["P"], cache["N"], cache["Q"], cache["PHI"]
    A = P.shape[0] if subset is None else min(subset, P.shape[0])
    dense = P.shape[1]
    net = net or CoeffNet(d_embed=d_embed, n_layers=n_layers,
                          supertoroid=supertoroid, dropout=dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    hist = []
    net.train()
    for ep in range(epochs):
        order = torch.randperm(A, generator=g).tolist()
        run, nb = 0.0, 0
        for s in range(0, A, batch):
            idx = torch.as_tensor(order[s:s + batch])
            sub = torch.argsort(torch.rand(len(idx), dense, generator=g), 1)[:, :n_points]
            bi = torch.arange(len(idx))[:, None]
            pts = P[idx][bi, sub].to(device)
            nrm = Nn[idx][bi, sub].to(device)
            if noise_std > 0:
                m = (torch.rand(pts.shape[:2], device=device) < frac_noisy).unsqueeze(-1)
                pts = pts + m * torch.randn_like(pts) * noise_std
            q = Q[idx].to(device); phi_true = PHI[idx].to(device)
            loss, ld, le = tori_blend_loss(net, pts, nrm, q, phi_true, k, C=C, eik=eik)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()); nb += 1
            if log_every and nb % log_every == 0:
                print(f"  tori ep{ep} {min(s + batch, A)}/{A} loss {run / nb:.4f}",
                      flush=True)
        hist.append({"epoch": ep, "loss": run / max(nb, 1)})
        print(f"tori epoch {ep}: loss {run / max(nb, 1):.4f}", flush=True)
    return net, hist


# --------------------------------------------------------------------------- #
#  Chamfer distance between two meshes
# --------------------------------------------------------------------------- #
def _surf_pts(verts, faces, n):
    import trimesh
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=False)
    return np.asarray(m.sample(n))                    # area-weighted surface sample


def chamfer(vA, fA, vB, fB, n=4000, seed=0):
    """Symmetric Chamfer distance (mean nearest-point L2, both directions).

    Samples ``n`` points from each surface; returns ``mean_a min_b ||a-b|| +
    mean_b min_a ||a-b||``.  Lower is a closer reconstruction.  Returns ``nan`` if
    either mesh is empty.
    """
    from scipy.spatial import cKDTree

    if vA is None or vB is None or len(fA) == 0 or len(fB) == 0:
        return float("nan")
    a = _surf_pts(vA, fA, n)
    b = _surf_pts(vB, fB, n)
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float(da.mean() + db.mean())


# --------------------------------------------------------------------------- #
#  One-mesh head-to-head
# --------------------------------------------------------------------------- #
class _SdfAdapter:
    """Wrap an ``sdf(q)`` callable as an object with a ``.sdf`` method (for eval3d)."""

    def __init__(self, fn):
        self._fn = fn

    def sdf(self, q):
        return self._fn(q)


def head_to_head(mesh, tori_net, wave_net, *, n_cloud=1536, noise=0.01, k=16,
                 res_recon=96, res_wave=32, trunc=0.1, bound=1.1, n_metric=40000,
                 device="cpu", render_path=None, name=""):
    """Reconstruct one (already unit-normalized) mesh with both models and compare.

    Both models receive the **same** noisy surface cloud.  Returns a dict::

        {"name", "tori": {iou, vol_err, chamfer, ...}, "wavelet": {...}}

    and, if ``render_path`` is given, writes a 3-panel *ground-truth / tori /
    wavelet* render.  Quality is the **voxel-free** Monte-Carlo IoU\\* of
    :func:`pat.eval3d.proper_metrics` plus Chamfer distance to the GT mesh surface.
    """
    from . import eval3d as E, render3d as R3
    from .wavelet import WaveletReconstruction

    P, N = E.sample_cloud(mesh, n=n_cloud, noise=noise, seed=0)
    gt = E.mesh_gt(mesh)
    gv, gf = gt.mesh.vertices, gt.mesh.faces

    # --- original tori network ---
    pat = PAT(P, N, model=tori_net, k=k, C=64.0, device=device)
    m_t = E.proper_metrics(gt, _SdfAdapter(lambda q: pat.sdf(q, neighbors=64)), n=n_metric)
    vt, ft = pat.reconstruct(res=res_recon, bound=bound, neighbors=64)
    m_t["chamfer"] = chamfer(gv, gf, vt, ft)

    # --- wavelet denoiser ---
    wr = WaveletReconstruction(P, N, wave_net, res=res_wave, trunc=trunc,
                               bound=bound, device=device)
    m_w = E.proper_metrics(gt, wr, n=n_metric)
    vw, fw = wr.reconstruct()
    m_w["chamfer"] = chamfer(gv, gf, vw, fw)

    if render_path is not None:
        panels = [("ground truth", gv, gf),
                  (f"tori  IoU* {m_t['iou']:.2f}", vt, ft),
                  (f"wavelet  IoU* {m_w['iou']:.2f}", vw, fw)]
        try:
            R3.render_meshes(panels, render_path, title=name)
        except Exception as exc:                     # rendering must never fail the eval
            print("render skip", name, exc)

    return {"name": name, "tori": m_t, "wavelet": m_w}
