"""WaveTori — combine the per-point TORI blend with the wavelet denoiser.

The wavelet denoiser (``pat.wavelet``) refines a TSDF.  Here its input is NOT the raw
noisy-cloud TSDF but the **tori blend field**: the per-point (super)torus blend
(``pat.core``) of the *same* cloud.  That blend is already a topology-clean, globally
coherent, continuous surface (no floating islands), so the wavelet only has to remove
the residual surface bumps in the multi-scale domain — an easier, faster-converging
job than denoising raw voxel noise.  In short: **tori provide the coherent base field,
the wavelet provides the multi-scale refinement.**

`tori_blend_tsdf` builds the prior on a grid entirely on the GPU (no cKDTree); the tori
net is used FROZEN (e.g. ``assets/pat_torus.pt``).  `WaveToriReconstruction` exposes the
usual ``.sdf`` / ``.reconstruct`` so it drops into ``eval3d`` / ``render3d``.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from . import core
from .compare import _gpu_knn
from .wavelet import WaveletDenoiser, dwt3d, grid_trilinear, haar_filters_3d, tsdf_from_clouds, _grad3d


@torch.no_grad()
def _tori_params(Ps, Ns, tori_net, k, chunk_nbr, device):
    """Per-point torus params from one kNN scale ``k`` (the points-grouping scope)."""
    B, N, _ = Ps.shape
    k = min(int(k), N - 1)                                   # can't group more neighbors than exist
    idx = _gpu_knn(Ps, k)                                    # (B,N,k+1)
    bi = torch.arange(B, device=device)[:, None, None]
    nbr_pos = Ps[bi, idx].reshape(B * N, k + 1, 3)
    nbr_nrm = Ns[bi, idx].reshape(B * N, k + 1, 3)
    cs = []
    for s in range(0, B * N, chunk_nbr):
        c, _, _ = tori_net(nbr_pos[s:s + chunk_nbr], nbr_nrm[s:s + chunk_nbr])
        cs.append(c)
    coeffs = torch.cat(cs, 0).reshape(B, N, 6)
    p = core.coeffs_to_torus(Ps, Ns, coeffs)
    return (p["center"].unsqueeze(1), p["axis"].unsqueeze(1),
            p["R"].unsqueeze(1), p["r"].unsqueeze(1), p["sign"].unsqueeze(1))


def tori_blend_tsdf(Ps, Ns, tori_net, res=48, trunc=0.1, bound=1.1, device="cpu",
                    k=(24, 64), qchunk=2048, C=64.0, chunk_nbr=4096):
    """Truncated SDF on a ``res^3`` grid from the per-point TORI blend of cloud(s).

    ``Ps``/``Ns`` are ``(B,N,3)`` (or a single ``(N,3)``).  Runs ``tori_net`` (a ``CoeffNet``) on
    each cloud's kNN neighborhoods → per-point tori → the self-normalized blend (Eq. 25), clipped to
    ``±trunc``.  Returns ``(B,1,res,res,res)`` (``<0`` inside).

    **Multi-scale grouping (wider scope):** ``k`` may be a list of neighborhood sizes (default
    ``(24, 64)``).  Each point is fit at every scale and the per-point signed values are averaged
    before the blend, so the prior carries both fine local detail (small ``k``) and wider context
    (large ``k``) instead of a single tiny patch.  Pass a single int for the old single-scale prior.
    """
    Ps = torch.as_tensor(Ps, dtype=torch.float32, device=device)
    Ns = torch.as_tensor(Ns, dtype=torch.float32, device=device)
    if Ps.dim() == 2:
        Ps, Ns = Ps[None], Ns[None]
    B, N, _ = Ps.shape
    ks = [k] if isinstance(k, int) else list(k)
    scales = [_tori_params(Ps, Ns, tori_net, kk, chunk_nbr, device) for kk in ks]

    lin = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    G = grid.shape[0]
    out = torch.empty(B, G, device=device)
    for a in range(0, G, qchunk):
        x = grid[a:a + qchunk].unsqueeze(0).expand(B, -1, -1)        # (B,q,3)
        g = sum(s0 * core.torus_sdf(x.unsqueeze(2), c0, u0, R0, r0)   # avg signed value over scales
                for (c0, u0, R0, r0, s0) in scales) / len(scales)     # (B,q,N)
        phi = core.blend_batched(x, Ps, g, C=C)                      # (B,q)
        out[:, a:a + qchunk] = phi.clamp(-trunc, trunc)
    return out.reshape(B, 1, res, res, res)


@torch.no_grad()
def superellipse_blend_tsdf(Ps, Ns, net, res=64, trunc=0.1, bound=1.1, device="cpu",
                            k=24, qchunk=2048, C=64.0, chunk_nbr=4096,
                            gate_beta=6.0, gate_m=1.0):
    """Per-point **SUPERELLIPSOID** blend prior — NO torus, NO CSG cutout box.

    ``net`` is a *supertoroid* ``CoeffNet`` (predicts the 6 coeffs **and** the two squareness
    logits).  From its per-point frame + curvature (``coeffs_to_torus``) and its learned
    squareness ``(p_tube, p_ring)`` we build, at each cloud point, an **osculating
    superellipsoid** that:

    * is **anchored at the sample** ``P`` (its pole along the outward normal sits exactly on
      ``P``), so the self-normalized blend's zero-set is pinned to the real surface — not to the
      polynomial centre ``q*`` (anchoring there collapses the field to a shrunken shell);
    * is aligned to the principal frame ``[n*, v_min, v_max]`` with half-extents matched to the
      two radii of curvature (``1/|kappa_min|`` along the flat direction, ``r = 1/|kappa_max|``
      along the tight direction and the normal);
    * carries the net's learned squareness exponent ``e`` (sweeps pinched→diamond→circle→
      rounded-square→cube — see :func:`pat.core.superellipsoid_sdf`).

    **One-sided cutoff (matches the supertoroid splats' clip, Sec. pat/splat.py).** Each
    primitive is a *closed* solid, but we only ever want its near-surface patch — a one-sided
    plane/corner, not the whole blob (the back side would corrupt thin parts).  Rather than a
    CSG box-cut (``max(g, g_box)``), which carves spurious internal faces inside an *averaging*
    blend, we make each primitive one-sided in **influence**: its blend weight is gated by a
    smooth front-ness factor ``sigmoid(beta*(proj/h_c + m))`` where ``proj`` is the outward-
    normal projection of the query past ``P``.  Points behind a primitive's tangent plane get
    ~0 weight from it, so it contributes only its front cap.  This is non-destructive (``g`` is
    untouched) and *improves* the prior (sharper sphere/cube than the closed solid).
    """
    Ps = torch.as_tensor(Ps, dtype=torch.float32, device=device)
    Ns = torch.as_tensor(Ns, dtype=torch.float32, device=device)
    if Ps.dim() == 2:
        Ps, Ns = Ps[None], Ns[None]
    B, N, _ = Ps.shape
    idx = _gpu_knn(Ps, k)
    bi = torch.arange(B, device=device)[:, None, None]
    nbr_pos = Ps[bi, idx].reshape(B * N, k + 1, 3)
    nbr_nrm = Ns[bi, idx].reshape(B * N, k + 1, 3)
    cs, ss = [], []
    for s in range(0, B * N, chunk_nbr):
        c, _, sq = net(nbr_pos[s:s + chunk_nbr], nbr_nrm[s:s + chunk_nbr])
        cs.append(c); ss.append(sq)
    coeffs = torch.cat(cs, 0).reshape(B, N, 6)
    assert ss[0] is not None, "superellipse prior needs a SUPERTOROID CoeffNet (predicts squareness)"
    sq = torch.cat(ss, 0).reshape(B, N, 2)                          # (p_tube, p_ring)
    params = core.coeffs_to_torus(Ps, Ns, coeffs)
    ns_ = params["n_star"]; r_ = params["r"]; sgn = params["sign"]; vmin = params["ea"]
    rmin = 1.0 / params["kappa_min"].abs().clamp_min(0.05)          # large radius (flat dir)
    ha = rmin.clamp(0.02, 2.0)                                      # along v_min
    hb = r_.clamp(0.02, 2.0)                                        # along v_max (tight)
    hc = r_.clamp(0.02, 2.0)                                        # along normal
    e = (0.5 * (sq[..., 0] + sq[..., 1])).clamp(1.0, 12.0)          # learned squareness (B,N)
    center = Ps - (sgn * hc).unsqueeze(-1) * ns_                    # pole anchored at P
    on = sgn.unsqueeze(-1) * ns_                                    # outward normal (B,N,3)
    Pon = (Ps * on).sum(-1)                                         # (B,N)  for proj = x.on - P.on
    c0 = center.unsqueeze(1); u0 = ns_.unsqueeze(1); ea0 = vmin.unsqueeze(1)
    h0a = ha.unsqueeze(1); h0b = hb.unsqueeze(1); h0c = hc.unsqueeze(1)
    s0 = sgn.unsqueeze(1); e0 = e.unsqueeze(1)
    lin = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    G = grid.shape[0]
    out = torch.empty(B, G, device=device)
    for a in range(0, G, qchunk):
        x = grid[a:a + qchunk].unsqueeze(0).expand(B, -1, -1)
        g = s0 * core.superellipsoid_sdf(x.unsqueeze(2), c0, u0, ea0, h0a, h0b, h0c, e0)
        proj = torch.einsum("bqd,bnd->bqn", x, on) - Pon.unsqueeze(1)        # (B,q,N)
        gate = torch.sigmoid(gate_beta * (proj / hc.unsqueeze(1) + gate_m))  # one-sided front weight
        out[:, a:a + qchunk] = core.blend_batched(x, Ps, g, C=C, wmul=gate).clamp(-trunc, trunc)
    return out.reshape(B, 1, res, res, res)


def _blend(Ps, Ns, net, res, trunc, bound, device, k, superellipse):
    fn = superellipse_blend_tsdf if superellipse else tori_blend_tsdf
    return fn(Ps, Ns, net, res, trunc, bound, device, k)


class WaveToriReconstruction:
    """Reconstruct an SDF: TORI blend prior → wavelet refiner → mesh.

    Same surface as ``pat.wavelet.WaveletReconstruction`` (``.sdf`` / ``.reconstruct``),
    but the wavelet input is the tori blend field (``tori_blend_tsdf``) rather than the
    raw-cloud TSDF.
    """

    def __init__(self, P, N, tori_net, wave_net, *, res=48, trunc=0.1, bound=1.1,
                 device="cpu", k=24, superellipse=False):
        self.res, self.trunc, self.bound, self.M = res, trunc, bound, ""
        wave_net = wave_net.to(device).eval()
        with torch.no_grad():
            prior = _blend(P, N, tori_net, res, trunc, bound, device, k, superellipse) / trunc
            pred, _, _ = wave_net(prior)
        self.grid = (pred[0, 0].detach().cpu().numpy() * trunc).astype(np.float64)

    def sdf(self, q, neighbors=None):
        # ``neighbors`` accepted (and ignored) so this is a drop-in for the paper renderer's
        # slice lambda (which calls ``.sdf(x, neighbors=...)`` for every model).
        return grid_trilinear(self.grid, q, self.bound, self.trunc)

    def reconstruct(self, res=None, bound=None, neighbors=None, level: float = 0.0):
        # res/bound/neighbors accepted (and ignored) so this is a drop-in for the paper
        # renderer ``pat.render.render_comparison`` alongside PAT; the grid is fixed at build time.
        from skimage import measure
        vol = self.grid
        if not (vol.min() < level < vol.max()):
            return None, None
        v, f, _, _ = measure.marching_cubes(vol, level=level)
        v = v / (self.res - 1) * (2 * self.bound) - self.bound
        return v, f


# --------------------------------------------------------------------------- #
#  Training the wavelet refiner on the TORI-blend prior (full ModelNet40)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def wavetori_val_error(wave, tori_net, P, N, val_idx, *, res, trunc, bound, noise_std,
                       device, k=24, mb=8, seed=0):
    """Held-out denoise error of the refiner: |wave(tori_blend(noisy)) − clean_tsdf| (no grad)."""
    wave.eval()
    gv = torch.Generator().manual_seed(seed)
    tot, cnt = 0.0, 0
    for s in range(0, len(val_idx), mb):
        idx = val_idx[s:s + mb]
        Pc, Nc = P[idx], N[idx]
        clean = tsdf_from_clouds(Pc.to(device), Nc.to(device), res, trunc, bound, device) / trunc
        noise = torch.randn(Pc.shape, generator=gv) * noise_std
        prior = tori_blend_tsdf((Pc + noise).to(device), Nc.to(device), tori_net,
                                res, trunc, bound, device, k) / trunc
        err = (wave(prior)[0] - clean).abs().mean()
        if torch.isfinite(err):
            tot += float(err) * len(idx); cnt += len(idx)
        del clean, prior, err
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    wave.train()
    return tot / cnt if cnt else float("inf")


def train_wavetori(cache, tori_net, *, res=64, trunc=0.1, bound=1.1, epochs=4, batch=16,
                   n_points=None, noise_lo=0.005, noise_hi=0.03, lr=2e-3, lam_wave=0.3,
                   lam_grad=0.05, base=40, k=24, device="cuda", subset=None, n_val=None,
                   clean_every=50, log_every=50, seed=0, wave=None):
    """Train the WaveTori wavelet refiner on a dense ``{P,N,...}`` cache.

    The (FROZEN) ``tori_net`` produces the per-point blend PRIOR (`tori_blend_tsdf`) of a
    DYNAMICALLY-noised cloud (fresh noise, random magnitude in ``[noise_lo, noise_hi]``); the
    refiner learns to map that already-coherent prior to the clean TSDF.  Best-by-val
    selection + the proven NaN/spike guard + periodic ``empty_cache`` (RAM hygiene for the
    long full-ModelNet40 run).  Returns ``(wave, history)``.
    """
    P, N = cache["P"], cache["N"]
    A = P.shape[0] if subset is None else min(subset, P.shape[0])
    dense = P.shape[1]
    tori_net = tori_net.to(device).eval()
    for p in tori_net.parameters():
        p.requires_grad_(False)
    wave = wave or WaveletDenoiser(base=base).to(device)
    opt = torch.optim.Adam(wave.parameters(), lr=lr)
    haar = haar_filters_3d(device)

    g0 = torch.Generator().manual_seed(seed)
    perm = torch.randperm(A, generator=g0)
    if n_val is None:
        n_val = min(256, max(1, A // 5))
    n_val = max(0, min(int(n_val), A - 1)) if A > 1 else 0
    val_idx, train_pool = perm[:n_val], perm[n_val:]

    g = torch.Generator().manual_seed(seed + 1)
    hist, loss_ema = [], None
    best_val, best_ep, best_state = float("inf"), -1, None
    wave.train()
    for ep in range(epochs):
        tr = train_pool[torch.randperm(len(train_pool), generator=g)]
        run, nb, skipped = 0.0, 0, 0
        for s in range(0, len(tr), batch):
            idx = tr[s:s + batch]
            Pc, Nc = P[idx], N[idx]
            if n_points is not None and n_points < dense:
                sub = torch.argsort(torch.rand(len(idx), dense, generator=g), 1)[:, :n_points]
                bi = torch.arange(len(idx))[:, None]
                Pc, Nc = Pc[bi, sub], Nc[bi, sub]
            Pc, Nc = Pc.to(device), Nc.to(device)
            with torch.no_grad():
                clean = tsdf_from_clouds(Pc, Nc, res, trunc, bound, device) / trunc
                target_c = dwt3d(clean, haar)
                ns = torch.empty(Pc.shape[0], 1, 1, device=device).uniform_(noise_lo, noise_hi)
                prior = tori_blend_tsdf(Pc + torch.randn(Pc.shape, device=device) * ns,
                                        Nc, tori_net, res, trunc, bound, device, k) / trunc
            pred, _, c_pred = wave(prior)
            l_t = F.smooth_l1_loss(pred, clean, beta=0.1)
            l_w = (c_pred - target_c).abs().mean()
            gp, gc = _grad3d(pred), _grad3d(clean)
            l_g = sum((a - b).abs().mean() for a, b in zip(gp, gc)) / 3.0
            loss = l_t + lam_wave * l_w + lam_grad * l_g
            opt.zero_grad()
            v = float(loss.detach()) if torch.isfinite(loss) else float("inf")
            spike = (loss_ema is not None) and (v > 3.0 * loss_ema)
            if torch.isfinite(loss) and not spike:
                loss.backward()
                for p in wave.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, 0., 0., 0.)
                torch.nn.utils.clip_grad_norm_(wave.parameters(), 1.0)
                opt.step()
                loss_ema = v if loss_ema is None else 0.98 * loss_ema + 0.02 * v
                run += v; nb += 1
            else:
                skipped += 1
            del clean, target_c, prior, pred, c_pred, loss
            if str(device).startswith("cuda") and clean_every and nb % clean_every == 0:
                torch.cuda.empty_cache()
            if log_every and nb > 0 and (nb + skipped) % log_every == 0:
                print(f"  wavetori ep{ep} {min(s + batch, len(tr))}/{len(tr)} "
                      f"loss {run / nb:.4f}", flush=True)
        val = (wavetori_val_error(wave, tori_net, P, N, val_idx, res=res, trunc=trunc, bound=bound,
                                  noise_std=0.5 * (noise_lo + noise_hi), device=device, k=k, seed=seed)
               if n_val else float("nan"))
        hist.append({"epoch": ep, "loss": run / max(nb, 1), "val": val, "skipped": skipped})
        if n_val and val < best_val:
            best_val, best_ep = val, ep
            best_state = copy.deepcopy({kk: vv.detach().cpu() for kk, vv in wave.state_dict().items()})
        print(f"wavetori epoch {ep}: loss {run / max(nb, 1):.4f} | val {val:.4f} | "
              f"skipped {skipped}", flush=True)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if best_state is not None:
        wave.load_state_dict(best_state)
        print(f"wavetori: selected BEST epoch {best_ep} (val {best_val:.4f})", flush=True)
    return wave, hist
