"""Batched Stage-A teacher: optimize MANY meshes' supertoroid-splat fields in parallel on one GPU.

The single-mesh :func:`pat.teacher.fit_teacher` underuses a GPU (one mesh's ~2k queries x ~60 splats is a
tiny kernel).  This module stacks ``B`` meshes into a leading batch axis -- the splat axis ``M`` is already
vectorized -- so the GPU processes ``B x M`` splats at once (parallel "on both splats count and meshes
level").  Pruning is **speculative**: instead of 14 sequential single-drop+refit rounds per mesh, it drops
many low-ownership splats per refit over a short keep-schedule and keeps, per mesh, the SMALLEST field that
still met ``md_target`` (independently per mesh, so easy and hard meshes share the same batched refits).

Correctness is identical to the single-mesh teacher: occupancy = the self-normalized blend sign, GT built
hole-respecting from each mesh's cached P+N.  Each mesh's result is converted back to a plain
:class:`pat.splat.SuperToroidSplats` and cached with the exact same artifact format.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import core
from . import splat as _S
from . import teacher as _T
from .splat import SuperToroidSplats, ROW_W, _ROW_SLICES, farthest_point_sample, _init_from_coeffs

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except Exception:                                   # pragma: no cover
    _HAVE_TQDM = False

EPS = 1e-9


# --------------------------------------------------------------------------- #
#  Batched supertoroid-splat field  (B meshes, M_max splats each, alive mask)
# --------------------------------------------------------------------------- #
class BatchSplats:
    """``B`` independent supertoroid+box splat fields sharing the splat-count axis ``M`` (padded with an
    ``alive`` mask for per-mesh variable counts).  Mirrors :class:`pat.splat.SuperToroidSplats` math with
    a leading batch dim; all parameter tensors are leaf tensors with ``requires_grad`` for one shared Adam.
    """

    _PNAMES = ("center", "raw_u", "raw_ea", "log_R", "log_r", "raw_pt", "raw_pr",
               "log_sigma", "log_b", "box_offset")

    def __init__(self, rows, alive, p_max=6.0, device="cpu"):
        rows = rows.to(device, torch.float32)                       # (B, M, ROW_W)
        self.p_max = float(p_max)
        self.device = device
        self.alive = alive.to(device)                               # (B, M) bool
        self.sign = torch.sign(rows[..., _ROW_SLICES["sign"]].squeeze(-1)).clamp_min(-1.0).to(device)
        self.params = {}
        for name in self._PNAMES:
            col = rows[..., _ROW_SLICES[name]].clone()
            if col.shape[-1] == 1 and name in ("log_R", "log_r", "raw_pt", "raw_pr"):
                col = col.squeeze(-1)                                # (B,M)
            self.params[name] = col.requires_grad_(True)
        for k, v in self.params.items():
            setattr(self, k, v)

    @property
    def B(self): return self.alive.shape[0]

    @property
    def M(self): return self.alive.shape[1]

    def parameters(self):
        return list(self.params.values())

    def _frame(self):
        u = F.normalize(self.raw_u, dim=-1, eps=EPS)
        ea = self.raw_ea - (self.raw_ea * u).sum(-1, keepdim=True) * u
        ea = F.normalize(ea, dim=-1, eps=EPS)
        eb = torch.cross(u, ea, dim=-1)
        return u, ea, eb

    def _gw(self, q):
        """Per-splat solid value ``g`` and window weight ``w`` at ``q (B,Q,3)`` -> each ``(B,Q,M)``."""
        u, ea, eb = self._frame()
        R = self.log_R.clamp(-5, 2).exp(); r = self.log_r.clamp(-5, 2).exp()
        pt = core.raw_to_p(self.raw_pt, p_max=self.p_max); pr = core.raw_to_p(self.raw_pr, p_max=self.p_max)
        b = self.log_b.clamp(-4, 1.5).exp()
        sig = self.log_sigma.clamp(-5, 1).exp().clamp_min(1e-3)
        boxc = self.center + self.box_offset.clamp(-1, 1)
        qE = q[:, :, None, :]                                       # (B,Q,1,3)
        cE = self.center[:, None]; uE = u[:, None]; eaE = ea[:, None]; ebE = eb[:, None]
        g_s = self.sign[:, None] * core.supertoroid_sdf(
            qE, cE, uE, eaE, R[:, None], r[:, None], pt[:, None], pr[:, None])      # (B,Q,M)
        relb = qE - boxc[:, None]
        lx = (relb * uE).sum(-1); ly = (relb * eaE).sum(-1); lz = (relb * ebE).sum(-1)
        qb = torch.stack([lx, ly, lz], -1).abs() - b[:, None]
        g_box = qb.clamp_min(0.0).norm(dim=-1) + qb.amax(-1).clamp_max(0.0)
        g = torch.maximum(g_s, g_box)                              # (B,Q,M)
        rel = qE - cE
        ru = (rel * uE).sum(-1); ra = (rel * eaE).sum(-1); rc = (rel * ebE).sum(-1)
        loc = torch.stack([ru, ra, rc], -1)
        w = torch.exp(-0.5 * ((loc / sig[:, None]) ** 2).sum(-1))   # (B,Q,M)
        w = w * self.alive[:, None].to(w.dtype)                     # mask dead splats
        return g, w

    def blend_sdf(self, q, chunk=8192):
        """Self-normalized blend SDF at ``q (B,Q,3)`` -> ``(B,Q)`` (chunked over Q)."""
        out = []
        for a in range(0, q.shape[1], chunk):
            g, w = self._gw(q[:, a:a + chunk])
            out.append((w * g).sum(-1) / w.sum(-1).clamp_min(EPS))
        return torch.cat(out, 1)

    @torch.no_grad()
    def surface_ownership(self, P, tau=0.02):
        """Per-point soft owner ``(B,N,M)`` by surface proximity (dead splats -> ~0)."""
        g, _ = self._gw(P)
        g = g.abs().masked_fill(~self.alive[:, None], 1e4)
        return torch.softmax(-g / tau, dim=-1)

    @torch.no_grad()
    def param_rows(self):
        """``(B, M, ROW_W)`` rows (inverse of the constructor); use with the alive mask to extract."""
        cols = []
        for name in BatchSplats._PNAMES:
            v = self.params[name].detach()
            cols.append(v if v.dim() == 3 else v[..., None])
        cols.append(self.sign[..., None])
        return torch.cat(cols, dim=-1)

    @torch.no_grad()
    def to_single(self, b):
        """Extract mesh ``b``'s ALIVE splats as a plain :class:`SuperToroidSplats` (for caching)."""
        rows = self.param_rows()[b][self.alive[b]]
        return SuperToroidSplats.from_rows(rows.cpu(), p_max=self.p_max)

    @classmethod
    def from_clouds(cls, Ps, Ns, m_max, m_alive=None, device="cuda", seed=0, sigma_init=0.18):
        """Pre-place ``m_max`` FPS splats per mesh (coeffs init); ``m_alive`` of them start alive (the
        rest are dormant capacity for :func:`_grow_batch`).  FPS order is itself well-spread, so the
        first ``m_alive`` give good initial coverage."""
        rows = torch.zeros(len(Ps), m_max, ROW_W)
        for i, (P, N) in enumerate(zip(Ps, Ns)):
            P = np.asarray(P, np.float32); N = np.asarray(N, np.float32)
            idx = farthest_point_sample(P, m_max, seed=seed)
            s = _init_from_coeffs(P, N, idx, np.full(len(idx), sigma_init, np.float32))
            rows[i] = s.param_rows()
        alive = torch.zeros(len(Ps), m_max, dtype=torch.bool)
        alive[:, :(m_max if m_alive is None else m_alive)] = True
        return cls(rows, alive, device=device)

    @torch.no_grad()
    def set_rows(self, b, slots, rows_k):
        """Write ``rows_k (k, ROW_W)`` into mesh ``b``'s splat ``slots`` and mark them alive (for grow)."""
        rows_k = rows_k.to(self.device)
        for name in BatchSplats._PNAMES:
            col = rows_k[:, _ROW_SLICES[name]]
            if name in ("log_R", "log_r", "raw_pt", "raw_pr"):
                col = col.squeeze(-1)
            self.params[name].data[b, slots] = col
        self.sign[b, slots] = torch.sign(rows_k[:, _ROW_SLICES["sign"]].squeeze(-1)).clamp_min(-1.0)
        self.alive[b, slots] = True


# --------------------------------------------------------------------------- #
#  Batched MD metric (occupancy = blend sign; the fixed grid is shared across meshes)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def md_batch(bs, occ_gt, res=64, bound=1.0, return_iou=False, chunk=8192):
    """Per-mesh Minkowski filled-volume distance ``(B,)`` between the batched fields and the cached GT
    occupancies ``occ_gt (B,4,res^3)`` (bool).  Same definition as :func:`pat.teacher.md_filled_volume`."""
    base = torch.as_tensor(_T.grid_centers(res, bound), device=bs.device)   # (res^3,3) shared
    occ_gt = occ_gt.to(bs.device)
    sym = torch.zeros(bs.B, device=bs.device)
    inter = torch.zeros(bs.B, device=bs.device); uni = torch.zeros(bs.B, device=bs.device)
    for i, o in enumerate(_T._offsets(res, bound)):
        q = (base + torch.as_tensor(o, device=bs.device))[None].expand(bs.B, -1, -1)
        occ = bs.blend_sdf(q, chunk=chunk) < 0                     # (B,res^3)
        gt = occ_gt[:, i]
        sym = sym + (occ ^ gt).float().mean(1)
        if return_iou:
            inter = inter + (occ & gt).float().sum(1); uni = uni + (occ | gt).float().sum(1)
    md = sym / 4.0                                                 # (B,) FRACTION vol(A xor B)/vol(cube)
    if return_iou:
        return md, inter / uni.clamp_min(1.0)
    return md


# --------------------------------------------------------------------------- #
#  Batched optimization (one Adam over all meshes; per-mesh GT pool sampled on-GPU)
# --------------------------------------------------------------------------- #
def _optimize_batch(bs, qpool, phipool, *, steps, lr, n_query, lam_eik=0.05, grow=1e-3, seed=0):
    opt = torch.optim.Adam(bs.parameters(), lr=lr)
    Npool = qpool.shape[1]
    gen = torch.Generator(device=bs.device).manual_seed(seed)
    rng = range(steps)
    for _ in rng:
        idx = torch.randint(0, Npool, (bs.B, n_query), generator=gen, device=bs.device)
        q = torch.gather(qpool, 1, idx[..., None].expand(-1, -1, 3))
        phi_true = torch.gather(phipool, 1, idx)
        qt = q.detach().clone().requires_grad_(True)
        phi = bs.blend_sdf(qt, chunk=n_query)
        l_dist = (phi - phi_true).abs().mean()
        grad, = torch.autograd.grad(phi.sum(), qt, create_graph=True)
        l_eik = (1.0 - torch.nan_to_num(grad).norm(dim=-1)).abs().mean()
        l_grow = -(bs.log_sigma.mean(-1) * bs.alive).sum() / bs.alive.sum().clamp_min(1)
        loss = l_dist + lam_eik * l_eik + grow * l_grow
        opt.zero_grad()
        if not torch.isfinite(loss):
            continue
        loss.backward()
        for p in bs.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
        torch.nn.utils.clip_grad_norm_(bs.parameters(), 1.0)
        opt.step()
    return float(l_dist.detach())


def _grow_batch(bs, Pt, Ns_np, need, grow_add, seed):
    """Activate dormant splats for the meshes flagged in ``need (B,)``: place ``grow_add`` new splats at
    each such mesh's WORST-fit surface points (coeffs init), filling coverage gaps before the next refit."""
    Pnp = Pt.detach().cpu().numpy()
    with torch.no_grad():
        err = bs.blend_sdf(Pt).abs()                                # (B,N) surface residual (~0 where fit)
    for b in torch.where(need)[0].tolist():
        dormant = torch.where(~bs.alive[b])[0]
        if dormant.numel() == 0:
            continue
        k = int(min(grow_add, dormant.numel()))
        worst = torch.topk(err[b], max(k * 4, 32)).indices.cpu().numpy()
        pick = farthest_point_sample(Pnp[b][worst], k, seed=seed)
        idx = worst[pick]
        s = _init_from_coeffs(Pnp[b], np.asarray(Ns_np[b], np.float32), idx,
                              np.full(k, 0.18, np.float32))
        bs.set_rows(b, dormant[:k], s.param_rows())


# --------------------------------------------------------------------------- #
#  The batched teacher
# --------------------------------------------------------------------------- #
def prepare_gt(Ps, Ns, *, res=64, k_dense=50_000, seed=0):
    """Build per-mesh GT (occupancy ``(B,4,res^3)`` + query pool ``(B,Npool,*)``) on CPU as stacked CPU
    tensors -- PREFETCH-able on a background thread while the GPU fits another batch (the KD-tree GT is the
    GPU-idle phase; overlapping it keeps the GPU busy)."""
    occ, qp, pp = [], [], []
    for P, N in zip(Ps, Ns):
        shape = _T.CloudShape(P, N, k_dense=k_dense, seed=seed)
        occ.append(torch.as_tensor(_T.gt_occupancy(shape, res=res)))
        q, p = _T.build_gt_pool(shape, np.asarray(P, np.float32), device="cpu", seed=seed)
        qp.append(q); pp.append(p)
    return torch.stack(occ), torch.stack(qp), torch.stack(pp)


def fit_teacher_batch(Ps, Ns, *, m_init=40, m_max=128, grow_add=16, max_grow=4, md_target=1e-3, iou_ok=0.7,
                      res=64, steps_warm=300, steps_refit=70, n_query=2048,
                      keep_schedule=(0.8, 0.6, 0.45, 0.33, 0.25), min_keep=8,
                      k_dense=50_000, device="cuda", seed=0, lr=1e-2, verbose=False, gt=None):
    """Optimize ``B = len(Ps)`` meshes' minimal splat sets in parallel.  Returns a list of per-mesh
    ``(splat, md, iou, status)`` (``splat`` is a plain :class:`SuperToroidSplats`).

    Capacity is ``m_max`` splats/mesh with ``m_init`` initially alive.  Batched warm-fit, then a **grow**
    loop (activate dormant splats at the worst-fit regions of meshes still above ``md_target``, refit),
    then a **speculative** prune (descending keep-schedule; keep each mesh's smallest field still meeting
    the target).  GPU tensors are freed + the cache emptied before returning.
    """
    B = len(Ps)
    # per-mesh GT (KD-tree occupancy + query pool).  Build on CPU here unless prebuilt `gt` was passed
    # (the notebook PREFETCHES the next batch's GT on a CPU thread while this batch fits on the GPU, so
    # the GPU never stalls on the KD-tree).  Then move to the GPU.
    if gt is None:
        gt = prepare_gt(Ps, Ns, res=res, k_dense=k_dense, seed=seed)
    occ_gt = gt[0].to(device); qpool = gt[1].to(device); phipool = gt[2].to(device)
    Pt = torch.stack([torch.as_tensor(np.asarray(p), dtype=torch.float32) for p in Ps]).to(device)
    Ns_np = [np.asarray(n, np.float32) for n in Ns]

    bs = BatchSplats.from_clouds(Ps, Ns, m_max, m_alive=m_init, device=device, seed=seed)
    _optimize_batch(bs, qpool, phipool, steps=steps_warm, lr=lr, n_query=n_query, seed=seed)

    # GROW: activate dormant splats for meshes still above target, refit; stop when all met or capped
    for r in range(max_grow):
        md = md_batch(bs, occ_gt, res=res)
        need = (md > md_target) & (bs.alive.sum(1) < m_max)
        if not bool(need.any()):
            break
        _grow_batch(bs, Pt, Ns_np, need, grow_add, seed + r)
        _optimize_batch(bs, qpool, phipool, steps=steps_refit, lr=lr, n_query=n_query, seed=seed + 100 + r)
        if verbose:
            print(f"  grow {r+1}: still-need {int(need.sum())}/{B} | max-alive {int(bs.alive.sum(1).max())}", flush=True)

    # speculative prune: remember each mesh's smallest FEASIBLE field. A mesh is feasible if MD <= target
    # OR IoU >= iou_ok (the absolute MD target is hard to hit on detailed meshes; IoU is the scale-free
    # quality gate that keeps the prune minimizing splats for well-fit-but-not-perfect meshes too).
    best_rows = bs.param_rows().clone(); best_alive = bs.alive.clone()
    best_md, best_iou = md_batch(bs, occ_gt, res=res, return_iou=True)
    feas0 = (best_md <= md_target) | (best_iou >= iou_ok)
    best_cnt = torch.where(feas0, bs.alive.sum(1), torch.full((B,), 10_000, device=device)).clone()
    for f in keep_schedule:
        with torch.no_grad():
            share = bs.surface_ownership(Pt).sum(1).masked_fill(~bs.alive, -1.0)      # (B,M)
            keep_n = torch.clamp((f * bs.alive.sum(1)).long(), min=min_keep)          # (B,)
            new_alive = torch.zeros_like(bs.alive)
            for b in range(B):
                new_alive[b, torch.topk(share[b], int(keep_n[b].item())).indices] = True
            bs.alive = new_alive
        _optimize_batch(bs, qpool, phipool, steps=steps_refit, lr=lr, n_query=n_query, seed=seed + 1)
        md, iou = md_batch(bs, occ_gt, res=res, return_iou=True); cnt = bs.alive.sum(1)
        feasible = (md <= md_target) | (iou >= iou_ok)
        improve = feasible & (cnt < best_cnt)
        if bool(improve.any()):
            rows_now = bs.param_rows()
            best_rows[improve] = rows_now[improve]; best_alive[improve] = bs.alive[improve]
            best_md[improve] = md[improve]; best_iou[improve] = iou[improve]; best_cnt[improve] = cnt[improve]
        if verbose:
            print(f"  keep {f:.2f}: feasible {int(feasible.sum())}/{B} "
                  f"min-cnt {int(best_cnt[best_cnt<10000].min().item()) if (best_cnt<10000).any() else 0}", flush=True)

    # assemble per-mesh results from the best feasible field (fall back to full field if none feasible)
    final = BatchSplats(best_rows, best_alive, p_max=bs.p_max, device=device)
    md_f, iou_f = md_batch(final, occ_gt, res=res, return_iou=True)
    out = [(final.to_single(b), float(md_f[b]), float(iou_f[b]),
            "ok" if best_cnt[b] < 10_000 else "hard") for b in range(B)]
    del occ_gt, qpool, phipool, Pt, bs, final                                        # free GPU memory
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    return out


def auto_batch_size(P0, N0, *, m_max=128, res=64, n_query=2048, k_dense=50_000, device="cuda",
                    safety=0.8, max_b=64, probe_steps=6, seed=0):
    """Estimate the largest SAFE ``BATCH_MESHES`` for the current GPU so the teacher actually FILLS VRAM:
    probe peak memory at B=1 and B=2 (worst case = all ``m_max`` splats alive + the GPU GT build),
    linearly extrapolate against ``safety`` * free VRAM, and clean up.  Returns ``4`` on non-CUDA.
    """
    if "cuda" not in str(device) or not torch.cuda.is_available():
        return 4
    import gc
    torch.cuda.empty_cache(); gc.collect()
    free, _ = torch.cuda.mem_get_info()
    shape = _T.CloudShape(P0, N0, k_dense=k_dense, seed=seed)
    occ1 = torch.as_tensor(_T.gt_occupancy(shape, res=res)).to(device)
    q1, p1 = _T.build_gt_pool(shape, np.asarray(P0, np.float32), device=device, seed=seed)

    def probe(b):
        occ = occ1[None].expand(b, -1, -1).contiguous()
        qp = q1[None].expand(b, -1, -1).contiguous(); pp = p1[None].expand(b, -1).contiguous()
        bs = BatchSplats.from_clouds([P0] * b, [N0] * b, m_max, m_alive=m_max, device=device, seed=seed)
        torch.cuda.reset_peak_memory_stats()
        _optimize_batch(bs, qp, pp, steps=probe_steps, lr=1e-2, n_query=n_query, seed=seed)
        md_batch(bs, occ, res=res)                                                    # the memory-peak op
        peak = torch.cuda.max_memory_allocated()
        del bs, occ, qp, pp; torch.cuda.empty_cache()
        return peak

    p1m = probe(1); p2m = probe(2)
    slope = max(p2m - p1m, 1); intercept = p1m - slope                                # peak ~ slope*B + intercept
    B = int((free * safety - intercept) / slope)
    del occ1, q1, p1; torch.cuda.empty_cache(); gc.collect()
    return int(max(1, min(max_b, B)))


# --------------------------------------------------------------------------- #
#  Batched cache driver (presence-checked, atomic, resumable)
# --------------------------------------------------------------------------- #
def fit_and_cache_batch(Ps, Ns, gids, outdir, *, force=False, device="cuda", **kw):
    """Teacher-fit a batch of meshes (skipping any already cached) and write each shard.  Returns a list
    of ``(gid, status, M, md)`` for the batch (status ``"cached"`` for skipped meshes)."""
    todo = [(P, N, g) for P, N, g in zip(Ps, Ns, gids)
            if force or not _T.shard_is_current(_T.shard_path(outdir, g))]   # regen MISSING or STALE shards
    results = {g: ("cached",) for g in gids if g not in {t[2] for t in todo}}
    for g in list(results):                                          # backfill M/md for skipped (current) shards
        a = torch.load(_T.shard_path(outdir, g), weights_only=False, map_location="cpu")
        results[g] = ("cached", a["M"], a["md"])
    if todo:
        fits = fit_teacher_batch([t[0] for t in todo], [t[1] for t in todo], device=device, **kw)
        for (P, N, g), (splat, md, iou, status) in zip(todo, fits):
            _T.save_teacher(_T.teacher_artifact(splat, P, N, md, iou, status, g), _T.shard_path(outdir, g))
            results[g] = (status, int(splat.M), float(md))
    return [(g, *results[g]) for g in gids]
