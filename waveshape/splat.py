"""Adaptive **supertoroid splats** -- a 3D-Gaussian-Splatting-style sparse primitive field.

The paper's per-point blend fits ONE torus per cloud point and averages all ~1024 of them.
Because every per-point primitive is anchored at (and passes through) its own noisy point, the
blended zero-set is forced to interpolate the whole point cloud -> flat regions come out **bumpy**
no matter how good each local fit is.

This module takes the 3DGS view: a **small, adaptive set of supertoroid splats**, each a full
supertoroid (:func:`pat.core.supertoroid_sdf`) with a **learned spatial extent** (a Gaussian
window).  A splat can *grow to consume its neighbors*, so a whole region collapses into ONE
supertoroid -> far fewer, larger primitives -> a much smoother surface.  The set is optimized
per-shape (like 3DGS optimizes per-scene) and **pruned** so redundant splats disappear.

Splat ``i`` carries a supertoroid ``(center c_i, axis u_i, in-plane axis ea_i, major R_i,
minor r_i, squareness p_tube_i / p_ring_i, sign s_i)`` plus a log-extent ``log sigma_i``::

    g_i(x) = s_i * supertoroid_sdf(x; c_i, u_i, ea_i, R_i, r_i, p_tube_i, p_ring_i)
    w_i(x) = exp( -||x - c_i||^2 / (2 sigma_i^2) )            (Gaussian window)
    phi(x) = sum_i w_i g_i / sum_i w_i                        (self-normalized blend)

Splats are **initialized from the tested** :func:`pat.core.coeffs_to_torus` fit at
farthest-point-sampled centers (a sensible per-region torus), then jointly optimized; the
squareness ``p`` lets each splat go boxy to hug flat/angular regions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import core

try:                                            # progress bars (Colab/terminal); optional
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except Exception:                               # pragma: no cover
    _HAVE_TQDM = False

EPS = 1e-9

# --------------------------------------------------------------------------- #
#  Flat 23-column splat-parameter row layout. SINGLE SOURCE OF TRUTH shared by
#  the teacher cache (`splat_param_rows`), FitNet's decode head, and
#  `SuperToroidSplats.from_rows` -- so the amortizer and the optimizer can never
#  disagree on what a "splat point" is. Width == ROW_W (23).
# --------------------------------------------------------------------------- #
ROW_LAYOUT = [                                      # (name, width); buffers and params alike
    ("center", 3), ("raw_u", 3), ("raw_ea", 3),
    ("log_R", 1), ("log_r", 1), ("raw_pt", 1), ("raw_pr", 1),
    ("log_sigma", 3), ("log_b", 3), ("box_offset", 3), ("sign", 1),
]
ROW_W = sum(w for _, w in ROW_LAYOUT)              # 23
_ROW_SLICES = {}
_off = 0
for _name, _w in ROW_LAYOUT:
    _ROW_SLICES[_name] = slice(_off, _off + _w)
    _off += _w


def farthest_point_sample(xyz: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point sampling -> indices of ``m`` spread-out points."""
    n = len(xyz)
    m = min(m, n)
    rng = np.random.default_rng(seed)
    idx = np.empty(m, dtype=np.int64)
    idx[0] = rng.integers(n)
    d = np.full(n, np.inf)
    for i in range(1, m):
        d = np.minimum(d, ((xyz - xyz[idx[i - 1]]) ** 2).sum(-1))
        idx[i] = int(d.argmax())
    return idx


class SuperToroidSplats(nn.Module):
    """A sparse, adaptive field of supertoroid splats with learned extents."""

    def __init__(self, center, axis, ea, R, r, sign, sigma, p_max=6.0, box_init=2.0):
        super().__init__()
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        M = len(center)
        self.center = nn.Parameter(t(center))                       # (M,3)
        self.raw_u = nn.Parameter(t(axis))                          # -> unit axis
        self.raw_ea = nn.Parameter(t(ea))                           # -> in-plane axis
        self.log_R = nn.Parameter(t(R).clamp_min(1e-3).log())       # major radius
        self.log_r = nn.Parameter(t(r).clamp_min(1e-3).log())       # minor radius
        self.raw_pt = nn.Parameter(torch.full((M,), core.P2_RAW))   # p_tube (init 2)
        self.raw_pr = nn.Parameter(torch.full((M,), core.P2_RAW))   # p_ring (init 2)
        s = t(sigma).reshape(M, -1)
        if s.shape[1] == 1:                                         # isotropic -> 3 equal axes
            s = s.expand(M, 3).contiguous()
        self.log_sigma = nn.Parameter(s.clamp_min(1e-3).log())     # (M,3) ANISOTROPIC window
        # (a flat-face splat becomes a thin disk in its own frame -> covers the face without
        #  bleeding across edges -> sharper corners than an isotropic ball window.)
        # --- cut-out / clip box (our extension): an oriented box intersected with the
        # supertoroid (g = max(super, box)).  Its faces are FLAT, so a clipped supertoroid
        # makes planar/angular surfaces directly.  Box uses the splat frame [u, ea, eb] and is
        # centered at center + box_offset (it "moves with" the splat).  half-extents init large
        # (box_init) -> no clipping at init, so each splat starts as a pure supertoroid; the
        # optimizer shrinks the box only where a flat face fits the data better.
        self.log_b = nn.Parameter(torch.full((M, 3), float(np.log(box_init))))  # box half-extents
        self.box_offset = nn.Parameter(torch.zeros(M, 3))           # box center offset from splat
        self.register_buffer("sign", t(sign).reshape(-1))           # fixed inside/outside sign
        self.p_max = float(p_max)

    @property
    def M(self):
        return self.center.shape[0]

    def _params(self):
        u = self.raw_u / self.raw_u.norm(dim=1, keepdim=True).clamp_min(EPS)
        ea = self.raw_ea - (self.raw_ea * u).sum(-1, keepdim=True) * u    # orthogonalize to u
        ea = ea / ea.norm(dim=1, keepdim=True).clamp_min(EPS)
        eb = torch.cross(u, ea, dim=-1)                              # 3rd box axis
        R = self.log_R.clamp(-5.0, 2.0).exp()
        r = self.log_r.clamp(-5.0, 2.0).exp()
        pt = core.raw_to_p(self.raw_pt, p_max=self.p_max)
        pr = core.raw_to_p(self.raw_pr, p_max=self.p_max)
        b = self.log_b.clamp(-4.0, 1.5).exp()                       # box half-extents (M,3), bounded
        return u, ea, eb, R, r, pt, pr, b

    def _g_splat(self, xb, u, ea, eb, R, r, pt, pr, b, boxc):
        """Per-splat signed value ``g_i`` at ``xb (q,1,3)`` -> ``(q,M)`` (supertoroid clipped by box)."""
        g_s = self.sign[None] * core.supertoroid_sdf(
            xb, self.center[None], u[None], ea[None], R[None], r[None], pt[None], pr[None])
        relb = xb - boxc[None]
        lx = (relb * u[None]).sum(-1); ly = (relb * ea[None]).sum(-1); lz = (relb * eb[None]).sum(-1)
        qb = torch.stack([lx, ly, lz], dim=-1).abs() - b[None]
        g_box = qb.clamp_min(0.0).norm(dim=-1) + qb.amax(dim=-1).clamp_max(0.0)
        return torch.maximum(g_s, g_box)                            # CSG intersection (clip)

    def sdf_torch(self, x, chunk=8000, edge_gamma=0.0):
        """Blended SDF at query points ``x`` ``(Q,3)`` -> ``(Q,)`` (differentiable).

        ``edge_gamma > 0`` enables the edge-aware gate: each splat's weight is multiplied by
        ``cos(angle)**edge_gamma`` between its own surface normal (per-splat ``grad g_i``, via
        finite differences, detached) and the query's window-dominant normal -- so the blend
        does NOT average across a sharp crease (where two faces' normals disagree) -> sharp
        dihedral angles instead of rounded edges.  Costs 3 extra (no-grad) forward evals.
        """
        u, ea, eb, R, r, pt, pr, b = self._params()
        boxc = self.center + self.box_offset.clamp(-1.0, 1.0)       # box center (moves with splat)
        sig3 = self.log_sigma.clamp(-5.0, 1.0).exp().clamp_min(1e-3)   # (M,3) anisotropic extents
        out = []
        for a in range(0, x.shape[0], chunk):
            xb = x[a:a + chunk][:, None, :]                          # (q,1,3)
            g = self._g_splat(xb, u, ea, eb, R, r, pt, pr, b, boxc)  # (q,M)
            # anisotropic Gaussian window in the splat frame [u, ea, eb]
            rel = xb - self.center[None]
            ru = (rel * u[None]).sum(-1); ra = (rel * ea[None]).sum(-1); rc = (rel * eb[None]).sum(-1)
            locw = torch.stack([ru, ra, rc], dim=-1)                # (q,M,3) splat-frame
            w = torch.exp(-0.5 * ((locw / sig3[None]) ** 2).sum(-1))
            if edge_gamma > 0:
                with torch.no_grad():                              # per-splat normal via finite diff
                    eps = 2e-3
                    cols = []
                    for k in range(3):
                        e = torch.zeros(3, device=xb.device); e[k] = eps
                        cols.append((self._g_splat(xb + e, u, ea, eb, R, r, pt, pr, b, boxc) - g) / eps)
                    ni = torch.stack(cols, dim=-1)                  # (q,M,3)
                    ni = ni / ni.norm(dim=-1, keepdim=True).clamp_min(EPS)
                    nsum = (w.unsqueeze(-1) * ni).sum(1)            # (q,3) dominant normal
                    nbar = nsum / torch.sqrt((nsum * nsum).sum(-1, keepdim=True) + 1e-6)
                    cos = (nbar.unsqueeze(1) * ni).sum(-1).clamp(-1.0, 1.0).clamp_min(0.0)  # (q,M)
                w = w * cos.pow(edge_gamma)
            out.append((w * g).sum(1) / w.sum(1).clamp_min(EPS))
        return torch.cat(out, 0)

    def sdf(self, x, edge_gamma=0.0):
        with torch.no_grad():
            return self.sdf_torch(torch.as_tensor(np.asarray(x), dtype=torch.float32)
                                  .to(self.center.device), edge_gamma=edge_gamma).cpu().numpy()

    @torch.no_grad()
    def union_sdf(self, x, chunk=200_000):
        """CSG-UNION signed distance at ``x (Q,3)`` -> ``(Q,)``: ``min_i sign_i*max(super_i, box_i)``.

        This is a GENUINE solid SDF (negative strictly inside the union of the per-splat solids), so
        ``union_sdf(x) < 0`` is a 0%-interior-sign-error occupancy -- UNLIKE the self-normalized
        ``sdf_torch`` blend, whose sign is unreliable in the interior. The filled-volume Minkowski
        distance is computed against THIS, never the blend.
        """
        x = (x.detach().to(self.center.device, torch.float32) if torch.is_tensor(x)
             else torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.center.device))
        u, ea, eb, R, r, pt, pr, b = self._params()
        boxc = self.center + self.box_offset.clamp(-1.0, 1.0)
        out = torch.empty(x.shape[0], device=x.device)
        for a in range(0, x.shape[0], chunk):
            xb = x[a:a + chunk][:, None, :]                              # (q,1,3)
            g = self._g_splat(xb, u, ea, eb, R, r, pt, pr, b, boxc)      # (q,M) signed solids
            out[a:a + chunk] = g.amin(dim=1)                            # CSG union = min over solids
        return out

    def responsibility(self, cloud):
        """Per-point ownership matrix ``(N, M)``: each cloud point's normalized window weight over
        the splats (rows sum to 1). ``total_weight(cloud) == responsibility(cloud).sum(0)`` -- the
        single source of truth for "which splat owns which points" (teacher labels for GroupNet)."""
        with torch.no_grad():
            u, ea, eb, R, r, pt, pr, b = self._params()
            sig3 = self.log_sigma.clamp(-5.0, 1.0).exp().clamp_min(1e-3)
            rel = torch.as_tensor(np.asarray(cloud), dtype=torch.float32,
                                  device=self.center.device)[:, None, :] - self.center[None]
            ru = (rel * u[None]).sum(-1); ra = (rel * ea[None]).sum(-1); rc = (rel * eb[None]).sum(-1)
            locw = torch.stack([ru, ra, rc], dim=-1)
            w = torch.exp(-0.5 * ((locw / sig3[None]) ** 2).sum(-1))         # (N,M)
            return w / w.sum(1, keepdim=True).clamp_min(EPS)                 # (N,M) rows sum to 1

    def total_weight(self, cloud):
        return self.responsibility(cloud).sum(0)                            # (M,) per-splat share

    @torch.no_grad()
    def surface_ownership(self, cloud, tau=0.02):
        """Per-point SOFT membership ``(N, M)`` by SURFACE proximity: ``softmax_i(-|g_i_solid|/tau)``.

        Unlike :meth:`responsibility` (the Gaussian-window share, which the ``grow`` reward inflates so
        one splat can dominate everywhere), this is anchored to each splat's actual surface, so it stays
        a balanced, local partition -- the robust grouping label for the Stage-B student.  ``argmax``
        gives the hard owner; the column-sum gives each splat's surface share (prune victim = argmin)."""
        u, ea, eb, sR, sr, pt, pr, b = self._params()
        boxc = self.center + self.box_offset.clamp(-1.0, 1.0)
        x = torch.as_tensor(np.asarray(cloud), dtype=torch.float32,
                            device=self.center.device)[:, None, :]
        g = self._g_splat(x, u, ea, eb, sR, sr, pt, pr, b, boxc).abs()      # (N,M) |single-splat solid|
        return torch.softmax(-g / tau, dim=1)

    def param_rows(self):
        """This field's params as a flat ``(M, ROW_W)`` tensor in :data:`ROW_LAYOUT` order
        (the teacher's per-splat label / FitNet target). Inverse of :meth:`from_rows`."""
        cols = []
        for name, _w in ROW_LAYOUT:
            v = getattr(self, name).detach()
            cols.append(v if v.dim() == 2 else v[:, None].float())
        return torch.cat(cols, dim=1)                                       # (M, ROW_W)

    @classmethod
    def from_rows(cls, rows, p_max=6.0):
        """Assemble a ``SuperToroidSplats`` of ``K`` splats from a ``(K, ROW_W)`` param tensor (the
        inverse of :meth:`param_rows`; consumes FitNet's decoded rows directly).  Accepts CPU/CUDA
        tensors or arrays; the assembled module is on CPU (callers ``.to(device)`` as needed)."""
        if torch.is_tensor(rows):                                   # avoid np.asarray on a CUDA tensor
            rows = rows.detach().to(device="cpu", dtype=torch.float32)
        else:
            rows = torch.as_tensor(np.asarray(rows), dtype=torch.float32)
        K = rows.shape[0]
        sp = cls(np.zeros((K, 3), np.float32),
                 np.tile([0, 0, 1.0], (K, 1)).astype(np.float32),
                 np.tile([1.0, 0, 0], (K, 1)).astype(np.float32),
                 np.full(K, 0.5, np.float32), np.full(K, 0.2, np.float32),
                 np.ones(K, np.float32), np.full(K, 0.18, np.float32), p_max=p_max)
        with torch.no_grad():
            for name, _w in ROW_LAYOUT:
                col = rows[:, _ROW_SLICES[name]]
                if name == "sign":
                    sp.sign = torch.sign(col.squeeze(1)).clamp_min(-1.0)    # buffer; +-1
                elif col.shape[1] == 1:
                    getattr(sp, name).data.copy_(col.squeeze(1))
                else:
                    getattr(sp, name).data.copy_(col)
        return sp

    @torch.no_grad()
    def prune(self, cloud, min_share=0.5, min_keep=8):
        share = self.total_weight(cloud)
        keep = share >= min_share
        if int(keep.sum()) < min_keep:                      # never collapse below a floor
            keep = torch.zeros_like(keep)
            keep[torch.topk(share, min(min_keep, self.M)).indices] = True
        if keep.all() or int(keep.sum()) < 1:
            return 0
        idx = torch.where(keep)[0]
        for name in ("center", "raw_u", "raw_ea", "log_R", "log_r", "raw_pt", "raw_pr",
                     "log_sigma", "log_b", "box_offset"):
            p = getattr(self, name)
            setattr(self, name, nn.Parameter(p.data[idx].clone()))
        self.sign = self.sign[idx].clone()
        return int((~keep).sum())


def _init_from_coeffs(cloud, normals, idx, sigma):
    """Initialize splat supertoroids from the coeffs_to_torus fit at centers ``idx``."""
    from .lstsq import fit_coeffs_lstsq
    coeffs = fit_coeffs_lstsq(cloud, normals, k=24)
    P = torch.as_tensor(cloud, dtype=torch.float32)
    N = torch.as_tensor(normals, dtype=torch.float32)
    p = core.coeffs_to_torus(P[idx], N[idx], coeffs[idx])
    return SuperToroidSplats(p["center"].detach(), p["axis"].detach(), p["ea"].detach(),
                             p["R"].detach(), p["r"].detach(), p["sign"].detach(), sigma)


def reconstruct(splat, res=96, bound=1.0, edge_gamma=0.0):
    """Marching-cubes the splat field -> (verts, faces)."""
    from skimage import measure
    lin = np.linspace(-bound, bound, res)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    vol = splat.sdf(grid, edge_gamma=edge_gamma).reshape(res, res, res)
    v, f, _, _ = measure.marching_cubes(vol, level=0.0)
    return v / (res - 1) * (2 * bound) - bound, f


def optimize_splats(splat, shape, cloud, *, steps=600, lr=1e-2, bound=1.2, n_query=4000,
                    lam_eik=0.05, grow=1e-3, square_reg=0.0, edge_gamma=0.0, prune_every=150,
                    min_share=0.3, min_keep=12, device="cpu", seed=0, log=None, desc="fit splats",
                    q_pool=None, phi_pool=None):
    """Run the splat optimization loop on an EXISTING :class:`SuperToroidSplats` (no FPS re-init).

    This is the shared core of :func:`fit_shape` (init + this) and the teacher's *refit-after-delete*
    (``pat.teacher._refit``): deleting a splat reweights its neighbors under the self-normalized blend,
    so the survivors MUST re-optimize -- a delete-and-recheck without refit is unsound.  Mutates and
    returns ``splat``.

    ``q_pool (Npool,3)`` + ``phi_pool (Npool,)`` (precomputed GT, on ``device``): when given, each step
    SAMPLES queries from the pool on-GPU instead of calling ``shape.sdf`` -- this removes the per-step
    CPU KD-tree query + CPU<->GPU sync that otherwise starves the GPU (the teacher uses this path).
    """
    splat = splat.to(device)
    cloud = np.asarray(cloud, np.float32)
    opt = torch.optim.Adam(splat.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    pooled = q_pool is not None and phi_pool is not None
    gen = torch.Generator(device=device).manual_seed(seed) if pooled else None
    bar = tqdm(range(steps), desc=desc, leave=False) if _HAVE_TQDM else range(steps)
    for it in bar:
        if pooled:                                                 # sample precomputed GT on-GPU (fast)
            idx = torch.randint(0, q_pool.shape[0], (n_query,), generator=gen, device=device)
            qt = q_pool[idx].detach().clone().requires_grad_(True)
            phi_true = phi_pool[idx]
        else:                                                      # per-step CPU sampling + shape.sdf
            nb = n_query // 2
            band = cloud[rng.integers(0, len(cloud), nb)] + rng.normal(scale=0.03, size=(nb, 3))
            bulk = rng.uniform(-bound, bound, size=(n_query - nb, 3))
            q = np.concatenate([band, bulk], 0).astype(np.float32)
            phi_true = torch.as_tensor(shape.sdf(q), dtype=torch.float32, device=device)
            qt = torch.tensor(q, dtype=torch.float32, device=device, requires_grad=True)
        phi = splat.sdf_torch(qt, edge_gamma=edge_gamma)
        l_dist = (phi - phi_true).abs().mean()
        grad, = torch.autograd.grad(phi.sum(), qt, create_graph=True)
        grad = torch.nan_to_num(grad)
        l_eik = (1.0 - grad.norm(dim=-1)).abs().mean()
        # reward large extents (consume neighbors); keep squareness near 2 unless data needs boxy
        l_sq = ((splat.raw_pt - core.P2_RAW) ** 2 + (splat.raw_pr - core.P2_RAW) ** 2).mean()
        loss = l_dist + lam_eik * l_eik - grow * splat.log_sigma.mean() + square_reg * l_sq
        opt.zero_grad()
        if not torch.isfinite(loss):                               # skip a bad (NaN/Inf) step
            continue
        loss.backward()
        for p in splat.parameters():                              # sanitize NaN/Inf grads
            if p.grad is not None:
                torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
        torch.nn.utils.clip_grad_norm_(splat.parameters(), 1.0)
        opt.step()
        if prune_every and (it + 1) % prune_every == 0 and (it + 1) < steps:
            splat.prune(cloud, min_share=min_share, min_keep=min_keep)
            opt = torch.optim.Adam(splat.parameters(), lr=lr)
        if (it + 1) % 50 == 0 or it == 0:
            rec = dict(step=it + 1, loss=float(loss.detach()), dist=float(l_dist.detach()),
                       eik=float(l_eik.detach()), n_splats=int(splat.M))
            if log is not None:
                log.append(rec)
            if _HAVE_TQDM:
                bar.set_postfix_str(f"dist {rec['dist']:.4f} eik {rec['eik']:.3f} splats {splat.M}")
    return splat


def fit_shape(shape, cloud, normals, *, n_init=96, steps=600, lr=1e-2, bound=1.2,
              n_query=4000, sigma_init=0.18, lam_eik=0.05, grow=1e-3, square_reg=0.0,
              edge_gamma=0.0, prune_every=150, min_share=0.3, min_keep=12, device="cpu",
              seed=0, log=None, q_pool=None, phi_pool=None):
    """Optimize an adaptive supertoroid-splat field to a ``shape`` exposing ``.sdf(q)`` (3DGS-style).

    Initializes ``n_init`` splats from the :func:`pat.core.coeffs_to_torus` fit at FPS centers, then
    runs :func:`optimize_splats`.  ``log`` (if given) collects per-step ``{step,loss,dist,eik,n_splats}``.
    ``q_pool/phi_pool`` (precomputed GT on ``device``) make the loop fully on-GPU.  Returns the fitted
    :class:`SuperToroidSplats`.
    """
    cloud = np.asarray(cloud, np.float32)
    normals = np.asarray(normals, np.float32)
    idx = farthest_point_sample(cloud, n_init, seed=seed)
    splat = _init_from_coeffs(cloud, normals, idx,
                              np.full(len(idx), sigma_init, np.float32)).to(device)
    return optimize_splats(splat, shape, cloud, steps=steps, lr=lr, bound=bound, n_query=n_query,
                           lam_eik=lam_eik, grow=grow, square_reg=square_reg, edge_gamma=edge_gamma,
                           prune_every=prune_every, min_share=min_share, min_keep=min_keep,
                           device=device, seed=seed, log=log, q_pool=q_pool, phi_pool=phi_pool)


# --------------------------------------------------------------------------- #
#  Save / load + plotting helpers (used by the Colab notebook)
# --------------------------------------------------------------------------- #
def save_splats(splat, path):
    """Persist a fitted splat field (params + the fixed sign buffer + count)."""
    torch.save({"state_dict": splat.state_dict(), "M": splat.M,
                "p_max": splat.p_max}, path)


def load_splats(path, device="cpu"):
    """Reload a fitted :class:`SuperToroidSplats` from :func:`save_splats`."""
    ck = torch.load(path, map_location=device, weights_only=False)
    M = ck["M"]
    z3 = np.zeros((M, 3), np.float32); z1 = np.zeros(M, np.float32)
    sp = SuperToroidSplats(z3, z3 + np.array([0, 0, 1.0], np.float32), z3 + np.array([1.0, 0, 0], np.float32),
                           z1 + 0.5, z1 + 0.2, z1 + 1.0, z1 + 0.2, p_max=ck["p_max"])
    sp.load_state_dict(ck["state_dict"])
    return sp.to(device)


def progress_plot(log, path, title=""):
    """Save the optimization progress (SDF L1 + splat count vs step)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [r["step"] for r in log]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, [r["dist"] for r in log], "-o", color="C0", ms=3, label="SDF L1")
    ax1.plot(steps, [r["eik"] for r in log], "-s", color="C2", ms=3, label="eikonal")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.set_yscale("log"); ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(steps, [r["n_splats"] for r in log], "-^", color="C3", ms=3, label="# splats")
    ax2.set_ylabel("# splats", color="C3")
    ax1.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def render_comparison(shape, splat, path, title="", res=110, edge_gamma=0.0):
    """Save a ground-truth vs splat-reconstruction figure (two 3D surfaces)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4))
    splat_fn = (lambda g: splat.sdf(g, edge_gamma=edge_gamma))
    for i, (lab, fn) in enumerate([("ground truth", shape.sdf), ("supertoroid splats", splat_fn)]):
        v, f = _mc(fn, res)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if v is not None:
            ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color="#6b7f99",
                            edgecolor="none", antialiased=True)
        ax.set_title(lab, fontsize=10); ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1)); ax.view_init(22, -62)
    fig.suptitle(title + f"  ({splat.M} splats)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _mc(sdf_fn, res, bound=1.0):
    from skimage import measure
    lin = np.linspace(-bound, bound, res)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    vol = sdf_fn(grid).reshape(res, res, res)
    if not (vol.min() < 0 < vol.max()):
        return None, None
    v, f, _, _ = measure.marching_cubes(vol, level=0.0)
    return v / (res - 1) * (2 * bound) - bound, f
