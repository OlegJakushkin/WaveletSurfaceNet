"""Wavelet-domain denoising reconstruction network — a contrast model to PAT.

Where :class:`pat.pat.PAT` represents a surface as a **per-point parametric**
blend of (super)tori, this module implements a very different idea (the one
sketched in the project brief): infer a *clean implicit surface* from a *noisy
point cloud* by **denoising a truncated signed-distance field in the wavelet
domain**.  The wavelet transform is used as the multi-scale *representation /
regularizer* — not as the whole model — and a small 3-D U-Net does the actual
inference of clean structure from noisy coefficients::

    noisy points
       ↓   tsdf_from_clouds   (voxelize to a truncated SDF grid)
    noisy TSDF  (B,1,R,R,R)
       ↓   dwt3d              (1-level 3-D Haar: 1 coarse + 7 detail subbands)
    wavelet coefficients  (B,8,R/2,R/2,R/2)
       ↓   U-Net denoiser     (residual: clean = noisy_coeffs + Δ)
    clean wavelet coefficients
       ↓   idwt3d             (exact inverse — orthonormal Haar)
    clean TSDF
       ↓   marching cubes      (WaveletReconstruction.reconstruct)
    mesh

The key intuition (brief): random sensor noise lives in *incoherent* high-
frequency wavelet coefficients, while real repeated/structured geometry appears
*coherently across scales and positions* — so a network trained on noisy/clean
coefficient pairs learns to keep the coherent detail and drop the noise, rather
than blurring all high frequencies away.

Everything is differentiable and **device-agnostic** (trains/tests on CPU; runs
faster on a GPU).  :class:`WaveletReconstruction` exposes ``.sdf(q)`` and
``.reconstruct()`` with the *same call shape* as ``PAT`` / the splat models, so it
drops straight into :mod:`pat.eval3d` and :mod:`pat.render3d`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  3-D Haar wavelet transform (single level, perfectly invertible)
# --------------------------------------------------------------------------- #
def haar_filters_3d(device=None, dtype=torch.float32) -> torch.Tensor:
    """The 8 separable 3-D Haar filters as a ``(8, 1, 2, 2, 2)`` conv weight.

    Each subband is the tensor product of a low-pass ``l = [1, 1]/√2`` or a
    high-pass ``h = [1, -1]/√2`` filter along each of the three axes.  The eight
    combinations are ``LLL`` (the coarse approximation) followed by the seven
    detail bands ``LLH … HHH``.  The bank is **orthonormal**, so the very same
    filters synthesize the signal back via a transposed convolution (see
    :func:`idwt3d`), giving exact reconstruction.
    """
    l = torch.tensor([1.0, 1.0], dtype=dtype) / np.sqrt(2.0)
    h = torch.tensor([1.0, -1.0], dtype=dtype) / np.sqrt(2.0)
    banks = (l, h)
    filt = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                filt.append(torch.einsum("i,j,k->ijk", banks[a], banks[b], banks[c]))
    w = torch.stack(filt, 0).unsqueeze(1)            # (8, 1, 2, 2, 2)
    return w.to(device=device)


def dwt3d(x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """1-level 3-D DWT: ``(B, 1, D, H, W) -> (B, 8, D/2, H/2, W/2)`` (even dims)."""
    if w is None:
        w = haar_filters_3d(x.device, x.dtype)
    return F.conv3d(x, w, stride=2)


def idwt3d(c: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse 1-level 3-D DWT: ``(B, 8, d, h, w) -> (B, 1, 2d, 2h, 2w)``.

    Exact because the Haar bank is orthonormal (synthesis == transpose of
    analysis), so ``idwt3d(dwt3d(x)) == x`` to floating-point precision.
    """
    if w is None:
        w = haar_filters_3d(c.device, c.dtype)
    return F.conv_transpose3d(c, w, stride=2)


# --------------------------------------------------------------------------- #
#  Point cloud  ->  truncated signed-distance field (TSDF) grid
# --------------------------------------------------------------------------- #
def tsdf_from_cloud(P, N, res: int = 32, trunc: float = 0.1, bound: float = 1.1):
    """Truncated SDF on a ``res^3`` grid in ``[-bound, bound]^3`` (numpy / cKDTree).

    For every grid point the unsigned distance is to the nearest cloud point and
    the sign comes from the dot product with that point's normal (the standard
    surface-pseudonormal test, robust on open / non-watertight clouds).  The field
    is clipped to ``±trunc`` and returned in **distance units** (not normalized),
    so ``< 0`` means inside.  Shape ``(res, res, res)`` float32.

    This is the CPU reference; :func:`tsdf_from_clouds` is the batched GPU version.
    """
    from scipy.spatial import cKDTree

    P = np.asarray(P, np.float64)
    N = np.asarray(N, np.float64)
    lin = np.linspace(-bound, bound, res, dtype=np.float64)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    d, idx = cKDTree(P).query(grid)
    sign = np.einsum("ij,ij->i", grid - P[idx], N[idx])
    sign = np.where(sign >= 0.0, 1.0, -1.0)
    sdf = np.clip(sign * d, -trunc, trunc)
    return sdf.reshape(res, res, res).astype(np.float32)


def tsdf_from_clouds(Ps, Ns, res: int = 32, trunc: float = 0.1, bound: float = 1.1,
                     device="cpu", qchunk: int = 4096) -> torch.Tensor:
    """Batched TSDF grids for many clouds — ``(B, 1, res, res, res)`` (torch).

    ``Ps``/``Ns`` are ``(B, Npts, 3)`` (tensors or arrays).  Nearest-point lookup
    is a chunked :func:`torch.cdist` (GPU-friendly, no KD-tree), so the whole batch
    of meshes is voxelized in parallel.  Values are in distance units, clipped to
    ``±trunc``; ``< 0`` is inside.  Memory is bounded by ``qchunk`` grid points per
    step (peak ≈ ``B * qchunk * Npts`` floats).
    """
    Ps = torch.as_tensor(Ps, dtype=torch.float32, device=device)
    Ns = torch.as_tensor(Ns, dtype=torch.float32, device=device)
    if Ps.dim() == 2:                                # single cloud -> add batch dim
        Ps, Ns = Ps[None], Ns[None]
    B = Ps.shape[0]
    lin = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    G = grid.shape[0]
    out = torch.empty(B, G, device=device)
    for a in range(0, G, qchunk):
        gq = grid[a:a + qchunk]                      # (q, 3)
        d = torch.cdist(gq.unsqueeze(0).expand(B, -1, -1), Ps)   # (B, q, Npts)
        dmin, idx = d.min(dim=2)                     # (B, q)
        ix = idx.unsqueeze(-1).expand(-1, -1, 3)
        near = torch.gather(Ps, 1, ix)               # (B, q, 3)
        nn = torch.gather(Ns, 1, ix)
        sign = ((gq.unsqueeze(0) - near) * nn).sum(-1)           # (B, q)
        sign = torch.where(sign >= 0.0, torch.ones_like(sign), -torch.ones_like(sign))
        out[:, a:a + qchunk] = (sign * dmin).clamp(-trunc, trunc)
    return out.reshape(B, 1, res, res, res)


# --------------------------------------------------------------------------- #
#  Trilinear sampler (TSDF grid  ->  callable SDF)
# --------------------------------------------------------------------------- #
def grid_trilinear(grid: np.ndarray, q, bound: float, fill: float) -> np.ndarray:
    """Trilinearly sample a ``(res, res, res)`` field at world points ``q (Q, 3)``.

    Maps ``[-bound, bound]`` to grid index ``[0, res-1]``.  Queries outside the cube
    return ``fill`` (use ``+trunc`` so the exterior reads as "outside the surface").
    """
    res = grid.shape[0]
    q = np.asarray(q, np.float64)
    c = (q + bound) / (2.0 * bound) * (res - 1)
    inside = np.all((c >= 0.0) & (c <= res - 1), axis=1)
    c = np.clip(c, 0.0, res - 1 - 1e-6)
    i0 = np.floor(c).astype(np.int64)
    i1 = np.minimum(i0 + 1, res - 1)
    f = c - i0
    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    c00 = grid[x0, y0, z0] * (1 - fx) + grid[x1, y0, z0] * fx
    c10 = grid[x0, y1, z0] * (1 - fx) + grid[x1, y1, z0] * fx
    c01 = grid[x0, y0, z1] * (1 - fx) + grid[x1, y0, z1] * fx
    c11 = grid[x0, y1, z1] * (1 - fx) + grid[x1, y1, z1] * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    val = c0 * (1 - fz) + c1 * fz
    return np.where(inside, val, fill)


# --------------------------------------------------------------------------- #
#  The denoiser network (3-D U-Net over the wavelet subbands)
# --------------------------------------------------------------------------- #
def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class _ConvBlock(nn.Module):
    """Two 3×3×3 convs with GroupNorm + SiLU (the U-Net's basic unit)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), _gn(cout), nn.SiLU(),
            nn.Conv3d(cout, cout, 3, padding=1), _gn(cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class WaveletDenoiser(nn.Module):
    """Denoise a TSDF in the wavelet domain.

    ``forward(noisy_tsdf)`` decomposes the ``(B, 1, R, R, R)`` field into its 8
    Haar subbands at ``R/2``, runs a small two-scale 3-D U-Net over them, predicts
    a **residual** correction to the coefficients (``clean = noisy_coeffs + Δ``),
    and inverts the transform back to a clean ``(B, 1, R, R, R)`` TSDF.

    Predicting a residual (with a zero-initialized output head) makes the *identity*
    the network's starting point — an untrained model simply returns the input TSDF,
    so training only ever has to learn the denoising *correction*, which is far
    easier to optimize than learning the whole field from scratch.

    Args:
        base:   width of the first U-Net stage (channels grow ``base → 2·base →
                4·base``).  Must be a multiple of 8 (GroupNorm groups).
        clamp:  if not ``None``, the output TSDF is ``tanh``-squashed to
                ``±clamp`` (in the normalized ``[-1, 1]`` TSDF scale) so a wild
                prediction can never blow up marching cubes.  Default ``None``.

    The grid resolution ``R`` must be divisible by 8 (one DWT halving + two U-Net
    poolings): ``R = 32`` (subbands ``16 → 8 → 4``) is the default; ``64`` works too.
    """

    def __init__(self, base: int = 32, clamp: float | None = None):
        super().__init__()
        c0 = 8                                       # the 8 wavelet subbands
        self.clamp = clamp
        self.in_block = _ConvBlock(c0, base)
        self.pool = nn.AvgPool3d(2)
        self.down1 = _ConvBlock(base, base * 2)
        self.down2 = _ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.out = nn.Conv3d(base, c0, 1)
        nn.init.zeros_(self.out.weight)              # residual starts at 0 -> identity
        nn.init.zeros_(self.out.bias)
        self.register_buffer("haar", haar_filters_3d())

    def forward(self, tsdf):
        c = dwt3d(tsdf, self.haar)                   # (B, 8, R/2, R/2, R/2)
        x0 = self.in_block(c)                        # R/2
        x1 = self.down1(self.pool(x0))               # R/4
        x2 = self.down2(self.pool(x1))               # R/8
        y1 = self.dec2(torch.cat([self.up2(x2), x1], 1))     # R/4
        y0 = self.dec1(torch.cat([self.up1(y1), x0], 1))     # R/2
        c_clean = c + self.out(y0)                   # residual correction
        out = idwt3d(c_clean, self.haar)             # (B, 1, R, R, R)
        if self.clamp is not None:
            out = torch.tanh(out / self.clamp) * self.clamp
        return out, c, c_clean

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #
def _grad3d(v):
    gx = v[..., 1:, :, :] - v[..., :-1, :, :]
    gy = v[..., :, 1:, :] - v[..., :, :-1, :]
    gz = v[..., :, :, 1:] - v[..., :, :, :-1]
    return gx, gy, gz


def wavelet_loss(pred, clean, c_pred=None, c_clean=None, lam_wave=1.0, lam_grad=0.1):
    """Reconstruction + wavelet-coefficient + gradient-consistency loss.

    * **TSDF L1** ``|pred − clean|`` — the field matches the clean surface.
    * **Wavelet L1** ``|c_pred − c_clean|`` (Δ-band of the design's "wavelet
      coefficient loss") — denoises multi-scale structure directly where the
      coherent detail vs. incoherent noise distinction lives.
    * **Gradient L1** — a smoothness / eikonal-flavored term matching the finite-
      difference gradient of the predicted and clean fields, suppressing residual
      voxel noise.

    Returns ``(loss, parts_dict)`` where ``parts_dict`` holds detached floats.
    """
    l_tsdf = (pred - clean).abs().mean()
    loss = l_tsdf
    parts = {"tsdf": float(l_tsdf.detach())}
    if c_pred is not None and lam_wave > 0.0:
        l_w = (c_pred - c_clean).abs().mean()
        loss = loss + lam_wave * l_w
        parts["wave"] = float(l_w.detach())
    if lam_grad > 0.0:
        gp, gc = _grad3d(pred), _grad3d(clean)
        l_g = sum((a - b).abs().mean() for a, b in zip(gp, gc)) / 3.0
        loss = loss + lam_grad * l_g
        parts["grad"] = float(l_g.detach())
    parts["loss"] = float(loss.detach())
    return loss, parts


# --------------------------------------------------------------------------- #
#  Training over the dense {P, N, ...} cache
# --------------------------------------------------------------------------- #
def train_wavelet(cache, *, res: int = 32, trunc: float = 0.1, bound: float = 1.1,
                  epochs: int = 4, batch: int = 8, n_points: int | None = None,
                  noise_std: float = 0.015, lr: float = 1e-3, lam_wave: float = 1.0,
                  lam_grad: float = 0.1, device="cpu", subset: int | None = None,
                  base: int = 32, log_every: int = 50, seed: int = 0, net=None):
    """Train a :class:`WaveletDenoiser` on noisy→clean TSDF pairs from a mesh cache.

    For every mesh in ``cache`` (a dict of ``P (A,Npts,3)``, ``N (A,Npts,3)`` CPU
    tensors, e.g. from :func:`pat.datasets.build_mesh_cache`) each step builds:

    * a **clean target** TSDF from the cached (clean) surface cloud, and
    * a **noisy input** TSDF from the same cloud with fresh Gaussian noise added,

    then supervises the network to map noisy → clean (plus the wavelet/gradient
    terms).  Fresh noise every step is the denoising signal.  Returns
    ``(net, history)`` where ``history`` is a list of per-epoch loss dicts.

    ``n_points`` optionally subsamples each cloud per step (``None`` = use all
    cached points); ``subset`` caps the number of meshes (``None`` = all).
    """
    assert res % 8 == 0, "res must be divisible by 8 (one DWT halving + two poolings)"
    P, N = cache["P"], cache["N"]
    A = P.shape[0] if subset is None else min(subset, P.shape[0])
    dense = P.shape[1]
    net = net or WaveletDenoiser(base=base).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    haar = haar_filters_3d(device)
    g = torch.Generator().manual_seed(seed)          # CPU generator (cache is on CPU)
    hist = []
    net.train()
    for ep in range(epochs):
        order = torch.randperm(A, generator=g).tolist()
        run, nb = 0.0, 0
        for s in range(0, A, batch):
            idx = torch.as_tensor(order[s:s + batch])
            Pc = P[idx]; Nc = N[idx]                 # (b, dense, 3) on CPU
            if n_points is not None and n_points < dense:
                sub = torch.argsort(torch.rand(len(idx), dense, generator=g), 1)[:, :n_points]
                bi = torch.arange(len(idx))[:, None]
                Pc, Nc = Pc[bi, sub], Nc[bi, sub]
            Pc = Pc.to(device); Nc = Nc.to(device)
            with torch.no_grad():
                clean = tsdf_from_clouds(Pc, Nc, res, trunc, bound, device) / trunc
                Pn = Pc + torch.randn(Pc.shape, generator=None, device=device) * noise_std
                noisy = tsdf_from_clouds(Pn, Nc, res, trunc, bound, device) / trunc
                target_c = dwt3d(clean, haar)
            pred, _c_noisy, c_pred = net(noisy)
            loss, parts = wavelet_loss(pred, clean, c_pred, target_c, lam_wave, lam_grad)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += parts["loss"]; nb += 1
            if log_every and nb % log_every == 0:
                print(f"  wavelet ep{ep} {min(s + batch, A)}/{A} loss {run / nb:.4f}",
                      flush=True)
        hist.append({"epoch": ep, "loss": run / max(nb, 1)})
        print(f"wavelet epoch {ep}: loss {run / max(nb, 1):.4f}", flush=True)
    return net, hist


# --------------------------------------------------------------------------- #
#  Inference wrapper (drop-in for PAT / splat in eval3d + render3d)
# --------------------------------------------------------------------------- #
class WaveletReconstruction:
    """A fitted SDF over one (noisy) cloud, produced by a trained denoiser.

    Builds the noisy-cloud TSDF, runs the network once to denoise it, and caches
    the resulting field (back in **distance units**).  Exposes the same surface as
    ``PAT`` / the splat models so it plugs into :func:`pat.eval3d.proper_metrics`
    and :mod:`pat.render3d`:

    * ``sdf(q)`` — trilinearly-sampled signed distance at world queries ``q (Q,3)``
      (``< 0`` inside; queries outside ``[-bound, bound]^3`` read as ``+trunc``).
    * ``reconstruct(level=0.0)`` — marching cubes of the field → ``(verts, faces)``.
    * ``M`` — a label placeholder (so ``eval3d.gallery_render`` formats cleanly).
    """

    def __init__(self, P, N, net, *, res: int = 32, trunc: float = 0.1,
                 bound: float = 1.1, device="cpu"):
        self.res, self.trunc, self.bound = res, trunc, bound
        self.M = ""                                  # label placeholder for galleries
        net = net.to(device).eval()
        with torch.no_grad():
            noisy = tsdf_from_clouds(P, N, res, trunc, bound, device) / trunc
            pred, _, _ = net(noisy)
        # back to distance units so `.sdf` is a true (truncated) signed distance
        self.grid = (pred[0, 0].detach().cpu().numpy() * trunc).astype(np.float64)

    def sdf(self, q):
        return grid_trilinear(self.grid, q, self.bound, self.trunc)

    def reconstruct(self, level: float = 0.0):
        from skimage import measure

        vol = self.grid
        if not (vol.min() < level < vol.max()):
            return None, None
        v, f, _, _ = measure.marching_cubes(vol, level=level)
        v = v / (self.res - 1) * (2 * self.bound) - self.bound
        return v, f
