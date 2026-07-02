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

import copy

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


def point_thinness(P, N, thin: float = 0.10, opp: float = -0.3, k: int = 24):
    """Per-point thin-sheet gate ``(B, n)``: 1 where a point has a NEAR neighbour with an OPPOSING normal
    (a thin / open shell -> use the UNSIGNED base); 0 where no such neighbour exists (the surface bounds a
    volume -> use the SIGNED base).  This is the analytic per-point BASE SELECTOR for the mixed field."""
    P = torch.as_tensor(P, dtype=torch.float32); N = torch.as_tensor(N, dtype=torch.float32)
    if P.dim() == 2:
        P, N = P[None], N[None]
    d = torch.cdist(P, P)                                       # (B,n,n)
    nd = torch.bmm(N, N.transpose(1, 2))                        # (B,n,n) normal alignment
    near = d.topk(min(k, P.shape[1]), dim=-1, largest=False)   # k nearest per point (self has d=0, nd=1 > opp -> excluded)
    ndk = torch.gather(nd, 2, near.indices)
    return ((near.values < thin) & (ndk < opp)).any(2).float()  # (B,n)


def tsdf_from_clouds(Ps, Ns, res: int = 32, trunc: float = 0.1, bound: float = 1.1,
                     device="cpu", qchunk: int = 4096, unsigned: bool = False,
                     mode: str | None = None, band: float = 0.04) -> torch.Tensor:
    """Batched TSDF grids for many clouds — ``(B, 1, res, res, res)`` (torch).

    ``Ps``/``Ns`` are ``(B, Npts, 3)`` (tensors or arrays).  Nearest-point lookup
    is a chunked :func:`torch.cdist` (GPU-friendly, no KD-tree), so the whole batch
    of meshes is voxelized in parallel.  Values are in distance units, clipped to
    ``±trunc``; ``< 0`` is inside.  Memory is bounded by ``qchunk`` grid points per
    step (peak ≈ ``B * qchunk * Npts`` floats).

    ``mode`` selects the field (``unsigned=True`` is a back-compat alias for ``mode='unsigned'``):
      * ``'signed'``   — truncated SIGNED distance (``<0`` inside); for closed solids.
      * ``'unsigned'`` — sign dropped, distance clamped to ``[0,trunc]`` (UDF); for open shells.
      * ``'mixed'``    — PER-POINT base selection: the unified field used by the mixed model.  Each grid
        voxel inherits its nearest point's :func:`point_thinness` gate ``g`` and blends signed (closed,
        ``g=0``) with ``unsigned-band`` (thin, ``g=1``); meshed at ``0`` it gives crisp solids in closed
        regions and clean shells in thin regions -- both bases in one field, selected per point.
    """
    Ps = torch.as_tensor(Ps, dtype=torch.float32, device=device)
    Ns = torch.as_tensor(Ns, dtype=torch.float32, device=device)
    if Ps.dim() == 2:                                # single cloud -> add batch dim
        Ps, Ns = Ps[None], Ns[None]
    if mode is None:
        mode = "unsigned" if unsigned else "signed"
    B = Ps.shape[0]
    t = point_thinness(Ps, Ns).to(device) if mode == "mixed" else None    # (B,n) per-point base selector
    lin = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    G = grid.shape[0]
    out = torch.empty(B, G, device=device)
    for a in range(0, G, qchunk):
        gq = grid[a:a + qchunk]                      # (q, 3)
        d = torch.cdist(gq.unsqueeze(0).expand(B, -1, -1), Ps)   # (B, q, Npts)
        dmin, idx = d.min(dim=2)                     # (B, q)
        if mode == "unsigned":                        # UDF: no sign, surface at 0
            out[:, a:a + qchunk] = dmin.clamp(0.0, trunc)
            continue
        ix = idx.unsqueeze(-1).expand(-1, -1, 3)
        near = torch.gather(Ps, 1, ix)               # (B, q, 3)
        nn = torch.gather(Ns, 1, ix)
        sign = ((gq.unsqueeze(0) - near) * nn).sum(-1)           # (B, q)
        sign = torch.where(sign >= 0.0, torch.ones_like(sign), -torch.ones_like(sign))
        signed = (sign * dmin).clamp(-trunc, trunc)
        if mode == "mixed":                           # per-point: signed (closed) blended with unsigned-band (thin)
            gate = torch.gather(t, 1, idx)            # (B,q) nearest-point thinness
            ub = dmin.clamp(0.0, trunc) - band        # 0 at distance=band -> clean shell at level 0
            out[:, a:a + qchunk] = ((1 - gate) * signed + gate * ub).clamp(-trunc, trunc)
        else:
            out[:, a:a + qchunk] = signed
    return out.reshape(B, 1, res, res, res)


# --------------------------------------------------------------------------- #
#  Trilinear sampler (TSDF grid  ->  callable SDF)
# --------------------------------------------------------------------------- #
def region_labels(P, N, k: int = 12, thresh: float = 0.80, min_pts: int = 48):
    """DYNAMIC point->surface-region allocation (the edge detector): greedy region growing on the point kNN
    graph where a neighbour joins the region if its normal is locally coherent (``|n_i . n_j| > thresh`` --
    UNSIGNED so a two-sided thin sheet is ONE region).  Smoothly-curved surfaces chain into a single region
    (sphere/bunny -> 1); normal JUMPS (edges/creases) split regions (cube -> 6 faces) -- so region boundaries
    ARE the detected edges.  Tiny fragments (< ``min_pts``) are merged into their nearest big region.
    ``P``/``N``: (n,3) numpy or tensors (single cloud).  Returns int labels (n,)."""
    P = torch.as_tensor(P, dtype=torch.float32); N = torch.as_tensor(N, dtype=torch.float32)
    n = P.shape[0]
    d = torch.cdist(P, P); knn = d.topk(min(k + 1, n), largest=False).indices[:, 1:]      # (n,k) neighbours
    lab = np.full(n, -1, dtype=np.int64); Nn = N.cpu().numpy(); knn_np = knn.cpu().numpy(); cur = 0
    for seed in range(n):
        if lab[seed] >= 0:
            continue
        stack = [seed]; lab[seed] = cur
        while stack:                                        # BFS: chain-coherent normals grow the region
            i = stack.pop()
            for j in knn_np[i]:
                if lab[j] < 0 and abs(float(Nn[i] @ Nn[j])) > thresh:
                    lab[j] = cur; stack.append(int(j))
        cur += 1
    # merge tiny fragments (noise/edge slivers) into the nearest big region
    sizes = np.bincount(lab); big = np.flatnonzero(sizes >= min_pts)
    if len(big) == 0:
        return np.zeros(n, dtype=np.int64)
    if len(big) < len(sizes):
        big_mask = np.isin(lab, big)
        dd = d.cpu().numpy(); dd[:, ~big_mask] = np.inf
        for i in np.flatnonzero(~big_mask):
            lab[i] = lab[dd[i].argmin()]
    _, lab = np.unique(lab, return_inverse=True)            # compact 0..R-1
    return lab


def region_pair_ops(P, N, lab, k: int = 12):
    """Per region-pair composition op from junction geometry: for boundary point pairs (a in region A with a
    kNN neighbour b in region B), the junction is CONVEX if each side's points lie BEHIND the other's tangent
    plane (``(p_a-p_b).n_b < 0`` and vice versa) -> intersection -> ``max``; else CONCAVE -> union -> ``min``.
    Majority vote over boundary pairs.  Returns dict {(A,B): +1 (max) | -1 (min)} with A<B."""
    P = torch.as_tensor(P, dtype=torch.float32); N = torch.as_tensor(N, dtype=torch.float32)
    n = P.shape[0]
    knn = torch.cdist(P, P).topk(min(k + 1, n), largest=False).indices[:, 1:].cpu().numpy()
    votes = {}
    Pn, Nn = P.cpu().numpy(), N.cpu().numpy()
    for i in range(n):
        for j in knn[i]:
            a, b = int(lab[i]), int(lab[j])
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            cv = (float((Pn[i] - Pn[j]) @ Nn[j]) < 0) and (float((Pn[j] - Pn[i]) @ Nn[i]) < 0)
            s, c = votes.get(key, (0, 0)); votes[key] = (s + (1 if cv else -1), c + 1)
    return {key: (1 if s >= 0 else -1) for key, (s, c) in votes.items()}


def tsdf_composed(P, N, lab, res: int = 32, trunc: float = 0.1, bound: float = 1.1, device="cpu",
                  band: float = 0.04, thin_tau: float = 0.5, ops=None, thin=None):
    """REGION-COMPOSED anchor/target TSDF (crust-free edges): build each region's field from ONLY its own
    points (sign coherent within a region -> smooth), then compose the per-voxel TWO nearest-surface regions
    with their junction op (convex -> max = intersection, concave -> min = union).  The edge becomes the exact
    intersection curve of two smooth fields instead of the ragged nearest-point Voronoi seam -> NO edge crust.
    Per-region S/UDF: a region whose mean :func:`point_thinness` exceeds ``thin_tau`` is a THIN sheet -> its
    field is the unsigned band (and any pair involving it composes with min/union).  Single region -> the plain
    per-region field (curved shapes unchanged).  ``P``/``N``: (n,3); returns ``(1,1,res,res,res)`` tensor."""
    P = torch.as_tensor(P, dtype=torch.float32, device=device); N = torch.as_tensor(N, dtype=torch.float32, device=device)
    R = int(lab.max()) + 1
    if thin is None:                                                 # (n,) per-point thin gate (cacheable)
        thin = point_thinness(P[None], N[None])[0]
    thin = torch.as_tensor(thin, device=device)
    fields, is_u = [], []
    for r in range(R):
        sel = torch.as_tensor(lab == r, device=device)
        u = bool(thin[sel].float().mean() > thin_tau)                # region S/UDF selection (dynamic)
        f = tsdf_from_clouds(P[sel][None], N[sel][None], res, trunc, bound, device,
                             mode="unsigned" if u else "signed")[0, 0]
        if u:
            f = f - band                                             # unsigned band -> zero-level shell
        fields.append(f); is_u.append(u)
    F_ = torch.stack(fields)                                         # (R, res,res,res)
    if R == 1:
        # single region: still apply the OPEN-CLOUD rule (an open one-sided scan encloses no volume -- its
        # signed field is a half-space sheet-mess; rebuild as an unsigned band shell).  Closed shapes
        # (normals cover the sphere, ||mean N|| ~ 0) keep their plain signed field.
        if not is_u[0] and bool(N.mean(0).norm() > 0.25):
            f = tsdf_from_clouds(P[None], N[None], res, trunc, bound, device, mode="unsigned")[0, 0] - band
            return f[None, None]
        return F_[0][None, None]
    # CONVEX-CLUSTER composition: regions linked by CONVEX junctions form one convex piece -> INTERSECTION
    # (max) within the cluster (a cube's 6 faces -> one cluster -> exact sharp box); clusters and thin/U
    # regions then combine by UNION (min) -- correct for concave junctions (L-shapes: deep inside one arm the
    # other arm's field is saturated-outside, min keeps it inside) and safe for every attachment.  No per-voxel
    # nearest-region selection -> no arbitrary tie-breaks in the saturated far zone (the crater bug).
    if ops is None:                                                  # junction votes (cacheable per shape)
        ops = region_pair_ops(P.cpu(), N.cpu(), lab)
    parent = list(range(R))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (a, b), s in ops.items():
        if s > 0 and not is_u[a] and not is_u[b]:                    # convex signed junction -> same cluster
            parent[find(a)] = find(b)
    roots = {}
    for r in range(R):
        roots.setdefault(find(r), []).append(r)
    # PER-PIECE SUPPORT BOUND (the +5% rule, applied per piece): a piece has NO OPINION beyond its own points'
    # bbox+5% (a small patch's signed field otherwise extends as an infinite half-space -> phantom petals /
    # slabs when union'd), and within the bbox any near-zero surface farther than ~5% of the piece diagonal
    # from its points is forced OUTSIDE (floaters).  Deep-inside voxels (f < -trunc/2) INSIDE the bbox are
    # protected, so solid cores survive.  Being per-piece, this bounds each surface to ITS OWN extent -- the
    # global-AABB flaw (can't respect internal holes / multi-part extents) does not apply.
    lin = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1)          # (res,res,res,3)
    pieces = []
    multi = len(roots) > 1
    for m in roots.values():
        sel = torch.as_tensor(np.isin(lab, m), device=device)
        Pp, Np = P[sel], N[sel]
        lo = Pp.amin(0); hi = Pp.amax(0); ext = (hi - lo).clamp_min(1e-6)
        # OPEN pieces get an UNSIGNED band shell instead of a signed half-space fill: (a) FLAT piece = an open
        # one-sided surface (scan ground/wall -- no volume along its normal; a cube FACE clusters into the 3D
        # cube piece first so it never hits this); (b) OPEN CLOUD = the whole cloud's normals are one-sided
        # (||mean N|| large -> an open scan encloses no volume; closed shapes' normals cover the sphere).
        flat = bool(ext.min() < 0.10 * ext.norm()) or bool(N.mean(0).norm() > 0.25)
        if flat and not all(is_u[r] for r in m):
            f = tsdf_from_clouds(Pp[None], Np[None], res, trunc, bound, device, mode="unsigned")[0, 0] - band
        else:
            f = F_[m].max(0).values if len(m) > 1 else F_[m[0]]
        if multi:
            # PER-PIECE SUPPORT BOUND (union artifacts only exist with >1 piece; a single piece defers to the
            # net's distance clamp): no opinion beyond the piece bbox+5%, and near-surface opinions farther
            # than the support radius (5% diag, >= sampling density x2, >= 3 voxels) are dropped -> a small
            # patch cannot leak an infinite half-space (petals/slabs) into the union.
            out_box = ((grid < lo - 0.05 * ext) | (grid > hi + 0.05 * ext)).any(-1)
            nn_med = torch.cdist(Pp[None], Pp[None])[0].topk(2, largest=False).values[:, 1].median()
            thr = torch.maximum(0.05 * ext.norm(), 2.0 * nn_med).clamp_min(3.0 * bound / res)
            gq = grid.reshape(-1, 3)
            far = torch.empty(gq.shape[0], dtype=torch.bool, device=device)
            for a0 in range(0, gq.shape[0], 8192):
                far[a0:a0 + 8192] = torch.cdist(gq[a0:a0 + 8192][None], Pp[None])[0].amin(1) > thr
            far = far.view(res, res, res)
            kill = out_box | (far & (f > -0.5 * trunc))              # no-opinion zone -> force OUTSIDE
            f = torch.where(kill, torch.full_like(f, trunc), f)
        pieces.append(f)
    out = torch.stack(pieces).min(0).values                          # union of convex pieces + thin shells
    return out[None, None]


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

    def __init__(self, base: int = 32, levels: int = 3, global_ctx: bool = True,
                 clamp: float | None = None):
        super().__init__()
        c0 = 8                                       # the 8 wavelet subbands
        self.clamp = clamp; self.levels = levels; self.use_gctx = global_ctx
        chans = [base * (2 ** i) for i in range(levels + 1)]   # base, 2b, 4b, ... -> wider+deeper
        self.in_block = _ConvBlock(c0, chans[0])
        self.pool = nn.AvgPool3d(2)
        self.downs = nn.ModuleList([_ConvBlock(chans[i], chans[i + 1]) for i in range(levels)])
        # bottleneck: a DILATED conv widens the receptive field without extra pooling, and a
        # GLOBAL-CONTEXT branch (global avg-pool -> MLP -> broadcast-add) injects whole-field
        # information into every voxel -- so the refiner is no longer confined to a local window.
        cb = chans[-1]
        self.bottleneck = nn.Sequential(nn.Conv3d(cb, cb, 3, padding=2, dilation=2), _gn(cb), nn.SiLU(),
                                        nn.Conv3d(cb, cb, 3, padding=1), _gn(cb), nn.SiLU())
        if global_ctx:
            self.gctx = nn.Sequential(nn.Linear(cb, cb), nn.SiLU(), nn.Linear(cb, cb))
        self.ups = nn.ModuleList([nn.ConvTranspose3d(chans[i + 1], chans[i], 2, stride=2)
                                  for i in reversed(range(levels))])
        self.decs = nn.ModuleList([_ConvBlock(chans[i] * 2, chans[i]) for i in reversed(range(levels))])
        self.out = nn.Conv3d(base, c0, 1)
        nn.init.zeros_(self.out.weight)              # residual starts at 0 -> identity
        nn.init.zeros_(self.out.bias)
        self.register_buffer("haar", haar_filters_3d())

    def forward(self, tsdf):
        c = dwt3d(tsdf, self.haar)                   # (B, 8, R/2, R/2, R/2)
        x = self.in_block(c)
        skips = [x]
        for d in self.downs:
            x = d(self.pool(x)); skips.append(x)     # coarsen toward a (near-)global bottleneck
        x = self.bottleneck(x)
        if self.use_gctx:
            g = self.gctx(x.mean(dim=(2, 3, 4)))     # (B, cb) whole-field summary
            x = x + g[:, :, None, None, None]        # broadcast global context to every voxel
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], 1))
        c_clean = c + self.out(x)                    # residual correction
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
@torch.no_grad()
def wavelet_val_error(net, P, N, val_idx, *, res, trunc, bound, noise_std,
                      device="cpu", mb=8, seed=0):
    """Mean held-out TSDF denoising error over ``val_idx`` meshes (no grad).

    For each validation mesh builds the clean + a **fixed-noise** noisy TSDF, runs
    the denoiser, and returns ``mean |pred − clean|`` (normalized TSDF scale) — the
    model-selection metric.  Returns ``inf`` if nothing finite was measured.
    """
    net.eval()
    gv = torch.Generator().manual_seed(seed)
    tot, cnt = 0.0, 0
    for s in range(0, len(val_idx), mb):
        idx = val_idx[s:s + mb]
        Pc = P[idx]; Nc = N[idx]
        noise = torch.randn(Pc.shape, generator=gv) * noise_std
        clean = tsdf_from_clouds(Pc.to(device), Nc.to(device), res, trunc, bound, device) / trunc
        noisy = tsdf_from_clouds((Pc + noise).to(device), Nc.to(device), res, trunc, bound, device) / trunc
        pred, _, _ = net(noisy)
        err = (pred - clean).abs().mean()
        if torch.isfinite(err):
            tot += float(err) * len(idx); cnt += len(idx)
    net.train()
    return tot / cnt if cnt else float("inf")


def train_wavelet(cache, *, res: int = 32, trunc: float = 0.1, bound: float = 1.1,
                  epochs: int = 4, batch: int = 8, n_points: int | None = None,
                  noise_std: float = 0.015, lr: float = 1e-3, lam_wave: float = 1.0,
                  lam_grad: float = 0.1, device="cpu", subset: int | None = None,
                  base: int = 32, n_val: int | None = None, log_every: int = 50,
                  seed: int = 0, net=None):
    """Train a :class:`WaveletDenoiser` on noisy→clean TSDF pairs from a mesh cache.

    For every mesh in ``cache`` (a dict of ``P (A,Npts,3)``, ``N (A,Npts,3)`` CPU
    tensors, e.g. from :func:`pat.datasets.build_mesh_cache`) each step builds:

    * a **clean target** TSDF from the cached (clean) surface cloud, and
    * a **noisy input** TSDF from the same cloud with fresh Gaussian noise added,

    then supervises the network to map noisy → clean (plus the wavelet/gradient
    terms).  Fresh noise every step is the denoising signal.  Returns
    ``(net, history)`` (per-epoch loss / val dicts).

    **Best-by-validation selection.**  A fixed random slice of ``n_val`` meshes is
    held out; after each epoch the held-out TSDF denoising error is measured
    (:func:`wavelet_val_error`) and the **best-by-val** weights are snapshotted.  The
    returned net is loaded with those best weights, so saving it persists the best
    epoch (pass ``n_val=0`` to keep the final weights).

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

    g0 = torch.Generator().manual_seed(seed)
    perm = torch.randperm(A, generator=g0)
    if n_val is None:
        n_val = min(256, max(1, A // 5))
    n_val = max(0, min(int(n_val), A - 1)) if A > 1 else 0
    val_idx = perm[:n_val]
    train_pool = perm[n_val:]

    g = torch.Generator().manual_seed(seed + 1)      # CPU generator (cache is on CPU)
    hist = []
    best_val, best_ep, best_state = float("inf"), -1, None
    net.train()
    for ep in range(epochs):
        tr = train_pool[torch.randperm(len(train_pool), generator=g)]
        run, nb, skipped = 0.0, 0, 0
        for s in range(0, len(tr), batch):
            idx = tr[s:s + batch]
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
            if not torch.isfinite(loss):              # skip a degenerate batch (don't poison weights)
                skipped += 1
                continue
            loss.backward()
            for p in net.parameters():                # sanitize any NaN/Inf grads before clipping
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += parts["loss"]; nb += 1
            if log_every and nb % log_every == 0:
                print(f"  wavelet ep{ep} {min(s + batch, len(tr))}/{len(tr)} loss {run / nb:.4f}",
                      flush=True)
        val = (wavelet_val_error(net, P, N, val_idx, res=res, trunc=trunc, bound=bound,
                                 noise_std=noise_std, device=device, seed=seed) if n_val else float("nan"))
        hist.append({"epoch": ep, "loss": run / max(nb, 1), "val": val, "skipped": skipped})
        if n_val and val < best_val:
            best_val, best_ep = val, ep
            best_state = copy.deepcopy({kk: vv.detach().cpu() for kk, vv in net.state_dict().items()})
        print(f"wavelet epoch {ep}: loss {run / max(nb, 1):.4f} | val {val:.4f} | "
              f"skipped {skipped} bad steps", flush=True)
    if best_state is not None:                        # restore the best-by-val weights
        net.load_state_dict(best_state)
        print(f"wavelet: selected BEST epoch {best_ep} (val {best_val:.4f})", flush=True)
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


# --------------------------------------------------------------------------- #
#  Wavelet-native surface model (NO primitives): point splat -> wavelet U-Net -> SDF
# --------------------------------------------------------------------------- #
@torch.no_grad()
def splat_input_grid(P, N, res: int = 64, trunc: float = 0.1, bound: float = 1.1, device="cpu"):
    """Primitive-free 5-channel input grid straight from a point cloud (NO tori / kNN / CoeffNet).

    Channels: ``0`` occupancy (voxel hit), ``1:4`` mean unit normal, ``4`` the direct point TSDF
    (:func:`tsdf_from_clouds`).  Returns ``(B, 5, res, res, res)``.  This replaces the per-point
    primitive blend as the model's input -- the shape is defined only by *where the points are* and
    *which way they face*, then the wavelet net turns that into a clean SDF.
    """
    P = torch.as_tensor(P, dtype=torch.float32, device=device)
    N = torch.as_tensor(N, dtype=torch.float32, device=device)
    if P.dim() == 2:
        P, N = P[None], N[None]
    B, Np, _ = P.shape
    vi = (((P + bound) / (2 * bound)) * res).long().clamp(0, res - 1)            # (B,Np,3) voxel idx
    flat = (vi[..., 0] * res + vi[..., 1]) * res + vi[..., 2]                    # (B,Np) row-major
    ones = torch.ones_like(flat, dtype=torch.float32)
    occ = torch.zeros(B, res ** 3, device=device).scatter_add_(1, flat, ones)   # point count / voxel
    nrm = torch.zeros(B, 3, res ** 3, device=device)
    for c in range(3):
        nrm[:, c].scatter_add_(1, flat, N[..., c])
    nrm = nrm / occ.clamp_min(1.0)[:, None]                                      # mean normal
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-6)                    # re-normalize
    grid = torch.zeros(B, 5, res, res, res, device=device)
    grid[:, 0] = (occ > 0).float().reshape(B, res, res, res)
    grid[:, 1:4] = nrm.reshape(B, 3, res, res, res)
    grid[:, 4:5] = tsdf_from_clouds(P, N, res, trunc, bound, device) / trunc
    return grid


def wavelet_side_labels(target_c, surf: float = 0.5):
    """Per-voxel 'flat side' pseudo-labels from wavelet sparsity (no extra data).

    ``target_c = dwt3d(clean_tsdf)`` ``(B,8,r,r,r)``.  Flat faces are SPARSE in the 7 detail bands;
    edges/creases concentrate high-frequency energy.  Inside a near-surface band (``|coarse LLL| <
    surf``), voxels with below-median detail energy are labelled ``1`` (a similarity segment / side),
    else ``0``.  Restricting to the surface band stops far/empty voxels (also hf-sparse) from
    dominating.  Returns ``(B,1,r,r,r)``.
    """
    coarse = target_c[:, 0:1]
    hf = target_c[:, 1:8].abs().sum(1, keepdim=True)
    near = coarse.abs() < surf
    label = torch.zeros_like(hf)
    for b in range(hf.shape[0]):
        m = near[b]
        if m.any():
            med = hf[b][m].median()
            label[b] = ((hf[b] < med) & m).float()
    return label


class WaveletSurfaceNet(nn.Module):
    """Primitive-free, wavelet-DOMAIN surface net (sibling of :class:`WaveletDenoiser`).

    Input is the 5-channel point splat ``(B,5,R,R,R)`` from :func:`splat_input_grid` (occupancy +
    mean-normal + direct TSDF -- NO primitives).  Every channel is Haar-transformed → ``(B,40,R/2³)``
    (5·8 subbands); a 3-D U-Net over the subbands predicts a **residual to the TSDF channel's 8
    coefficients**; the inverse transform gives the clean SDF.  Anchoring the residual to the input
    TSDF's coefficients makes the *untrained* net return exactly the direct point TSDF (identity
    start -- the correct primitive-free baseline).  A light 1×1 head emits per-voxel side-segment
    logits at ``R/2`` for the wavelet-sparsity auxiliary loss.
    """

    def __init__(self, base: int = 40, levels: int = 3, in_ch: int = 5,
                 global_ctx: bool = True, with_seg: bool = True, clamp: float | None = None):
        super().__init__()
        self.in_ch = in_ch; self.clamp = clamp; self.levels = levels
        self.use_gctx = global_ctx; self.with_seg = with_seg
        c0 = in_ch * 8
        chans = [base * (2 ** i) for i in range(levels + 1)]
        self.band_gate = nn.Parameter(torch.ones(c0))               # per-(channel,subband) gate
        self.in_block = _ConvBlock(c0, chans[0])
        self.pool = nn.AvgPool3d(2)
        self.downs = nn.ModuleList([_ConvBlock(chans[i], chans[i + 1]) for i in range(levels)])
        cb = chans[-1]
        self.bottleneck = nn.Sequential(nn.Conv3d(cb, cb, 3, padding=2, dilation=2), _gn(cb), nn.SiLU(),
                                        nn.Conv3d(cb, cb, 3, padding=1), _gn(cb), nn.SiLU())
        if global_ctx:
            self.gctx = nn.Sequential(nn.Linear(cb, cb), nn.SiLU(), nn.Linear(cb, cb))
        self.ups = nn.ModuleList([nn.ConvTranspose3d(chans[i + 1], chans[i], 2, stride=2)
                                  for i in reversed(range(levels))])
        self.decs = nn.ModuleList([_ConvBlock(chans[i] * 2, chans[i]) for i in reversed(range(levels))])
        self.out = nn.Conv3d(base, 8, 1)                            # residual on the TSDF subbands only
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # -> identity start
        if with_seg:
            self.seg = nn.Conv3d(base, 1, 1)                        # light side-segmentation head
        self.register_buffer("haar", haar_filters_3d())

    def forward(self, grid5):
        B = grid5.shape[0]
        c = dwt3d(grid5.reshape(B * self.in_ch, 1, *grid5.shape[2:]), self.haar)
        c = c.reshape(B, self.in_ch * 8, *c.shape[2:])              # (B,40,R/2³)
        c_tsdf = c[:, (self.in_ch - 1) * 8:self.in_ch * 8]          # the TSDF channel's 8 subbands
        x = self.in_block(c * self.band_gate[None, :, None, None, None])
        skips = [x]
        for d in self.downs:
            x = d(self.pool(x)); skips.append(x)
        x = self.bottleneck(x)
        if self.use_gctx:
            x = x + self.gctx(x.mean(dim=(2, 3, 4)))[:, :, None, None, None]
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], 1))
        c_clean = c_tsdf + self.out(x)                              # residual ANCHORED to direct-TSDF coeffs
        out = idwt3d(c_clean, self.haar)
        if self.clamp is not None:
            out = torch.tanh(out / self.clamp) * self.clamp
        seg = self.seg(x) if self.with_seg else None                # (B,1,R/2³) side logits
        return out, c_tsdf, c_clean, seg

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _smooth_grid(grid, sigma: float = 0.8):
    """Light Gaussian blur of an SDF grid -> fills speckle holes / pockets so marching cubes
    yields a CONTINUOUS, hole-free surface.  ``sigma`` in voxels; ``0`` disables."""
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(np.asarray(grid, dtype=np.float64), sigma=sigma)
    return grid


def keep_main_grid(grid, drop_frac: float = 1.0):
    """Collapse detached inside-blobs (floaters).  ``drop_frac=1.0`` -> keep ONLY the largest connected inside
    region (signed solids: one body).  ``drop_frac<1.0`` -> keep every inside component at least ``drop_frac``
    the size of the largest (mixed: collapse tiny floaters but preserve legitimately-disjoint solid parts)."""
    from scipy import ndimage
    inside = grid < 0
    if not inside.any():
        return grid                                          # pure open/unsigned shell: nothing to collapse
    lbl, n = ndimage.label(inside)
    if n <= 1:
        return grid
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    keep = np.flatnonzero(sizes >= max(1.0, drop_frac * sizes.max()))
    g = grid.copy(); g[inside & ~np.isin(lbl, keep)] = abs(grid).max()
    return g


def adaptive_eps_mesh(g, bound: float = 1.1, lo: float = 0.045, hi: float = 0.075, w: int = 3, delta: float = 0.013):
    """DYNAMIC eps (unsigned/UDF): per-voxel band tracking the local field floor, clamped to ``[lo,hi]``
    (detail->tight, sparse flat faces->wide; never 0=holes nor inf=fat).  Meshes ``{g = eps_field(x)}``."""
    from scipy import ndimage
    from skimage import measure
    eps_field = ndimage.gaussian_filter(np.clip(ndimage.minimum_filter(g, size=w) + delta, lo, hi), 0.8)
    gg = g - eps_field
    if not (gg.min() < 0 < gg.max()):
        return None, None
    v, f, _, _ = measure.marching_cubes(gg.astype(np.float64), 0.0)
    return v / (g.shape[0] - 1) * (2 * bound) - bound, f


def mesh_field(grid, field_mode: str = "mixed", *, bound: float = 1.1, trunc: float = 0.1, smooth: float = 0.5):
    """CANONICAL field-mode-aware mesher -- the single meshing entry point so DYNAMIC eps (unsigned band) and
    floater COLLAPSING can never be silently disengaged.  ``smooth`` is a LIGHT anti-alias gaussian only (0.5):
    edge quality comes from the region-COMPOSED field (see :func:`tsdf_composed` -- edges are exact
    intersections of smooth per-region fields), so heavy smoothing would only round the sharp edges.
    Returns ``(verts, faces)`` in world coords:
      * ``'unsigned'`` -> :func:`adaptive_eps_mesh` (dynamic per-voxel eps band);
      * ``'signed'``   -> :func:`keep_main_grid` (largest body only) + marching cubes at level 0;
      * ``'mixed'``    -> :func:`keep_main_grid` with a small drop-fraction (collapse tiny floaters, keep legit
                          disjoint parts / thin shells) + marching cubes at level 0."""
    from skimage import measure
    g = _smooth_grid(grid, smooth)
    if field_mode == "unsigned":
        return adaptive_eps_mesh(g, bound)
    g = keep_main_grid(g, drop_frac=1.0 if field_mode == "signed" else 0.03)
    if not (g.min() < 0 < g.max()):
        return None, None
    v, f, _, _ = measure.marching_cubes(g.astype(np.float64), 0.0)
    return v / (g.shape[0] - 1) * (2 * bound) - bound, f


def _crease_saliency(v, band: float = 0.3, eps: float = 1e-6):
    """Near-surface normal-INCOHERENCE weight ``(B,1,R,R,R)``: ~1 at creases/corners/crust (normals scatter),
    ~0 on flat faces and smooth curves (normals agree).  Used to (a) down-weight field-fidelity at creases so
    the wavelet edge-refiner can reshape them freely, and (b) locate where the crust penalty applies."""
    gx = F.pad((v[..., 2:, :, :] - v[..., :-2, :, :]) * 0.5, (0, 0, 0, 0, 1, 1))
    gy = F.pad((v[..., :, 2:, :] - v[..., :, :-2, :]) * 0.5, (0, 0, 1, 1, 0, 0))
    gz = F.pad((v[..., :, :, 2:] - v[..., :, :, :-2]) * 0.5, (1, 1, 0, 0, 0, 0))
    n = torch.cat([gx, gy, gz], 1); n = n / (n.norm(dim=1, keepdim=True) + eps)
    coh = F.avg_pool3d(n, 3, 1, 1).norm(dim=1, keepdim=True)
    return (1.0 - coh).clamp(0, 1) * (v.abs() < band).float()


def wavelet_surface_loss(pred, clean, c_clean, target_c, seg_logits=None, seg_label=None,
                         lam_wave: float = 0.3, lam_grad: float = 0.05, lam_seg: float = 0.05,
                         lam_smooth: float = 0.1, lam_sign: float = 0.25, lam_conn: float = 0.0,
                         lam_geo: float = 0.0, lam_corner: float = 0.0):
    """Composite best-loss for :class:`WaveletSurfaceNet`.

    Terms: field smooth-L1 + wavelet-coeff L1 + gradient + a CONTINUITY/de-speckle penalty (per-voxel
    high-frequency deviation from the 3³ local mean) + a SIGN-AGREEMENT penalty that targets noisy
    mesh artifacts -- ``relu(-pred*clean)`` is nonzero only where ``pred`` and ``clean`` disagree on
    inside/outside, weighted by ``|pred·clean|``, so a confident FLOATING fragment (pred≈−1 where the
    true field is +1) or a HOLE (pred≈+1 where clean<0) is penalized hard while correct voxels cost
    nothing -- + a CONNECTIVITY penalty (``lam_conn``) that keeps the detail bands from tearing the
    connected body, by penalising sign disagreement between the full field and the COARSE-only
    reconstruction ``\\phi_{lll}`` (a smooth, watertight body): detail may move the surface but may not
    flip inside/outside away from the connected coarse shape -- + side-segmentation BCE.
    """
    # (targets are the region-COMPOSED fields -- already crust-free and SHARP at edges; match them exactly)
    L = F.smooth_l1_loss(pred, clean, beta=0.1) + lam_wave * (c_clean - target_c).abs().mean()
    L = L + lam_grad * sum((a - b).abs().mean() for a, b in zip(_grad3d(pred), _grad3d(clean))) / 3.
    if lam_smooth:                                      # hole-less continuity: suppress hi-freq speckle
        L = L + lam_smooth * (pred - F.avg_pool3d(pred, 3, stride=1, padding=1)).abs().mean()
    if lam_sign:                                        # kill floating fragments / holes (sign disagreement)
        L = L + lam_sign * F.relu(-pred * clean).mean()
    if lam_conn:                                        # KEEP CONNECTED: the coarse band is a watertight
        c_co = c_clean.clone(); c_co[:, 1:] = 0         # body; penalise detail flipping the sign away
        phi_co = idwt3d(c_co).detach()                  # from it (no holes / no detached fragments)
        L = L + lam_conn * F.relu(-pred * phi_co).mean()
    if seg_logits is not None and seg_label is not None:
        L = L + lam_seg * F.binary_cross_entropy_with_logits(seg_logits, seg_label)
    # ---- geometry-quality block: differentiable FIELD proxies for the mesh metrics where ours is weak ----
    # NOTE: a previous version scaled this to lam_geo x the base-loss VALUE; because the geo terms are tiny
    # (the anchor already satisfies the anti-floater almost everywhere), that normalisation blew up the geo
    # GRADIENT and degraded the clean field.  We now apply lam_geo as a plain FIXED weight at the terms'
    # natural magnitude -- gentle, no gradient amplification.  Empirically the block still tends to trade clean
    # accuracy for little gain (cf. the v2 retrain), so it is OFF by default; opt in and verify with gen_table.
    if lam_geo:
        far = (clean.abs() >= 0.4).float()                          # CHAMFER / #COMPONENTS: no spurious zero-crossing
        l_float = (F.relu(0.15 - pred.abs()) * far).mean()          #   where clean is clearly in/out (floaters)
        band = (clean.abs() < 0.30).float()                         # HOLES / #COMPONENTS: de-speckle the near-surface
        l_band = ((pred - F.avg_pool3d(pred, 3, stride=1, padding=1)).abs() * band).mean()   # band (pinholes/fragments)
        interior = (clean < -0.1).float()                           # F-CLOSED: sharper interior -> crisper closed solids
        l_closed = (F.smooth_l1_loss(pred, clean, beta=0.1, reduction="none") * interior).mean()  # (open shells have none)
        L = L + lam_geo * (l_float + l_band + l_closed)             # FIXED weight (no magnitude normalisation)
    # ---- CRUST penalty (self-supervised signal that DRIVES the wavelet edge-refiner) ----------------------
    # Adjacent surface normals (field gradients) that flip >90 deg = the surface folding over itself = the
    # jagged crease crust.  Penalise those gradient-direction reversals in the near-surface band on `pred` (the
    # refined field) so the edge-refiner is rewarded for reshaping folded creases into clean edges.  A clean
    # sharp 90-deg edge (normals turn but don't reverse -> dot~=0) is NOT penalised, so edges stay crisp.
    if lam_corner:
        gx = F.pad((pred[..., 2:, :, :] - pred[..., :-2, :, :]) * 0.5, (0, 0, 0, 0, 1, 1))
        gy = F.pad((pred[..., :, 2:, :] - pred[..., :, :-2, :]) * 0.5, (0, 0, 1, 1, 0, 0))
        gz = F.pad((pred[..., :, :, 2:] - pred[..., :, :, :-2]) * 0.5, (1, 1, 0, 0, 0, 0))
        g = torch.cat([gx, gy, gz], 1)                              # (B,3,R,R,R) centered gradient (SDF normal)
        gn = g / (g.norm(dim=1, keepdim=True) + 1e-6)
        band = (pred.abs() < 0.3).float()
        lc = 0.0
        for ax in (2, 3, 4):                                        # x/y/z neighbour normal dot-products
            a = gn.narrow(ax, 1, gn.size(ax) - 1); b = gn.narrow(ax, 0, gn.size(ax) - 1)
            bnd = band.narrow(ax, 1, band.size(ax) - 1)
            lc = lc + (F.relu(-(a * b).sum(1, keepdim=True)) * bnd).mean()
        L = L + lam_corner * lc / 3.0
    return L


class WaveletSurfaceReconstruction:
    """Primitive-free wavelet-native reconstruction (drop-in for ``proper_metrics`` / renderers)."""

    def __init__(self, P, N, net, *, res: int = 64, trunc: float = 0.1, bound: float = 1.1,
                 device="cpu", smooth: float = 0.8):
        self.res, self.trunc, self.bound, self.M = res, trunc, bound, ""
        net = net.to(device).eval()
        with torch.no_grad():
            if isinstance(net, PerceiverWaveNet):                  # resolution-free point input
                Pb = torch.as_tensor(P, dtype=torch.float32, device=device)
                Nb = torch.as_tensor(N, dtype=torch.float32, device=device)
                if Pb.dim() == 2: Pb, Nb = Pb[None], Nb[None]
                pred = net(Pb, Nb)[0]
            else:
                grid5 = splat_input_grid(P, N, res, trunc, bound, device)
                pred = net(grid5)[0]
        self.grid = _smooth_grid((pred[0, 0].detach().cpu().numpy() * trunc).astype(np.float64), smooth)

    def sdf(self, q, neighbors=None):
        return grid_trilinear(self.grid, q, self.bound, self.trunc)

    def reconstruct(self, res=None, bound=None, neighbors=None, level: float = 0.0):
        from skimage import measure
        vol = self.grid
        if not (vol.min() < level < vol.max()):
            return None, None
        v, f, _, _ = measure.marching_cubes(vol, level=level)
        v = v / (self.res - 1) * (2 * self.bound) - self.bound
        return v, f


# =========================================================================== #
#  PerceiverWaveNet : wavelet-from-attention  (no conv U-Net, no input grid)
#  A Perceiver encoder (M latents cross-attend the point cloud, L self-attn
#  blocks) summarises the shape; a decoder queries a (res/2)^3 lattice and emits
#  the COARSE band from the global latents and the 7 DETAIL bands from each
#  query's k nearest point tokens.  Coeffs are anchored to the direct point-TSDF
#  (zero-init heads -> identity start) and idwt'd to the SDF.  Drop-in for
#  WaveletSurfaceNet: returns (out, c_anchor, c_clean, seg).
# =========================================================================== #
def fourier_encode(x, bands: int = 8):
    """``(...,3)`` in ~[-1,1] -> ``(...,3*2*bands)`` sinusoidal positional features."""
    freqs = (2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)) * np.pi
    xb = x[..., None] * freqs
    return torch.cat([torch.sin(xb), torch.cos(xb)], -1).flatten(-2)


def _fps(x, n):
    """Farthest-point sampling: ``x (B,N,3)`` -> indices ``(B,n)`` covering the cloud."""
    B, N = x.shape[0], x.shape[1]
    n = min(n, N)
    idx = torch.zeros(B, n, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), 1e10, device=x.device)
    far = torch.zeros(B, dtype=torch.long, device=x.device)
    ar = torch.arange(B, device=x.device)
    for i in range(n):
        idx[:, i] = far
        d = ((x - x[ar, far][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(1)
    return idx


class _MHA(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)

    def forward(self, x, ctx):
        B, Nq, D = x.shape; Nk = ctx.shape[1]
        q = self.q(x).view(B, Nq, self.h, self.dk).transpose(1, 2)
        k = self.k(ctx).view(B, Nk, self.h, self.dk).transpose(1, 2)
        v = self.v(ctx).view(B, Nk, self.h, self.dk).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.dk), -1)
        return self.o((a @ v).transpose(1, 2).reshape(B, Nq, D))


class _CrossBlock(nn.Module):
    def __init__(self, d, heads=8, mlp=4):
        super().__init__()
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d); self.att = _MHA(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(), nn.Linear(d * mlp, d))

    def forward(self, x, ctx):
        x = x + self.att(self.nq(x), self.nk(ctx))
        return x + self.ff(self.n2(x))


class _SelfBlock(nn.Module):
    def __init__(self, d, heads=8, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.att = _MHA(d, heads); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(), nn.Linear(d * mlp, d))

    def forward(self, x):
        h = self.n1(x); x = x + self.att(h, h)
        return x + self.ff(self.n2(x))


class _LocalAttn(nn.Module):
    """Per-query multi-head attention over each query's own ``k`` neighbour tokens ``(B,Q,k,d)``."""
    def __init__(self, d, heads=8, mlp=4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(), nn.Linear(d * mlp, d))

    def forward(self, x, nbr):                              # x:(B,Q,d) nbr:(B,Q,k,d)
        B, Q, k, D = nbr.shape
        qn, kn = self.nq(x), self.nk(nbr)
        q = self.q(qn).view(B, Q, self.h, self.dk)
        kk = self.k(kn).view(B, Q, k, self.h, self.dk)
        vv = self.v(kn).view(B, Q, k, self.h, self.dk)
        a = torch.softmax((q[:, :, None] * kk).sum(-1) / np.sqrt(self.dk), 2)   # (B,Q,k,h)
        out = (a[..., None] * vv).sum(2).reshape(B, Q, D)
        x = x + self.o(out)
        return x + self.ff(self.n2(x))


class EpsNet(nn.Module):
    """Learned, data-dependent UDF meshing band.  A small 3D CNN reads the (unsigned) field grid and
    predicts the band ``eps`` that minimises reconstruction loss (tight enough to hug the surface, wide
    enough to be speckle/hole-free).  Trained by regression to the per-mesh loss-optimal eps; replaces the
    analytic ``auto_eps`` heuristic with a single forward pass."""

    def __init__(self, lo=0.035, hi=0.085):
        super().__init__()
        self.lo, self.hi = lo, hi
        self.body = nn.Sequential(
            nn.Conv3d(1, 8, 3, 2, 1), nn.GroupNorm(2, 8), nn.GELU(),      # R -> R/2
            nn.Conv3d(8, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU(),    # R/2 -> R/4
            nn.Conv3d(16, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),   # R/4 -> R/8
            nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, g):                                      # g: (B,1,R,R,R) smoothed UDF field (distance units)
        e = torch.sigmoid(self.head(self.body(g)))[:, 0]
        return self.lo + (self.hi - self.lo) * e               # (B,) eps in distance units


def distance_field_clamp(v, P, bound: float = 1.1, thresh_frac: float = 0.05, inside_th: float = 0.5,
                         outside_val: float = 1.0, qchunk: int = 8192):
    """PER-VOXEL spatial clamp (req 1, generalises the AABB box): force the field OUTSIDE (>= ``outside_val``)
    at every grid voxel whose distance to the NEAREST input point exceeds ``thresh_frac`` of the point-cloud
    bbox diagonal -- UNLESS the voxel is confidently INSIDE a solid (``v < -inside_th``), which is protected so
    solid interiors are never hollowed.  Unlike a bounding box this also carves INTERNAL empty regions (e.g. a
    teapot spout/handle tube's hollow, a room's open interior) and kills external flying sheets, while a genuine
    solid core (far from surface points but enclosed) is kept.  ``P`` = (box-frame) cloud ``(B,n,3)``; ``v`` =
    ``(B,1,R,R,R)`` on the ``linspace(-bound,bound,R)`` lattice.  Nearest-point distance is a chunked cdist."""
    B, R = v.shape[0], v.shape[-1]
    lo = P.amin(1); hi = P.amax(1)
    thr = (thresh_frac * (hi - lo).norm(dim=1)).clamp_min(1.5 * 2 * bound / R)   # (B,) >= ~1.5 voxels
    lin = torch.linspace(-bound, bound, R, device=v.device, dtype=v.dtype)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)   # (G,3)
    far = torch.empty(B, grid.shape[0], device=v.device, dtype=torch.bool)
    for a in range(0, grid.shape[0], qchunk):
        d = torch.cdist(grid[a:a + qchunk][None].expand(B, -1, -1), P).amin(2)   # (B,c) nearest-point distance
        far[:, a:a + qchunk] = d > thr[:, None]
    far = far.view(B, 1, R, R, R)
    clamp = far & (v > -inside_th)                                        # far & NOT confidently inside a solid
    return torch.where(clamp, v.clamp_min(outside_val), v)


class WaveletEdgeRefiner(nn.Module):
    """Small (<1M param) WAVELET-DOMAIN edge/corner refiner -- the learned 'second pass' that reshapes the
    jagged crease crust.  The crust is high-frequency DETAIL-band energy at the corners/edges formed where the
    reconstructed surfaces meet; this 3D conv U-Net reads the 8 Haar coefficient bands + an edge-saliency
    channel (the detail-band energy itself -- the wavelet signature of edges) and predicts a RESIDUAL to the 7
    DETAIL bands only (zero-init -> identity start; the coarse LLL band = bulk shape is left untouched).  Being
    convolutional it is resolution-free (trained on the r=res/2 coeff lattice, applied at any res).  It is
    trained self-supervised (recon loss down-weighted at creases + a strong gradient-flip crust penalty in
    :func:`wavelet_surface_loss`), so it learns to reshape folded/ragged creases into clean edges."""

    def __init__(self, ch: int = 48):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.e1 = blk(9, ch)                     # in = 8 coeff bands + 1 edge-saliency (detail energy)
        self.e2 = blk(ch, ch * 2)
        self.mid = blk(ch * 2, ch * 2)
        self.d1 = blk(ch * 2 + ch, ch)           # upsample + skip
        self.out = nn.Conv3d(ch, 7, 1)           # residual for the 7 DETAIL bands
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # identity start

    def forward(self, c):                        # c: (B, 8, r, r, r) Haar coeff bands
        edge = c[:, 1:].abs().mean(1, keepdim=True)                     # detail energy = edge/corner saliency
        x = torch.cat([c, edge], 1)
        a = self.e1(x)
        b = self.mid(self.e2(F.avg_pool3d(a, 2)))
        b = F.interpolate(b, size=a.shape[2:], mode="nearest")
        resid = self.out(self.d1(torch.cat([a, b], 1)))                 # (B,7,r,r,r)
        c = c.clone(); c[:, 1:] = c[:, 1:] + resid                      # reshape ONLY detail (keep coarse bulk)
        return c


class PerceiverWaveNet(nn.Module):
    """Resolution-free point transformer that EMITS the Haar SDF coefficients (wavelet-from-attention).

    **Resolution-free input.**  The encoder reads a fixed ``seq_len``-token (128) sequence
    ``[context | SEP | main]``: an FPS summary of the WHOLE shape, a learned separator, and the dense
    region under reconstruction.  ``M`` latents cross-attend the tokens then ``L`` self-attention blocks
    refine them — a cost independent of the point count AND of any output grid, so neither the input nor
    the encoder depends on resolution.

    **Resolution-free output.**  The decoder is POSITION-CONDITIONED: each Fourier-encoded query position
    emits the coarse Haar band from the global latents and the 7 detail bands from its nearest point
    tokens.  Because the queries are arbitrary positions, coefficients can be emitted on ANY output
    lattice — one trained checkpoint reconstructs at res32, res64, ... alike (train at one res, query at
    any; see :func:`load_at_res`).  The only res-dependent piece is the ``qpos`` query lattice, which is
    NOT learned — it is recomputed for the chosen res.

    **Flexible token budget.**  The context/main split (``n_ctx``) is a free deployment-time choice; the
    trainer randomises it every step so the net learns to read ANY division of the 128 tokens.

    Drop-in for :class:`WaveletSurfaceNet`: returns ``(out, c_anchor, c_clean, seg)``.
    """

    def __init__(self, d=256, M=256, L=6, heads=8, k=16, res=64, trunc=0.1, bound=1.1,
                 with_seg=True, fourier_bands=8, detail_decay=2.5, seq_len=128, n_ctx=64, base=None,
                 unsigned=False, field_mode=None):
        super().__init__()
        self.d, self.k, self.res, self.trunc, self.bound = d, k, res, trunc, bound
        self.unsigned = unsigned                             # back-compat flag
        # field_mode: 'signed' | 'unsigned' | 'mixed'.  'mixed' = per-point base selection (signed for closed
        # regions, unsigned-band for thin/open) -> BOTH bases in one model call (anchor/target use it).
        self.field_mode = field_mode if field_mode is not None else ("unsigned" if unsigned else "signed")
        self.with_seg, self.fb, self.r = with_seg, fourier_bands, res // 2
        self.detail_decay = detail_decay                     # kept so set_res() can recompute detail_sigma
        self.detail_sigma = detail_decay * 2 * bound / res   # detail vanishes >~this far from any point
        # FLEXIBLE 128-token budget: the ctx | SEP | main split is a deployment-time choice (see forward),
        # so these are just the *defaults* — the trainer randomises n_ctx so the net reads any division.
        self.seq_len, self.n_ctx = seq_len, n_ctx            # encoder token budget: ctx | SEP | main
        self.n_main = seq_len - n_ctx - 1
        self.sep = nn.Parameter(torch.randn(1, d) * 0.02)    # learned separator between context and main
        self.type_emb = nn.Parameter(torch.zeros(2, d))      # 0 = context, 1 = main/dense (tells the two apart)
        fdim = 3 * 2 * fourier_bands
        self.tok = nn.Sequential(nn.Linear(fdim + 6, d), nn.LayerNorm(d))
        self.qemb = nn.Sequential(nn.Linear(fdim, d), nn.LayerNorm(d))
        self.latents = nn.Parameter(torch.randn(M, d) * 0.02)
        self.enc_in = _CrossBlock(d, heads)
        self.enc = nn.ModuleList([_SelfBlock(d, heads) for _ in range(L)])
        self.coarse_x = _CrossBlock(d, heads)              # query <- global latents  (coarse LLL)
        self.detail_x = _LocalAttn(d, heads)               # query <- k-NN point tokens (7 detail bands)
        self.coarse_head = nn.Linear(d, 1); self.detail_head = nn.Linear(d, 7)
        self.seg_head = nn.Linear(d, 1) if with_seg else None
        for hd in [self.coarse_head, self.detail_head] + ([self.seg_head] if with_seg else []):
            nn.init.zeros_(hd.weight); nn.init.zeros_(hd.bias)   # identity start: residual = 0
        # POST-PROCESSING (req 1, in forward train+eval): distance_field_clamp forces OUTSIDE any voxel farther
        # than this fraction of the bbox diagonal from ALL points (except confidently-inside solid cores) ->
        # carves INTERNAL holes (teapot tubes) + kills external sheets.  Req-2 edge crust = the learned
        # edge_refiner, driven by the crust penalty in wavelet_surface_loss.
        self.bound_margin = 0.05
        self.edge_refiner = WaveletEdgeRefiner(ch=32)   # learned wavelet-domain edge/corner crust refiner (req 2)
        self.register_buffer("haar", haar_filters_3d())
        # qpos is the ONLY res-dependent piece and is NOT learned: the (res//2)^3 coeff-lattice centres the
        # position-conditioned decoder queries.  load_at_res() recomputes it for any output res and loads
        # the learned weights (everything except qpos) -> one checkpoint queries at res32, res64, ... alike.
        lin = (torch.arange(self.r) + 0.5) / self.r * 2 * bound - bound
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        self.register_buffer("qpos", torch.stack([gx, gy, gz], -1).reshape(-1, 3))  # (r^3,3) coeff-lattice centres

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_res(self, res):
        """Re-point the ONLY resolution-dependent pieces to output at ``res`` (the encoder + position-
        conditioned decoder are resolution-free; see :func:`load_at_res`).  Recomputes ``res``/``r``/``qpos``
        and the detail-gate ``sigma`` -> ONE trained net queries at any res (train 42 -> eval 128).  The smax
        head is a fixed 3^3 fillet (res-independent).  Returns ``self`` for chaining."""
        self.res, self.r = int(res), int(res) // 2
        self.detail_sigma = self.detail_decay * 2 * self.bound / self.res
        lin = (torch.arange(self.r, device=self.qpos.device, dtype=self.qpos.dtype) + 0.5) / self.r * 2 * self.bound - self.bound
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        self.qpos = torch.stack([gx, gy, gz], -1).reshape(-1, 3)   # replaces the registered buffer, same device
        return self

    def _postprocess(self, out, P):
        """Req 1 (in the forward, train+eval): per-voxel distance clamp -- force OUTSIDE any voxel farther than
        ~5% of the bbox diagonal from every input point, except confidently-inside solid cores.  Kills external
        flying sheets AND carves internal empty holes (teapot tubes, room interiors) without hollowing solids."""
        return distance_field_clamp(out, P, self.bound, self.bound_margin)

    def _tok_at(self, P, N, idx):
        """Tokenise the points at ``idx`` ``(B,n)`` -> ``(B,n,d)``."""
        e = idx[..., None].expand(-1, -1, 3)
        Pg, Ng = torch.gather(P, 1, e), torch.gather(N, 1, e)
        return self.tok(torch.cat([fourier_encode(Pg / self.bound, self.fb), Ng, Pg / self.bound], -1))

    def forward(self, P, N, ctx_P=None, ctx_N=None, center=None, half=None, qchunk=2048, n_ctx=None,
                regions=None):
        """Whole-mesh: ``forward(P, N)``.  Region / SUPER-RESOLUTION: ``forward(dense_P, dense_N,
        ctx_P=whole, ctx_N=whole, center=c, half=h)`` normalises the sub-box ``[c-h, c+h]`` to the unit
        frame (so the lattice covers only the box = higher effective resolution there), encodes the whole
        shape as zoomed-out global CONTEXT, and reads DETAIL from the dense box points.

        ``n_ctx`` chooses how the fixed ``seq_len``-token budget is split: ``n_ctx`` context tokens, one
        separator, and ``seq_len-n_ctx-1`` main tokens.  Trained over a RANGE of splits, so the caller can
        divide the budget freely at inference (context-heavy for super-resolution, main-heavy / ``n_ctx``
        small for reading a whole small shape).  ``None`` -> the default ``self.n_ctx``."""
        B, dev = P.shape[0], P.device
        nctx = self.n_ctx if n_ctx is None else int(n_ctx)
        # flexible budget: split the fixed seq_len tokens as nctx context + 1 SEP + nmain main, any division
        nctx = max(1, min(nctx, self.seq_len - 2)); nmain = self.seq_len - nctx - 1
        if center is not None:                             # normalise the box to the standard frame
            sc = self.bound / half
            Pd = (P - center) * sc                         # dense points -> [-bound, bound]
            Pc, Nc = ((ctx_P - center) * sc, ctx_N) if ctx_P is not None else (Pd, N)
        else:
            Pd, Pc, Nc = P, P, N
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):   # anchor stays FP32 under bf16
            # REGION-COMPOSED anchor (crust-free edges): per-region fields composed at junctions.  ``regions``
            # = per-item (labels, ops, thin) precomputed/cached by the trainer; None -> computed here per item
            # (inference convenience).  The identity start is therefore already edge-clean at ANY eval res.
            # (autocast disabled: nearest-point distances / edge sign tests need fp32 precision.)
            if regions is None:
                regions = [(region_labels(Pd[b].cpu(), N[b].cpu()), None, None) for b in range(B)]
            tsdf = torch.cat([tsdf_composed(Pd[b].float(), N[b].float(), regions[b][0], self.res, self.trunc,
                                            self.bound, dev, ops=regions[b][1], thin=regions[b][2])
                              for b in range(B)], 0) / self.trunc
            c_anchor = dwt3d(tsdf, self.haar)              # (B,8,r,r,r)
        tok_d = self.tok(torch.cat([fourier_encode(Pd / self.bound, self.fb), N, Pd / self.bound], -1))  # full dense -> detail
        # encoder reads a compact CONTEXT | SEP | MAIN sequence (flexible split of seq_len tokens):
        # nctx FPS tokens summarising the WHOLE shape, the learned SEP, then nmain FPS tokens of the dense
        # region under reconstruction; type_emb tags which half each token belongs to. Cost is independent
        # of point count and of output res, so the encoder stays resolution-free.
        ctx_tok = self._tok_at(Pc, Nc, _fps(Pc, nctx)) + self.type_emb[0]
        main_tok = torch.gather(tok_d, 1, _fps(Pd, nmain)[..., None].expand(-1, -1, self.d)) + self.type_emb[1]
        seq = torch.cat([ctx_tok, self.sep[None].expand(B, -1, -1), main_tok], 1)   # (B, seq_len, d)
        lat = self.enc_in(self.latents[None].expand(B, -1, -1), seq)
        for blk in self.enc:
            lat = blk(lat)                                 # (B,M,d)
        # POSITION-CONDITIONED decode: every qpos lattice centre is queried independently from its own
        # Fourier-encoded position -> coarse band from the global latents, 7 detail bands from its nearest
        # point tokens. Queries are arbitrary positions, so swapping qpos for any other lattice (any res)
        # just re-points the decode -> the output is resolution-free.
        Q = self.qpos.shape[0]
        q_all = self.qemb(fourier_encode(self.qpos / self.bound, self.fb))   # (Q,d)
        kk = min(self.k, Pd.shape[1])
        c_lll, c_det, seg_l = [], [], []
        for s in range(0, Q, qchunk):
            qpos_c = self.qpos[s:s + qchunk]               # (c,3)
            qc = q_all[s:s + qchunk][None].expand(B, -1, -1)
            cf = self.coarse_x(qc, lat)                    # global latents -> coarse feature
            cc = qpos_c.shape[0]
            d2 = torch.cdist(qpos_c[None].expand(B, -1, -1), Pd)          # (B,c,Ndense)
            tk = d2.topk(kk, dim=-1, largest=False)
            gate = torch.exp(-(tk.values[..., 0] / self.detail_sigma) ** 2)[..., None]  # ~0 far from any point
            nbr = torch.gather(tok_d, 1, tk.indices.reshape(B, cc * kk, 1).expand(-1, -1, self.d)
                               ).reshape(B, cc, kk, self.d)               # (B,c,k,d) dense detail
            df = self.detail_x(qc, nbr)                    # local -> detail feature
            c_lll.append(self.coarse_head(cf))
            c_det.append(self.detail_head(df) * gate)      # GATED detail: no hallucinated floaters
            if self.with_seg:
                seg_l.append(self.seg_head(cf))
        c_resid = torch.cat([torch.cat(c_lll, 1), torch.cat(c_det, 1)], -1)        # (B,Q,8)
        c_resid = c_resid.view(B, self.r, self.r, self.r, 8).permute(0, 4, 1, 2, 3)
        c_clean = c_anchor + c_resid
        c_clean = self.edge_refiner(c_clean)               # learned wavelet-domain edge/corner refiner (reshapes crease crust)
        out = idwt3d(c_clean, self.haar)                   # (B,1,res,res,res)
        out = self._postprocess(out, Pd)                   # req1 AABB+5% spatial bound (box-frame cloud Pd)
        seg = (torch.cat(seg_l, 1).view(B, self.r, self.r, self.r, 1).permute(0, 4, 1, 2, 3)
               if self.with_seg else None)
        return out, c_anchor, c_clean, seg


def load_at_res(ck, res=None, bound=1.1):
    """Load a PerceiverWaveNet checkpoint at ANY output resolution.  The encoder reads the resolution-free
    128-token cloud and the decoder is position-conditioned, so only the ``qpos`` query lattice depends on
    ``res`` --- we recompute it for the requested ``res`` and load the learned weights.  One checkpoint
    therefore reconstructs at res32, res64, ... alike (train at one res, query at any)."""
    r = res if res is not None else ck.get("res", 32)
    net = PerceiverWaveNet(res=r, trunc=ck.get("trunc", 0.1), bound=bound, with_seg=ck.get("with_seg", True),
                           unsigned=ck.get("unsigned", False), field_mode=ck.get("field_mode"))
    # drop qpos (res-dependent, recomputed) and any smax.* (the retired learned corner head -> now analytic
    # edge_tangential_smooth; a checkpoint from the smax era carries dead smax.* keys, skip them).
    state = {k: v for k, v in ck["state"].items() if k != "qpos" and not k.startswith("smax.")}
    miss = net.load_state_dict(state, strict=False)
    # qpos recomputed; edge_refiner.* may be absent in a PRE-refiner ckpt (zero-init residual = identity no-op).
    allowed = {"qpos"} | {k for k in net.state_dict() if k.startswith("edge_refiner.")}
    assert not miss.unexpected_keys and set(miss.missing_keys) <= allowed, miss
    return net
