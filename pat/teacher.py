"""Stage A -- the per-mesh TEACHER optimizer.

For each Objaverse++ mesh, optimize a field of supertoroid + cut-out-box splats that MINIMIZES the
number of splats ("supertori points") while keeping the **Minkowski filled-volume distance** to the
ground-truth solid at or below ``md_target`` (default ``1e-3``), **respecting holes**.  The optimized
splat set + the point->splat grouping is cached per mesh; those are the supervised labels the Stage-B
amortizer (``pat.student``) learns to reproduce.

Two correctness invariants (hard-won from prior failures, see the design synthesis):

* **Occupancy is the CSG UNION of the per-splat solids** (:meth:`SuperToroidSplats.union_sdf`), never
  the self-normalized blend's sign (which has ~43% interior sign errors).  So the MD gate is sound.
* **Ground-truth occupancy is built self-contained from the cached surface cloud P + normals N** via a
  densified point-cloud SDF (nearest surface point, sign from its normal) -- no source mesh needed, and
  holes are respected because the pseudo-normal sign is purely local (no winding-number closure).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import splat as _S
from .splat import SuperToroidSplats, fit_shape, _HAVE_TQDM

if _HAVE_TQDM:
    from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
#  Hole-respecting ground-truth occupancy from cached (P, N) only
# --------------------------------------------------------------------------- #
def densify_surface(P, N, k_dense=50_000, seed=0):
    """Upsample a sparse oriented cloud ``(P,N)`` to ``~k_dense`` points by jittering each point in
    its local TANGENT plane (from its normal), keeping the parent normal.  Needed because the cached
    1536 points are too coarse (~0.05 spacing) to resolve a 1e-3-level occupancy boundary; ~50k drops
    spacing below the voxel size.  No mesh required.  Returns ``(ds (D,3), dn (D,3))`` float32."""
    P = np.asarray(P, np.float32); N = np.asarray(N, np.float32)
    Nn = N / np.clip(np.linalg.norm(N, axis=1, keepdims=True), 1e-9, None)
    n = len(P)
    if k_dense <= n:
        return P.copy(), Nn
    spacing = float(np.median(cKDTree(P).query(P, k=2)[0][:, 1]))    # median nearest-neighbor gap
    rng = np.random.default_rng(seed)
    par = rng.integers(0, n, k_dense - n)                           # parent point per new sample
    npar = Nn[par]
    t1 = np.cross(npar, np.array([0, 0, 1.0], np.float32))          # a tangent (handle the polar case)
    deg = np.linalg.norm(t1, axis=1) < 1e-4
    t1[deg] = np.cross(npar[deg], np.array([0, 1.0, 0], np.float32))
    t1 /= np.clip(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9, None)
    t2 = np.cross(npar, t1)
    t2 /= np.clip(np.linalg.norm(t2, axis=1, keepdims=True), 1e-9, None)
    off = rng.normal(scale=0.5 * spacing, size=(k_dense - n, 2)).astype(np.float32)
    jit = P[par] + off[:, :1] * t1 + off[:, 1:] * t2
    ds = np.concatenate([P, jit], 0).astype(np.float32)
    dn = np.concatenate([Nn, npar], 0).astype(np.float32)
    return ds, dn


class CloudShape:
    """Hole-respecting ground-truth SDF built ONLY from a cached oriented cloud ``(P, N)``.

    Exposes ``.sdf(q) -> signed distance`` (negative inside) so it drops straight into
    :func:`pat.splat.fit_shape`, which only ever calls ``shape.sdf(q)``.
    """

    def __init__(self, P, N, k_dense=50_000, seed=0, k_sign=16):
        self.ds, self.dn = densify_surface(P, N, k_dense, seed)
        self.tree = cKDTree(self.ds)
        self.k_sign = k_sign

    def sdf(self, q, chunk=200_000):
        """Signed distance: magnitude = nearest-surface distance; SIGN = distance-weighted vote of the
        ``k_sign`` nearest surface points' pseudo-normals.  The k-NN vote is essential on REAL meshes --
        a SINGLE nearest normal (k=1) flips wherever one face normal is inconsistent (flipped/double-
        sided geometry), producing spurious interior pockets / spikes in the occupancy; voting over a
        neighborhood suppresses those, leaving a clean solid."""
        q = np.asarray(q, np.float64)
        out = np.empty(len(q), np.float64)
        K = int(min(self.k_sign, len(self.ds)))
        for a in range(0, len(q), chunk):
            qq = q[a:a + chunk]
            d, idx = self.tree.query(qq, k=K, workers=-1)                   # (q,K) nearest surface pts
            if K == 1:                                                     # scipy returns 1-D for k=1
                d = d[:, None]; idx = idx[:, None]
            diff = qq[:, None, :] - self.ds[idx]                           # (q,K,3)
            dots = np.einsum("qkc,qkc->qk", diff, self.dn[idx])            # signed projection per neighbor
            s = np.sign((dots / (d + 1e-6)).sum(1))                        # distance-weighted sign vote
            out[a:a + chunk] = np.where(s >= 0, d[:, 0], -d[:, 0])
        return out


# --------------------------------------------------------------------------- #
#  Minkowski filled-volume distance (occupancy from the CSG UNION, holes respected)
# --------------------------------------------------------------------------- #
# 4 antithetic sub-voxel offsets cancel grid-quantization bias to 2nd order, so a 128^3 grid certifies
# ~1e-3 at 1/8 the cost of 256^3 (deep-dive result).
_OFF_FRAC = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]], np.float32) * 0.5


def grid_centers(res, bound=1.0):
    lin = -bound + (np.arange(res) + 0.5) * (2.0 * bound / res)      # voxel centers in [-bound, bound]
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float32)


def _offsets(res, bound=1.0):
    return _OFF_FRAC * (2.0 * bound / res)                          # sub-voxel shifts, res-scaled


def gt_occupancy(cloud_shape, res=128, bound=1.0):
    """Ground-truth occupancy ``bool (4, res^3)`` (4 antithetic offsets) -- cache this per mesh."""
    base = grid_centers(res, bound)
    return np.stack([cloud_shape.sdf(base + o) < 0 for o in _offsets(res, bound)])


@torch.no_grad()
def _blend_occ(splat, pts, device):
    """Occupancy ``(Q,)`` bool from the self-normalized blend sign ``sdf_torch < 0``.

    Empirically (validated on shapes with holes/flats) the blend sign is an accurate filled-volume
    indicator for a CONVERGED per-shape fit -- the ~43% interior-sign error only afflicts under-trained
    AMORTIZED estimators, not the teacher.  This is ~5-10x cheaper than marching-cubes-of-the-blend and
    gives a near-identical MD, so it is the metric inside the greedy minimization loop.
    """
    x = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=device)
    return splat.sdf_torch(x) < 0


def md_filled_volume(splat, occ_gt, res=128, bound=1.0, device="cuda", return_iou=False):
    """Minkowski filled-volume distance ``vol(A xor B)`` between the splat field's solid and the cached
    GT occupancy ``occ_gt (4, res^3)`` (holes respected -- the GT was built hole-aware).  Occupancy is
    the blend sign (:func:`_blend_occ`).  ``return_iou`` also returns the scale-free IoU ``|A&B|/|A|B||``.
    """
    splat = splat.to(device)                                        # tolerate a CPU-assembled field
    base = grid_centers(res, bound)
    occ_gt_t = torch.as_tensor(np.asarray(occ_gt), device=device)
    sym = torch.zeros((), device=device)
    inter = union = torch.zeros((), device=device)
    for i, o in enumerate(_offsets(res, bound)):
        occ = _blend_occ(splat, base + o, device)
        gt = occ_gt_t[i]
        sym = sym + (occ ^ gt).float().mean()
        if return_iou:
            inter = inter + (occ & gt).float().sum()
            union = union + (occ | gt).float().sum()
    md = float(sym / len(_OFF_FRAC))                                 # FRACTION of cube volume vol(A xor B)/vol(cube)
    if return_iou:
        return md, float(inter / union.clamp_min(1.0))
    return md


# --------------------------------------------------------------------------- #
#  Minimize #splats s.t. MD <= md_target  (over-provision -> warm-fit -> grow -> greedy-prune+refit)
# --------------------------------------------------------------------------- #
def build_gt_pool(shape, P, n_pool=50_000, bound=1.05, device="cuda", seed=0):
    """Precompute a GT query pool ON ``device`` ONCE: a band hugging the cloud + a uniform bulk, with
    ground-truth ``phi`` from the KD-tree (one parallel query).  The optimizer then samples from this
    pool on-GPU every step instead of a per-step CPU KD-tree query -- the key GPU-utilization fix."""
    rng = np.random.default_rng(seed)
    nb = n_pool // 2
    band = P[rng.integers(0, len(P), nb)] + rng.normal(scale=0.03, size=(nb, 3))
    bulk = rng.uniform(-bound, bound, size=(n_pool - nb, 3))
    q = np.concatenate([band, bulk], 0).astype(np.float32)
    phi = shape.sdf(q).astype(np.float32)                            # ONE parallel KD-tree query
    return (torch.as_tensor(q, device=device), torch.as_tensor(phi, device=device))


def _refit(splat, shape, cloud, steps, device, q_pool=None, phi_pool=None, n_query=2048):
    """Re-optimize an existing splat field WITHOUT FPS re-init (single source of truth: optimize_splats)."""
    return _S.optimize_splats(splat, shape, cloud, steps=steps, prune_every=0, device=device,
                              n_query=n_query, q_pool=q_pool, phi_pool=phi_pool)


def _drop_and_refit(splat, victim, shape, cloud, steps, device, q_pool=None, phi_pool=None, n_query=2048):
    """Build a fresh field without splat ``victim``, then refit so survivors close the gap."""
    keep = [i for i in range(splat.M) if i != victim]
    cand = SuperToroidSplats.from_rows(splat.param_rows()[keep], p_max=splat.p_max)
    return _refit(cand, shape, cloud, steps, device, q_pool=q_pool, phi_pool=phi_pool, n_query=n_query)


def grow_at_residual(splat, shape, P, N, add, steps, device, seed=0, q_pool=None, phi_pool=None, n_query=2048):
    """Add ``add`` splats at the WORST-fit surface points (largest ``|sdf_torch(P)|``), init from
    coeffs there, then refit -- targets under-covered regions when the warm fit missed the target."""
    with torch.no_grad():
        err = splat.sdf_torch(torch.as_tensor(P, dtype=torch.float32, device=device)).abs().cpu().numpy()
    cand = np.argsort(-err)[:max(add * 8, 64)]                       # worst-fit pool
    pick = _S.farthest_point_sample(P[cand], min(add, len(cand)), seed=seed)
    new = _S._init_from_coeffs(P, N, cand[pick], np.full(len(pick), 0.18, np.float32)).to(device)
    merged = SuperToroidSplats.from_rows(                            # both rows on `device` -> no mismatch
        torch.cat([splat.param_rows(), new.param_rows()], 0), p_max=splat.p_max)
    return _refit(merged, shape, P, steps, device, q_pool=q_pool, phi_pool=phi_pool, n_query=n_query)


def fit_teacher(P, N, *, n_init=64, md_target=1e-3, iou_ok=0.7, res=64, steps_warm=300, steps_refit=80,
                n_query=2048, grow_add=16, max_grow=3, max_prune=14, max_splats=160, min_keep=8,
                time_budget_s=45.0, k_dense=50_000, device="cuda", seed=0, verbose=False):
    """Optimize the MINIMAL supertoroid-splat set with Minkowski filled-volume distance <= ``md_target``
    (holes respected, GT from cached P+N).  Returns ``(splat, md, iou, status, occ_gt)`` where ``status``
    is ``"ok"`` if the target was met (else ``"hard"`` -- those meshes are gated out of student training).

    The optimizer runs fully on ``device``: GT occupancy + a query pool are precomputed once, so every
    optimization step samples on-GPU (no per-step CPU KD-tree).  ``res`` is the MD grid resolution
    (64 is ~8x cheaper than 128 and adequate for the prune decisions); ``max_prune`` caps the greedy
    rounds so a stubborn mesh can't run the per-mesh budget dry.
    """
    P = np.asarray(P, np.float32); N = np.asarray(N, np.float32)
    t0 = time.time()
    shape = CloudShape(P, N, k_dense=k_dense, seed=seed)
    occ_gt = gt_occupancy(shape, res=res)
    qp, pp = build_gt_pool(shape, P, device=device, seed=seed)       # GT pool on GPU (sampled each step)
    # (1) over-provision + warm fit (fit_shape's own pruning ON)
    splat = fit_shape(shape, P, N, n_init=n_init, steps=steps_warm, prune_every=150, n_query=n_query,
                      min_share=0.3, min_keep=min_keep, device=device, seed=seed, q_pool=qp, phi_pool=pp)
    md = md_filled_volume(splat, occ_gt, res=res, device=device)
    # (2) grow at residual until target met or budget/round/size cap
    rounds = 0
    while md > md_target and rounds < max_grow and splat.M < max_splats and (time.time() - t0) < time_budget_s:
        splat = grow_at_residual(splat, shape, P, N, add=grow_add, steps=steps_refit, device=device,
                                 seed=seed + rounds, q_pool=qp, phi_pool=pp, n_query=n_query)
        md = md_filled_volume(splat, occ_gt, res=res, device=device); rounds += 1
        if verbose:
            print(f"  grow {rounds}: M={splat.M} MD={md:.5f}", flush=True)
    # (3) greedy prune-to-minimum: drop the splat owning the FEWEST surface points (the redundant one;
    #     surface-ownership is robust to window inflation, unlike total_weight), refit, accept iff MD ok
    pruned = 0
    while splat.M > min_keep and pruned < max_prune and (time.time() - t0) < time_budget_s:
        victim = int(splat.surface_ownership(P).sum(0).argmin())
        cand = _drop_and_refit(splat, victim, shape, P, steps_refit, device, q_pool=qp, phi_pool=pp, n_query=n_query)
        md_c = md_filled_volume(cand, occ_gt, res=res, device=device)
        pruned += 1
        if md_c <= md_target:
            splat, md = cand, md_c
            if verbose:
                print(f"  prune -> M={splat.M} MD={md:.5f}", flush=True)
        else:
            break                                                    # cannot shrink further
    md, iou = md_filled_volume(splat, occ_gt, res=res, device=device, return_iou=True)
    status = "ok" if (md <= md_target or iou >= iou_ok) else "hard"
    return splat, md, iou, status, occ_gt


# --------------------------------------------------------------------------- #
#  Per-mesh cached artifact (the student labels) -- sharded, presence-checked, atomic
# --------------------------------------------------------------------------- #
def teacher_artifact(splat, P, N, md, iou, status, gid):
    """Pack the optimized field + the point->splat grouping into the cached student-label dict."""
    P = np.asarray(P, np.float32); N = np.asarray(N, np.float32)
    resp = splat.surface_ownership(P).detach().cpu()                 # (Npts, M) surface-proximity share
    return {
        "gid": int(gid), "M": int(splat.M), "p_max": float(splat.p_max),
        "state": {k: v.cpu() for k, v in splat.state_dict().items()},
        "params": splat.param_rows().detach().cpu().float(),         # (M, ROW_W) FitNet target
        "resp": resp.half(),                                         # (Npts, M) soft ownership
        "owner": resp.argmax(1).to(torch.int16),                     # (Npts,) hard owner (GroupNet label)
        "P": torch.as_tensor(P).half(), "N": torch.as_tensor(N).half(),
        "md": float(md), "iou": float(iou), "status": str(status),
    }


def shard_path(outdir, gid, per_shard=256):
    return os.path.join(outdir, f"shard_{gid // per_shard:04d}", f"mesh_{gid:06d}.pt")


def save_teacher(artifact, path):
    """Atomic save (.tmp -> os.replace) so an interrupted Colab session loses at most the in-flight mesh."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(artifact, tmp)
    os.replace(tmp, path)


def fit_and_cache(P, N, gid, outdir, *, force=False, device="cuda", **kw):
    """Presence-checked teacher fit for one mesh: skip if its shard file already exists (regen only if
    missing or ``force``).  Returns ``(status, M, md)`` -- ``status="cached"`` when skipped."""
    path = shard_path(outdir, gid)
    if os.path.exists(path) and not force:
        a = torch.load(path, weights_only=False, map_location="cpu")
        return "cached", a["M"], a["md"]
    splat, md, iou, status, _ = fit_teacher(P, N, device=device, **kw)
    save_teacher(teacher_artifact(splat, P, N, md, iou, status, gid), path)
    return status, int(splat.M), float(md)


def load_teacher(path, device="cpu"):
    """Reload a cached artifact and rebuild its :class:`SuperToroidSplats` from ``params``."""
    a = torch.load(path, weights_only=False, map_location=device)
    a["splat"] = SuperToroidSplats.from_rows(a["params"], p_max=a["p_max"]).to(device)
    return a
