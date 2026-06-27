"""GPU (+Docker) trainer for BOTH the torus and supertoroid CoeffNets.

Training regime (this is the "do it properly with noise robustness" trainer):

* **>=10k assets**, drawn to exercise the supertoroid's extra subsurfaces
  (supertoroids over a wide squareness range) plus sharp/faceted shapes (cube,
  knurled cylinder, bolt plate) and smooth ones.  A *dense* cloud is sampled once
  per asset and cached.
* **Per-epoch re-randomization.**  Every epoch, for every asset, we (a) fetch a
  fresh random subset of points, and (b) add fresh noise to a random **50%** of
  those points, leaving the other 50% noiseless -- so the network sees a different
  partly-noisy cloud of every asset on each epoch.  Ground-truth distance is always
  to the clean surface.
* **Eval split.**  A held-out set is evaluated with **50% fully-noisy** and **50%
  clean** clouds, reported separately, so we can see noise robustness directly.
* **Batched on the GPU** (GPU kNN + batched blend) so >=5 epochs over >=10k assets
  is feasible on a single laptop GPU.  Trains the plain-torus and supertoroid nets
  on the identical data.

GPU + Docker only (aborts on CPU; see the `train-gpu-docker` skill).  Writes
``assets/pat_torus.pt`` and ``assets/pat_supertoroid.pt``.

Usage:  docker compose run --rm train      (uses the command in docker-compose.yml)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from pat import core, shapes
from pat.assets import BoltPlate, BoxWithCylinders, Cube, TexturedCylinder
from pat.model import CoeffNet
from pat.neighbors import neighborhood_features, rescale_coeffs
from pat.train import pat_loss

if not torch.cuda.is_available():
    raise SystemExit(
        "train_gpu.py requires a CUDA GPU and is meant to run via Docker.\n"
        "Use:  docker compose run --rm train\n"
        "(training is GPU-only by policy; see the train-gpu-docker skill).")
DEVICE = "cuda"


# --------------------------------------------------------------------------- #
#  Weight EMA (smooths the saved checkpoint past transient spikes)
# --------------------------------------------------------------------------- #
class EMA:
    """Exponential moving average of a model's parameters.

    We validate and **save** the EMA weights, not the raw training weights, so a
    single bad (corner-heavy) mini-batch can never land in the released model --
    it smooths the supertoroid's epoch-4 dip out of the checkpoint and makes the
    best-by-val selection stable.  The averaged ``state_dict`` is structurally
    identical, so ``pat.PAT`` / ``CoeffNet(**config)`` load it unchanged.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def store_and_copy_to(self, model):
        """Stash the live weights and load the EMA weights into ``model``."""
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n].to(p.device))

    @torch.no_grad()
    def restore(self, model):
        """Undo :meth:`store_and_copy_to`, putting the live weights back."""
        if self._backup is None:
            return
        for n, p in model.named_parameters():
            p.data.copy_(self._backup[n].to(p.device))
        self._backup = None


# --------------------------------------------------------------------------- #
#  Assets -> dense per-asset clouds (cached once)
# --------------------------------------------------------------------------- #
def random_analytic_shape(rng):
    """A shape drawn to exercise the supertoroid's range + sharp/faceted features."""
    r = rng.random()
    if r < 0.34:
        R = rng.uniform(0.4, 0.7); rr = rng.uniform(0.15, 0.32) * R / 0.6
        return shapes.SuperToroid(R=R, r=rr, p_tube=rng.uniform(2.0, 6.0),
                                  p_ring=rng.uniform(2.0, 4.0), axis=rng.normal(size=3))
    if r < 0.54:
        R = rng.uniform(0.4, 0.7); rr = rng.uniform(0.14, 0.30) * R / 0.6
        return shapes.Torus(R=R, r=rr, axis=rng.normal(size=3))
    if r < 0.70:
        c = rng.random()
        if c < 0.4:
            return shapes.Sphere(rng.uniform(0.3, 0.8))
        if c < 0.8:
            return shapes.RoundedBox(half=rng.uniform(0.3, 0.6, size=3),
                                     radius=rng.uniform(0.05, 0.2))
        return shapes.Plane(normal=rng.normal(size=3))
    if r < 0.86:
        return Cube(half=rng.uniform(0.4, 0.6), rounding=rng.uniform(0.01, 0.06))
    c = rng.random()
    if c < 0.55:
        return TexturedCylinder(radius=rng.uniform(0.28, 0.38), amp=rng.uniform(0.03, 0.06),
                                n_around=int(rng.integers(18, 30)),
                                n_axial=int(rng.integers(14, 26)))
    if c < 0.8:
        return BoltPlate()
    return BoxWithCylinders()


# The per-mesh GT example + query sampler live in pat.datasets (so the Colab
# notebook can build the cache incrementally, per ABC chunk, without importing
# this CUDA-gated module).
from pat.datasets import mesh_dense_example, surface_band_queries as _queries


def build_dense_cache(n_analytic, dense, n_query, bound=1.0, seed=0,
                      n_meshes=0, mesh_root="data", max_faces=None):
    """Sample a dense cloud + GT queries per asset (analytic + optional real meshes).

    Returns stacked CPU tensors of ``n_analytic + (cached real meshes)`` assets.
    On a big-memory host (e.g. an A100 80 GB) set ``n_meshes`` to mix in a real
    CAD corpus -- by default whatever the notebook downloads into ``data/`` (the
    **ABC CAD dataset**, ≥50k ``.obj``; or Objaverse / ModelNet40), found via
    :func:`pat.datasets.mesh_index`.
    """
    rng = np.random.default_rng(seed)
    paths = []
    if n_meshes > 0:
        from pat.datasets import mesh_index
        paths = mesh_index(mesh_root)
        rng.shuffle(paths)
        print(f"  real-mesh pool: {len(paths)} meshes under {mesh_root!r}; "
              f"caching up to {n_meshes}", flush=True)

    P, N, Q, PHI = [], [], [], []
    t0 = time.time()
    # analytic assets
    for i in range(n_analytic):
        sh = random_analytic_shape(rng)
        pts, nrm = sh.sample_surface(dense, rng)
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        surf, _ = sh.sample_surface(n_query, rng)
        q = _queries(rng, surf, n_query, bound)
        P.append(pts.astype(np.float32)); N.append(nrm.astype(np.float32))
        Q.append(q.astype(np.float32)); PHI.append(sh.sdf(q).astype(np.float32))
        if (i + 1) % 2000 == 0:
            print(f"  analytic {i+1}/{n_analytic}  ({(i+1)/(time.time()-t0):.0f}/s)", flush=True)
    # real meshes (ABC / Objaverse / ModelNet ...)
    got = 0
    tried = 0
    for path in paths:
        if got >= n_meshes:
            break
        tried += 1
        try:
            ex = mesh_dense_example(path, dense, n_query, rng, bound, max_faces=max_faces)
        except Exception:
            continue
        if not all(np.isfinite(a).all() for a in ex):       # drop degenerate meshes
            continue
        P.append(ex[0]); N.append(ex[1]); Q.append(ex[2]); PHI.append(ex[3])
        got += 1
        if got % 1000 == 0:
            print(f"  meshes {got}/{n_meshes}  (kept {got}/{tried}; "
                  f"{got/(time.time()-t0):.0f}/s)", flush=True)
    if n_meshes > 0 and got < n_meshes:
        print(f"  NOTE: only cached {got}/{n_meshes} real meshes (pool had "
              f"{len(paths)}, {tried} tried) -- download more chunks for the full set.",
              flush=True)

    total = len(P)
    print(f"dense cache: {total} assets ({n_analytic} analytic + {got} real meshes) "
          f"in {time.time()-t0:.0f}s", flush=True)
    return {"P": torch.from_numpy(np.stack(P)), "N": torch.from_numpy(np.stack(N)),
            "Q": torch.from_numpy(np.stack(Q)), "PHI": torch.from_numpy(np.stack(PHI))}


# --------------------------------------------------------------------------- #
#  Batched GPU ops
# --------------------------------------------------------------------------- #
def gpu_knn(pts, k):
    """k-NN indices (incl. self) for a batch of clouds ``pts (B, N, 3)`` -> ``(B, N, k+1)``."""
    d = torch.cdist(pts, pts)                                  # (B, N, N)
    return d.topk(k + 1, dim=2, largest=False).indices         # nearest incl. self at 0


def sample_epoch_clouds(cache, idx, n_points, noise_std, frac_noisy, rng_t):
    """Fetch a random point subset + add noise to a random fraction (per epoch).

    Returns ``pts (B, n_points, 3)``, ``nrm (B, n_points, 3)`` on the GPU.
    ``rng_t`` is a torch.Generator on DEVICE for reproducible per-epoch randomness.
    """
    B = len(idx)
    dense = cache["P"].shape[1]
    # different random point subset per asset, per epoch
    sub = torch.argsort(torch.rand(B, dense, generator=rng_t, device=DEVICE),
                        dim=1)[:, :n_points]                    # (B, n_points)
    bi = torch.arange(B, device=DEVICE)[:, None]
    pts = cache["P"][idx][bi, sub]                              # (B, n, 3)
    nrm = cache["N"][idx][bi, sub]
    # noise on a random fraction of points, fresh each epoch
    noisy = torch.rand(B, n_points, generator=rng_t, device=DEVICE) < frac_noisy
    noise = torch.randn(B, n_points, 3, generator=rng_t, device=DEVICE) * noise_std
    pts = pts + noisy.unsqueeze(-1) * noise
    return pts, nrm


def batched_coeffs(net, pts, nrm, k, chunk=3072):
    """Run the net on every neighborhood of a batch of clouds -> coeffs (B,N,6), sq.

    The neighborhoods are pushed through the transformer in chunks of ``chunk``
    sequences: the fused transformer-encoder kernel raises a CUDA "invalid
    configuration argument" if launched with too many sequences at once.
    """
    B, N, _ = pts.shape
    idx = gpu_knn(pts, k)                                       # (B, N, k+1)
    bi = torch.arange(B, device=DEVICE)[:, None, None]
    nbr_pos = pts[bi, idx].reshape(B * N, k + 1, 3)            # (B*N, k+1, 3)
    nbr_nrm = nrm[bi, idx].reshape(B * N, k + 1, 3)
    cs, ss = [], []
    for s in range(0, B * N, chunk):
        c, _, sq = net(nbr_pos[s:s + chunk], nbr_nrm[s:s + chunk])
        cs.append(c); ss.append(sq)
    coeffs = torch.cat(cs, 0).reshape(B, N, 6)
    sq = torch.cat(ss, 0).reshape(B, N, 2) if ss[0] is not None else None
    return coeffs, sq


def batched_loss(net, pts, nrm, q, phi_true, k, C=64.0, eik=0.1, chunk=3072,
                 square_reg=0.0):
    """L1 + eikonal blend loss over a batch of clouds (all on GPU).

    ``square_reg`` adds ``square_reg * mean((p - 2)^2)`` over the supertoroid
    squareness exponents -- a soft pull toward ``p = 2`` (an ordinary torus).
    Annealed from a small value to 0 over the first few epochs by the trainer,
    it keeps the squareness near the well-conditioned torus regime while the
    geometry settles, which (together with the ``p`` cap) cures the supertoroid's
    epoch-4 squareness blow-up.  A no-op for the plain-torus net (``sq is None``).
    """
    coeffs, sq = batched_coeffs(net, pts, nrm, k, chunk=chunk)
    params = core.coeffs_to_torus(pts, nrm, coeffs)            # batched (B,N,...)
    q = q.detach().clone().requires_grad_(True)
    x = q.unsqueeze(2)                                         # (B, Q, 1, 3)
    c = params["center"].unsqueeze(1)                          # (B, 1, N, 3)
    u = params["axis"].unsqueeze(1)
    R = params["R"].unsqueeze(1); r = params["r"].unsqueeze(1)
    sign = params["sign"].unsqueeze(1)
    if sq is not None:
        ea = params["ea"].unsqueeze(1)
        pt = sq[..., 0].unsqueeze(1); pr = sq[..., 1].unsqueeze(1)
        sdf = core.supertoroid_sdf(x, c, u, ea, R, r, pt, pr)
    else:
        sdf = core.torus_sdf(x, c, u, R, r)
    g = sign * sdf                                            # (B, Q, N)
    phi = core.blend_batched(q, pts, g, C=C)                  # (B, Q)
    grad, = torch.autograd.grad(phi.sum(), q, create_graph=True)
    grad = torch.nan_to_num(grad)                             # guard 0/0 eikonal gradients
    l_dist = (phi - phi_true).abs().mean()
    l_eik = (1.0 - grad.norm(dim=-1)).abs().mean()
    loss = l_dist + eik * l_eik
    if sq is not None and square_reg > 0.0:
        loss = loss + square_reg * ((sq - 2.0) ** 2).mean()
    return loss, l_dist.detach(), l_eik.detach()


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate_recon(net, shapes, C=16, npoints=1024, seed=123, bound=1.2):
    """Reconstruct each shape from a clean cloud; return mean abs SDF error per shape.

    One CPU round-trip for the whole list (PAT inference runs on CPU).  We test a
    default **torus** (the acceptance metric) and a sharp flat-sided **cube** (where
    the supertoroid's boxy cross-section should beat the plain torus).
    """
    from pat import PAT
    net.eval(); net.to("cpu")
    rng = np.random.default_rng(seed)
    errs = []
    for sh in shapes:
        pts, nrm = sh.sample_surface(npoints, rng)
        pat = PAT(pts, nrm, model=net, k=16, C=C)
        grid = rng.uniform(-bound, bound, (4000, 3))
        errs.append(float(np.mean(np.abs(pat.sdf(grid, neighbors=64) - sh.sdf(grid)))))
    net.to(DEVICE).train()
    return errs


@torch.no_grad()
def eval_noise_split(net, eval_cache, k, noise_std=0.015, mb=24, n_points=512):
    """Held-out eval: 50% clouds clean, 50% fully noisy; return (clean_err, noisy_err).

    Processed in mini-batches of ``mb`` clouds so the GPU kNN / blend stay small.
    """
    net.eval()
    A = eval_cache["P"].shape[0]
    half = A // 2

    def err_for(sl, ns):
        idxs = list(range(*sl.indices(A)))
        tot, cnt = 0.0, 0
        for s in range(0, len(idxs), mb):
            j = idxs[s:s + mb]
            pts = eval_cache["P"][j, :n_points].to(DEVICE).clone()
            nrm = eval_cache["N"][j, :n_points].to(DEVICE)
            if ns > 0:
                pts = pts + torch.randn_like(pts) * ns
            q = eval_cache["Q"][j].to(DEVICE)
            phi_true = eval_cache["PHI"][j].to(DEVICE)
            coeffs, sq = batched_coeffs(net, pts, nrm, k)
            params = core.coeffs_to_torus(pts, nrm, coeffs)
            x = q.unsqueeze(2)
            c = params["center"].unsqueeze(1); u = params["axis"].unsqueeze(1)
            R = params["R"].unsqueeze(1); r = params["r"].unsqueeze(1)
            sign = params["sign"].unsqueeze(1)
            if sq is not None:
                ea = params["ea"].unsqueeze(1)
                sdf = core.supertoroid_sdf(x, c, u, ea, R, r,
                                           sq[..., 0].unsqueeze(1), sq[..., 1].unsqueeze(1))
            else:
                sdf = core.torus_sdf(x, c, u, R, r)
            phi = core.blend_batched(q, pts, sign * sdf, C=16.0)
            tot += float((phi - phi_true).abs().mean()) * len(j)
            cnt += len(j)
        return tot / max(cnt, 1)

    clean = err_for(slice(0, half), 0.0)
    noisy = err_for(slice(half, A), noise_std)
    net.train()
    return clean, noisy


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=int, default=10000, help="analytic training assets")
    # Real meshes mixed in.  --meshes is the new generic name (any corpus found by
    # pat.datasets.mesh_index under --mesh-root: ABC .obj, Objaverse .glb, ModelNet
    # .off); --modelnet / --modelnet-root remain as back-compat aliases.
    ap.add_argument("--meshes", "--modelnet", dest="meshes", type=int, default=0,
                    help="real meshes to mix in (download first; big-RAM hosts). "
                         ">=50k real CAD meshes (ABC dataset) is the intended setting.")
    ap.add_argument("--mesh-root", "--modelnet-root", dest="mesh_root", default="data",
                    help="root dir scanned for real meshes (ABC/Objaverse/ModelNet)")
    ap.add_argument("--max-faces", type=int, default=200000,
                    help="skip real meshes above this face count (heavy CAD/scan models)")
    ap.add_argument("--epochs", type=int, default=8, help=">= 5 epochs (8 gives cosine room)")
    ap.add_argument("--dense", type=int, default=1024, help="dense points cached per asset")
    ap.add_argument("--n-points", type=int, default=512, help="points fetched per cloud per epoch")
    ap.add_argument("--n-query", type=int, default=160)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=12, help="clouds per GPU step (raise on big GPUs)")
    ap.add_argument("--chunk", type=int, default=3072, help="neighborhoods per transformer launch")
    ap.add_argument("--frac-noisy", type=float, default=0.5, help="fraction of points noised (train)")
    ap.add_argument("--noise", type=float, default=0.015)
    ap.add_argument("--eval-assets", type=int, default=400)
    ap.add_argument("--lr", type=float, default=8e-4, help="AdamW lr for the plain-torus net")
    ap.add_argument("--lr-super", type=float, default=9e-4,
                    help="AdamW lr for the (wider, stiffer) supertoroid net")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW weight decay (regularization; curbs the torus overfitting)")
    ap.add_argument("--dropout", type=float, default=0.05,
                    help="transformer dropout (regularization)")
    # --- architecture (2x-wider-but-snappy CoeffNet) ---
    ap.add_argument("--d-embed", type=int, default=192, help="token width (was 128)")
    ap.add_argument("--n-layers", type=int, default=6, help="encoder depth (held = latency)")
    ap.add_argument("--n-heads", type=int, default=12, help="attention heads (head_dim 192/12=16)")
    ap.add_argument("--d-ff", type=int, default=672, help="FFN width (was 512); ~2x params total")
    ap.add_argument("--p-max", type=float, default=6.0,
                    help="cap on the supertoroid squareness exponent (stability)")
    # --- training-stability knobs (cure the supertoroid epoch-4/5 spike + stall) ---
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="linear LR warmup as a fraction of total steps, then cosine")
    ap.add_argument("--square-reg", type=float, default=0.05,
                    help="initial p->2 squareness pull (annealed to 0 over --square-reg-epochs)")
    ap.add_argument("--square-reg-epochs", type=float, default=3.0,
                    help="anneal the squareness pull to 0 by this epoch")
    ap.add_argument("--eik", type=float, default=0.1, help="eikonal weight (ramped over 1st epoch)")
    ap.add_argument("--eik-warmup-epochs", type=float, default=1.0,
                    help="ramp the eikonal weight 0->--eik over this many epochs")
    ap.add_argument("--clip", type=float, default=1.0, help="global grad-norm clip (torus net)")
    ap.add_argument("--clip-super", type=float, default=0.5,
                    help="global grad-norm clip (supertoroid net; tighter)")
    ap.add_argument("--head-clip", type=float, default=0.5,
                    help="extra grad-norm clip on the supertoroid head (squareness) only")
    ap.add_argument("--spike-factor", type=float, default=3.0,
                    help="skip a finite step whose loss exceeds this x the trailing mean")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="weight-EMA decay; the EMA weights are validated and saved")
    ap.add_argument("--outdir", default="assets")
    ap.add_argument("--cache-file", default="",
                    help="persist the dense cache here; reused if present (skips re-caching)")
    ap.add_argument("--mesh-cache-file", default="",
                    help="pre-built mesh-only cache (P/N/Q/PHI) from the notebook's "
                         "incremental per-chunk build; concatenated after the analytic "
                         "assets so the trainer never needs the raw meshes on disk")
    ap.add_argument("--log-every", type=int, default=80, help="log every N steps")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # The per-epoch point-subset re-sampling only varies if we cache MORE points than
    # we fetch; otherwise every epoch just permutes the same set (identical kNN).
    args.n_points = min(args.n_points, args.dense)
    if args.dense <= args.n_points:
        print(f"WARNING: --dense ({args.dense}) <= --n-points ({args.n_points}): the fetched "
              f"point set will NOT vary across epochs (only the noise will). Use --dense > "
              f"--n-points for per-epoch nearest-set variation.", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)} | analytic {args.assets} + "
          f"meshes {args.meshes} | epochs {args.epochs} | batch {args.batch}", flush=True)

    # Build the dense cache once and reuse it: caching tens of thousands of real
    # meshes is slow (CPU/disk-bound), so a persisted --cache-file (e.g. on Google
    # Drive) lets re-runs skip it entirely.  The cache may hold fewer real meshes
    # than requested if the corpus on disk is smaller (partial download), so we
    # accept any cache whose total asset count does not exceed the request.
    cache = None
    if args.cache_file and os.path.exists(args.cache_file):
        print(f"loading dense cache {args.cache_file}", flush=True)
        c = torch.load(args.cache_file, weights_only=False)
        if (c["P"].shape[0] <= args.assets + args.meshes and c["P"].shape[1] == args.dense
                and c["Q"].shape[1] == args.n_query):
            cache = c
        else:
            print(f"  cache shape {tuple(c['P'].shape)} != request "
                  f"(assets+meshes<={args.assets+args.meshes}, dense={args.dense}, "
                  f"n_query={args.n_query}) -> rebuilding", flush=True)
    if cache is None:
        if args.mesh_cache_file and os.path.exists(args.mesh_cache_file):
            # Analytic assets built here (cheap, no disk); the real meshes were
            # pre-cached by the notebook incrementally (download -> cache -> delete
            # the extracted .obj per chunk), so no raw meshes are touched here.
            ana = build_dense_cache(args.assets, args.dense, args.n_query, seed=0, n_meshes=0)
            mc = torch.load(args.mesh_cache_file, weights_only=False)
            if mc["P"].shape[1] != args.dense or mc["Q"].shape[1] != args.n_query:
                raise SystemExit(
                    f"--mesh-cache-file shape (dense={mc['P'].shape[1]}, "
                    f"n_query={mc['Q'].shape[1]}) != --dense {args.dense} / "
                    f"--n-query {args.n_query}")
            cache = {k: torch.cat([ana[k], mc[k]], 0) for k in ("P", "N", "Q", "PHI")}
            print(f"assembled cache: {ana['P'].shape[0]} analytic + {mc['P'].shape[0]} "
                  f"pre-cached meshes = {cache['P'].shape[0]} assets", flush=True)
        else:
            cache = build_dense_cache(args.assets, args.dense, args.n_query, seed=0,
                                      n_meshes=args.meshes, mesh_root=args.mesh_root,
                                      max_faces=args.max_faces)
        if args.cache_file:
            os.makedirs(os.path.dirname(args.cache_file) or ".", exist_ok=True)
            torch.save(cache, args.cache_file)
            print(f"saved dense cache -> {args.cache_file}", flush=True)
    A = cache["P"].shape[0]
    cache = {kk: v.to(DEVICE) for kk, v in cache.items()}
    eval_cache = build_dense_cache(args.eval_assets, args.dense, args.n_query, seed=999)

    # ~2x-wider-but-snappy CoeffNet (width, not depth: at the k+1=17-token
    # neighborhood the transformer is launch/memory-bound, so widening d_embed/d_ff
    # ~2x the params while holding n_layers keeps per-neighborhood latency ~constant).
    # The supertoroid net additionally caps its squareness exponent at --p-max.
    arch = dict(d_embed=args.d_embed, n_layers=args.n_layers, n_heads=args.n_heads,
                d_ff=args.d_ff, dropout=args.dropout)
    cfg_t = dict(supertoroid=False, **arch)
    cfg_s = dict(supertoroid=True, p_max=args.p_max, **arch)
    net_t = CoeffNet(**cfg_t).to(DEVICE)
    net_s = CoeffNet(**cfg_s).to(DEVICE)
    n_par_t = sum(p.numel() for p in net_t.parameters())
    n_par_s = sum(p.numel() for p in net_s.parameters())
    # AdamW (decoupled weight decay) regularizes -- it most helps the plain torus,
    # which otherwise overfits by contorting its curvature coefficients to fake the
    # boxy shapes its fixed circular cross-section can't represent.  The supertoroid
    # net gets its own (slightly higher) lr.
    opt_t = torch.optim.AdamW(net_t.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_s = torch.optim.AdamW(net_s.parameters(), lr=args.lr_super, weight_decay=args.weight_decay)
    steps_per_epoch = A // args.batch
    total = args.epochs * steps_per_epoch
    warmup = max(1, int(args.warmup_frac * total))

    def warmup_cosine(opt, lr):
        """Linear LR warmup (no high-LR kick on an early stiff batch) then cosine."""
        w = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup)
        c = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total - warmup),
                                                       eta_min=lr * 0.05)
        return torch.optim.lr_scheduler.SequentialLR(opt, [w, c], milestones=[warmup])

    sch_t = warmup_cosine(opt_t, args.lr)
    sch_s = warmup_cosine(opt_s, args.lr_super)
    ema_t = EMA(net_t, args.ema_decay)
    ema_s = EMA(net_s, args.ema_decay)
    # per-net trailing-mean loss for the finite-spike skip guard
    loss_ema = {"t": None, "s": None}
    eik_warm_steps = max(1, int(args.eik_warmup_epochs * steps_per_epoch))
    sq_reg_steps = max(1, int(args.square_reg_epochs * steps_per_epoch))

    print(f"arch d_embed={args.d_embed} n_layers={args.n_layers} n_heads={args.n_heads} "
          f"d_ff={args.d_ff} | params torus {n_par_t/1e6:.2f}M supertoroid {n_par_s/1e6:.2f}M "
          f"| p_max {args.p_max}", flush=True)
    print(f"{total} steps ({args.epochs} epochs x {steps_per_epoch} batches of {args.batch}) "
          f"x 2 models over {A} assets | warmup {warmup} steps + cosine | "
          f"lr T {args.lr:g} S {args.lr_super:g} | EMA {args.ema_decay}", flush=True)
    print(f"per-epoch (re-rolled each epoch): resample {args.n_points}/{args.dense} pts per asset "
          f"(set varies={args.dense > args.n_points}); re-noise {args.frac_noisy:.0%} of them with "
          f"fresh noise; kNN recomputed per step", flush=True)

    gen = torch.Generator(device=DEVICE)
    rng = np.random.default_rng(0)
    from pat.shapes import Torus
    from pat.assets import Cube
    val_shapes = [Torus(0.6, 0.24), Cube()]   # acceptance torus + a sharp flat-sided cube
    best = {"t": 1e9, "s": 1e9}
    history = []
    done = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        gen.manual_seed(1000 + epoch)                         # fresh per-epoch randomness
        order = rng.permutation(A)
        noise_e = float(rng.uniform(0.008, args.noise))       # noise magnitude varies per epoch
        rt = rs = 0.0
        for b in range(steps_per_epoch):
            idx = torch.as_tensor(order[b * args.batch:(b + 1) * args.batch],
                                  dtype=torch.long, device=DEVICE)
            if len(idx) < 2:
                continue
            pts, nrm = sample_epoch_clouds(cache, idx, args.n_points, noise_e,
                                           args.frac_noisy, gen)
            q = cache["Q"][idx]; phi = cache["PHI"][idx]
            # Per-step schedules: ramp the (2nd-order) eikonal term up over the first
            # epoch so it isn't shaping a still-random squareness geometry, and anneal
            # the p->2 square pull to 0 once the geometry has settled.
            gstep = epoch * steps_per_epoch + b
            eik_e = args.eik * min(1.0, (gstep + 1) / eik_warm_steps)
            sq_reg_e = args.square_reg * max(0.0, 1.0 - gstep / sq_reg_steps)
            for net, opt, sch, ema, key, clip, is_super in (
                    (net_t, opt_t, sch_t, ema_t, "t", args.clip, False),
                    (net_s, opt_s, sch_s, ema_s, "s", args.clip_super, True)):
                loss, ld, le = batched_loss(net, pts, nrm, q, phi, args.k, chunk=args.chunk,
                                            eik=eik_e,
                                            square_reg=(sq_reg_e if is_super else 0.0))
                opt.zero_grad()
                v = float(loss.detach()) if torch.isfinite(loss) else float("inf")
                # Skip a step that is non-finite OR a finite spike (loss >> trailing
                # mean): a single bad (degenerate / boxy-corner) batch must not poison
                # the weights -- once Adam writes a NaN/huge moment it never recovers,
                # and clip_grad_norm sanitizes neither NaNs nor a finite-but-huge step.
                le_ref = loss_ema[key]
                spike = (le_ref is not None) and (v > args.spike_factor * le_ref)
                if torch.isfinite(loss) and not spike:
                    loss.backward()
                    for p in net.parameters():
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
                    # Per-group clip on the squareness head FIRST (a localized
                    # squareness explosion would otherwise be diluted below the global
                    # threshold), then the global clip.
                    if is_super and args.head_clip > 0:
                        torch.nn.utils.clip_grad_norm_(net.head.parameters(), args.head_clip)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                    opt.step()
                    ema.update(net)
                    loss_ema[key] = v if le_ref is None else 0.98 * le_ref + 0.02 * v
                sch.step()
                vlog = v if (torch.isfinite(loss) and not spike) else 0.0
                if key == "t":
                    rt += vlog
                else:
                    rs += vlog
            done += 1
            if done % args.log_every == 0:
                rate = done / (time.time() - t0)
                eta = (total - done) / max(rate, 1e-6) / 60
                print(f"  [ep {epoch} {b+1}/{steps_per_epoch}] step {done}/{total} "
                      f"noise {noise_e:.3f} loss T {rt/args.log_every:.4f} S {rs/args.log_every:.4f} "
                      f"| {rate:.1f} it/s | ETA {eta:.1f} min", flush=True)
                rt = rs = 0.0
        # end-of-epoch validation + save on the EMA weights (smooths transient
        # spikes out of the released model); restore the live weights afterwards.
        ema_t.store_and_copy_to(net_t)
        ema_s.store_and_copy_to(net_s)
        # reconstruct a default torus AND a sharp cube
        vt, vt_cube = validate_recon(net_t, val_shapes)
        vs, vs_cube = validate_recon(net_s, val_shapes)
        ct, nt = eval_noise_split(net_t, eval_cache, args.k, args.noise, n_points=args.n_points)
        cs, ns = eval_noise_split(net_s, eval_cache, args.k, args.noise, n_points=args.n_points)
        print(f"epoch {epoch}  val-torus-err T {vt:.4f} S {vs:.4f} | "
              f"val-cube-err T {vt_cube:.4f} S {vs_cube:.4f} | "
              f"eval clean/noisy  T {ct:.4f}/{nt:.4f}  S {cs:.4f}/{ns:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        history.append(dict(epoch=epoch, val_torus_t=vt, val_torus_s=vs,
                            val_cube_t=vt_cube, val_cube_s=vs_cube,
                            eval_clean_t=ct, eval_noisy_t=nt, eval_clean_s=cs, eval_noisy_s=ns))
        import json
        with open(os.path.join(args.outdir, "train_history.json"), "w") as fh:
            json.dump(history, fh, indent=1)
        for net, cfg, key, name, v in ((net_t, cfg_t, "t", "pat_torus.pt", vt),
                                       (net_s, cfg_s, "s", "pat_supertoroid.pt", vs)):
            if v < best[key]:
                best[key] = v
                torch.save({"state_dict": net.state_dict(), "config": cfg,
                            "val_torus_err": v, "history": history},
                           os.path.join(args.outdir, name))
        # put the live training weights back (validation/save used the EMA copy)
        ema_t.restore(net_t)
        ema_s.restore(net_s)
    print(f"DONE. best val-torus-err  torus {best['t']:.4f}  supertoroid {best['s']:.4f}", flush=True)
    save_curves(history, os.path.join(args.outdir, "training_curves.png"))


def save_curves(history, path):
    """Save the val + clean/noisy eval curves to ``path`` (headless)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in history]
        nan = float("nan")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(ep, [h["val_torus_t"] for h in history], "-o", color="C0", label="Feng26 net | torus")
        ax[0].plot(ep, [h["val_torus_s"] for h in history], "-o", color="C1", label="ours net | torus")
        ax[0].plot(ep, [h.get("val_cube_t", nan) for h in history], "--s", color="C0", label="Feng26 net | cube")
        ax[0].plot(ep, [h.get("val_cube_s", nan) for h in history], "--s", color="C1", label="ours net | cube")
        ax[0].axhline(0.01, ls=":", c="gray", label="invisible-by-eye bar")
        ax[0].set_title("val: reconstruct a default torus / sharp cube")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("mean abs SDF err"); ax[0].legend(fontsize=8)
        for m, lab in [("eval_clean_s", "supertoroid clean"), ("eval_noisy_s", "supertoroid noisy"),
                       ("eval_clean_t", "Feng26 torus clean"), ("eval_noisy_t", "Feng26 torus noisy")]:
            ax[1].plot(ep, [h[m] for h in history], "-o", label=lab)
        ax[1].set_title("held-out eval (50% clean / 50% noisy)")
        ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
        print(f"saved training curves -> {path}", flush=True)
    except Exception as e:
        print(f"(curve plot not saved: {e})", flush=True)


if __name__ == "__main__":
    main()
