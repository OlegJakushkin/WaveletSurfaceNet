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


def _queries(rng, surf, n_query, bound):
    nb = n_query // 2
    band = surf[:nb] + rng.normal(scale=0.04, size=(nb, 3))
    bulk = rng.uniform(-bound, bound, size=(n_query - nb, 3))
    return np.concatenate([band, bulk], 0)


def mesh_dense_example(path, dense, n_query, rng, bound=1.0, dense_surf=50000):
    """One dense (cloud, normals, queries, GT SDF) tuple from a real mesh.

    Ground-truth signed distance uses a KD-tree over a dense surface sample
    (fast and accurate to the surface spacing) -- the same trick as the local
    trainer, which keeps real-data caching fast on a hosted GPU/disk.
    """
    from scipy.spatial import cKDTree
    from pat.datasets import load_mesh_normalized
    from pat.shapes import sample_mesh
    mesh = load_mesh_normalized(path)
    pts, nrm = sample_mesh(mesh, dense, rng)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    surf, _ = sample_mesh(mesh, n_query, rng)
    q = _queries(rng, surf, n_query, bound)
    ds, dn = sample_mesh(mesh, dense_surf, rng)
    tree = cKDTree(ds)
    d, idx = tree.query(q)
    sign = np.einsum("ij,ij->i", q - ds[idx], dn[idx])
    phi = np.where(sign >= 0, d, -d)
    return (pts.astype(np.float32), nrm.astype(np.float32),
            q.astype(np.float32), phi.astype(np.float32))


def build_dense_cache(n_analytic, dense, n_query, bound=1.0, seed=0,
                      n_modelnet=0, modelnet_root="data"):
    """Sample a dense cloud + GT queries per asset (analytic + optional ModelNet).

    Returns stacked CPU tensors of ``n_analytic + (cached ModelNet)`` assets.  On a
    big-memory host (e.g. an A100 80 GB) set ``n_modelnet`` to mix in real CAD models.
    """
    rng = np.random.default_rng(seed)
    paths = []
    if n_modelnet > 0:
        from pat.datasets import modelnet_index
        paths = modelnet_index(modelnet_root)
        rng.shuffle(paths)
        print(f"  ModelNet pool: {len(paths)} models; caching up to {n_modelnet}", flush=True)

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
    # real ModelNet assets
    got = 0
    for path in paths:
        if got >= n_modelnet:
            break
        try:
            ex = mesh_dense_example(path, dense, n_query, rng, bound)
        except Exception:
            continue
        P.append(ex[0]); N.append(ex[1]); Q.append(ex[2]); PHI.append(ex[3])
        got += 1
        if got % 1000 == 0:
            print(f"  modelnet {got}/{n_modelnet}  ({got/(time.time()-t0):.0f}/s)", flush=True)

    total = len(P)
    print(f"dense cache: {total} assets ({n_analytic} analytic + {got} ModelNet) "
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


def batched_loss(net, pts, nrm, q, phi_true, k, C=64.0, eik=0.1, chunk=3072):
    """L1 + eikonal blend loss over a batch of clouds (all on GPU)."""
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
    l_dist = (phi - phi_true).abs().mean()
    l_eik = (1.0 - grad.norm(dim=-1)).abs().mean()
    return l_dist + eik * l_eik, l_dist.detach(), l_eik.detach()


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate_default_torus(net, C=16, npoints=1024, seed=123):
    from pat import PAT
    from pat.shapes import Torus
    net.eval()
    rng = np.random.default_rng(seed)
    sh = Torus(0.6, 0.24)
    pts, nrm = sh.sample_surface(npoints, rng)
    pat = PAT(pts, nrm, model=net.to("cpu"), k=16, C=C)
    grid = rng.uniform(-1.2, 1.2, (4000, 3))
    err = float(np.mean(np.abs(pat.sdf(grid, neighbors=64) - sh.sdf(grid))))
    net.to(DEVICE).train()
    return err


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
    ap.add_argument("--modelnet", type=int, default=0,
                    help="real ModelNet40 models to mix in (download first; big-RAM hosts)")
    ap.add_argument("--modelnet-root", default="data")
    ap.add_argument("--epochs", type=int, default=6, help=">= 5 epochs")
    ap.add_argument("--dense", type=int, default=1024, help="dense points cached per asset")
    ap.add_argument("--n-points", type=int, default=512, help="points fetched per cloud per epoch")
    ap.add_argument("--n-query", type=int, default=160)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=12, help="clouds per GPU step (raise on big GPUs)")
    ap.add_argument("--chunk", type=int, default=3072, help="neighborhoods per transformer launch")
    ap.add_argument("--frac-noisy", type=float, default=0.5, help="fraction of points noised (train)")
    ap.add_argument("--noise", type=float, default=0.015)
    ap.add_argument("--eval-assets", type=int, default=400)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--outdir", default="assets")
    ap.add_argument("--log-every", type=int, default=80, help="log every N steps")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"device: {torch.cuda.get_device_name(0)} | analytic {args.assets} + "
          f"modelnet {args.modelnet} | epochs {args.epochs} | batch {args.batch}", flush=True)

    cache = build_dense_cache(args.assets, args.dense, args.n_query, seed=0,
                              n_modelnet=args.modelnet, modelnet_root=args.modelnet_root)
    A = cache["P"].shape[0]
    cache = {kk: v.to(DEVICE) for kk, v in cache.items()}
    eval_cache = build_dense_cache(args.eval_assets, args.dense, args.n_query, seed=999)

    cfg_t = dict(d_embed=128, n_layers=6, n_heads=8, d_ff=512, supertoroid=False)
    cfg_s = dict(d_embed=128, n_layers=6, n_heads=8, d_ff=512, supertoroid=True)
    net_t = CoeffNet(**cfg_t).to(DEVICE)
    net_s = CoeffNet(**cfg_s).to(DEVICE)
    opt_t = torch.optim.Adam(net_t.parameters(), lr=args.lr)
    opt_s = torch.optim.Adam(net_s.parameters(), lr=args.lr)
    steps_per_epoch = A // args.batch
    total = args.epochs * steps_per_epoch
    sch_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=total)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=total)
    print(f"{total} steps ({args.epochs} epochs x {steps_per_epoch} batches of {args.batch}) "
          f"x 2 models over {A} assets", flush=True)

    gen = torch.Generator(device=DEVICE)
    rng = np.random.default_rng(0)
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
            for net, opt, sch, key in ((net_t, opt_t, sch_t, "t"), (net_s, opt_s, sch_s, "s")):
                loss, ld, le = batched_loss(net, pts, nrm, q, phi, args.k, chunk=args.chunk)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sch.step()
                if key == "t":
                    rt += float(loss.detach())
                else:
                    rs += float(loss.detach())
            done += 1
            if done % args.log_every == 0:
                rate = done / (time.time() - t0)
                eta = (total - done) / max(rate, 1e-6) / 60
                print(f"  [ep {epoch} {b+1}/{steps_per_epoch}] step {done}/{total} "
                      f"noise {noise_e:.3f} loss T {rt/args.log_every:.4f} S {rs/args.log_every:.4f} "
                      f"| {rate:.1f} it/s | ETA {eta:.1f} min", flush=True)
                rt = rs = 0.0
        # end-of-epoch validation
        vt = validate_default_torus(net_t)
        vs = validate_default_torus(net_s)
        ct, nt = eval_noise_split(net_t, eval_cache, args.k, args.noise, n_points=args.n_points)
        cs, ns = eval_noise_split(net_s, eval_cache, args.k, args.noise, n_points=args.n_points)
        print(f"epoch {epoch}  val-torus-err T {vt:.4f} S {vs:.4f} | "
              f"eval clean/noisy  T {ct:.4f}/{nt:.4f}  S {cs:.4f}/{ns:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        history.append(dict(epoch=epoch, val_torus_t=vt, val_torus_s=vs,
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
    print(f"DONE. best val-torus-err  torus {best['t']:.4f}  supertoroid {best['s']:.4f}", flush=True)


if __name__ == "__main__":
    main()
