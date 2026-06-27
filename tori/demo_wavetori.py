"""GPU test of the WaveTori network on the 5 shapes (torus, teapot, bunny, cube, sphere).

Frozen pre-trained tori CoeffNet (assets/pat_torus.pt) supplies the per-point blend PRIOR;
a wavelet refiner is trained (DYNAMIC noise, refreshed bank) to clean it.  Renders, per shape,
`ground truth | tori blend (prior) | WaveTori (refined)` so the wavelet's contribution is visible.

GPU + Docker only:  docker compose run --rm train python demo_wavetori.py
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as Fn
import trimesh
from skimage import measure

from pat import wavelet as WV
from pat import wavetori as WT
from pat import compare as CMP
from pat import eval3d as E
from pat import render3d as R3
from pat.shapes import normalize_to_unit_cube
from pat.bunny import load_bunny


def five_shapes():
    return [
        ("torus",  normalize_to_unit_cube(trimesh.creation.torus(major_radius=0.5, minor_radius=0.2))),
        ("teapot", normalize_to_unit_cube(E._teapot_mesh())),
        ("bunny",  load_bunny(normalize=True)),
        ("cube",   normalize_to_unit_cube(trimesh.creation.box(extents=[1.0, 1.0, 1.0]))),
        ("sphere", normalize_to_unit_cube(trimesh.creation.uv_sphere(radius=0.7, count=[48, 48]))),
    ]


class _GridSDF:
    def __init__(self, grid, bound, trunc):
        self.grid, self.bound, self.trunc, self.res = grid, bound, trunc, grid.shape[0]
    def sdf(self, q):
        return WV.grid_trilinear(self.grid, q, self.bound, self.trunc)
    def reconstruct(self, level=0.0):
        if not (self.grid.min() < level < self.grid.max()):
            return None, None
        v, f, _, _ = measure.marching_cubes(self.grid, level=level)
        return v / (self.res - 1) * (2 * self.bound) - self.bound, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--trunc", type=float, default=0.1)
    ap.add_argument("--base", type=int, default=40)
    ap.add_argument("--dense", type=int, default=1536)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--tori-ckpt", default="assets/pat_torus.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--refresh", type=int, default=6, help="rebuild the noisy tori-blend bank every N epochs")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--noise-lo", type=float, default=0.005)
    ap.add_argument("--noise-hi", type=float, default=0.03)
    ap.add_argument("--lam-wave", type=float, default=0.3)
    ap.add_argument("--lam-grad", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-noise", type=float, default=0.015)
    ap.add_argument("--tag", default="wt")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU only — run via docker compose run --rm train python test_wavetori.py")
    dev = "cuda"; bound = 1.1; res = args.res; trunc = args.trunc
    os.makedirs("renders", exist_ok=True); os.makedirs("assets", exist_ok=True)
    print(f"GPU {torch.cuda.get_device_name(0)} | res {res} base {args.base}", flush=True)

    shapes = five_shapes(); gts = [E.mesh_gt(m) for _, m in shapes]
    Ps, Ns = [], []
    for _, m in shapes:
        P, N = E.sample_cloud(m, n=args.dense, noise=0.0, seed=0); Ps.append(P); Ns.append(N)
    Pt = torch.tensor(np.stack(Ps)).to(dev); Nt = torch.tensor(np.stack(Ns)).to(dev)
    S = len(shapes); M = args.draws

    ck = torch.load(args.tori_ckpt, weights_only=False)
    cfg = ck.get("config", {"d_embed": ck.get("d_embed", 128), "n_layers": ck.get("n_layers", 8)})
    tori = CMP.CoeffNet(**cfg).to(dev); tori.load_state_dict(ck.get("state_dict", ck.get("state"))); tori.eval()
    print(f"loaded frozen tori {args.tori_ckpt} (config {cfg})", flush=True)

    haar = WV.haar_filters_3d(dev)
    with torch.no_grad():
        clean = WV.tsdf_from_clouds(Pt, Nt, res, trunc, bound, dev) / trunc
        target_c = WV.dwt3d(clean, haar)
        clean_r = clean.repeat(M, 1, 1, 1, 1); tc_r = target_c.repeat(M, 1, 1, 1, 1)
        Pr, Nr = Pt.repeat(M, 1, 1), Nt.repeat(M, 1, 1)

    wave = WV.WaveletDenoiser(base=args.base).to(dev)
    print(f"wavelet refiner params {wave.count_params():,}", flush=True)
    opt = torch.optim.Adam(wave.parameters(), lr=args.lr)
    g = torch.Generator(device="cpu").manual_seed(0)
    prior = None
    for ep in range(args.epochs):
        if ep % args.refresh == 0:                                # DYNAMIC: refresh tori-blend bank
            with torch.no_grad():
                ns = torch.empty(S * M, 1, 1, device=dev).uniform_(args.noise_lo, args.noise_hi)
                Pn = Pr + torch.randn(Pr.shape, device=dev) * ns
                prior = WT.tori_blend_tsdf(Pn, Nr, tori, res, trunc, bound, dev, args.k) / trunc
        for _ in range(args.substeps):
            idx = torch.randperm(S * M, generator=g)[: max(S, (S * M) // args.substeps)].to(dev)
            pred, _, c_pred = wave(prior[idx])
            l_t = Fn.smooth_l1_loss(pred, clean_r[idx], beta=0.1)
            l_w = (c_pred - tc_r[idx]).abs().mean()
            gp, gc = WV._grad3d(pred), WV._grad3d(clean_r[idx])
            l_g = sum((a - b).abs().mean() for a, b in zip(gp, gc)) / 3.0
            loss = l_t + args.lam_wave * l_w + args.lam_grad * l_g
            opt.zero_grad(); loss.backward()
            for p in wave.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, 0., 0., 0.)
            torch.nn.utils.clip_grad_norm_(wave.parameters(), 1.0); opt.step()
        if ep % max(1, args.epochs // 12) == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                held = float((wave(prior[:S])[0] - clean).abs().mean())
            print(f"  ep {ep:4d}: loss {float(loss.detach()):.4f} | bank TSDF-L1 {held:.4f}", flush=True)
    torch.save({"state": wave.state_dict(), "base": args.base, "res": res, "trunc": trunc},
               "assets/wavetori_refiner.pt")

    print(f"\n{'shape':8s} | {'tori MD':>8s} {'tori IoU':>8s} | {'WTori MD':>8s} {'WTori IoU':>9s} {'WTori vts':>9s}",
          flush=True)
    rng = np.random.default_rng(0); mt, it, mw, iw = [], [], [], []
    for (name, m), P, N, gt in zip(shapes, Ps, Ns, gts):
        Pn = (P + rng.normal(scale=args.eval_noise, size=P.shape)).astype(np.float32)   # same noisy cloud
        with torch.no_grad():
            tori_grid = (WT.tori_blend_tsdf(Pn, N, tori, res, trunc, bound, dev, args.k)[0, 0]
                         .cpu().numpy().astype(np.float64) * 1.0)   # already distance units
        tori_sdf = _GridSDF(tori_grid, bound, trunc)
        pm_t = E.proper_metrics(gt, tori_sdf, n=40000); vt, ft = tori_sdf.reconstruct()
        wr = WT.WaveToriReconstruction(Pn, N, tori, wave, res=res, trunc=trunc, bound=bound, device=dev, k=args.k)
        pm_w = E.proper_metrics(gt, wr, n=40000); vw, fw = wr.reconstruct()
        mt.append(pm_t["md"]); it.append(pm_t["iou"]); mw.append(pm_w["md"]); iw.append(pm_w["iou"])
        nvw = 0 if vw is None else len(vw)
        print(f"{name:8s} | {pm_t['md']:8.3f} {pm_t['iou']:8.3f} | {pm_w['md']:8.3f} {pm_w['iou']:9.3f} {nvw:9d}",
              flush=True)
        try:
            R3.render_meshes([("ground truth", m.vertices, m.faces),
                              (f"tori blend  MD {pm_t['md']:.3f} | IoU* {pm_t['iou']:.2f}", vt, ft),
                              (f"WaveTori  MD {pm_w['md']:.3f} | IoU* {pm_w['iou']:.2f}", vw, fw)],
                             f"renders/wavetori_{name}_{args.tag}.png", title=name)
        except Exception as exc:
            print("  render skip", name, exc, flush=True)
    print(f"\nMEAN: tori blend MD {np.mean(mt):.3f} IoU* {np.mean(it):.3f} | "
          f"WaveTori MD {np.mean(mw):.3f} IoU* {np.mean(iw):.3f} | renders -> renders/wavetori_*_{args.tag}.png",
          flush=True)


if __name__ == "__main__":
    main()
