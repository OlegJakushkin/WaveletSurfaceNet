"""Train the PRIMITIVE-FREE wavelet-native surface model (WaveletSurfaceNet) on MeshNet (ModelNet40).

No tori / superellipse / per-point primitives: each cloud is splatted to a 5-channel grid (occupancy +
mean-normal + direct point TSDF) and a wavelet-DOMAIN U-Net reconstructs the SDF directly, plus a light
side-segmentation head supervised by wavelet-sparsity pseudo-labels (flat faces = sparse high-freq).

Dynamic noise (<=10% mesh size) for robustness; the model-selection metric is the CLEAN (no-noise) raw
mean-abs SDF error (README metric).  Per epoch: best-by-val weights + renders of 6 favourites and 4
held-out ModelNet40-test meshes.  GPU only.
  /opt/conda/bin/python train_waveshape.py --epochs 16 --batch 48
  docker compose run --rm train python train_waveshape.py --smoke

RESOLUTION-FREE + FLEXIBLE-CONTEXT (PerceiverWaveNet, waveshape/wavelet.py).  The net never sees a grid on its
INPUT: each cloud is encoded as a fixed 128-token sequence [context | SEP | main] -- a sparse farthest-point
summary of the WHOLE shape, a learned separator, then the dense region under reconstruction.  A Perceiver
encoder (M latents cross-attend the tokens, then L self-attention blocks) summarises this at a cost
independent of the point count and of any output grid.  The DECODER is POSITION-CONDITIONED: each
Fourier-encoded query position emits the coarse Haar band from the global latents and the 7 detail bands
from its nearest point tokens, so coefficients can be emitted on ANY output lattice -- the model is
resolution-free at the output.  The ONLY res-dependent piece is the non-learned `qpos` query lattice
(r = res//2 grid centres), recomputed per res; WV.load_at_res(ck, res) rebuilds the net at any output res
and loads every learned weight except `qpos`, so ONE checkpoint reconstructs at res32, res64, ... alike
(train at one res, query at any -- a res mismatch is only a qpos-buffer detail, never a model limit).
The `res` below is therefore just the training/output lattice, not a property of the trained model.
"""
import argparse, glob, json, os, shutil, time
import numpy as np, torch, torch.nn.functional as F, trimesh
from skimage import measure
from waveshape import wavelet as WV, eval3d as E, render3d as R3
from waveshape.shapes import normalize_to_unit_cube
from waveshape.bunny import load_bunny

dev = "cuda"; bound = 1.1; RES = 64; TRUNC = 0.1; DENSE = 1536   # RES = training/output lattice only; the net is res-free (query any res via WV.load_at_res)
NOISE_LO, NOISE_HI = 0.0, 0.10
DRAWS = 2; LR = 2e-3; BASE = 40; SEED = 0; NVAL = 4; CLEAN_SAMPLE = 128
LAM_WAVE, LAM_GRAD, LAM_SEG = 0.4, 0.05, 0.05      # target-aware wavelet term (preserves texture)
LAM_SMOOTH, LAM_SIGN = 0.05, 0.30                  # v1 recipe (v2's boost measured WORSE on every metric -> reverted)
LAM_CONN = 0.4
LAM_GEO = 0.5                                       # NEW geometry-quality block at +50% of base loss: targets
#                                                    Chamfer/floaters, #components, holes, F-closed (see wavelet_surface_loss)
UDF_BAND_VOX = 0.7                                  # UDF mode: MC band in VOXELS (the field floor is ~half a voxel at any res)
CTX_MIN, CTX_MAX = 16, 104                          # FLEXIBLE 128-token budget: the context/main split (n_ctx) is a free
                                                    # deploy-time choice; randomising it per step here teaches the net to read ANY division


def _upright_teapot():                                  # rotate +90 about X so it STANDS on its base
    tp = trimesh.load("assets/teapot.obj", force="mesh")
    tp.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
    return normalize_to_unit_cube(tp)


def make_solids(n, seed=0):
    """Synthetic flat-faced solids (cubes, boxes, slabs, cylinders) at random sizes/orientations -- the
    flat-face exposure the ModelNet-trained UDF lacks, so its field on flat faces becomes clean and the
    dynamic band meshes hole-free.  Returns (P, N) tensors of DENSE-point clouds."""
    rng = np.random.default_rng(seed); Ps, Ns = [], []
    for i in range(n):
        k = i % 5
        if k == 0:
            m = trimesh.creation.box(extents=[1, 1, 1])
        elif k == 1:
            m = trimesh.creation.box(extents=rng.uniform(0.4, 1.5, 3))
        elif k == 2:
            m = trimesh.creation.cylinder(radius=float(rng.uniform(0.3, 0.6)), height=float(rng.uniform(0.7, 1.6)), sections=32)
        elif k == 3:
            m = trimesh.creation.box(extents=[rng.uniform(0.8, 1.2), rng.uniform(0.18, 0.4), rng.uniform(0.8, 1.2)])  # slab
        else:
            m = trimesh.creation.box(extents=rng.uniform(0.3, 1.4, 3))
        for ang, ax in zip(rng.uniform(0, np.pi, 3), np.eye(3)):
            m.apply_transform(trimesh.transformations.rotation_matrix(ang, ax))
        m = normalize_to_unit_cube(m)
        P, N = E.sample_cloud(m, n=DENSE, noise=0.0, seed=i)
        Ps.append(P.astype(np.float32)); Ns.append(N.astype(np.float32))
    return torch.tensor(np.stack(Ps)), torch.tensor(np.stack(Ns))


def base_meshes():
    return [("cube", normalize_to_unit_cube(trimesh.creation.box(extents=[1, 1, 1]))),
            ("torus", normalize_to_unit_cube(trimesh.creation.torus(major_radius=0.5, minor_radius=0.2))),
            ("sphere", normalize_to_unit_cube(trimesh.creation.uv_sphere(radius=0.7, count=[48, 48]))),
            ("bunny", load_bunny(normalize=True)),
            ("teapot", _upright_teapot()),
            ("knurl", normalize_to_unit_cube(E._knurl_mesh()))]


class GridSDF:
    def __init__(s, g): s.grid = g.astype(np.float64); s.res = g.shape[0]; s.bound = bound; s.trunc = TRUNC
    def sdf(s, q): return WV.grid_trilinear(s.grid, q, bound, TRUNC)


def mc(grid, level=0.0):
    if not (grid.min() < level < grid.max()): return None, None
    v, f, _, _ = measure.marching_cubes(grid.astype(np.float64), level=level)
    return v / (grid.shape[0] - 1) * (2 * bound) - bound, f


def auto_eps(g, thresh=0.02, margin=0.015, lo=0.035, hi=0.085, K=12):
    """Self-configuring UDF meshing band.  The speckle on flat faces is a CONNECTIVITY failure: at a tight
    band the unsigned iso-surface fragments into tiny double-wall specks.  Pick the smallest band whose
    near-surface region {g<=eps} is connected (speck-free), plus a small margin so the noisy iso-surface is
    hole-free.  Adapts per shape (thin/textured -> tight band keeps detail; flat solids -> wider band)."""
    from scipy import ndimage
    for eps in np.linspace(lo, hi, K):
        band = g <= eps
        if band.sum() < 8:
            continue
        sizes = np.bincount(ndimage.label(band)[0].ravel())[1:]
        if sizes.size and 1.0 - sizes[sizes >= max(15, 0.04 * sizes.max())].sum() / band.sum() < thresh:
            return float(min(eps + margin, hi))
    return float(hi)


def adaptive_eps_mesh(g, lo=0.045, hi=0.075, w=3, delta=0.013):
    """Per-voxel UDF band that tracks the local field floor, CLAMPED to the [lo,hi] corridor (the range
    that meshes cleanly for every model): detail -> lo (sharp), sparse flat faces -> up to hi (hole-free).
    Never 0 (holes) nor inf (fat).  Meshes the spatially-varying level set {g = eps_field(x)}."""
    from scipy import ndimage
    eps_field = ndimage.gaussian_filter(np.clip(ndimage.minimum_filter(g, size=w) + delta, lo, hi), 0.8)
    gg = g - eps_field
    if not (gg.min() < 0 < gg.max()):
        return None, None
    v, f, _, _ = measure.marching_cubes(gg.astype(np.float64), 0.0)
    return v / (g.shape[0] - 1) * (2 * bound) - bound, f


def plane_flat_grid(P, N, res_, k=64):
    """Per-voxel 'true-planar' mask from the INPUT cloud (robust to field noise): flat = locally CONSTANT
    normals (cube faces, alignment ~1), which a wide kNN + strict threshold separates from smooth curvature
    (a sphere's normals drift) and edges (normals turn).  Rasterised to the grid by nearest point."""
    from scipy.spatial import cKDTree
    tree = cKDTree(P); _, idx = tree.query(P, k=min(k, len(P)))
    flat_pt = (N[idx] * N[:, None]).sum(-1).mean(1)
    lin = np.linspace(-bound, bound, res_)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    _, gi = tree.query(grid)
    return np.clip((flat_pt[gi].reshape(res_, res_, res_) - 0.985) / 0.015, 0, 1)


def seg_adaptive_mesh(P, N, g):
    """Segment-aware UDF meshing: denoise the field HARDER on true-planar groups (kills the flat-face noise
    that holes flat solids) while leaving curved / detail / thin regions sharp; then the per-voxel dynamic
    band.  ``g`` is the raw field (net output * trunc)."""
    s = plane_flat_grid(P, N, g.shape[0])
    return adaptive_eps_mesh(WV._smooth_grid(g, 0.5) * (1 - s) + WV._smooth_grid(g, 2.4) * s)


_EPSNET = None


def learned_eps(g):
    """Trained, data-dependent UDF band: EpsNet (assets/eps_net.pt) reads the field and predicts the
    loss-optimal eps in one forward pass.  Falls back to the analytic auto_eps if the net is unavailable."""
    global _EPSNET
    if not os.path.exists("assets/eps_net.pt"):
        return auto_eps(g)
    if _EPSNET is None:
        ck = torch.load("assets/eps_net.pt", weights_only=False)
        _EPSNET = WV.EpsNet(lo=ck["lo"], hi=ck["hi"]).to(dev); _EPSNET.load_state_dict(ck["state"]); _EPSNET.eval()
    with torch.no_grad():
        return float(_EPSNET(torch.tensor(g[None, None]).float().to(dev) / TRUNC)[0])


def keep_main_grid(grid):
    """Drop detached inside-blobs: keep only the largest connected inside region (removes floaters)."""
    from scipy import ndimage
    inside = grid < 0
    if not inside.any(): return grid
    lbl, n = ndimage.label(inside)
    if n <= 1: return grid
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    g = grid.copy(); g[inside & (lbl != sizes.argmax())] = abs(grid).max()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--cap", type=int, default=0, help="train-set cap (0 = all cached clouds)")
    ap.add_argument("--res", type=int, default=0, help="override grid resolution (0 = RES)")
    ap.add_argument("--region", action="store_true", help="train context+dense region mode (super-resolution)")
    ap.add_argument("--no-seg", action="store_true")
    ap.add_argument("--resume", default="", help="warm-start from a checkpoint (e.g. assets/waveshape.pt)")
    ap.add_argument("--unsigned", action="store_true",
                    help="UDF mode: unsigned-distance anchor/target + MC at a band (open-shell shapes)")
    ap.add_argument("--base", choices=["signed", "unsigned", "mixed"], default=None,
                    help="field base: signed | unsigned | MIXED (per-point: signed for closed, unsigned for thin -- BOTH in one model)")
    ap.add_argument("--out", default="waveshape", help="checkpoint basename under assets/ (e.g. waveshape_udf)")
    ap.add_argument("--solids", type=int, default=0, help="mix in N synthetic flat-faced solids (clean flat-face fields for UDF)")
    # loss-term ablations (reviewer): override a weight; <0 keeps the default constant. e.g. --lam-conn 0 drops connectivity.
    for _t in ("wave", "grad", "smooth", "sign", "conn", "seg", "geo"):
        ap.add_argument(f"--lam-{_t}", type=float, default=-1.0, help=f"ablation: override LAM_{_t.upper()} (>=0)")
    a = ap.parse_args()
    base = a.base or ("unsigned" if a.unsigned else "signed")
    assert torch.cuda.is_available(), "GPU only"
    res, trunc, batch = RES, TRUNC, a.batch
    with_seg = not a.no_seg
    if a.smoke:
        res, batch, a.epochs, a.cap = 16, 8, 2, 24
    if a.res:
        res = a.res
    out_best, out_latest = f"assets/{a.out}.pt", f"assets/{a.out}_latest.pt"
    mc_level = (UDF_BAND_VOX * 2 * bound / res) if base == "unsigned" else 0.0   # unsigned meshes at a band; signed/mixed at 0
    # effective loss weights (ablation overrides; <0 = keep constant)
    lam_wave = a.lam_wave if a.lam_wave >= 0 else LAM_WAVE
    lam_grad = a.lam_grad if a.lam_grad >= 0 else LAM_GRAD
    lam_smooth = a.lam_smooth if a.lam_smooth >= 0 else LAM_SMOOTH
    lam_seg = a.lam_seg if a.lam_seg >= 0 else LAM_SEG
    lam_geo = a.lam_geo if a.lam_geo >= 0 else LAM_GEO        # geometry-quality block (Chamfer/holes/#comp/F-closed)
    lam_sign = 0.0 if base == "unsigned" else (a.lam_sign if a.lam_sign >= 0 else LAM_SIGN)   # signed-only terms
    lam_conn = 0.0 if base == "unsigned" else (a.lam_conn if a.lam_conn >= 0 else LAM_CONN)
    os.makedirs("renders", exist_ok=True)
    haar = WV.haar_filters_3d(dev); t0 = time.time()

    # ---- favourites (clouds + GT meshes) ------------------------------------
    favs = base_meshes(); fav_names = [n for n, _ in favs]
    Pf, Nf, fav_gt = [], [], []
    for nm, m in favs:
        P, N = E.sample_cloud(m, n=DENSE, noise=0.0, seed=0)
        Pf.append(P.astype(np.float32)); Nf.append(N.astype(np.float32)); fav_gt.append(E.mesh_gt(m))
    Pf = torch.tensor(np.stack(Pf)); Nf = torch.tensor(np.stack(Nf))

    # ---- training pool (cached MN40/Obj++ clouds) ---------------------------
    if a.smoke:
        P, N = Pf.repeat(4, 1, 1), Nf.repeat(4, 1, 1)
    else:
        blob = torch.load("data/se_clouds.pt", weights_only=False)
        P, N = blob["P"], blob["N"]; assert P.shape[1] == DENSE
        sh = torch.randperm(P.shape[0], generator=torch.Generator().manual_seed(SEED))
        P, N = P[sh], N[sh]                                          # SAME permutation keeps P/N aligned
        if a.cap: P, N = P[:a.cap], N[:a.cap]
    if a.solids:
        Ps, Ns = make_solids(a.solids)
        P, N = torch.cat([P, Ps], 0), torch.cat([N, Ns], 0)     # flat-face exposure for the UDF field
        print(f"+ {a.solids} synthetic flat-faced solids", flush=True)
    M = P.shape[0]
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(1))
    n_val = min(CLEAN_SAMPLE, M // 6); val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"train {len(train_idx)} | clean-val {n_val} | res {res} batch {batch} | seg {with_seg}", flush=True)

    # ---- held-out MN40-test validation (real, structured, never trained) ----
    val_names, Pv, Nv, val_gt = [], [], [], []
    if a.smoke:
        for i in range(NVAL):
            val_names.append(f"val{i}"); Pv.append(Pf[i].numpy()); Nv.append(Nf[i].numpy()); val_gt.append(fav_gt[i])
    else:
        from waveshape.datasets import load_mesh_normalized
        CLEAN = ["airplane", "chair", "guitar", "car", "bottle", "sofa", "monitor", "piano", "bench", "laptop"]
        cand = []
        for c in CLEAN: cand += sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))
        np.random.default_rng(SEED).shuffle(cand)
        for p in cand:
            if len(val_names) >= NVAL: break
            try:
                m = load_mesh_normalized(p, max_faces=200000)
                Pp, Nn = E.sample_cloud(m, n=DENSE, noise=0.0, seed=0)
                if not (np.isfinite(Pp).all() and np.isfinite(Nn).all()): continue
                parts = p.replace("\\", "/").split("/")
                val_names.append(f"{parts[-3]}_{parts[-1][:-4].split('_')[-1]}")
                Pv.append(Pp.astype(np.float32)); Nv.append(Nn.astype(np.float32)); val_gt.append(E.mesh_gt(m))
            except Exception:
                continue
    Pval = torch.tensor(np.stack(Pv)); Nval = torch.tensor(np.stack(Nv))

    # ---- model -------------------------------------------------------------
    torch.manual_seed(SEED)
    net = WV.PerceiverWaveNet(with_seg=with_seg, res=res, trunc=trunc, bound=bound, field_mode=base).to(dev)
    best = float("inf")
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, weights_only=False); net.load_state_dict(ck["state"])
        best = ck.get("val_sdferr", float("inf"))
        print(f"resumed from {a.resume} (ep{ck.get('epoch')}, val {best:.4f})", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    print(f"PerceiverWaveNet {net.count_params():,} params (wavelet-from-attention, primitive-free)", flush=True)

    def clean_sdferr(idx_pool):
        net.eval(); tot = cnt = 0
        with torch.no_grad():
            for s in range(0, len(idx_pool), batch):
                ii = idx_pool[s:s + batch]
                Pc, Nc = P[ii].to(dev), N[ii].to(dev)
                clean = WV.tsdf_from_clouds(Pc, Nc, res, trunc, bound, dev, mode=base) / trunc
                pred = net(Pc, Nc)[0]
                tot += float((pred - clean).abs().mean()) * trunc * len(ii); cnt += len(ii)
        net.train(); return tot / max(cnt, 1)

    @torch.no_grad()
    def render_recon(tag, ep):
        net.eval()
        Pc, Nc, names, gts = (Pf, Nf, fav_names, fav_gt) if tag == "fav" else (Pval, Nval, val_names, val_gt)
        panels = []
        for i, nm in enumerate(names):
            grid = WV._smooth_grid(net(Pc[i:i + 1].to(dev), Nc[i:i + 1].to(dev))[0][0, 0].cpu().numpy() * trunc, 0.5)
            if base == "signed":
                grid = keep_main_grid(grid)              # signed only: remove detached floaters (mixed/UDF parts are legitimately disjoint)
            v, f = mc(grid, mc_level); lab = nm
            if v is not None:
                lab = (f"{nm} ({len(f)}f)" if base == "unsigned"               # IoU-vs-solid is meaningless for a pure shell
                       else f"{nm} {E.proper_metrics(gts[i], GridSDF(grid), n=20000)['iou']:.2f}")
            panels.append((lab, v, f))
        try:
            out = f"renders/wsn_{tag}_live.png"
            R3.render_meshes(panels, out, title=f"PerceiverWaveNet {tag} (ep{ep})", size=(300 * len(panels), 340))
            if ep % 2 == 0 or ep == 1: shutil.copy(out, f"renders/wsn_{tag}_ep{ep}.png")
        except Exception as e:
            print(f"  render skip {tag}:", e, flush=True)
        net.train()

    g = torch.Generator().manual_seed(2); hist = []                 # `best` carried over from --resume
    for ep in range(a.epochs):
        tr = train_idx[torch.randperm(len(train_idx), generator=g)]; run = nb = 0
        for s in range(0, len(tr), batch):
            ii = tr[s:s + batch]
            Pc = P[ii].repeat(DRAWS, 1, 1).to(dev); Nc = N[ii].repeat(DRAWS, 1, 1).to(dev)
            with torch.no_grad():
                Bc = Pc.shape[0]
                center = half = None
                Pt_ = Pc                                              # the cloud the TARGET TSDF is built on
                if a.region:                                          # equal whole-mesh / random-region (req 1)
                    si = torch.randint(0, Pc.shape[1], (Bc,), device=dev)
                    center = Pc[torch.arange(Bc, device=dev), si].unsqueeze(1)        # (B,1,3) random surface point
                    whole = torch.rand(Bc, 1, 1, device=dev) < 0.5                    # 50% whole-mesh, 50% region
                    center = torch.where(whole, torch.zeros_like(center), center)
                    half = torch.where(whole, torch.full((Bc, 1, 1), bound, device=dev),
                                       torch.empty(Bc, 1, 1, device=dev).uniform_(0.3, bound))
                    Pt_ = (Pc - center) * (bound / half)              # box-local cloud -> target frame
                clean = WV.tsdf_from_clouds(Pt_, Nc, res, trunc, bound, dev, mode=base) / trunc
                tc = WV.dwt3d(clean, haar)
                ns = torch.full((Bc, 1, 1), NOISE_HI, device=dev)      # noise aug: each mesh twice --
                ns[:len(ii)] = NOISE_LO                               # draw 0 = clean, draw 1 = 10% noise (diff region)
                Pn = Pc + torch.randn(Pc.shape, device=dev) * ns       # noisy cloud -> resolution-free input
                seg_label = WV.wavelet_side_labels(tc) if with_seg else None
            # FLEXIBLE 128-token split: draw a fresh context/main division [CTX_MIN, CTX_MAX] every step so the
            # network learns to read any [context | SEP | main] partition, not one fixed n_ctx.
            nctx = int(torch.randint(CTX_MIN, CTX_MAX + 1, (1,), generator=g).item())
            pred, c_anchor, c_clean, seg = net(Pn, Nc, ctx_P=Pn, ctx_N=Nc, center=center, half=half, n_ctx=nctx)
            loss = WV.wavelet_surface_loss(pred, clean, c_clean, tc, seg, seg_label,
                                           lam_wave, lam_grad, lam_seg, lam_smooth, lam_sign, lam_conn, lam_geo)
            opt.zero_grad()
            if torch.isfinite(loss) and (nb < 5 or loss < 3 * (run / max(nb, 1))):
                loss.backward()
                for p in net.parameters():
                    if p.grad is not None: torch.nan_to_num_(p.grad, 0., 0., 0.)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                run += float(loss.detach()); nb += 1
            del clean, tc, Pn, pred, c_anchor, c_clean, seg, loss
            if nb == 1 or nb % 25 == 0:
                print(f"  ep{ep+1} {min(s+batch,len(tr))}/{len(tr)} step{nb} loss {run/max(nb,1):.4f} "
                      f"| GPU {torch.cuda.max_memory_allocated()/1e9:.1f}GB | {time.time()-t0:.0f}s", flush=True)
            torch.cuda.empty_cache()
        sdferr = clean_sdferr(val_idx)
        render_recon("fav", ep + 1); render_recon("val", ep + 1)
        improved = sdferr < best; hist.append({"epoch": ep + 1, "train": run / max(nb, 1), "val_sdferr": sdferr})
        meta = {"state": net.state_dict(), "base": BASE, "res": res, "trunc": trunc, "with_seg": with_seg,
                "model": "PerceiverWaveNet", "epoch": ep + 1, "val_sdferr": sdferr,
                "unsigned": base == "unsigned", "field_mode": base}
        torch.save(meta, out_latest)                          # always keep the latest (de-floatered) model
        if improved:
            best = sdferr
            torch.save(meta, out_best)                        # best-by-val (raw SDF error)
        print(f"epoch {ep+1}/{a.epochs}: loss {run/max(nb,1):.4f} | val SDFerr {sdferr:.4f} (best {best:.4f})"
              f"{'  *SAVED*' if improved else ''} | {time.time()-t0:.0f}s", flush=True)
        json.dump(hist, open("renders/wsn_train_hist.json", "w"), indent=1)
        torch.cuda.empty_cache()
    print(f"DONE in {time.time()-t0:.0f}s | best val SDFerr {best:.4f} | weights {out_best}", flush=True)


if __name__ == "__main__":
    main()
