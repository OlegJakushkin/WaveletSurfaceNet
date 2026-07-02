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
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F, trimesh
from waveshape import wavelet as WV, eval3d as E
from waveshape.shapes import normalize_to_unit_cube

dev = "cuda"; bound = 1.1; RES = 64; TRUNC = 0.1; DENSE = 1536   # TRAIN lattice (coeff base r=32 -> enough edge signal for the refiner); net is res-free (query any res)
EVAL_RES = 128                                     # ALWAYS eval/render/select at 128^3 (res-free query of the 42^3-trained net)
EVAL_CAP = 16                                      # clean-val shapes scored at 128^3 (B=1; dense 128^3 decode is heavy)
NOISE_LO, NOISE_HI = 0.0, 0.20     # input-noise range: draw 0 = CLEAN, draw 1 = per-sample U[0,0.2] (robustness 0-20%)
DRAWS = 2; LR = 2e-3; BASE = 40; SEED = 0; NVAL = 4; CLEAN_SAMPLE = 128
LAM_WAVE, LAM_GRAD, LAM_SEG = 0.4, 0.05, 0.05      # target-aware wavelet term (preserves texture)
LAM_SMOOTH, LAM_SIGN = 0.05, 0.30                  # v1 recipe (v2's boost measured WORSE on every metric -> reverted)
LAM_CONN = 0.4
LAM_CORNER = 0.5                                    # crust penalty (gradient-flip) that DRIVES the wavelet edge-refiner to
#                                                    reshape folded crease crust into clean edges (see wavelet_surface_loss)
LAM_GEO = 0.2                                       # geometry-quality block ON by default at 20%: differentiable FIELD
#                                                    proxies for the mesh metrics ours is weak on (Chamfer/floaters,
#                                                    #components, holes, F-closed; see wavelet_surface_loss).  Applied as a
#                                                    FIXED weight now (the magnitude-normalisation that amplified its gradient
#                                                    is fixed) -- override with --lam-geo 0 to disable and verify any change
#                                                    with compare/gen_table.py before trusting it.
UDF_BAND_VOX = 0.7                                  # UDF mode: MC band in VOXELS (the field floor is ~half a voxel at any res)
CTX_MIN, CTX_MAX = 16, 104                          # FLEXIBLE 128-token budget: the context/main split (n_ctx) is a free
                                                    # deploy-time choice; randomising it per step here teaches the net to read ANY division


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--cap", type=int, default=0, help="train-set cap (0 = all cached clouds)")
    ap.add_argument("--res", type=int, default=0, help="override grid resolution (0 = RES)")
    ap.add_argument("--no-region", action="store_true",
                    help="disable the ALWAYS-ON context+region training (default: 50%% whole / 50%% random box every step)")
    ap.add_argument("--no-seg", action="store_true")
    ap.add_argument("--resume", default="", help="resume from a checkpoint (weights + best-val + EPOCH continue)")
    ap.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast (default: bf16 ON)")
    ap.add_argument("--unsigned", action="store_true",
                    help="UDF mode: unsigned-distance anchor/target + MC at a band (open-shell shapes)")
    ap.add_argument("--base", choices=["signed", "unsigned", "mixed"], default=None,
                    help="field base: signed | unsigned | MIXED (per-point: signed for closed, unsigned for thin -- BOTH in one model)")
    ap.add_argument("--out", default="waveshape", help="checkpoint basename under assets/ (e.g. waveshape_udf)")
    ap.add_argument("--solids", type=int, default=0, help="mix in N synthetic flat-faced solids (clean flat-face fields for UDF)")
    # loss-term ablations (reviewer): override a weight; <0 keeps the default constant. e.g. --lam-conn 0 drops connectivity.
    for _t in ("wave", "grad", "smooth", "sign", "conn", "seg", "geo", "corner"):
        ap.add_argument(f"--lam-{_t}", type=float, default=-1.0, help=f"ablation: override LAM_{_t.upper()} (>=0)")
    a = ap.parse_args()
    region_on = not a.no_region                              # ALWAYS train with context+region unless explicitly disabled
    base = a.base or ("unsigned" if a.unsigned else "mixed")   # DEFAULT = mixed (per-point dynamic signed/unsigned selection)
    assert torch.cuda.is_available(), "GPU only"
    res, trunc, batch = RES, TRUNC, a.batch
    eval_res = EVAL_RES                                       # always eval/render at 128^3 (smoke shrinks it below)
    with_seg = not a.no_seg
    if a.smoke:
        res, batch, a.epochs, a.cap, eval_res = 16, 8, 2, 24, 32
    if a.res:
        res = a.res
    out_best, out_latest = f"assets/{a.out}.pt", f"assets/{a.out}_latest.pt"
    # effective loss weights (ablation overrides; <0 = keep constant)
    lam_wave = a.lam_wave if a.lam_wave >= 0 else LAM_WAVE
    lam_grad = a.lam_grad if a.lam_grad >= 0 else LAM_GRAD
    lam_smooth = a.lam_smooth if a.lam_smooth >= 0 else LAM_SMOOTH
    lam_seg = a.lam_seg if a.lam_seg >= 0 else LAM_SEG
    lam_geo = a.lam_geo if a.lam_geo >= 0 else LAM_GEO        # geometry-quality block (Chamfer/holes/#comp/F-closed)
    lam_corner = a.lam_corner if a.lam_corner >= 0 else LAM_CORNER   # smax corner head training signal
    lam_sign = 0.0 if base == "unsigned" else (a.lam_sign if a.lam_sign >= 0 else LAM_SIGN)   # signed-only terms
    lam_conn = 0.0 if base == "unsigned" else (a.lam_conn if a.lam_conn >= 0 else LAM_CONN)
    os.makedirs("renders", exist_ok=True)
    haar = WV.haar_filters_3d(dev); t0 = time.time()

    # ---- training pool (cached MN40/Obj++ clouds) ---------------------------
    if a.smoke:
        P, N = make_solids(24)                                   # synthetic solids stand in for the cache in smoke
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

    # ---- per-shape REGION cache (labels + junction ops + thinness) -----------------------------------------
    # The region-composed anchor/target needs each shape's dynamic point->region allocation.  Clouds are fixed,
    # so compute ONCE (GPU cdists + python BFS/vote, ~50 ms/shape) and cache; free on every restart/resume.
    rp_path = f"assets/region_pool_{M}.pt"
    if os.path.exists(rp_path):
        region_pool = torch.load(rp_path, weights_only=False)
        print(f"region pool: loaded {len(region_pool)} shapes from {rp_path}", flush=True)
    else:
        region_pool = []; t_rp = time.time()
        for i in range(M):
            Pg, Ng = P[i].to(dev), N[i].to(dev)
            lab = WV.region_labels(Pg, Ng)
            ops = WV.region_pair_ops(Pg, Ng, lab)
            thin = WV.point_thinness(Pg[None], Ng[None])[0].cpu()
            region_pool.append((lab.astype(np.int16), ops, thin))
            if (i + 1) % 500 == 0:
                print(f"  region pool {i+1}/{M} | {time.time()-t_rp:.0f}s", flush=True)
        torch.save(region_pool, rp_path)
        print(f"region pool: built {M} shapes in {time.time()-t_rp:.0f}s -> {rp_path}", flush=True)
    n_regs = np.array([int(r[0].max()) + 1 for r in region_pool])
    print(f"regions/shape: median {np.median(n_regs):.0f} | mean {n_regs.mean():.1f} | max {n_regs.max()}", flush=True)

    def composed_batch(Pb, Nb, regs, res_):
        """Per-item region-composed TSDF batch ``(B,1,res_,res_,res_)`` in /trunc units."""
        return torch.cat([WV.tsdf_composed(Pb[b], Nb[b], regs[b][0], res_, trunc, bound, dev,
                                           ops=regs[b][1], thin=regs[b][2]) for b in range(Pb.shape[0])], 0) / trunc

    # ---- model -------------------------------------------------------------
    torch.manual_seed(SEED)
    net = WV.PerceiverWaveNet(with_seg=with_seg, res=res, trunc=trunc, bound=bound, field_mode=base).to(dev)
    best = float("inf"); start_ep = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, weights_only=False); net.load_state_dict(ck["state"])
        best = ck.get("val_sdferr", float("inf")); start_ep = int(ck.get("epoch", 0))
        print(f"resumed from {a.resume} (continuing at epoch {start_ep+1}, best val {best:.4f})", flush=True)
    # bf16 autocast (memory ~halves -> bigger batch) + 8-bit Adam (bitsandbytes) with graceful fallback
    bf16 = (not a.no_bf16) and torch.cuda.is_bf16_supported()
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(net.parameters(), lr=LR)
        opt_name = "Adam8bit(bnb)"
    except Exception as e:
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        opt_name = f"Adam(fp32 states; bitsandbytes unavailable: {type(e).__name__})"
    from contextlib import nullcontext
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else nullcontext
    print(f"PerceiverWaveNet {net.count_params():,} params (wavelet-from-attention, primitive-free)", flush=True)
    print(f"RECIPE: base={base} | train {res}^3 / eval {eval_res}^3 | context+region {'ON' if region_on else 'OFF'} "
          f"| seg {with_seg} | REGION-COMPOSED anchor+target (sharp edges) | per-piece +5% bound | "
          f"edge-refiner ON (lam_corner {lam_corner}) | dyn-eps+collapse | "
          f"{'bf16' if bf16 else 'fp32'} + {opt_name}", flush=True)

    def _meta(epoch_val, sdferr_val):
        return {"state": net.state_dict(), "base": BASE, "res": res, "eval_res": eval_res, "trunc": trunc,
                "with_seg": with_seg, "model": "PerceiverWaveNet", "epoch": epoch_val, "val_sdferr": sdferr_val,
                "unsigned": base == "unsigned", "field_mode": base, "composed": True}

    def _atomic_save(meta, path):
        tmp = path + ".tmp"; torch.save(meta, tmp); os.replace(tmp, path)   # a crash mid-write can't corrupt the ckpt

    CKPT_EVERY = 25                                           # mid-epoch checkpoint cadence (long run insurance)

    def clean_sdferr(idx_pool):
        # ALWAYS scored at eval_res (128^3) against the region-COMPOSED target: B=1 loop over a capped subset.
        net.eval(); net.set_res(eval_res); tot = cnt = 0
        with torch.no_grad():
            for ii in idx_pool[:EVAL_CAP].tolist():
                Pc, Nc = P[ii:ii + 1].to(dev), N[ii:ii + 1].to(dev); regs = [region_pool[ii]]
                clean = composed_batch(Pc, Nc, regs, eval_res)
                with amp():
                    pred = net(Pc, Nc, regions=regs)[0]
                tot += float((pred.float() - clean).abs().mean()) * trunc; cnt += 1
        net.set_res(res); net.train(); return tot / max(cnt, 1)

    g = torch.Generator().manual_seed(2); hist = []                 # `best`/`start_ep` carried over from --resume
    for ep in range(start_ep, a.epochs):
        tr = train_idx[torch.randperm(len(train_idx), generator=g)]; run = nb = 0
        for s in range(0, len(tr), batch):
            ii = tr[s:s + batch]
            Pc = P[ii].repeat(DRAWS, 1, 1).to(dev); Nc = N[ii].repeat(DRAWS, 1, 1).to(dev)
            regs = [region_pool[gi] for gi in ii.tolist()] * DRAWS      # per-item cached (labels, ops, thin)
            with torch.no_grad():
                Bc = Pc.shape[0]
                center = half = None
                Pt_ = Pc                                              # the cloud the TARGET TSDF is built on
                if region_on:                                         # ALWAYS ON: equal whole-mesh / random-region
                    si = torch.randint(0, Pc.shape[1], (Bc,), device=dev)
                    center = Pc[torch.arange(Bc, device=dev), si].unsqueeze(1)        # (B,1,3) random surface point
                    whole = torch.rand(Bc, 1, 1, device=dev) < 0.5                    # 50% whole-mesh, 50% region
                    center = torch.where(whole, torch.zeros_like(center), center)
                    half = torch.where(whole, torch.full((Bc, 1, 1), bound, device=dev),
                                       torch.empty(Bc, 1, 1, device=dev).uniform_(0.3, bound))
                    Pt_ = (Pc - center) * (bound / half)              # box-local cloud -> target frame
                clean = composed_batch(Pt_, Nc, regs, res)            # region-COMPOSED target (crust-free sharp edges)
                tc = WV.dwt3d(clean, haar)
                ns = torch.empty((Bc, 1, 1), device=dev).uniform_(NOISE_LO, NOISE_HI)   # per-sample noise ~ U[0,0.2]
                ns[:len(ii)] = 0.0                                   # draw 0 = CLEAN (keep clean examples: no thin-collapse)
                Pn = Pc + torch.randn(Pc.shape, device=dev) * ns       # noisy cloud -> resolution-free input
                seg_label = WV.wavelet_side_labels(tc) if with_seg else None
            # FLEXIBLE 128-token split: draw a fresh context/main division [CTX_MIN, CTX_MAX] every step so the
            # network learns to read any [context | SEP | main] partition, not one fixed n_ctx.
            nctx = int(torch.randint(CTX_MIN, CTX_MAX + 1, (1,), generator=g).item())
            with amp():                                               # bf16 forward + loss (fp32 master weights)
                pred, c_anchor, c_clean, seg = net(Pn, Nc, ctx_P=Pn, ctx_N=Nc, center=center, half=half, n_ctx=nctx,
                                                   regions=regs)      # composed ANCHOR from the cached regions
                loss = WV.wavelet_surface_loss(pred, clean, c_clean, tc, seg, seg_label,
                                               lam_wave, lam_grad, lam_seg, lam_smooth, lam_sign, lam_conn, lam_geo, lam_corner)
            opt.zero_grad()
            if torch.isfinite(loss) and (nb < 5 or loss < 3 * (run / max(nb, 1))):
                loss.backward()
                for p in net.parameters():
                    if p.grad is not None: torch.nan_to_num_(p.grad, 0., 0., 0.)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                run += float(loss.detach()); nb += 1
                if nb % CKPT_EVERY == 0: _atomic_save(_meta(ep, best), out_latest)   # mid-epoch resume point
            del clean, tc, Pn, pred, c_anchor, c_clean, seg, loss
            if nb == 1 or nb % 25 == 0:
                print(f"  ep{ep+1} {min(s+batch,len(tr))}/{len(tr)} step{nb} loss {run/max(nb,1):.4f} "
                      f"| GPU {torch.cuda.max_memory_allocated()/1e9:.1f}GB | {time.time()-t0:.0f}s", flush=True)
            torch.cuda.empty_cache()
        sdferr = clean_sdferr(val_idx)                        # cheap best-by-val scalar; the SUITE (render_suite.py) is the eval
        improved = sdferr < best; hist.append({"epoch": ep + 1, "train": run / max(nb, 1), "val_sdferr": sdferr})
        meta = _meta(ep + 1, sdferr)
        _atomic_save(meta, out_latest)                        # always keep the latest model
        if improved:
            best = sdferr
            _atomic_save(meta, out_best)                      # best-by-val (raw SDF error)
        print(f"epoch {ep+1}/{a.epochs}: loss {run/max(nb,1):.4f} | val SDFerr {sdferr:.4f} (best {best:.4f})"
              f"{'  *SAVED*' if improved else ''} | {time.time()-t0:.0f}s", flush=True)
        json.dump(hist, open("renders/wsn_train_hist.json", "w"), indent=1)
        torch.cuda.empty_cache()
    print(f"DONE in {time.time()-t0:.0f}s | best val SDFerr {best:.4f} | weights {out_best}", flush=True)


if __name__ == "__main__":
    main()
