"""Generate notebooks/compare_tori_vs_wavelet_colab.ipynb (generator -> no hand-edited JSON).

A head-to-head Colab notebook that trains, on the **complete ModelNet40 train split**:

  * Network A -- the ORIGINAL tori network (pat.model.CoeffNet; the paper's Sec. 4.3
    coefficient predictor) with the paper's L1 + eikonal blend loss; and
  * Network B -- a NEW wavelet-domain denoising reconstruction model
    (pat.wavelet.WaveletDenoiser): noisy points -> TSDF grid -> 3-D Haar wavelet
    transform -> U-Net denoiser on the coefficients -> inverse wavelet -> clean TSDF
    -> mesh.

Both train from the SAME cached meshes and the SAME noise model, then reconstruct
the held-out ModelNet40 TEST meshes from the SAME noisy clouds, scored with the
voxel-free IoU* + Chamfer distance + side-by-side renders.  Every stage is cached
to Drive and skipped if present, so an interrupted session resumes.

NOTE: the notebook clones REPO_URL@main and imports pat.wavelet + pat.compare, so
push those modules before running it in Colab.  Tuned for a Colab T4/G4 (16 GB) but
runs on any CUDA GPU (lower the batch / resolution knobs for a small GPU).
"""
import json, os

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _src(lines)}
def _src(lines):
    flat = []
    for l in lines:
        flat.extend(l.split("\n"))
    return [s + "\n" for s in flat[:-1]] + [flat[-1]] if flat else []

cells = []

# ----------------------------------------------------------------------------- title
cells.append(md(
"# Original **tori network** vs. a **wavelet-domain denoiser** — on all of ModelNet40",
"",
"Two very different point-cloud → surface models, trained on the **complete ModelNet40 train",
"split** (all ~9843 meshes, 40 categories) and compared head-to-head on the held-out **test** split.",
"",
"### Network A — the original tori network (the paper's method)",
"`pat.model.CoeffNet` predicts, *per point*, the six coefficients of a local height function; each",
"point becomes one fitted **torus** and the tori are blended into a global SDF (Feng–Gkioulekas–Crane,",
"*Points as Tori*, Eq. 25–27).  It is a **per-point parametric** surface — no voxel grid.",
"",
"### Network B — a wavelet-domain denoising reconstruction model (new)",
"The idea from the brief: *train a denoising reconstruction model in a wavelet domain, but don't make",
"wavelets the whole model* — use wavelets as the multi-scale **representation/regularizer** and let a",
"neural net infer the clean surface from noisy points.  Pipeline:",
"",
"```text",
"noisy points",
"   ↓  voxelize to a truncated SDF (TSDF) grid",
"   ↓  3-D Haar wavelet transform  (1 coarse + 7 detail subbands)",
"   ↓  3-D U-Net denoiser  (residual: clean = noisy_coeffs + Δ)",
"   ↓  inverse wavelet transform  (exact — orthonormal Haar)",
"   ↓  clean TSDF  →  marching cubes  →  mesh",
"```",
"",
"Losses: **TSDF L1** + **wavelet-coefficient L1** (denoise multi-scale structure) + **gradient",
"L1** (smoothness/eikonal).  The intuition: random noise lives in *incoherent* high-frequency",
"coefficients while real structure is *coherent across scales*, so the net keeps coherent detail",
"and drops noise instead of blurring all high frequencies.  Closest prior work: neural",
"wavelet-domain TSDF modeling ([arXiv:2209.08725](https://arxiv.org/abs/2209.08725)) and",
"self-prior point→mesh denoising ([Point2Mesh, arXiv:2005.11084](https://arxiv.org/abs/2005.11084)).",
"",
"### Fair comparison",
"Both networks see the **same cached meshes**, the **same Gaussian noise model**, and at eval the",
"**same noisy input cloud** per test mesh.  Quality is the **voxel-free** Monte-Carlo IoU\\*",
"(continuous on both sides, no occupancy grid) plus symmetric **Chamfer distance** to the GT surface.",
"",
"> Set the Colab runtime to a **GPU** (T4 / G4 / A100).  Every result is cached to Drive and reused.",
))

# ----------------------------------------------------------------------------- 1. setup
cells.append(md("## 1 · Setup — Google Drive, repo, deps"))
cells.append(code(
"import os, sys, subprocess",
"from google.colab import drive",
"drive.mount('/content/drive')",
"DRIVE_DIR = '/content/drive/MyDrive/points_as_supertoroids'",
"os.makedirs(DRIVE_DIR, exist_ok=True)",
"print('outputs will be saved under:', DRIVE_DIR)",
"",
"REPO_URL    = 'https://github.com/OlegJakushkin/Points_as_supertoroids.git'",
"REPO_BRANCH = 'main'   # branch holding pat/wavelet.py + pat/compare.py",
"REPO_DIR    = 'Points_as_supertoroids'",
"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'trimesh', 'scikit-image', 'scipy', 'pyvista', 'rtree'], check=False)",
"subprocess.run('apt-get install -y -qq xvfb libgl1-mesa-glx >/dev/null 2>&1', shell=True, check=False)   # headless 3D rendering",
"if not any(os.path.isdir(os.path.join(c, 'pat')) for c in [REPO_DIR, '.', '..']):",
"    subprocess.run(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, REPO_DIR], check=True)",
"for cand in [REPO_DIR, '.', '..']:",
"    if os.path.isdir(os.path.join(cand, 'pat')):",
"        os.chdir(cand); break",
"sys.path.insert(0, os.getcwd())",
"import torch, pat",
"from pat import wavelet as WV, compare as CMP   # the new modules (push them first!)",
"assert torch.cuda.is_available(), 'Set the Colab runtime to a GPU (T4 / G4 / A100).'",
"print('cwd', os.getcwd(), '| pat ready | GPU', torch.cuda.get_device_name(0))",
))

# ----------------------------------------------------------------------------- 2. config
cells.append(md("## 2 · Config — every knob commented"))
cells.append(code(
"# ---- dataset — the COMPLETE ModelNet40 TRAIN split (all ~9843 meshes) ----",
"MN40_URL   = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'   # official Princeton OFF meshes (~2GB)",
"DENSE      = 1536    # dense surface points cached per mesh (the cloud both nets consume).",
"NQUERY     = 160     # GT signed-distance query points per mesh (used by the tori net's loss).",
"MAXFACES   = 200000  # skip pathologically heavy meshes.",
"SUBSET     = None    # None = train BOTH nets on EVERY cached train mesh. Set an int (e.g. 1000) to test faster.",
"NOISE_STD  = 0.015   # training noise std (unit-cube units) added to the cloud — the noise-robustness regime.",
"",
"# ---- Network A: original tori network (pat.model.CoeffNet) ----",
"TORI_EPOCHS  = 6      # epochs over the cache.",
"TORI_BATCH   = 24     # clouds per GPU step (raise on a big GPU; lower on OOM).",
"TORI_NPOINTS = 512    # points fetched per cloud per step (< DENSE so the subset varies).",
"K_NBR        = 16     # kNN neighborhood size for the CoeffNet transformer.",
"D_EMBED      = 128    # transformer width (paper CoeffNet = 128).",
"N_LAYERS     = 8      # transformer depth (paper = 8).",
"TORI_LR      = 8e-4   # AdamW lr.",
"",
"# ---- Network B: wavelet-domain denoiser (pat.wavelet.WaveletDenoiser) ----",
"WAVE_RES    = 32      # TSDF grid resolution (must be divisible by 8). 32 is fast+light; 64 = finer detail,",
"                      #   ~8x the voxel cost (drop the batch on a small GPU). The net is fully-conv, so it",
"                      #   runs at any res, but TRAIN and EVAL must use the same res (voxel scale).",
"WAVE_TRUNC  = 0.1     # SDF truncation band (distance units) — the field is clipped to +-this.",
"WAVE_BASE   = 32      # first U-Net stage width (multiple of 8); 16 is lighter, 48/64 heavier.",
"WAVE_EPOCHS = 6       # epochs over the cache.",
"WAVE_BATCH  = 8       # meshes per GPU step (each builds a clean+noisy TSDF; lower on OOM).",
"WAVE_LR     = 1e-3    # AdamW lr.",
"LAM_WAVE    = 1.0     # weight of the wavelet-coefficient L1 term.",
"LAM_GRAD    = 0.1     # weight of the gradient-consistency (smoothness/eikonal) term.",
"",
"# ---- eval ----",
"EVAL_NOISE = 0.01     # noise std on the held-out test clouds BOTH nets reconstruct from.",
"N_EVAL     = 12       # held-out ModelNet40 TEST meshes to reconstruct + score (spread across categories).",
"RES_RECON  = 96       # marching-cubes grid for the tori net's mesh (its field is continuous/parametric).",
"",
"FORCE_TORI = False    # set True to retrain the tori net even if a checkpoint exists.",
"FORCE_WAVE = False    # set True to retrain the wavelet net even if a checkpoint exists.",
"",
"MESH_CACHE = os.path.join(DRIVE_DIR, 'mesh_cache.pt')",
"MODELS_DIR = os.path.join(DRIVE_DIR, 'compare_models'); os.makedirs(MODELS_DIR, exist_ok=True)",
"EVAL_DIR   = os.path.join(DRIVE_DIR, 'compare_eval');   os.makedirs(EVAL_DIR, exist_ok=True)",
"print('config set | subset', SUBSET, '| wavelet res', WAVE_RES)",
))

# ----------------------------------------------------------------------------- 3. dataset
cells.append(md(
"## 3 · Dataset — the COMPLETE ModelNet40 TRAIN split (download once, cache reused/resumed)",
"",
"Downloads ModelNet40 (40 categories of OFF meshes) and caches **every train mesh** as `{P,N,Q,PHI}`",
"(dense surface cloud + normals + GT signed-distance queries) — the **same cache both networks train",
"on**.  Built in batches and saved incrementally, so a Colab disconnect just resumes.  The held-out",
"**test** split feeds the eval cell.  (If you already built this cache with another notebook in the",
"same Drive folder, it is reused as-is.)",
))
cells.append(code(
"import json, zipfile, urllib.request, glob",
"PROG = os.path.join(DRIVE_DIR, 'mn40_progress.json')",
"MN40_DIR = os.path.join(DRIVE_DIR, 'ModelNet40'); MN40_ZIP = os.path.join(DRIVE_DIR, 'ModelNet40.zip')",
"if not os.path.isdir(MN40_DIR):                   # download + extract once (~2GB)",
"    if not os.path.exists(MN40_ZIP):",
"        print('downloading ModelNet40 (~2GB, one-time)...', flush=True)",
"        urllib.request.urlretrieve(MN40_URL, MN40_ZIP)",
"    print('extracting ModelNet40...', flush=True)",
"    with zipfile.ZipFile(MN40_ZIP) as z: z.extractall(DRIVE_DIR)",
"train_paths = sorted(glob.glob(os.path.join(MN40_DIR, '*', 'train', '*.off')))",
"test_paths  = sorted(glob.glob(os.path.join(MN40_DIR, '*', 'test',  '*.off')))",
"assert train_paths, f'no ModelNet40 train .off meshes under {MN40_DIR} (download/extract failed?)'",
"ncat = len({os.path.basename(os.path.dirname(os.path.dirname(p))) for p in train_paths})",
"print(f'ModelNet40: {len(train_paths)} train / {len(test_paths)} test meshes, {ncat} categories')",
"from pat.datasets import build_mesh_cache",
"parts = [torch.load(MESH_CACHE, weights_only=False)] if os.path.exists(MESH_CACHE) else []",
"cached = parts[0]['P'].shape[0] if parts else 0",
"i = json.load(open(PROG))['idx'] if (parts and os.path.exists(PROG)) else 0",
"BATCH = 400",
"while i < len(train_paths):",
"    d = build_mesh_cache(train_paths[i:i+BATCH], DENSE, NQUERY, max_faces=MAXFACES, shuffle=False)",
"    i += BATCH",
"    if d is not None:",
"        parts.append(d); cached += d['P'].shape[0]",
"        torch.save({k: torch.cat([p[k] for p in parts], 0) for k in ('P','N','Q','PHI')}, MESH_CACHE)",
"        json.dump({'idx': i, 'target': len(train_paths)}, open(PROG, 'w'))",
"    print(f'  cached {cached} meshes ({min(i,len(train_paths))}/{len(train_paths)} paths)', flush=True)",
"cache = torch.load(MESH_CACHE, weights_only=False)",
"print('dataset ready:', cache['P'].shape[0], 'ModelNet40 train meshes |',",
"      'cloud', tuple(cache['P'].shape[1:]), 'queries', tuple(cache['Q'].shape[1:]))",
))

# ----------------------------------------------------------------------------- 4. train tori
cells.append(md(
"## 4 · Network A — train the ORIGINAL tori network on all of ModelNet40",
"",
"`pat.compare.train_tori_cache` trains `pat.model.CoeffNet` (the paper's Sec. 4.3 predictor) with the",
"paper's **L1 + eikonal blend loss** (Eq. 27) — a device-agnostic copy of `train_gpu.py`'s batched",
"regime, but trained from the ModelNet40 cache only (no analytic assets), so it sees exactly the meshes",
"the wavelet net does.  Each step draws a random point subset of each cloud and adds fresh Gaussian",
"noise (GT distance is always to the clean surface).  The checkpoint is cached to Drive and reused.",
))
cells.append(code(
"TORI_PATH = os.path.join(MODELS_DIR, 'tori_modelnet40.pt')",
"if os.path.exists(TORI_PATH) and not FORCE_TORI:",
"    ck = torch.load(TORI_PATH, weights_only=False)",
"    tori = CMP.CoeffNet(d_embed=ck['d_embed'], n_layers=ck['n_layers']).cuda()",
"    tori.load_state_dict(ck['state']); tori_hist = ck.get('hist', [])",
"    print('reused', TORI_PATH, '| final loss', round(tori_hist[-1]['loss'], 4) if tori_hist else '?')",
"else:",
"    tori, tori_hist = CMP.train_tori_cache(cache, k=K_NBR, epochs=TORI_EPOCHS, batch=TORI_BATCH,",
"        n_points=TORI_NPOINTS, noise_std=NOISE_STD, lr=TORI_LR, d_embed=D_EMBED, n_layers=N_LAYERS,",
"        device='cuda', subset=SUBSET, log_every=50)",
"    torch.save({'state': tori.state_dict(), 'hist': tori_hist, 'd_embed': D_EMBED, 'n_layers': N_LAYERS}, TORI_PATH)",
"    json.dump(tori_hist, open(os.path.join(MODELS_DIR, 'tori_log.json'), 'w'), indent=1)",
"print('tori net params:', sum(p.numel() for p in tori.parameters()))",
))

# ----------------------------------------------------------------------------- 5. train wavelet
cells.append(md(
"## 5 · Network B — train the WAVELET-DOMAIN denoiser on all of ModelNet40",
"",
"`pat.wavelet.train_wavelet` builds, per mesh per step, a **clean target TSDF** from the cached (clean)",
"cloud and a **noisy input TSDF** from the same cloud with fresh Gaussian noise, then trains the",
"`WaveletDenoiser` to map noisy → clean (TSDF L1 + wavelet-coefficient L1 + gradient L1).  The TSDFs are",
"built on the GPU (batched nearest-point), so the whole batch of meshes is voxelized in parallel.  The",
"net is residual-initialized to the identity, so training only learns the denoising *correction*.",
))
cells.append(code(
"WAVE_PATH = os.path.join(MODELS_DIR, 'wavelet_modelnet40.pt')",
"if os.path.exists(WAVE_PATH) and not FORCE_WAVE:",
"    ck = torch.load(WAVE_PATH, weights_only=False)",
"    wave = WV.WaveletDenoiser(base=ck['base']).cuda(); wave.load_state_dict(ck['state'])",
"    wave_hist = ck.get('hist', []); WAVE_RES, WAVE_TRUNC = ck['res'], ck['trunc']",
"    print('reused', WAVE_PATH, '| res', WAVE_RES, '| final loss', round(wave_hist[-1]['loss'], 4) if wave_hist else '?')",
"else:",
"    wave, wave_hist = WV.train_wavelet(cache, res=WAVE_RES, trunc=WAVE_TRUNC, epochs=WAVE_EPOCHS,",
"        batch=WAVE_BATCH, noise_std=NOISE_STD, lr=WAVE_LR, lam_wave=LAM_WAVE, lam_grad=LAM_GRAD,",
"        base=WAVE_BASE, device='cuda', subset=SUBSET, log_every=50)",
"    torch.save({'state': wave.state_dict(), 'hist': wave_hist, 'base': WAVE_BASE,",
"               'res': WAVE_RES, 'trunc': WAVE_TRUNC}, WAVE_PATH)",
"    json.dump(wave_hist, open(os.path.join(MODELS_DIR, 'wavelet_log.json'), 'w'), indent=1)",
"print('wavelet net params:', wave.count_params())",
))

# ----------------------------------------------------------------------------- 6. curves
cells.append(md("## 6 · Training curves"))
cells.append(code(
"import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt",
"fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))",
"if tori_hist: ax[0].plot([h['epoch'] for h in tori_hist], [h['loss'] for h in tori_hist], '-o', c='C0')",
"ax[0].set_title('Network A: tori (L1 + eikonal)'); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss')",
"if wave_hist: ax[1].plot([h['epoch'] for h in wave_hist], [h['loss'] for h in wave_hist], '-o', c='C3')",
"ax[1].set_title('Network B: wavelet denoiser (TSDF + wave + grad)'); ax[1].set_xlabel('epoch')",
"fig.tight_layout(); fig.savefig(os.path.join(EVAL_DIR, 'training_curves.png'), dpi=130)",
"from IPython.display import Image, display; display(Image(os.path.join(EVAL_DIR, 'training_curves.png')))",
))

# ----------------------------------------------------------------------------- 7. head-to-head eval
cells.append(md(
"## 7 · Head-to-head on the held-out ModelNet40 TEST split (voxel-free, same noisy input)",
"",
"For each test mesh the **same** noisy surface cloud is reconstructed by both networks.  `head_to_head`",
"reports the **voxel-free** Monte-Carlo IoU\\* (our continuous `sdf<0` vs the exact mesh occupancy — no",
"grid on either side), symmetric **Chamfer distance** to the GT surface, and a 3-panel render",
"(ground-truth | tori | wavelet).  These meshes were never seen in training, so this is true",
"generalization.",
))
cells.append(code(
"from pat.datasets import load_mesh_normalized",
"import numpy as np",
"sel = test_paths[:: max(1, len(test_paths)//N_EVAL)][:N_EVAL]   # spread across categories",
"records = []",
"for p in sel:",
"    try: mesh = load_mesh_normalized(p, max_faces=MAXFACES)",
"    except Exception as e: print('skip', os.path.basename(p), e); continue",
"    name = os.path.splitext(os.path.basename(p))[0]",
"    r = CMP.head_to_head(mesh, tori, wave, n_cloud=DENSE, noise=EVAL_NOISE, k=K_NBR,",
"        res_recon=RES_RECON, res_wave=WAVE_RES, trunc=WAVE_TRUNC, n_metric=40000, device='cuda',",
"        render_path=os.path.join(EVAL_DIR, f'cmp_{name}.png'), name=name)",
"    records.append(r)",
"    print(f\"{name:22s} | tori  IoU* {r['tori']['iou']:.3f} ch {r['tori']['chamfer']:.4f}\"",
"          f\"   |  wavelet  IoU* {r['wavelet']['iou']:.3f} ch {r['wavelet']['chamfer']:.4f}\")",
"json.dump(records, open(os.path.join(EVAL_DIR, 'compare_metrics.json'), 'w'), indent=1)",
"from IPython.display import Image, display",
"import glob as _g",
"for q in sorted(_g.glob(os.path.join(EVAL_DIR, 'cmp_*.png')))[:6]: display(Image(q))",
))

# ----------------------------------------------------------------------------- 8. summary
cells.append(md(
"## 8 · Summary — mean metrics, who wins where",
"",
"Mean IoU\\* (higher is better), Chamfer distance and volume error (lower is better) across the held-out",
"meshes, plus a per-mesh IoU\\* scatter so you can see which shapes favor the **parametric tori** field",
"vs. the **wavelet TSDF** denoiser.",
))
cells.append(code(
"import numpy as np, matplotlib.pyplot as plt",
"def col(side, key): return np.array([r[side][key] for r in records], float)",
"rows = [('IoU* (↑)', 'iou'), ('Chamfer (↓)', 'chamfer'), ('vol_err (↓)', 'vol_err')]",
"print(f\"{'metric':14s} {'tori':>12s} {'wavelet':>12s}\")",
"for lab, key in rows:",
"    print(f'{lab:14s} {np.nanmean(col(\"tori\", key)):12.4f} {np.nanmean(col(\"wavelet\", key)):12.4f}')",
"wins = int((col('wavelet', 'iou') > col('tori', 'iou')).sum())",
"print(f'\\nwavelet beats tori on IoU* for {wins}/{len(records)} meshes')",
"fig, ax = plt.subplots(1, 2, figsize=(11, 4))",
"x = np.arange(len(rows)); w = 0.36",
"ax[0].bar(x - w/2, [np.nanmean(col('tori', k)) for _, k in rows], w, label='tori', color='C0')",
"ax[0].bar(x + w/2, [np.nanmean(col('wavelet', k)) for _, k in rows], w, label='wavelet', color='C3')",
"ax[0].set_xticks(x); ax[0].set_xticklabels([l for l, _ in rows]); ax[0].legend(); ax[0].set_title('mean metrics')",
"it, iw = col('tori', 'iou'), col('wavelet', 'iou')",
"ax[1].scatter(it, iw, s=28, c='C2'); lim = [0, max(0.01, np.nanmax([it.max(), iw.max()]))]",
"ax[1].plot(lim, lim, ':', c='gray'); ax[1].set_xlabel('tori IoU*'); ax[1].set_ylabel('wavelet IoU*')",
"ax[1].set_title('per-mesh IoU* (above line = wavelet wins)')",
"fig.tight_layout(); fig.savefig(os.path.join(EVAL_DIR, 'summary.png'), dpi=130)",
"from IPython.display import Image, display; display(Image(os.path.join(EVAL_DIR, 'summary.png')))",
"print('done — models + eval cached under', DRIVE_DIR)",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "compare_tori_vs_wavelet_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "(%d cells)" % len(cells))
