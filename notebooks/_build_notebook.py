"""Generate notebooks/train_pat_colab.ipynb (kept as a generator to avoid hand-editing JSON).

The notebook is a *complete* training run for a big GPU (tuned for an A100 **80 GB**):
clone the repo, download **≥50k real CAD meshes** from the **ABC dataset**, build a
large dense cache (analytic + real models), run the batched noise-robust trainer with
the **2x-wider CoeffNet** and the **epoch-5-stall stability fixes**, plot the curves,
and plug the models back in.

NOTE: the notebook trains the *repo's* code (it clones REPO_URL@REPO_BRANCH), so push
these changes before running it in Colab.
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

cells.append(md(
"# Points as **Supertoroids** — complete training run (A100 80 GB)",
"",
"Trains BOTH the plain-torus and the supertoroid coefficient networks with the project's",
"**latest** regime and a **2× larger (but equally snappy) architecture**, on a big GPU",
"(tuned for an **A100 80 GB**). End to end it:",
"",
"1. sets up the `pat` package and the `train_gpu.py` trainer (cloned from the repo),",
"2. **downloads ≥50,000 real CAD meshes** from the **ABC dataset** (*A Big CAD Model",
"   Dataset*, ~750k Onshape parts) — no login required,",
"3. builds a large dense cache (analytic assets + the real CAD models, cached once on Drive),",
"4. runs the **batched, noise-robust** trainer: ≥8 epochs over the assets, with per-epoch",
"   point + noise re-randomization (50% of each cloud's points noised, 50% noiseless) and a",
"   50/50 clean/noisy eval split,",
"5. plots the val / clean / noisy curves, and",
"6. plugs `pat_torus.pt` + `pat_supertoroid.pt` straight into `pat.PAT` and the renderer.",
"",
"### What changed vs. the earlier run",
"",
"* **Dataset → ABC (≥50k real CAD meshes).** Replaces the ~12k ModelNet40 set with the",
"  ABC CAD dataset — mechanical parts (brackets, gears, bolts, machined bodies with flat",
"  faces, cylinders, holes, fillets) that are *exactly* the torus/supertoroid/cube/bolt",
"  domain. Fetched no-auth as Wavefront `.obj`; trimesh-loadable; `pat.datasets.mesh_index`",
"  finds them by walking `data/`.",
"* **Architecture → 2× wider, same depth.** `CoeffNet` grows `d_embed 128→192`,",
"  `n_heads 8→12` (head_dim stays 16), `d_ff 512→672` → **~2.06× parameters**. Depth is",
"  held at 6 layers: at the 17-token neighborhood the transformer is launch/memory-bound,",
"  so **width is ~free in latency while depth is not** — it stays as snappy per neighborhood.",
"* **Stability fixes for the epoch-4/5 supertoroid spike.** The squareness exponent `p` is",
"  now **capped** (`p_max=6`), pulled toward `p=2` early with an annealed **square-reg**,",
"  trained with **LR warmup + cosine**, a tighter **per-net / per-head grad clip**, a",
"  **finite-loss-spike skip**, and **weight EMA** (the saved model is the EMA / best-by-val",
"  one). These cure the runaway `p` that made the supertoroid val jump at epoch 4 and stall.",
"",
"Set the runtime to **GPU (A100 80 GB)** before running.",
))

cells.append(md(
"## 1. Setup — Google Drive, repo, deps",
"",
"The notebook trains the **repo's** code, so make sure these changes are pushed to",
"`REPO_BRANCH` first.",
))
cells.append(code(
"import os, sys, subprocess",
"",
"# --- Connect Google Drive and create an output folder (weights + plots land here) ---",
"from google.colab import drive",
"drive.mount('/content/drive')",
"DRIVE_DIR = '/content/drive/MyDrive/points_as_supertoroids'",
"DRIVE_ASSETS = os.path.join(DRIVE_DIR, 'assets')   # everything lands here (== repo assets/)",
"os.makedirs(DRIVE_ASSETS, exist_ok=True)",
"print('outputs (weights + images + curves) will be saved to:', DRIVE_ASSETS)",
"",
"# --- Get the repo (pat package + train_gpu.py + make_renders.py) ---",
"REPO_URL    = \"https://github.com/OlegJakushkin/Points_as_supertoroids.git\"",
"REPO_BRANCH = \"main\"   # branch holding the 2x-arch + ABC-dataset changes",
"REPO_DIR    = \"Points_as_supertoroids\"",
"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
"                'trimesh', 'scikit-image', 'scipy', 'py7zr'], check=False)",
"# clone unless a pat/ package is already present somewhere obvious",
"if not any(os.path.isdir(os.path.join(c, 'pat')) for c in [REPO_DIR, '.', '..']):",
"    subprocess.run(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH,",
"                    REPO_URL, REPO_DIR], check=True)",
"for cand in [REPO_DIR, '.', '..']:",
"    if os.path.isdir(os.path.join(cand, 'pat')):",
"        os.chdir(cand); break",
"sys.path.insert(0, os.getcwd())",
"assert os.path.isdir('pat'), f'pat package not found in {os.getcwd()} (clone of {REPO_URL} failed?)'",
"import torch; import pat",
"print('cwd', os.getcwd(), '| pat ready')",
))
cells.append(code(
"assert torch.cuda.is_available(), 'Set the Colab runtime to a GPU (A100 80 GB).'",
"name = torch.cuda.get_device_name(0)",
"mem = torch.cuda.get_device_properties(0).total_memory / 1e9",
"print(f'GPU: {name}  ({mem:.0f} GB)')",
"if mem < 70:",
"    print('NOTE: <70 GB GPU — lower BATCH / CHUNK / ASSETS / MESHES below to fit your GPU.')",
))

cells.append(md(
"## 2. Build the training cache (≥50k CAD meshes, disk-light)",
"",
"**Default: the ABC dataset** (*A Big CAD Model Dataset*), fetched **with no login** from NYU's",
"public archive. We **never keep the whole corpus on disk.** Each ~7.75 GB OBJ chunk is processed",
"**incrementally**: download → extract → sample each mesh into the compact training cache (points",
"+ GT signed distance) → **delete the chunk's `.7z` and extracted `.obj` files** before the next",
"chunk. Peak disk stays at ~one chunk (~28 GB) instead of ~100 GB.",
"",
"The result is a small **mesh cache (~2 GB)** that is **persisted to Google Drive** (`MESH_CACHE`)",
"and **reused on every later run** — so you download the meshes **only once, only if absent**, and",
"never re-tessellate or re-sample. For diversity we pick chunks **spread across** the 100-chunk",
"corpus (e.g. 0, 20, 40, 60, 80) rather than the first 5.",
"",
"The NYU host has no `Content-Length` and ignores HTTP `Range`, so a dropped download is **not",
"resumable** — we retry the whole chunk. Lower `ABC_CHUNKS` (e.g. 3 → ~30k) if disk/time-limited;",
"switch `DATASET='modelnet'` for the lighter ~12k no-auth fallback.",
))
cells.append(code(
"import os, sys, subprocess, glob, shutil, time, torch",
"from pat.datasets import build_mesh_cache, mesh_index",
"",
"# ---- config shared with the training cell ----",
"DATASET    = 'abc'     # 'abc' (≥50k real CAD, recommended) | 'modelnet' (~12k no-auth fallback)",
"ABC_CHUNKS = 5         # diverse chunks spread across the 100-chunk corpus (~10k models each)",
"ASSETS     = 4000      # analytic assets (built by the trainer itself; no disk)",
"DENSE      = 1536      # dense points cached per mesh (> NPOINTS so the per-epoch subset varies)",
"NQUERY     = 160       # GT query points per mesh",
"MAXFACES   = 200000    # skip pathologically heavy meshes",
"MESH_ROOT  = 'data'",
"MESH_CACHE = os.path.join(DRIVE_DIR, 'mesh_cache.pt')   # ~2 GB, persisted on Drive; reused if present",
"CACHE      = os.path.join(DRIVE_DIR, 'dense_cache.pt')  # full analytic+mesh cache (trainer writes it)",
"os.makedirs(MESH_ROOT, exist_ok=True)",
"",
"if os.path.exists(MESH_CACHE):                            # ---- already downloaded: reuse, no fetch ----",
"    MESHES = torch.load(MESH_CACHE, weights_only=False)['P'].shape[0]",
"    print(f'mesh cache present on Drive ({MESHES} meshes) — skipping download/extract')",
"elif DATASET == 'abc':",
"    subprocess.run(['apt-get', '-qq', 'install', '-y', 'p7zip-full'], check=False)",
"    subprocess.run(['wget', '-q', '-O', 'obj_v00.txt',",
"                    'https://deep-geometry.github.io/abc-dataset/data/obj_v00.txt'], check=True)",
"    with open('obj_v00.txt') as f:                        # 100 lines: '<url> <filename>'",
"        lines = [l.split() for l in f.read().splitlines() if l.strip()]",
"    stepc = max(1, len(lines) // ABC_CHUNKS)",
"    picks = [lines[j] for j in range(0, len(lines), stepc)][:ABC_CHUNKS]   # spread for diversity",
"    parts, total = [], 0",
"    for i, (url, fname) in enumerate(picks):",
"        arc, ext = f'data/{fname}', 'data/abc_chunk'",
"        print(f'[{i+1}/{len(picks)}] {fname}: download (~7.75 GB, no resume)...', flush=True)",
"        t = time.time()",
"        for _ in range(3):                                # retry the WHOLE file on failure",
"            r = subprocess.run(['wget', '--no-check-certificate', '-q', '-O', arc, url])",
"            if r.returncode == 0 and os.path.exists(arc) and os.path.getsize(arc) > 1e8:",
"                break",
"            print('   download failed/short — retrying whole chunk...', flush=True)",
"        print(f'   extract + cache ({(time.time()-t)/60:.1f} min download)...', flush=True)",
"        os.makedirs(ext, exist_ok=True)",
"        subprocess.run(['7z', 'x', arc, f'-o{ext}', '-y', '-bso0', '-bsp0'], check=True)",
"        objs = sorted(glob.glob(f'{ext}/**/*.obj', recursive=True))",
"        d = build_mesh_cache(objs, DENSE, NQUERY, max_faces=MAXFACES, seed=1000 + i)",
"        os.remove(arc); shutil.rmtree(ext, ignore_errors=True)   # <-- delete what we no longer need",
"        if d is not None:",
"            parts.append(d); total += d['P'].shape[0]",
"        print(f'   chunk cached: {0 if d is None else d[\"P\"].shape[0]} meshes "
"(running total {total})', flush=True)",
"    mc = {k: torch.cat([p[k] for p in parts], 0) for k in ('P', 'N', 'Q', 'PHI')}",
"    os.makedirs(os.path.dirname(MESH_CACHE), exist_ok=True)",
"    torch.save(mc, MESH_CACHE); MESHES = mc['P'].shape[0]; del mc",
"    print(f'saved mesh cache -> {MESH_CACHE}  ({MESHES} meshes)')",
"elif DATASET == 'modelnet':                               # ---- light no-auth fallback (~12k) ----",
"    import urllib.request, zipfile",
"    from pat.datasets import modelnet_index",
"    if not modelnet_index(MESH_ROOT):",
"        zp = 'data/ModelNet40.zip'",
"        if not os.path.exists(zp):",
"            urllib.request.urlretrieve('http://modelnet.cs.princeton.edu/ModelNet40.zip', zp)",
"        with zipfile.ZipFile(zp) as z: z.extractall('data')",
"    mc = build_mesh_cache(mesh_index(MESH_ROOT), DENSE, NQUERY, max_faces=MAXFACES, seed=0)",
"    torch.save(mc, MESH_CACHE); MESHES = mc['P'].shape[0]; del mc",
"    print(f'saved mesh cache -> {MESH_CACHE}  ({MESHES} meshes)')",
"",
"print('real meshes cached:', MESHES)",
))

cells.append(md(
"## 3. Train (the complete run)",
"",
"`train_gpu.py` is the project's GPU trainer; we call it as a script so the run uses the latest",
"code verbatim — including the **2× CoeffNet** (`d_embed=192, n_heads=12, d_ff=672`, ~2.06×",
"params) and the **stability defaults** (`--p-max 6`, square-reg, LR warmup, EMA, spike-skip),",
"which are baked into `train_gpu.py`'s argparse defaults. A100-80 GB sizing below.",
"",
"The trainer **adds the analytic assets to the Drive mesh cache** and writes the assembled",
"`dense_cache.pt` (also on Drive). It **never touches raw meshes** — those were already sampled",
"and deleted in step 2. Re-runs reuse `CACHE` directly. Training escapes degenerate batches (it",
"skips any non-finite **or** finite-spike step), so a bad mesh can't poison the weights. Live",
"progress streams below.",
))
cells.append(code(
"EPOCHS  = 8          # ≥8 (gives the warmup+cosine schedule room past the old stall)",
"BATCH   = 96         # clouds per GPU step (A100 80 GB)",
"CHUNK   = 12288      # neighborhoods per transformer launch (a touch lower for the wider net)",
"NPOINTS = 1024       # points fetched per cloud per epoch (< DENSE so the subset varies/epoch)",
"# ASSETS / DENSE / NQUERY / MESHES / MESH_CACHE / CACHE all come from the dataset cell above.",
"",
"cmd = [sys.executable, 'train_gpu.py',",
"       '--assets', str(ASSETS), '--meshes', str(MESHES),",
"       '--mesh-cache-file', MESH_CACHE, '--cache-file', CACHE,",
"       '--dense', str(DENSE), '--n-query', str(NQUERY),",
"       '--epochs', str(EPOCHS), '--batch', str(BATCH), '--chunk', str(CHUNK),",
"       '--n-points', str(NPOINTS), '--eval-assets', '600', '--outdir', 'assets', '--log-every', '100']",
"print(' '.join(cmd))",
"# stream the trainer's stdout live",
"proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)",
"for line in proc.stdout:",
"    print(line, end='')",
"proc.wait(); print('exit', proc.returncode)",
))

cells.append(md(
"## 4. Training curves (and the torus-overfitting story)",
"",
"`train_gpu.py` already wrote `assets/training_curves.png`; we re-plot it here and also save a",
"copy. **Look for the correlation:** the plain **torus overfits** after a couple of epochs (its",
"val + held-out errors bend back up) while the **supertoroid keeps improving** — and, with the",
"stability fixes, the supertoroid's val no longer **spikes at epoch 4**. The supertoroid's",
"squareness DOF matches the boxy CAD data, so it fits it honestly; the torus, stuck at `p = 2`,",
"contorts its curvature coefficients to fake boxy shapes — which is overfitting. We curb it with",
"**weight decay + dropout**, cap/regularize the squareness, and ship the **EMA / best-by-val**",
"epoch (early stopping).",
))
cells.append(code(
"import json, numpy as np, matplotlib.pyplot as plt",
"hist = json.load(open('assets/train_history.json'))",
"ep = [h['epoch'] for h in hist]",
"fig, ax = plt.subplots(1, 2, figsize=(12, 4))",
"ax[0].plot(ep, [h['val_torus_t'] for h in hist], '-o', color='C0', label='Feng26 net | torus')",
"ax[0].plot(ep, [h['val_torus_s'] for h in hist], '-o', color='C1', label='ours net | torus')",
"ax[0].plot(ep, [h.get('val_cube_t', float('nan')) for h in hist], '--s', color='C0', label='Feng26 net | cube')",
"ax[0].plot(ep, [h.get('val_cube_s', float('nan')) for h in hist], '--s', color='C1', label='ours net | cube')",
"ax[0].axhline(0.01, ls=':', c='gray', label='invisible-by-eye bar')",
"ax[0].set_title('val: reconstruct a default torus / sharp cube'); ax[0].set_xlabel('epoch')",
"ax[0].set_ylabel('mean abs SDF err'); ax[0].legend(fontsize=8)",
"for m, lab in [('eval_clean_s','supertoroid clean'),('eval_noisy_s','supertoroid noisy'),",
"               ('eval_clean_t','Feng26 torus clean'),('eval_noisy_t','Feng26 torus noisy')]:",
"    ax[1].plot(ep, [h[m] for h in hist], '-o', label=lab)",
"ax[1].set_title('held-out eval (50% clean / 50% noisy)'); ax[1].set_xlabel('epoch'); ax[1].legend()",
"fig.tight_layout()",
"fig.savefig('assets/training_curves.png', dpi=130)        # save the plot to disc",
"import shutil; shutil.copy('assets/training_curves.png', DRIVE_ASSETS)",
"plt.show()",
"print('final val-torus-err: torus', hist[-1]['val_torus_t'], ' supertoroid', hist[-1]['val_torus_s'])",
"print('saved curves -> assets/training_curves.png and', DRIVE_ASSETS)",
))

cells.append(md(
"## 5. Save weights to Google Drive (do this FIRST)",
"",
"Persist the trained weights + history + curves before anything else, so a later step can't lose",
"them. Everything is written into **`assets/`** (next to the figures rendered in step 7), and",
"mirrored to `DRIVE_ASSETS`. That folder is identical to the repo's `assets/` — copy it back over",
"the repo's `assets/` to update the project with your freshly trained models, curves and figures.",
))
cells.append(code(
"import shutil",
"for f in ['assets/pat_torus.pt', 'assets/pat_supertoroid.pt',",
"          'assets/train_history.json', 'assets/training_curves.png']:",
"    if os.path.exists(f):",
"        shutil.copy(f, DRIVE_ASSETS)",
"print('weights + history + curves saved to', DRIVE_ASSETS)",
"try:",
"    from google.colab import files",
"    files.download('assets/pat_torus.pt'); files.download('assets/pat_supertoroid.pt')",
"except Exception:",
"    pass",
))

cells.append(md(
"## 6. Free memory",
"",
"Drop the training-time objects and empty the CUDA cache before rendering (reconstruction marches",
"a dense grid and is memory-hungry).",
))
cells.append(code(
"import gc, torch",
"for _v in ['hist', 'fig', 'ax', 'PATHS']:",
"    globals().pop(_v, None)",
"gc.collect()",
"if torch.cuda.is_available():",
"    torch.cuda.empty_cache(); torch.cuda.synchronize()",
"    print('GPU mem allocated (MB):', round(torch.cuda.memory_allocated() / 1e6, 1))",
"print('memory freed')",
))

cells.append(md(
"## 7. Render the comparison figures",
"",
"Now (and only now) regenerate the paper-style torus-vs-supertoroid figures with the trained",
"models — `make_renders.py` loads the weights fresh from disc. Saved to Drive and displayed.",
))
cells.append(code(
"import subprocess, sys, glob, shutil",
"# make_renders.py writes the figures into assets/ (next to the weights + curves).",
"subprocess.run([sys.executable, 'make_renders.py', '--points', '1024', '--scale', '2',",
"                '--outdir', 'assets'], check=False)",
"for p in glob.glob('assets/*.png'):",
"    shutil.copy(p, DRIVE_ASSETS)",
"print('figures saved to assets/ and', DRIVE_ASSETS)",
"from IPython.display import Image, display",
"for f in ['torus', 'bunny', 'textured', 'bolts', 'cube', 'composite_noise', 'buckyball']:",
"    p = f'assets/{f}.png'",
"    if os.path.exists(p): display(Image(p))",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(__file__), "train_pat_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
