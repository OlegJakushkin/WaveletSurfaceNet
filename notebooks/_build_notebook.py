"""Generate notebooks/train_pat_colab.ipynb (kept as a generator to avoid hand-editing JSON).

The notebook is a *complete* training run for a big GPU (tuned for an A100 80 GB):
download ModelNet40, build a large dense cache (analytic + real models), run the
batched noise-robust trainer, plot the curves, and plug the models back in.
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
"# Points as **Supertoroids** — complete training run (A100)",
"",
"Trains BOTH the plain-torus and the supertoroid coefficient networks with the project's",
"**latest** regime and architecture, on a big GPU (tuned for an **A100 80 GB**, since this is",
"too slow on a laptop). End to end it:",
"",
"1. sets up the `pat` package and the `train_gpu.py` trainer,",
"2. **downloads ModelNet40** (12,311 real CAD models),",
"3. builds a large dense cache (tens of thousands of analytic assets + the real models),",
"4. runs the **batched, noise-robust** trainer: >=5 epochs over >=10k assets, with per-epoch",
"   point + noise re-randomization (50% of each cloud's points noised, 50% noiseless) and a",
"   50/50 clean/noisy eval split,",
"5. plots the val / clean / noisy curves, and",
"6. plugs `pat_torus.pt` + `pat_supertoroid.pt` straight into `pat.PAT` and the renderer.",
"",
"Set the runtime to **GPU (A100)** before running.",
))

cells.append(md("## 1. Setup — Google Drive, repo, deps"))
cells.append(code(
"import os, sys, subprocess",
"",
"# --- Connect Google Drive and create an output folder (weights + plots land here) ---",
"from google.colab import drive",
"drive.mount('/content/drive')",
"DRIVE_DIR = '/content/drive/MyDrive/points_as_supertoroids'",
"os.makedirs(os.path.join(DRIVE_DIR, 'renders'), exist_ok=True)",
"print('outputs will be saved to:', DRIVE_DIR)",
"",
"# --- Get the pat package + train_gpu.py (point REPO_URL at YOUR fork) ---",
"REPO_URL = \"\"   # e.g. \"https://github.com/<you>/Points_as_supertoroids.git\"",
"REPO_DIR = \"Points_as_supertoroids\"",
"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
"                'trimesh', 'scikit-image', 'scipy'], check=False)",
"if REPO_URL and not os.path.isdir(REPO_DIR):",
"    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR], check=True)",
"for cand in [REPO_DIR, '.', '..']:",
"    if os.path.isdir(os.path.join(cand, 'pat')):",
"        os.chdir(cand); break",
"sys.path.insert(0, os.getcwd())",
"import torch; import pat",
"print('cwd', os.getcwd(), '| pat ready')",
))
cells.append(code(
"assert torch.cuda.is_available(), 'Set the Colab runtime to a GPU (A100).'",
"name = torch.cuda.get_device_name(0)",
"mem = torch.cuda.get_device_properties(0).total_memory / 1e9",
"print(f'GPU: {name}  ({mem:.0f} GB)')",
"if 'A100' not in name:",
"    print('NOTE: not an A100 — lower BATCH / ASSETS / MODELNET below to fit your GPU.')",
))

cells.append(md(
"## 2. Download ModelNet40 (real dataset)",
"",
"~2 GB; extracted once. `pat.datasets.modelnet_index` then finds the meshes by walking the dir.",
))
cells.append(code(
"import urllib.request, zipfile",
"from pat.datasets import modelnet_index",
"os.makedirs('data', exist_ok=True)",
"if not modelnet_index('data'):",
"    zp = 'data/ModelNet40.zip'",
"    if not os.path.exists(zp):",
"        print('downloading ModelNet40 (~2GB)...')",
"        urllib.request.urlretrieve('http://modelnet.cs.princeton.edu/ModelNet40.zip', zp)",
"    print('extracting...')",
"    with zipfile.ZipFile(zp) as z: z.extractall('data')",
"PATHS = modelnet_index('data')",
"print(f'ModelNet40 ready: {len(PATHS)} real models')",
))

cells.append(md(
"## 3. Train (the complete run)",
"",
"`train_gpu.py` is the project's GPU trainer; we call it as a script so the run uses the latest",
"code verbatim. A100-sized defaults below — drop `BATCH`/`ASSETS`/`MODELNET` for a smaller GPU.",
"Live progress (loss, it/s, ETA, and per-epoch val + clean/noisy eval) streams below.",
))
cells.append(code(
"ASSETS   = 20000      # analytic assets (diverse shapes incl. supertoroid facets)",
"MODELNET = 12000      # real ModelNet40 models mixed in (<= len(PATHS))",
"EPOCHS   = 8          # >= 5",
"BATCH    = 64         # clouds per GPU step (A100: 64-128)",
"CHUNK    = 16384      # neighborhoods per transformer launch (A100 can go large)",
"NPOINTS  = 512        # points fetched per cloud per epoch",
"",
"cmd = [sys.executable, 'train_gpu.py',",
"       '--assets', str(ASSETS), '--modelnet', str(MODELNET), '--epochs', str(EPOCHS),",
"       '--batch', str(BATCH), '--chunk', str(CHUNK), '--n-points', str(NPOINTS),",
"       '--eval-assets', '600', '--outdir', 'assets', '--log-every', '100']",
"print(' '.join(cmd))",
"# stream the trainer's stdout live",
"proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)",
"for line in proc.stdout:",
"    print(line, end='')",
"proc.wait(); print('exit', proc.returncode)",
))

cells.append(md("## 4. Training curves"))
cells.append(code(
"import json, numpy as np, matplotlib.pyplot as plt",
"hist = json.load(open('assets/train_history.json'))",
"ep = [h['epoch'] for h in hist]",
"fig, ax = plt.subplots(1, 2, figsize=(12, 4))",
"ax[0].plot(ep, [h['val_torus_t'] for h in hist], '-o', label='torus')",
"ax[0].plot(ep, [h['val_torus_s'] for h in hist], '-o', label='supertoroid')",
"ax[0].axhline(0.01, ls='--', c='gray', label='invisible-by-eye bar')",
"ax[0].set_title('val: reconstruct a default torus'); ax[0].set_xlabel('epoch')",
"ax[0].set_ylabel('mean abs SDF err'); ax[0].legend()",
"for m, lab in [('eval_clean_s','supertoroid clean'),('eval_noisy_s','supertoroid noisy'),",
"               ('eval_clean_t','torus clean'),('eval_noisy_t','torus noisy')]:",
"    ax[1].plot(ep, [h[m] for h in hist], '-o', label=lab)",
"ax[1].set_title('held-out eval (50% clean / 50% noisy)'); ax[1].set_xlabel('epoch'); ax[1].legend()",
"plt.tight_layout(); plt.show()",
"print('final val-torus-err: torus', hist[-1]['val_torus_t'], ' supertoroid', hist[-1]['val_torus_s'])",
))

cells.append(md(
"## 5. Plug the trained models in",
"",
"The checkpoints are plain `state_dict` + config for `pat.model.CoeffNet`, so they load straight",
"into `pat.PAT(model=...)` and the renderer / tests with no glue.",
))
cells.append(code(
"import numpy as np, torch",
"from pat import PAT, shapes",
"from pat.model import CoeffNet",
"def load(p):",
"    ck = torch.load(p, map_location='cpu', weights_only=False)",
"    m = CoeffNet(**ck['config']); m.load_state_dict(ck['state_dict']); m.eval(); return m",
"mt, ms = load('assets/pat_torus.pt'), load('assets/pat_supertoroid.pt')",
"",
"rng = np.random.default_rng(0)",
"for name, sh in [('torus', shapes.Torus(0.6, 0.24)),",
"                 ('supertoroid', shapes.SuperToroid(0.6, 0.28, p_tube=4.0))]:",
"    pts, nrm = sh.sample_surface(2048, rng)",
"    grid = rng.uniform(-1, 1, (4000, 3)); gt = sh.sdf(grid)",
"    et = np.mean(np.abs(PAT(pts, nrm, model=mt, k=16, C=16).sdf(grid, neighbors=64) - gt))",
"    es = np.mean(np.abs(PAT(pts, nrm, model=ms, k=16, C=16).sdf(grid, neighbors=64) - gt))",
"    print(f'{name:12s}  torus-net err {et:.4f}   supertoroid-net err {es:.4f}')",
))
cells.append(code(
"# Regenerate the paper-style comparison figures with the freshly trained models:",
"import subprocess, sys",
"subprocess.run([sys.executable, 'make_renders.py', '--points', '1024', '--scale', '2'], check=False)",
"from IPython.display import Image, display",
"for f in ['torus', 'bunny', 'textured', 'bolts']:",
"    p = f'renders/{f}.png'",
"    if os.path.exists(p): display(Image(p))",
))
cells.append(md(
"## 6. Save weights + plots to Google Drive",
))
cells.append(code(
"import shutil, glob",
"# trained weights + training history -> Drive",
"for f in ['assets/pat_torus.pt', 'assets/pat_supertoroid.pt', 'assets/train_history.json']:",
"    if os.path.exists(f):",
"        shutil.copy(f, DRIVE_DIR)",
"# rendered comparison plots -> Drive/renders",
"for p in glob.glob('renders/*.png'):",
"    shutil.copy(p, os.path.join(DRIVE_DIR, 'renders'))",
"print('saved to', DRIVE_DIR, ':', sorted(os.listdir(DRIVE_DIR)))",
"# also offer a browser download of the weights",
"try:",
"    from google.colab import files",
"    files.download('assets/pat_torus.pt'); files.download('assets/pat_supertoroid.pt')",
"except Exception:",
"    pass",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(__file__), "train_pat_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
