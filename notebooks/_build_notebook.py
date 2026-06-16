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
"# --- Get the repo (pat package + train_gpu.py + make_renders.py) ---",
"REPO_URL = \"https://github.com/OlegJakushkin/Points_as_supertoroids.git\"",
"REPO_DIR = \"Points_as_supertoroids\"",
"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
"                'trimesh', 'scikit-image', 'scipy'], check=False)",
"# clone unless a pat/ package is already present somewhere obvious",
"if not any(os.path.isdir(os.path.join(c, 'pat')) for c in [REPO_DIR, '.', '..']):",
"    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR], check=True)",
"for cand in [REPO_DIR, '.', '..']:",
"    if os.path.isdir(os.path.join(cand, 'pat')):",
"        os.chdir(cand); break",
"sys.path.insert(0, os.getcwd())",
"assert os.path.isdir('pat'), f'pat package not found in {os.getcwd()} (clone of {REPO_URL} failed?)'",
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
"The **dense cache is built once and saved on Drive** (`CACHE`); re-runs reuse it and skip the",
"~15-20 min of ModelNet caching. Training escapes degenerate batches (it skips any non-finite",
"step), so a bad mesh can't NaN-poison the weights. Live progress streams below.",
))
cells.append(code(
"ASSETS   = 2000      # analytic assets (diverse shapes incl. supertoroid facets)",
"MODELNET = 12000     # real ModelNet40 models mixed in (<= len(PATHS))",
"EPOCHS   = 5         # >= 5",
"BATCH    = 128       # clouds per GPU step (A100: 64-128)",
"CHUNK    = 16384     # neighborhoods per transformer launch (A100 can go large)",
"NPOINTS  = 1024      # points fetched per cloud per epoch",
"DENSE    = 1536      # dense points cached per asset (> NPOINTS so the fetched subset varies/epoch)",
"CACHE    = os.path.join(DRIVE_DIR, 'dense_cache.pt')  # built once, reused on re-runs",
"",
"cmd = [sys.executable, 'train_gpu.py',",
"       '--assets', str(ASSETS), '--modelnet', str(MODELNET), '--epochs', str(EPOCHS),",
"       '--batch', str(BATCH), '--chunk', str(CHUNK), '--n-points', str(NPOINTS),",
"       '--dense', str(DENSE), '--cache-file', CACHE,",
"       '--eval-assets', '600', '--outdir', 'assets', '--log-every', '100']",
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
"val + held-out errors bend back up) while the **supertoroid keeps improving**. The supertoroid's",
"squareness DOF matches the boxy data, so it fits it honestly; the torus, stuck at `p = 2`, has to",
"contort its curvature coefficients to fake boxy shapes — which is overfitting. We curb it with",
"**weight decay + dropout** and ship the **best-by-val** epoch (early stopping).",
))
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
"fig.tight_layout()",
"fig.savefig('assets/training_curves.png', dpi=130)        # save the plot to disc",
"import shutil; shutil.copy('assets/training_curves.png', DRIVE_DIR)",
"plt.show()",
"print('final val-torus-err: torus', hist[-1]['val_torus_t'], ' supertoroid', hist[-1]['val_torus_s'])",
"print('saved curves -> assets/training_curves.png and', DRIVE_DIR)",
))

cells.append(md(
"## 5. Save weights to Google Drive (do this FIRST)",
"",
"Persist the trained weights + history + curves before anything else, so a later step can't lose them.",
))
cells.append(code(
"import shutil",
"for f in ['assets/pat_torus.pt', 'assets/pat_supertoroid.pt',",
"          'assets/train_history.json', 'assets/training_curves.png']:",
"    if os.path.exists(f):",
"        shutil.copy(f, DRIVE_DIR)",
"print('weights + history + curves saved to', DRIVE_DIR)",
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
"subprocess.run([sys.executable, 'make_renders.py', '--points', '1024', '--scale', '2'], check=False)",
"for p in glob.glob('renders/*.png'):",
"    shutil.copy(p, os.path.join(DRIVE_DIR, 'renders'))",
"print('renders saved to', os.path.join(DRIVE_DIR, 'renders'))",
"from IPython.display import Image, display",
"for f in ['torus', 'bunny', 'textured', 'bolts', 'cube', 'composite_noise', 'buckyball']:",
"    p = f'renders/{f}.png'",
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
