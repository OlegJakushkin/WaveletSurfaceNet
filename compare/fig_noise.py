"""Noise robustness: reconstruct a closed (bunny) and an open (chair) shape from clouds with increasing
Gaussian position noise; compare ours vs SPSR (the strong public baseline).  Honest F-score @ tau=0.05."""
import sys; sys.path.insert(0, "compare")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import sample, recon_ours, fscore, draw3d
import baselines as B

NOISE = [0.0, 0.01, 0.02, 0.05]
SHAPES = ["bunny", "chair"]
spsr = B.available()["SPSR"]
rows = []   # (shape, method, [ (noise, v, f, F) ... ])
fig = plt.figure(figsize=(2.1 * len(NOISE), 2.2 * 2 * len(SHAPES)))
nrow = 2 * len(SHAPES)
curves = {}
for si, shape in enumerate(SHAPES):
    for mi, (mname, runner) in enumerate([("SPSR", lambda P, N: spsr(P, N)), ("ours", lambda P, N: recon_ours(P, N))]):
        r = si * 2 + mi
        Fs = []
        for j, nz in enumerate(NOISE):
            gt, P, N = sample(shape, n=4096, noise=nz, seed=0)
            v, f, _ = runner(P, N); F = fscore(v, f, gt)
            Fs.append(F)
            ax = fig.add_subplot(nrow, len(NOISE), r * len(NOISE) + j + 1, projection="3d")
            draw3d(ax, v, f)
            if r == 0:
                ax.set_title(f"noise {int(nz*100)}%", fontsize=11)
            ax.text2D(0.5, -0.02, f"F={F:.0f}", transform=ax.transAxes, ha="center", fontsize=8.5,
                      color=("#207020" if mname == "ours" else "#444"))
            if j == 0:
                ax.text2D(-0.16, 0.5, f"{shape}\n{mname}", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9.5)
        curves[f"{shape}/{mname}"] = Fs
        print(f"{shape}/{mname}: F " + " ".join(f"{x:.0f}" for x in Fs), flush=True)
fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.02, wspace=0.0, hspace=0.2)
fig.savefig("paper/figs/cmp_noise.png", dpi=130); plt.close(fig)
print("wrote paper/figs/cmp_noise.png", flush=True)
