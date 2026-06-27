"""Sparsity robustness: reconstruct from clouds of decreasing size (256..8000 points); ours vs SPSR vs tori.
Honestly shows where each degrades (our gate needs density on thin shells).  F-score @ tau=0.05."""
import sys; sys.path.insert(0, "compare")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import sample, recon_ours, recon_tori, fscore, draw3d
import baselines as B

NS = [256, 512, 1024, 2048, 4096]
SHAPE = "bunny"
spsr = B.available()["SPSR"]
methods = [("SPSR", lambda P, N: spsr(P, N)), ("tori", lambda P, N: recon_tori(P, N)), ("ours", lambda P, N: recon_ours(P, N))]
fig = plt.figure(figsize=(2.1 * len(NS), 2.2 * len(methods)))
for mi, (mname, runner) in enumerate(methods):
    Fs = []
    for j, n in enumerate(NS):
        gt, P, N = sample(SHAPE, n=n, noise=0.0, seed=0)
        v, f, _ = runner(P, N); F = fscore(v, f, gt); Fs.append(F)
        ax = fig.add_subplot(len(methods), len(NS), mi * len(NS) + j + 1, projection="3d")
        draw3d(ax, v, f)
        if mi == 0:
            ax.set_title(f"{n} pts", fontsize=11)
        ax.text2D(0.5, -0.02, f"F={F:.0f}", transform=ax.transAxes, ha="center", fontsize=8.5,
                  color=("#207020" if mname == "ours" else "#444"))
        if j == 0:
            ax.text2D(-0.16, 0.5, mname, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
    print(f"{mname}: F " + " ".join(f"{x:.0f}" for x in Fs), flush=True)
fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.02, wspace=0.0, hspace=0.18)
fig.savefig("paper/figs/cmp_sparsity.png", dpi=130); plt.close(fig)
print("wrote paper/figs/cmp_sparsity.png", flush=True)
