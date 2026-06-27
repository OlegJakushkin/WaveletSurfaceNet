"""Qualitative comparison vs public-library baselines (honest, real runs) + dump metrics for the charts.
Rows = shapes, cols = ground truth | SPSR | BPA | GWN | tori | ours.  Sparse clouds (n=2048) like the regime
our model targets.  Cite: SPSR [Kazhdan&Hoppe 2013] (Open3D), BPA [Bernardini 1999] (Open3D),
GWN [Barill 2018; Jacobson 2013] (libigl), tori [Feng et al. 2026]."""
import sys, os, json; sys.path.insert(0, "compare")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import run_all, draw3d, ORDER

SHAPES = ["cube", "teapot", "bunny", "knurl", "chair", "guitar", "table"]
N = 4096
metrics = {}
cols = ORDER                                                   # GT | SPSR | BPA | APSS | RIMLS | tori | ours
fig = plt.figure(figsize=(2.05 * len(cols), 2.2 * len(SHAPES)))
for i, shape in enumerate(SHAPES):
    gt, P, Np, res = run_all(shape, n=N, noise=0.0, seed=0)
    metrics[shape] = {m: {"chamfer": res[m][3], "fscore": res[m][4], "time": res[m][2],
                          "faces": (0 if res[m][1] is None else len(res[m][1])), "parts": res[m][5]} for m in res}
    for j, m in enumerate(cols):
        ax = fig.add_subplot(len(SHAPES), len(cols), i * len(cols) + j + 1, projection="3d")
        v, f = res[m][0], res[m][1]
        draw3d(ax, v, f)
        if i == 0:
            ax.set_title({"GT": "ground truth", "ours": "ours"}.get(m, m), fontsize=11, weight="bold")
        if m != "GT" and j > 0:
            fs = res[m][4]
            ax.text2D(0.5, -0.02, ("fail" if v is None else f"F={fs:.0f}"), transform=ax.transAxes,
                      ha="center", fontsize=8.5, color=("#207020" if m == "ours" else "#444"))
        if j == 0:
            ax.text2D(-0.12, 0.5, shape, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
    print(f"{shape}: " + " | ".join(f"{m} F={res[m][4]:.0f}/C={res[m][3]:.1f}" for m in cols if m != "GT"), flush=True)
fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.03, wspace=0.0, hspace=0.12)
os.makedirs("paper/figs", exist_ok=True)
fig.savefig("paper/figs/cmp_baselines.png", dpi=130); plt.close(fig)
json.dump(metrics, open("compare/metrics.json", "w"), indent=1)
print("wrote paper/figs/cmp_baselines.png + compare/metrics.json", flush=True)
