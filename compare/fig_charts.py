"""Quantitative bar charts from compare/metrics.json (written by fig_baselines.py): mean F-score split into
closed vs open shapes, and mean reconstruction time.  Honest: F-score at tau=0.05 (see core.fscore)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = json.load(open("compare/metrics.json"))
CLOSED = ["cube", "teapot", "bunny", "knurl"]
OPEN = ["chair", "guitar", "table"]
METHODS = ["SPSR", "BPA", "tori", "ours"]
COL = {"SPSR": "#8c6bb1", "BPA": "#9ecae1", "tori": "#a02020", "ours": "#207020"}


def mean(metric, shapes, m):
    vals = [M[s][m][metric] for s in shapes if m in M[s]]
    return float(np.mean(vals)) if vals else 0.0


alls = CLOSED + OPEN
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.4))
# --- F-score, closed vs open ---
groups = ["closed solids", "open shells"]
x = np.arange(len(groups)); w = 0.2
for k, m in enumerate(METHODS):
    vals = [mean("fscore", CLOSED, m), mean("fscore", OPEN, m)]
    ax[0].bar(x + (k - 1.5) * w, vals, w, label=m, color=COL[m])
ax[0].set_xticks(x); ax[0].set_xticklabels(groups); ax[0].set_ylabel("F-score @ tau=0.05 (%, higher better)")
ax[0].set_title("Reconstruction quality"); ax[0].set_ylim(0, 105); ax[0].legend(fontsize=8, ncol=2)
# --- components (watertightness) on open shells, log scale ---
xm = np.arange(len(METHODS))
parts = [np.mean([M[s][m]["parts"] for s in OPEN if m in M[s]]) for m in METHODS]
ax[1].bar(xm, parts, color=[COL[m] for m in METHODS]); ax[1].set_yscale("log")
ax[1].set_xticks(xm); ax[1].set_xticklabels(METHODS); ax[1].set_ylabel("# components on open shells (log, lower better)")
ax[1].set_title("Watertightness")
for i, p in enumerate(parts):
    ax[1].text(i, p, f"{p:.0f}", ha="center", va="bottom", fontsize=8)
# --- time, all shapes ---
times = [mean("time", alls, m) for m in METHODS]
ax[2].bar(xm, times, color=[COL[m] for m in METHODS])
ax[2].set_xticks(xm); ax[2].set_xticklabels(METHODS); ax[2].set_ylabel("seconds (lower better)")
ax[2].set_title("Reconstruction time")
for i, t in enumerate(times):
    ax[2].text(i, t, f"{t:.1f}s", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig("paper/figs/cmp_charts.png", dpi=140); plt.close(fig)
print("F-score closed/open + time:")
for m in METHODS:
    print(f"  {m:5s}: closed {mean('fscore',CLOSED,m):5.1f} | open {mean('fscore',OPEN,m):5.1f} | parts(open avg) {np.mean([M[s][m]['parts'] for s in OPEN if m in M[s]]):.0f} | {mean('time',alls,m):.2f}s")
print("wrote paper/figs/cmp_charts.png", flush=True)
