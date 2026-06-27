"""Quantitative bar charts over the ModelNet40 TEST-set sample (compare/metrics_val.json from bench_val.py).
Closed vs open is split by our gate's thin-fraction (>0.30 -> open).  F-score @ tau=0.05; #components is the
watertightness proxy; runtime in seconds.  Falls back to the 7-shape compare/metrics.json if val isn't present."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["SPSR", "BPA", "APSS", "RIMLS", "tori", "ours"]
COL = {"SPSR": "#8c6bb1", "BPA": "#9ecae1", "APSS": "#fdae6b", "RIMLS": "#f768a1", "tori": "#a02020", "ours": "#207020"}

rows = json.load(open("compare/metrics_val.json"))
closed = [r for r in rows if r["kind"] == "closed"]
opn = [r for r in rows if r["kind"] == "open"]
present = [m for m in METHODS if any(m in r["methods"] for r in rows)]


def agg(rs, m, metric):
    vals = [r["methods"][m][metric] for r in rs if m in r["methods"]
            and isinstance(r["methods"][m][metric], (int, float)) and r["methods"][m][metric] == r["methods"][m][metric]]
    return float(np.mean(vals)) if vals else 0.0


watertight = [r for r in rows if r.get("watertight")]            # signed distance only defined on closed/watertight GT
xm = np.arange(len(present))
fig, ax = plt.subplots(1, 4, figsize=(19, 3.5))

# (1) F-score, closed vs open
groups = [f"closed ({len(closed)})", f"open ({len(opn)})"]
x = np.arange(2); w = 0.8 / len(present)
for k, m in enumerate(present):
    vals = [agg(closed, m, "fscore"), agg(opn, m, "fscore")]
    ax[0].bar(x + (k - (len(present) - 1) / 2) * w, vals, w, label=m, color=COL[m])
ax[0].set_xticks(x); ax[0].set_xticklabels(groups); ax[0].set_ylabel("F-score @ tau=0.05 (%, higher better)")
ax[0].set_title(f"Surface accuracy, {len(rows)} ModelNet40 shapes"); ax[0].set_ylim(0, 105); ax[0].legend(fontsize=8, ncol=3)

# (2) mean signed-distance error (watertight subset) -- the metric our field is trained on
sdf = [agg(watertight, m, "sdf_err") for m in present]
ax[1].bar(xm, sdf, color=[COL[m] for m in present])
ax[1].set_xticks(xm); ax[1].set_xticklabels(present, fontsize=8)
ax[1].set_ylabel(r"mean signed-distance error $\times100$ (lower better)")
ax[1].set_title(f"SDF accuracy, {len(watertight)} watertight shapes")
for i, s in enumerate(sdf):
    ax[1].text(i, s, f"{s:.2f}", ha="center", va="bottom", fontsize=7.5)

# (3) watertightness (component count on open shells)
parts = [agg(opn, m, "parts") for m in present]
ax[2].bar(xm, parts, color=[COL[m] for m in present]); ax[2].set_yscale("log")
ax[2].set_xticks(xm); ax[2].set_xticklabels(present, fontsize=8); ax[2].set_ylabel("# components, open shells (log, lower better)")
ax[2].set_title("Watertightness")
for i, p in enumerate(parts):
    ax[2].text(i, max(p, 1), f"{p:.0f}", ha="center", va="bottom", fontsize=7.5)

# (4) runtime
times = [agg(rows, m, "time") for m in present]
ax[3].bar(xm, times, color=[COL[m] for m in present])
ax[3].set_xticks(xm); ax[3].set_xticklabels(present, fontsize=8); ax[3].set_ylabel("seconds (lower better)")
ax[3].set_title("Reconstruction time")
for i, t in enumerate(times):
    ax[3].text(i, t, f"{t:.1f}s", ha="center", va="bottom", fontsize=7.5)

fig.tight_layout()
fig.savefig("paper/figs/cmp_charts.png", dpi=140); plt.close(fig)
print(f"ModelNet40 sample: {len(rows)} shapes ({len(closed)} closed, {len(opn)} open, {len(watertight)} watertight)")
for m in present:
    print(f"  {m:5s}: closed F {agg(closed,m,'fscore'):5.1f} | open F {agg(opn,m,'fscore'):5.1f} | "
          f"sdf-err {agg(watertight,m,'sdf_err'):.2f} | open parts {agg(opn,m,'parts'):6.0f} | {agg(rows,m,'time'):.2f}s")
print("wrote paper/figs/cmp_charts.png", flush=True)
