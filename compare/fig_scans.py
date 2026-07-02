"""Qualitative reconstruction from REAL SCANS (no ground truth).  Reads the scan clouds prepared by
compare/scan_recon.py (baselines_ext/scans/<kind>.npz, points+ESTIMATED normals, NO gt mesh), reconstructs each
with our mixed-base model + the public baselines SPSR / BPA / APSS / RIMLS, and renders a grid to
paper/figs/cmp_scans.png:

    rows = scan kinds,  cols = input points | SPSR | BPA | APSS | RIMLS | ours.

Because real scans have NO ground-truth surface, there is NO F-score / Chamfer: evaluation is qualitative
(does the method survive real sensor noise + occlusion?) plus *mesh validity*.  Under each reconstructed panel
we print the connected-component count and self-intersecting-face count from core.mesh_defects -- the two
validity numbers that do not need a GT.  The model checkpoint is taken from MIXED_CKPT exactly as compare/core
reads it (set MIXED_CKPT=assets/waveshape_mixed_v2.pt to render the retrained model).  GPU-friendly but CPU-safe.
Run AFTER compare/scan_recon.py.
"""
import sys, os, glob
sys.path.insert(0, "compare")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import recon_ours, draw3d, mesh_defects, MIXED_CKPT
import baselines as B

SCANS = os.environ.get("SCAN_OUT", "baselines_ext/scans")
OUTPNG = os.environ.get("SCAN_FIG", "paper/figs/cmp_scans.png")
# fixed column order for the figure; baselines actually present are intersected with this.
BASELINE_COLS = ["SPSR", "BPA", "APSS", "RIMLS"]

kinds = sorted(f[:-4] for f in os.listdir(SCANS) if f.endswith(".npz")) if os.path.isdir(SCANS) else []
if not kinds:
    print(f"no scan clouds in {SCANS}/ -- run compare/scan_recon.py first", flush=True)
    sys.exit(0)

avail = B.available()                                            # {name: fn} for whatever libs are installed
baseline_cols = [m for m in BASELINE_COLS if m in avail]
cols = ["input"] + baseline_cols + ["ours"]
print(f"scans: {kinds}", flush=True)
print(f"checkpoint MIXED_CKPT={MIXED_CKPT}", flush=True)
print(f"baselines present: {baseline_cols}", flush=True)


def _defect_label(v, f):
    """One-line validity caption for a reconstructed panel: components + self-intersections (no GT needed)."""
    if v is None or not len(f):
        return "fail"
    d = mesh_defects(v, f)
    sx = d["self_intersections"]
    sx_txt = "?" if sx < 0 else f"{sx}"                          # -1 == pymeshlab unavailable
    return f"{d['components']} parts\\n{sx_txt} self-X"


fig = plt.figure(figsize=(2.05 * len(cols), 2.25 * len(kinds)))
for i, kind in enumerate(kinds):
    d = np.load(f"{SCANS}/{kind}.npz")
    P, N = d["points"].astype(np.float64), d["normals"].astype(np.float64)

    # reconstruct each method on the SAME scan cloud
    recon = {}
    for m in baseline_cols:
        try:
            v, f, _ = avail[m](P, N)
            recon[m] = (np.asarray(v), np.asarray(f))
        except Exception as e:
            print(f"  [{m}] failed on {kind}: {str(e)[:90]}", flush=True)
            recon[m] = (None, None)
    try:
        vo, fo, _ = recon_ours(P, N)
        recon["ours"] = (vo, fo)
    except Exception as e:
        print(f"  [ours] failed on {kind}: {str(e)[:90]}", flush=True)
        recon["ours"] = (None, None)

    for j, m in enumerate(cols):
        ax = fig.add_subplot(len(kinds), len(cols), i * len(cols) + j + 1, projection="3d")
        if m == "input":
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1.4, c="#3a4a66", depthshade=True)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(20, -55)
        else:
            v, f = recon[m]
            draw3d(ax, v, f)
            ax.text2D(0.5, -0.04, _defect_label(v, f), transform=ax.transAxes, ha="center",
                      fontsize=7.5, color=("#207020" if m == "ours" else "#444"))
        if i == 0:
            ax.set_title({"input": "input points", "ours": "ours"}.get(m, m), fontsize=11, weight="bold")
        if j == 0:
            ax.text2D(-0.12, 0.5, kind, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
    print(f"{kind}: " + " | ".join(
        f"{m}=" + _defect_label(*recon[m]).replace("\\n", "/") for m in cols if m != "input"), flush=True)

fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.04, wspace=0.0, hspace=0.16)
os.makedirs(os.path.dirname(OUTPNG), exist_ok=True)
fig.savefig(OUTPNG, dpi=130); plt.close(fig)
print(f"wrote {OUTPNG}", flush=True)
