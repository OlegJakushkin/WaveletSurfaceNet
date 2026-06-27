"""Gate-sensitivity analysis (reviewer: no gate-accuracy eval, no threshold sensitivity).  Sweeps the analytic
open/closed gate (waveshape.wavelet.point_thinness) over five nuisance factors and measures how well the
shape-level thin-fraction predicts the GT closed/open label (gt.is_watertight).  CPU-only -- safe to run while
the GPU trains.  Axes:  point density n;  normal noise;  point_thinness 'thin' (the near-distance band, the
reviewer's 'epsilon'); 'opp' (the opposing-normal cosine margin); and the shape-level decision threshold delta
(currently hard-coded 0.30 in bench_val) swept as an ROC operating point.  Writes compare/gate_sweep.json."""
import sys, os, glob, json; sys.path.insert(0, "compare")
import numpy as np, torch
from core import sample_path
from waveshape import wavelet as WV

CLOSED = ["bottle", "car", "sofa", "vase"]            # GT-watertight categories -> label 0 (closed)
OPEN = ["chair", "table", "bench", "lamp"]            # open-shell categories     -> label 1 (open)
PER = int(os.environ.get("GATE_PER", "6"))
files = [(c, 0, p) for c in CLOSED for p in sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))[:PER]] \
      + [(c, 1, p) for c in OPEN   for p in sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))[:PER]]
print(f"{len(files)} GT-labelled shapes ({len(CLOSED)} closed + {len(OPEN)} open categories)", flush=True)


def thinfrac(P, N, thin, opp, k=24):
    t = WV.point_thinness(torch.tensor(P[None]).float(), torch.tensor(N[None]).float(), thin=thin, opp=opp, k=k)
    return float(t.mean())


def perturb_normals(N, sigma, seed=0):
    if sigma <= 0: return N
    Nn = N + np.random.default_rng(seed).normal(scale=sigma, size=N.shape)
    return Nn / np.clip(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9, None)


rows = []
for cat, gt_open, path in files:
    for n_pts in (256, 512, 1024, 2048, 4096):
        try:
            gt, P, N = sample_path(path, n=n_pts, noise=0.0, seed=0)
        except Exception:
            continue
        for nn_sig in (0.0, 0.05, 0.1, 0.2):
            Nn = perturb_normals(N, nn_sig)
            for thin in (0.06, 0.08, 0.10, 0.14):
                for opp in (-0.1, -0.3, -0.5):
                    rows.append(dict(cat=cat, gt_open=gt_open, n=n_pts, nnoise=nn_sig, thin=thin, opp=opp,
                                     thin_frac=thinfrac(P, Nn, thin, opp), watertight=bool(gt.is_watertight)))
    print(f"  {cat}/{os.path.basename(path)} done", flush=True)


def acc_at_delta(sub, delta):
    y = np.array([r["gt_open"] for r in sub]); yh = np.array([1 if r["thin_frac"] > delta else 0 for r in sub])
    return float((y == yh).mean()) if len(sub) else float("nan")


def roc_auc(sub):
    if not sub: return float("nan")
    y = np.array([r["gt_open"] for r in sub]); s = np.array([r["thin_frac"] for r in sub])
    o = np.argsort(-s); y = y[o]; P_, Nn_ = y.sum(), (1 - y).sum()
    return float(np.trapz(np.cumsum(y) / max(P_, 1), np.cumsum(1 - y) / max(Nn_, 1)))


base = [r for r in rows if r["thin"] == 0.10 and r["opp"] == -0.3]    # point_thinness defaults
print("\n-- gate accuracy@delta=0.30 / ROC-AUC vs point DENSITY (clean normals) --")
for n_pts in (256, 512, 1024, 2048, 4096):
    sub = [r for r in base if r["n"] == n_pts and r["nnoise"] == 0.0]
    print(f"  {n_pts:5d} pts:  acc {acc_at_delta(sub,0.30):.2f}   AUC {roc_auc(sub):.2f}")
print("-- vs NORMAL NOISE (2048 pts) --")
for nn in (0.0, 0.05, 0.1, 0.2):
    sub = [r for r in base if r["nnoise"] == nn and r["n"] == 2048]
    print(f"  sigma {nn:.2f}:  acc {acc_at_delta(sub,0.30):.2f}   AUC {roc_auc(sub):.2f}")
clean = [r for r in base if r["nnoise"] == 0.0 and r["n"] == 2048]
deltas = np.linspace(0.05, 0.6, 23)
best = max(deltas, key=lambda d: acc_at_delta(clean, d))
print(f"\nbest delta {best:.2f} (acc {acc_at_delta(clean,best):.2f}) vs hard-coded 0.30 (acc {acc_at_delta(clean,0.30):.2f})")
json.dump(rows, open("compare/gate_sweep.json", "w"), indent=1)
print(f"wrote compare/gate_sweep.json ({len(rows)} points)", flush=True)
