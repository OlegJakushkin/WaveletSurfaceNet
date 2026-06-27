"""Quantitative benchmark over a ModelNet40 TEST-set sample (K shapes per category): every method on every
shape, F-score @ tau=0.05 + components + time, plus our gate's thin-fraction to split closed vs open
automatically.  Writes compare/metrics_val.json (consumed by fig_charts.py)."""
import sys, os, glob, json, time; sys.path.insert(0, "compare")
import numpy as np, trimesh
from core import sample_path, run_all_cloud, thin_fraction, normal_consistency, mesh_defects

K = int(os.environ.get("VAL_K", "3"))                          # shapes per category
N_PTS = 4096
cats = sorted(os.path.basename(d) for d in glob.glob("data/ModelNet40/*") if os.path.isdir(d))
shapes = [(c, p) for c in cats for p in sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))[:K]]
print(f"benchmarking {len(shapes)} shapes from {len(cats)} categories (K={K}, {N_PTS} pts)", flush=True)

rows = []
t0 = time.time()
for i, (cat, path) in enumerate(shapes):
    try:
        gt, P, N = sample_path(path, n=N_PTS, noise=0.0, seed=0)
        tf = thin_fraction(P, N)
        res = run_all_cloud(gt, P, N)
        # mean signed-distance error (Task 3): only where signed distance is defined (watertight GT); GT field once
        wt = bool(gt.is_watertight); qg = gd = None
        if wt:
            qg = np.random.default_rng(0).uniform(-1.0, 1.0, (4096, 3))
            try: gd = np.clip(trimesh.proximity.signed_distance(gt, qg), -0.1, 0.1)
            except Exception: gd = None

        def sderr(m):
            if gd is None or res[m][1] is None or not len(res[m][1]): return float("nan")
            try:
                rd = np.clip(trimesh.proximity.signed_distance(trimesh.Trimesh(res[m][0], res[m][1], process=False), qg), -0.1, 0.1)
                return float(np.abs(gd - rd).mean() * 100)
            except Exception: return float("nan")

        def methrec(m):
            d = mesh_defects(res[m][0], res[m][1])
            return {"fscore": res[m][4], "chamfer": res[m][3], "sdf_err": sderr(m), "time": res[m][2],
                    "parts": res[m][5], "faces": (0 if res[m][1] is None else len(res[m][1])),
                    "ncons": normal_consistency(res[m][0], res[m][1], gt),
                    "boundary_edges": d["boundary_edges"], "nonmanifold": d["nonmanifold_edges"],
                    "self_x": d["self_intersections"], "watertight_out": d["watertight"]}

        row = {"cat": cat, "file": os.path.basename(path), "thin_frac": tf, "watertight": wt,
               "kind": ("open" if tf > 0.30 else "closed"),
               "methods": {m: methrec(m) for m in res if m != "GT"}}
        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(shapes)}] {cat:14s} thin {tf:.2f} ({row['kind']}) | "
                  + " ".join(f"{m} F{res[m][4]:.0f}" for m in ("SPSR", "APSS", "RIMLS", "ours") if m in res)
                  + f" | {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  skip {cat}/{os.path.basename(path)}: {e}", flush=True)
json.dump(rows, open("compare/metrics_val.json", "w"), indent=1)
print(f"wrote compare/metrics_val.json ({len(rows)} shapes, {time.time()-t0:.0f}s)", flush=True)
