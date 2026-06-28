"""Re-score ONLY 'ours' on an existing benchmark, reusing the saved baseline results so a model change does NOT
re-run SPSR/BPA/APSS/RIMLS/tori (the slow part).  The baselines are deterministic, so their rows are kept
verbatim; only the 'ours' entry of each shape is recomputed with the new checkpoint.

  MIXED_CKPT=assets/your_new_model.pt SRC=results/local/metrics_val.json python compare/rescore_ours.py

Reuses the same clouds (same path + seed=0), so it stays apples-to-apples with the baselines.  Crash-safe
(checkpoints every 50) and resumable (skips shapes already scored with this exact checkpoint).  ~4-5x faster
than a full re-run, since the 5 baselines + GT sampling dominate the per-shape cost."""
import sys, os, json; sys.path.insert(0, "compare")
import torch
from core import (sample_path, recon_ours, fscore, sdf_error, normal_consistency,
                  mesh_defects, chamfer, ncomp, MIXED_CKPT)

SRC = os.environ.get("SRC", "results/local/metrics_val.json")
DST = os.environ.get("DST", SRC)
rows = json.load(open(SRC))
tag = MIXED_CKPT
print(f"re-scoring ours on {len(rows)} shapes with {tag} (baselines kept)", flush=True)

n_done = 0
for i, r in enumerate(rows):
    if r["methods"].get("ours", {}).get("model") == tag:     # resume: already scored with this model
        n_done += 1; continue
    path = f"data/ModelNet40/{r['cat']}/test/{r['file']}"
    try:
        gt, P, N = sample_path(path, n=4096, seed=0)
        v, f, t = recon_ours(P, N)
        d = mesh_defects(v, f)
        r["methods"]["ours"] = {
            "fscore": fscore(v, f, gt), "chamfer": chamfer(v, f, gt), "sdf_err": sdf_error(v, f, gt),
            "time": t, "parts": ncomp(v, f), "faces": (0 if f is None else len(f)),
            "ncons": normal_consistency(v, f, gt),
            "boundary_edges": d["boundary_edges"], "nonmanifold": d["nonmanifold_edges"],
            "self_x": d["self_intersections"], "watertight_out": d["watertight"], "model": tag}
        n_done += 1
        if n_done % 50 == 0:
            json.dump(rows, open(DST, "w"), indent=1)
            print(f"  {n_done}/{len(rows)} re-scored", flush=True)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  skip {r['cat']}/{r['file']}: {e}", flush=True)
json.dump(rows, open(DST, "w"), indent=1)
print(f"wrote {DST} ({n_done} shapes re-scored with ours={tag}; baselines untouched)", flush=True)
