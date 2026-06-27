"""Before/after sanity check for the v2 retrain: same shapes through the old and new mixed checkpoints,
reporting the things v2 was meant to fix -- component count on open shells (watertightness), F-score under
3% noise (noise robustness), F-score at 256 pts (sparsity), and signed-distance error on closed solids."""
import sys, os; sys.path.insert(0, "compare")
import core

OPEN = ["chair", "guitar", "table"]
CLOSED = ["cube", "teapot", "bunny"]


def run(ckpt):
    core._mixed = None; core.MIXED_CKPT = ckpt
    out = {}
    for s in OPEN + CLOSED:
        gt, P, N = core.sample(s, n=4096)
        v, f, _ = core.recon_ours(P, N)
        gtn, Pn, Nn = core.sample(s, n=4096, noise=0.03)
        vn, fn, _ = core.recon_ours(Pn, Nn)
        gts, Ps, Ns = core.sample(s, n=256)
        vs, fs, _ = core.recon_ours(Ps, Ns)
        out[s] = {"parts": core.ncomp(v, f), "F": core.fscore(v, f, gt),
                  "Fn": core.fscore(vn, fn, gtn), "Fs": core.fscore(vs, fs, gts),
                  "sdf": core.sdf_error(v, f, gt)}
    return out


a = run("assets/waveshape_mixed.pt")
b = run("assets/waveshape_mixed_v2.pt")
print(f"{'shape':8s} | {'parts v1->v2':14s} | {'F clean':12s} | {'F @3% noise':13s} | {'F @256pts':12s} | {'sdf-err':12s}")
for s in OPEN + CLOSED:
    print(f"{s:8s} | {a[s]['parts']:4d} -> {b[s]['parts']:<4d}   | {a[s]['F']:4.0f} -> {b[s]['F']:<4.0f} | "
          f"{a[s]['Fn']:4.0f} -> {b[s]['Fn']:<4.0f}  | {a[s]['Fs']:4.0f} -> {b[s]['Fs']:<4.0f} | "
          f"{a[s]['sdf']:5.2f} -> {b[s]['sdf']:<5.2f}", flush=True)
