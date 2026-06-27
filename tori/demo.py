"""Interactive polyscope demo for Points as (Super)Tori.

Usage::

    python demo.py                 # torus fit of a sampled supertoroid
    python demo.py --shape sphere  # sphere | torus | supertoroid | rbox | plane
    python demo.py --supertoroid   # fit supertoroids (with given exponents)
    python demo.py --model path.pt # use a trained CoeffNet checkpoint
    python demo.py --compare       # optimize torus vs supertoroid, print errors

Opens a polyscope window with the input point cloud, the reconstructed zero level
set, an SDF slice, and the fitted tori. Requires a display; the rest of the
library (and the tests) run fully headless.
"""

from __future__ import annotations

import argparse

import numpy as np

from pat import PAT
from pat import shapes as S

SHAPES = {
    "sphere": lambda: S.Sphere(0.7),
    "torus": lambda: S.Torus(0.6, 0.25),
    "supertoroid": lambda: S.SuperToroid(0.6, 0.28, p_tube=4.0, p_ring=2.0),
    "rbox": lambda: S.RoundedBox(half=(0.5, 0.5, 0.5), radius=0.1),
    "plane": lambda: S.Plane(extent=1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="supertoroid", choices=list(SHAPES))
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--supertoroid", action="store_true")
    ap.add_argument("--p-tube", type=float, default=4.0)
    ap.add_argument("--p-ring", type=float, default=2.0)
    ap.add_argument("--model", default=None, help="path to a trained CoeffNet .pt")
    ap.add_argument("--compare", action="store_true",
                    help="optimize torus vs supertoroid and print grid errors")
    ap.add_argument("--res", type=int, default=80)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    shape = SHAPES[args.shape]()
    pts, nrm = shape.sample_surface(args.points, rng)

    if args.compare:
        from pat.optimize import compare_torus_vs_supertoroid
        res = compare_torus_vs_supertoroid(shape, n_points=min(args.points, 400),
                                           steps=200, grid_res=32, verbose=True)
        print(f"\n  torus grid err       : {res['err_torus']:.4f}")
        print(f"  supertoroid grid err : {res['err_supertoroid']:.4f}")
        print(f"  improvement          : {res['improvement'] * 100:+.1f}%")
        print(f"  mean p_tube={res['p_tube_mean']:.2f}  p_ring={res['p_ring_mean']:.2f}")
        pat = res["pat_supertoroid"]
    elif args.model:
        import torch
        from pat.model import CoeffNet
        ckpt = torch.load(args.model, map_location="cpu")
        model = CoeffNet(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        pat = PAT(pts, nrm, model=model)
    else:
        pat = PAT(pts, nrm, supertoroid=args.supertoroid,
                  p_tube=args.p_tube, p_ring=args.p_ring)

    grid = rng.uniform(-1, 1, (5000, 3))
    err = np.mean(np.abs(pat.sdf(grid, neighbors=64) - shape.sdf(grid)))
    print(f"mean abs SDF error over [-1,1]^3: {err:.4f}")

    from pat import viz
    viz.init()
    viz.register_point_cloud("input", pts, nrm)
    viz.register_reconstruction("reconstruction", pat, res=args.res)
    viz.register_sdf_slice("sdf slice (z=0)", pat)
    viz.register_tori("fitted tori", pat, max_tori=300)
    print("opening polyscope window...")
    viz.show()


if __name__ == "__main__":
    main()
