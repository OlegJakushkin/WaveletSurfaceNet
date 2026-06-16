"""Render paper-style torus-vs-supertoroid comparisons for every asset.

Produces, in ``renders/``:
  * buckyball.png      -- the C60 / truncated-icosahedron lattice (the paper's hero shape), 1024 pts
  * cube.png           -- a sharp cube, 1024 pts
  * bunny.png          -- the Stanford bunny (complex traditional asset), 1024 pts
  * composite_noise.png-- box + cylinder boss + bored cylinder, sampled WITH noise

Each figure has ground truth, "torus (based on Feng 26)" and "supertoroid (ours)", in the
paper's layout.
If a trained checkpoint is given (``--model``), the per-point coefficients (and the
supertoroid's squareness) come from the network; otherwise they are optimized per cloud.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from pat import PAT
from pat.assets import BoltPlate, Buckyball, BoxWithCylinders, Cube, TexturedCylinder
from pat.bunny import bunny_shape
from pat.shapes import Torus
from pat.datasets import noisy_point_cloud
from pat.model import CoeffNet
from pat.optimize import fit_pair
from pat.render import render_comparison


def fit_models(shape, points, normals, *, model_t=None, model_s=None, model=None,
               C=16, steps=120, k=16, fast=False):
    """Return ``{label: PAT}`` for torus and supertoroid fit to the same cloud.

    * ``model_t`` / ``model_s``: the dedicated trained torus and supertoroid
      networks (best; the two models the GPU trainer produces).
    * ``model``: a single supertoroid network used for both columns (torus = same
      coeffs with squareness ignored).
    * ``fast``: least-squares torus + squareness-only optimization (quick baseline).
    * otherwise: full per-cloud optimization of both models.
    """
    if model_t is not None or model_s is not None:
        pat_t = PAT(points, normals, model=model_t, k=k, C=C)          # plain torus net
        pat_s = PAT(points, normals, model=model_s, k=k, C=C)          # supertoroid net
    elif model is not None:
        full = PAT(points, normals, model=model, k=k)          # learned coeffs + squareness
        coeffs = full.coeffs.numpy()
        pat_t = PAT(points, normals, coeffs=coeffs, C=C)
        pat_s = PAT(points, normals, coeffs=coeffs, supertoroid=True,
                    p_tube=full.p_tube.numpy(), p_ring=full.p_ring.numpy(), C=C)
    elif fast:
        from pat.optimize import optimize_cloud
        from pat.lstsq import fit_coeffs_lstsq
        coeffs = fit_coeffs_lstsq(points, normals, k=24).numpy()
        ps, _ = optimize_cloud(points, normals, shape, supertoroid=True, steps=70,
                               n_query=800, warm_coeffs=coeffs, freeze_coeffs=True)
        pat_t = PAT(points, normals, coeffs=coeffs, C=C)
        pat_s = PAT(points, normals, coeffs=coeffs, supertoroid=True,
                    p_tube=ps.p_tube.numpy(), p_ring=ps.p_ring.numpy(), C=C)
    else:
        pt, ps = fit_pair(points, normals, shape, steps=steps, n_query=1000, k=24)
        pat_t = PAT(points, normals, coeffs=pt.coeffs.numpy(), C=C)
        pat_s = PAT(points, normals, coeffs=ps.coeffs.numpy(), supertoroid=True,
                    p_tube=ps.p_tube.numpy(), p_ring=ps.p_ring.numpy(), C=C)
    return {"torus (based on Feng 26)": pat_t, "supertoroid (ours)": pat_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="single supertoroid checkpoint for both columns")
    ap.add_argument("--model-torus", default="assets/pat_torus.pt",
                    help="trained plain-torus checkpoint (used if it exists)")
    ap.add_argument("--model-supertoroid", default="assets/pat_supertoroid.pt",
                    help="trained supertoroid checkpoint (used if it exists)")
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--noise", type=float, default=0.015, help="noise std for the composite")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--scale", type=float, default=2.0,
                    help="output resolution multiplier (2.0 => 4x the pixels)")
    ap.add_argument("--fast", action="store_true",
                    help="least-squares torus + squareness-only supertoroid (quick)")
    ap.add_argument("--only", default=None, help="render only this asset name")
    args = ap.parse_args()
    dpi = int(130 * args.scale)                  # 2.0 -> 260 dpi (4x pixels)
    slice_res = int(220 * args.scale)

    def _load(path):
        if not path or not os.path.exists(path):
            return None
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = CoeffNet(**ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
        print(f"loaded {path}  (val-torus-err {ck.get('val_torus_err', '?')})")
        return m

    model = _load(args.model) if args.model else None
    model_t = None if model else _load(args.model_torus)
    model_s = None if model else _load(args.model_supertoroid)

    rng = np.random.default_rng(0)
    assets = {
        "torus": (Torus(0.6, 0.24), dict(smooth=12, C=16)),
        "buckyball": (Buckyball(), dict(smooth=8, C=14)),
        "cube": (Cube(), dict(smooth=10, C=18)),
        "bunny": (bunny_shape(), dict(smooth=14, C=16, view=(18, 210), slice_axis=0)),
        "composite_noise": (BoxWithCylinders(), dict(smooth=12, C=18, noisy=True)),
        "bolts": (BoltPlate(), dict(smooth=10, C=18)),
        "textured": (TexturedCylinder(), dict(smooth=4, C=22)),
    }
    for name, (shape, opt) in assets.items():
        if args.only and name != args.only:
            continue
        print(f"=== {name} ===", flush=True)
        if opt.get("noisy"):
            # the composite is sampled WITH noise (the paper's noisy-cloud figure)
            pts, nrm = shape.sample_surface(args.points, rng)
            pts = pts + rng.normal(scale=args.noise, size=pts.shape)
            label = f"{args.points}  (noise σ={args.noise})"
        else:
            pts, nrm = shape.sample_surface(args.points, rng)
            label = str(args.points)
        pats = fit_models(shape, pts, nrm, model=model, model_t=model_t,
                          model_s=model_s, C=opt["C"], fast=args.fast)
        render_comparison(shape, pats, pts, f"renders/{name}.png", recon_res=args.res,
                          neighbors=96, npoints_label=label, smooth_iters=opt["smooth"],
                          view=opt.get("view", (22, -62)), slice_axis=opt.get("slice_axis", 2),
                          slice_res=slice_res, dpi=dpi)
        print(f"  saved renders/{name}.png", flush=True)


if __name__ == "__main__":
    main()
