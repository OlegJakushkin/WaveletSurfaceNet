"""CLI: reconstruct a mesh (OBJ) from a shape's point cloud with PAT.

    python -m pat.meshgen --shape torus --model assets/pat_supertoroid.pt --out out.obj

Used by the Docker ``meshgen`` service. Without ``--model`` it uses the
training-free least-squares tori.
"""

from __future__ import annotations

import argparse

import numpy as np


def _shapes():
    from pat import shapes as S
    from pat.assets import BoltPlate, BoxWithCylinders, Buckyball, Cube, TexturedCylinder
    from pat.bunny import bunny_shape
    return {
        "sphere": lambda: S.Sphere(0.7),
        "torus": lambda: S.Torus(0.6, 0.24),
        "supertoroid": lambda: S.SuperToroid(0.6, 0.28, p_tube=4.0),
        "cube": lambda: Cube(),
        "buckyball": lambda: Buckyball(),
        "composite": lambda: BoxWithCylinders(),
        "textured": lambda: TexturedCylinder(),
        "bolts": lambda: BoltPlate(),
        "bunny": bunny_shape,
    }


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces + 1:                       # OBJ is 1-indexed
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="torus", choices=list(_shapes()))
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--model", default=None, help="trained CoeffNet checkpoint")
    ap.add_argument("--supertoroid", action="store_true",
                    help="fixed supertoroid (when no model); p-tube/p-ring below")
    ap.add_argument("--p-tube", type=float, default=4.0)
    ap.add_argument("--out", default="mesh.obj")
    args = ap.parse_args()

    from pat import PAT
    shape = _shapes()[args.shape]()
    rng = np.random.default_rng(0)
    pts, nrm = shape.sample_surface(args.points, rng)

    if args.model:
        import torch
        from pat.model import CoeffNet
        ck = torch.load(args.model, map_location="cpu", weights_only=False)
        model = CoeffNet(**ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()
        pat = PAT(pts, nrm, model=model, k=16, C=16)
    else:
        pat = PAT(pts, nrm, supertoroid=args.supertoroid, p_tube=args.p_tube, C=16)

    verts, faces = pat.reconstruct(res=args.res, bound=1.2, neighbors=64)
    save_obj(args.out, verts, faces)
    print(f"wrote {args.out}: {len(verts)} verts, {len(faces)} faces")


if __name__ == "__main__":
    main()
