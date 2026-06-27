#!/usr/bin/env python
"""WaveletSurfaceNet -- generate a mesh from a point cloud (the unified mixed-base model).

    points / mesh  (+ optional region box for super-resolution)   -->   watertight / shell mesh

One model, one forward, both bases: closed regions come back as crisp SOLIDS (signed path), thin/open
regions as clean SHELLS (unsigned path), meshed at the single level 0.  Runs on GPU (CUDA).

Examples (see README):
    python generate.py --shape bunny  --out out/bunny.obj
    python generate.py --shape teapot --out out/teapot.obj
    python generate.py --shape chair  --out out/chair.obj
    python generate.py --shape knurl  --region   --out out/knurl_region.obj    # one box, in the full pass
    python generate.py --shape knurl  --superres --out out/knurl_superres.obj   # same box, box-normalised
    python generate.py --points my_cloud.npy --out out/mine.obj                 # your own (N,6) xyz+normal cloud
"""
import argparse, os
import numpy as np, torch, trimesh
from skimage import measure
from waveshape import wavelet as WV, eval3d as E, shapes as S

BOUND, TRUNC = 1.1, 0.1


# ----------------------------------------------------------------- built-in example shapes
def _teapot():
    tp = trimesh.load("assets/teapot.obj", force="mesh")
    tp.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))  # stand upright
    return S.normalize_to_unit_cube(tp)


def builtin_mesh(name):
    if name == "cube":   return S.normalize_to_unit_cube(trimesh.creation.box(extents=[1, 1, 1]))
    if name == "sphere": return S.normalize_to_unit_cube(trimesh.creation.uv_sphere(radius=0.7, count=[48, 48]))
    if name == "torus":  return S.normalize_to_unit_cube(trimesh.creation.torus(major_radius=0.5, minor_radius=0.2))
    if name == "knurl":  return S.normalize_to_unit_cube(E._knurl_mesh())          # diamond-textured cylinder
    if name == "teapot": return _teapot()
    if name == "bunny":
        from waveshape.bunny import load_bunny
        return S.normalize_to_unit_cube(load_bunny(normalize=True))
    if name == "chair":                                                            # one sample ModelNet shell
        m = trimesh.load("assets/chair.off", force="mesh"); m.fix_normals()
        return S.normalize_to_unit_cube(m)
    raise ValueError(f"unknown --shape {name!r}")


def cloud_from_mesh(m, n, seed, fix=False):
    if fix:
        try: m.fix_normals()
        except Exception: pass
    sc = 1.0 / max(np.abs(m.vertices).max(), 1e-6)            # scale so the cloud fills [-1,1] (training frame)
    P, N = E.sample_cloud(m, n=n, noise=0.0, seed=seed)
    return (P * sc).astype(np.float32), N.astype(np.float32)


# ----------------------------------------------------------------- meshing
def _mc0(grid):
    g = WV._smooth_grid(grid, 0.5)
    if not (g.min() < 0 < g.max()):
        return None, None
    v, f, _, _ = measure.marching_cubes(g.astype(np.float64), 0.0)
    return v / (g.shape[0] - 1) * (2 * BOUND) - BOUND, f      # -> [-bound, bound]


def _keep_big(v, f, frac=0.05):
    """drop small disconnected components (box-edge shards) -> the clean surface only."""
    if v is None or not len(f):
        return v, f
    comps = trimesh.Trimesh(v, f, process=False).split(only_watertight=False)
    if len(comps) <= 1:
        return v, f
    big = max(len(c.faces) for c in comps)
    comps = [c for c in comps if len(c.faces) >= max(40, frac * big)]
    out = trimesh.util.concatenate(comps)
    return np.asarray(out.vertices), np.asarray(out.faces)


def generate(net, P, N, *, region=False, dev="cuda"):
    """Full-shape pass, or a box-normalised region pass (super-resolution).  Returns (verts, faces)."""
    Pt = torch.tensor(P[None]).float().to(dev); Nt = torch.tensor(N[None]).float().to(dev)
    if not region:
        with torch.no_grad():
            g = net(Pt, Nt)[0][0, 0].cpu().numpy() * TRUNC
        return _mc0(g)
    # region / super-resolution: a box on the camera-facing surface, normalised to fill the lattice,
    # the whole shape supplied as zoomed-out global context.
    cam = np.array([1., 1., 1.]) / np.sqrt(3)
    center = P[int(np.argmax(N @ cam))].astype(np.float64); half = 0.45
    ct = torch.tensor(center[None, None]).float().to(dev)
    with torch.no_grad():
        g = net(Pt, Nt, ctx_P=Pt, ctx_N=Nt, center=ct, half=float(half))[0][0, 0].cpu().numpy() * TRUNC
    v, f = _mc0(g)                                            # box-local frame
    if v is None:
        return None, None
    cen = v[f].mean(1); keep = np.all(np.abs(cen) < BOUND * 0.9, 1)   # trim the box-edge skirt
    f = f[keep]
    if len(f):
        used = np.unique(f); remap = {o: n for n, o in enumerate(used)}
        v, f = v[used], np.vectorize(remap.get)(f)
    return _keep_big(v, f)


# ----------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="WaveletSurfaceNet: points/mesh -> mesh (unified mixed-base model).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shape", choices=["cube", "sphere", "torus", "bunny", "teapot", "chair", "knurl"],
                     help="a built-in example shape")
    src.add_argument("--mesh", help="path to an input mesh (.obj/.off/.ply/...) to sample a cloud from")
    src.add_argument("--points", help="path to a .npy of shape (N,6): xyz + outward unit normal per point")
    ap.add_argument("--out", required=True, help="output mesh path (.obj/.ply/.stl)")
    ap.add_argument("--region", action="store_true", help="reconstruct ONE box region (in the full pass)")
    ap.add_argument("--superres", action="store_true", help="super-resolve that box (box-normalised + context)")
    ap.add_argument("--ckpt", default="assets/waveshape_mixed.pt", help="model checkpoint")
    ap.add_argument("--res", type=int, default=64, help="marching-cubes output lattice (R^3); 48 for region demos")
    ap.add_argument("--n", type=int, default=8000, help="points to sample from a mesh")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required (run via Docker with --gpus all)"
    dev = "cuda"
    res = a.res if not (a.region or a.superres) else min(a.res, 48)
    ck = torch.load(a.ckpt, weights_only=False)
    net = WV.load_at_res(ck, res=res, bound=BOUND).to(dev).eval()
    print(f"[model] {a.ckpt}  field_mode={ck.get('field_mode')}  ep{ck.get('epoch')}  @ res{res}", flush=True)

    if a.points:
        arr = np.load(a.points).astype(np.float32)
        assert arr.ndim == 2 and arr.shape[1] == 6, "--points must be (N,6): xyz + normal"
        P, Nn = arr[:, :3], arr[:, 3:]
        P = (P / max(np.abs(P).max(), 1e-6)).astype(np.float32)        # scale into the [-1,1] training frame
        src_name = os.path.basename(a.points)
    else:
        m = builtin_mesh(a.shape) if a.shape else trimesh.load(a.mesh, force="mesh")
        if a.mesh:
            m = S.normalize_to_unit_cube(m)
        P, Nn = cloud_from_mesh(m, a.n, a.seed, fix=bool(a.mesh))
        src_name = a.shape or os.path.basename(a.mesh)

    v, f = generate(net, P, Nn, region=(a.region or a.superres), dev=dev)
    if v is None or not len(f):
        raise SystemExit("no surface produced (the field did not cross 0)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    trimesh.Trimesh(v, f, process=False).export(a.out)
    mode = "super-res region" if a.superres else ("region" if a.region else "full")
    print(f"[done] {src_name} ({mode}) -> {a.out}   {len(f):,} faces", flush=True)


if __name__ == "__main__":
    main()
