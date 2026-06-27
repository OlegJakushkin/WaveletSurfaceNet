"""Proper, VOXEL-FREE evaluation of the supertoroid-splat reconstruction + canonical test shapes.

The teacher's training GT is a voxel occupancy grid (``gt_occupancy``) -- fine as a fitting target, but
comparing against the *rendered* voxel solid would bake in voxelization error.  Here every metric is
**Monte-Carlo and continuous on BOTH sides**: ground-truth occupancy from an EXACT analytic SDF (for the
CSG shapes) or trimesh's winding-number ``mesh.contains`` (the "library sample" reference for real meshes),
versus our reconstruction's continuous ``splat.sdf < 0`` -- no grid anywhere.

Canonical shapes (``canonical_shapes``): teapot, bunny, hole+bolts plate, cube+cylinder (noisy sampling),
diamond-knurl (sharp corners + texture).  ``proper_metrics`` returns IoU / volume error / precision /
recall; ``mesh_properties`` returns #faces, volume, area, watertight, genus, compactness, thinness;
``plot_metrics_matrix`` plots quality metrics against those properties.
"""

from __future__ import annotations

import numpy as np

try:
    import trimesh
except Exception:                                   # pragma: no cover
    trimesh = None

from . import shapes as _shapes
from .splat import _mc


# --------------------------------------------------------------------------- #
#  Analytic SDF primitives (exact ground truth -- no voxels)
# --------------------------------------------------------------------------- #
def _sd_box(q, b):
    d = np.abs(q) - b
    return np.linalg.norm(np.clip(d, 0, None), axis=1) + np.clip(d.max(1), None, 0)


def _sd_cyl(q, r, h, axis=2):
    o = [a for a in range(3) if a != axis]
    rad = np.linalg.norm(q[:, o], axis=1) - r
    d = np.stack([rad, np.abs(q[:, axis]) - h], 1)
    return np.linalg.norm(np.clip(d, 0, None), axis=1) + np.clip(d.max(1), None, 0)


class _SDFShape:
    """Wrap an exact SDF callable as a shape with ``.sdf(q)`` and ``.contains(q)``."""
    def __init__(self, fn, noise=0.0):
        self.fn = fn; self.noise = noise
    def sdf(self, q):
        return self.fn(np.asarray(q, np.float64))
    def contains(self, q):
        return self.sdf(q) < 0


def _cube_cylinder():                                # union of a cube and a crossing cylinder
    def fn(q):
        return np.minimum(_sd_box(q, np.array([0.42, 0.42, 0.42])), _sd_cyl(q, 0.26, 0.7, axis=1))
    return _SDFShape(fn, noise=0.02)                 # "sampled with noise"


def _plate_with_holes():                             # flat plate MINUS a central hole + 4 bolt holes
    def fn(q):
        plate = _sd_box(q, np.array([0.72, 0.72, 0.11]))
        holes = -_sd_cyl(q, 0.26, 1.0, axis=2)
        for cx, cy in [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]:
            holes = np.maximum(holes, -_sd_cyl(q - np.array([cx, cy, 0.0]), 0.09, 1.0, axis=2))
        return np.maximum(plate, holes)
    return _SDFShape(fn)


# --------------------------------------------------------------------------- #
#  Mesh test shapes (library-sampled ground truth via trimesh.contains)
# --------------------------------------------------------------------------- #
def _teapot_mesh():
    parts = []
    body = trimesh.creation.uv_sphere(radius=0.6, count=[48, 48]); body.apply_scale([1, 1, 0.75]); parts.append(body)
    spout = trimesh.creation.cylinder(radius=0.07, height=0.55, sections=24)
    spout.apply_transform(trimesh.transformations.rotation_matrix(np.radians(60), [0, 1, 0]))
    spout.apply_translation([0.5, 0, 0.05]); parts.append(spout)
    handle = trimesh.creation.torus(major_radius=0.22, minor_radius=0.05)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
    handle.apply_translation([-0.6, 0, 0.05]); parts.append(handle)
    lid = trimesh.creation.cone(radius=0.18, height=0.18, sections=24); lid.apply_translation([0, 0, 0.45]); parts.append(lid)
    return trimesh.util.concatenate(parts)


def _knurl_mesh(R=0.42, H=1.4, nt=180, nz=90, amp=0.05, k=16, pitch=6.0):
    """Diamond-knurled cylinder: a cylinder whose radius is modulated by two crossed triangle waves
    (sharp diamond ridges) -- the 'sharp corners & textures' stress shape."""
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    z = np.linspace(-H / 2, H / 2, nz)
    TH, Z = np.meshgrid(th, z)

    def tri(x):                                      # triangle wave in [0,1]
        u = x / (2 * np.pi); return 2 * np.abs(u - np.floor(u + 0.5))
    patt = tri(k * TH + pitch * 2 * np.pi * Z) + tri(k * TH - pitch * 2 * np.pi * Z)
    r = R + amp * (patt - 0.5)
    X = r * np.cos(TH); Y = r * np.sin(TH)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    F = []
    for i in range(nz - 1):
        for j in range(nt):
            a0 = i * nt + j; a1 = i * nt + (j + 1) % nt; b0 = (i + 1) * nt + j; b1 = (i + 1) * nt + (j + 1) % nt
            F += [[a0, a1, b1], [a0, b1, b0]]
    cb = len(V); ct = len(V) + 1                     # bottom / top cap centers
    V = np.vstack([V, [0, 0, -H / 2], [0, 0, H / 2]])
    for j in range(nt):
        F.append([cb, (j + 1) % nt, j])
        F.append([ct, (nz - 1) * nt + j, (nz - 1) * nt + (j + 1) % nt])
    m = trimesh.Trimesh(V, np.array(F), process=True)
    return m


class _MeshShapeGT:
    """A trimesh wrapped for evaluation.  Occupancy from :class:`pat.bunny.MeshShape` -- a kNN-pruned
    signed distance (trimesh.proximity / winding-number / pseudo-normal), the project's robust,
    memory-bounded, library-grade reference (trimesh.contains needs rtree and OOMs the raw GWN on a 69k-
    face mesh)."""
    def __init__(self, mesh, noise=0.0):
        from .bunny import MeshShape
        self.mesh = _shapes.normalize_to_unit_cube(mesh)
        # "normal" = kNN-pruned closest-surface-point pseudonormal sign: fast + memory-bounded (the
        # "trimesh"/"winding" methods are O(Q x faces) and stall on a 69k-face mesh).
        self._ms = MeshShape(self.mesh, sign_method="normal")
        self.noise = noise
    def sdf(self, q):
        return self._ms.sdf(np.asarray(q, np.float64))
    def contains(self, q):
        return self.sdf(q) < 0


def mesh_gt(mesh):
    """Wrap a trimesh as an evaluation ground truth with voxel-free occupancy (``.contains`` / ``.sdf``)."""
    return _MeshShapeGT(mesh)


def canonical_shapes(bunny=True):
    """The named test shapes -> list of ``(name, gt, mesh)`` where ``gt`` exposes ``.contains(q)`` (exact)
    and ``mesh`` is a trimesh for rendering/properties (CSG shapes are marching-cubed for the mesh)."""
    out = [("cube+cylinder", _cube_cylinder(), None),
           ("hole+bolts plate", _plate_with_holes(), None),
           ("teapot", _MeshShapeGT(_teapot_mesh()), None),
           ("diamond knurl", _MeshShapeGT(_knurl_mesh()), None)]
    if bunny:
        from .bunny import load_bunny
        out.insert(1, ("bunny", _MeshShapeGT(load_bunny(normalize=True)), None))
    res = []
    for name, gt, _ in out:
        mesh = gt.mesh if isinstance(gt, _MeshShapeGT) else _sdf_mesh(gt)
        res.append((name, gt, mesh))
    return res


def _sdf_mesh(gt, res=160):
    v, f = _mc(gt.sdf, res)
    return trimesh.Trimesh(v, f, process=False) if v is not None else None


def val_shapes(bunny=True):
    """Validation set: the named canonical shapes + procedural primitives spanning the property space
    (watertight/open, low/high genus, thin/bulky, few/many faces) -> enough points for the matrix plot."""
    prims = [
        ("sphere", trimesh.creation.uv_sphere(radius=0.7, count=[40, 40])),
        ("box", trimesh.creation.box(extents=[1.0, 0.8, 0.6])),
        ("cone", trimesh.creation.cone(radius=0.6, height=1.2, sections=40)),
        ("cylinder", trimesh.creation.cylinder(radius=0.45, height=1.2, sections=48)),
        ("torus", trimesh.creation.torus(major_radius=0.55, minor_radius=0.2)),
        ("icosphere", trimesh.creation.icosphere(subdivisions=3, radius=0.7)),
        ("thin plate", trimesh.creation.box(extents=[1.4, 1.4, 0.07])),
        ("capsule", trimesh.creation.capsule(radius=0.3, height=0.7)),
        ("annulus", trimesh.creation.annulus(r_min=0.25, r_max=0.6, height=0.4)),
    ]
    out = []
    for n, m in prims:
        gt = _MeshShapeGT(m)
        out.append((n, gt, gt.mesh))
    return canonical_shapes(bunny=bunny) + out


def sample_cloud(mesh, n=1536, noise=0.0, seed=0):
    """Surface cloud + normals from a trimesh (the teacher's input), optionally noised."""
    rng = np.random.default_rng(seed)
    P, N = _shapes.sample_mesh(mesh, n, rng)
    P = P + rng.normal(scale=noise, size=P.shape) if noise else P
    N = N / np.clip(np.linalg.norm(N, axis=1, keepdims=True), 1e-9, None)
    return P.astype(np.float32), N.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Voxel-free Monte-Carlo metrics + mesh properties
# --------------------------------------------------------------------------- #
def proper_metrics(gt, splat, *, n=120_000, bound=1.0, seed=0, chunk=40_000):
    """Continuous IoU / volume on BOTH sides: GT occupancy from ``gt.contains`` (exact / library), pred
    from ``splat.sdf < 0`` -- sampled Monte-Carlo, NO voxel grid.  Returns a metrics dict.

    Includes ``md`` -- the **Minkowski filled-volume distance** ``vol(A xor B)/vol(cube)`` that the
    supertoroid-splat teacher minimizes (:func:`pat.teacher.md_filled_volume`), i.e. the fraction of the
    cube where the reconstructed solid and the GT solid disagree (lower is better; 0 = identical solids).
    Here it is the **voxel-free** Monte-Carlo estimate (continuous on both sides), so any model exposing
    ``.sdf`` -- splats, tori, or the wavelet denoiser -- is judged by the same filled-volume loss."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-bound, bound, (n, 3)).astype(np.float32)
    g = np.zeros(n, bool); pr = np.zeros(n, bool)
    for a in range(0, n, chunk):
        s = slice(a, a + chunk)
        g[s] = np.asarray(gt.contains(pts[s]))
        pr[s] = np.asarray(splat.sdf(pts[s])) < 0
    inter = int((g & pr).sum()); union = int((g | pr).sum()); cube = (2 * bound) ** 3
    return dict(iou=inter / max(union, 1),
                md=(union - inter) / n,                  # vol(A xor B)/vol(cube) -- the splat-teacher MD loss
                vol_gt=g.mean() * cube, vol_pred=pr.mean() * cube,
                vol_err=abs(int(pr.sum()) - int(g.sum())) / n * cube,
                precision=inter / max(int(pr.sum()), 1), recall=inter / max(int(g.sum()), 1))


def gallery_render(name, mesh, splat, path, *, res=128, iou=None):
    """Render the GT mesh DIRECTLY (exact -- no voxelization) beside the splat reconstruction (marching-
    cubes of the continuous ``splat.sdf``).  Returns ``path``."""
    from . import render3d as R3
    rec = _mc(splat.sdf, res)
    panels = [("ground truth", mesh.vertices, mesh.faces),
              (f"supertoroid splats ({getattr(splat, 'M', '')} splats)",
               rec[0] if rec else None, rec[1] if rec else None)]
    title = name + (f"  |  IoU* {iou:.2f}" if iou is not None else "")
    return R3.render_meshes(panels, path, title=title)


def mesh_properties(mesh):
    """Geometric descriptors used as the x-axes of the property matrix."""
    wt = bool(mesh.is_watertight)
    vol = abs(float(mesh.volume)) if wt else float("nan")
    area = float(mesh.area)
    return dict(faces=len(mesh.faces), verts=len(mesh.vertices), area=area,
                volume=vol, watertight=int(wt),
                genus=float((2 - mesh.euler_number) / 2) if wt else float("nan"),
                compactness=float(area ** 1.5 / vol) if wt and vol > 0 else float("nan"),
                thinness=float(area / (mesh.bounding_box.volume + 1e-9)))   # area-to-bbox: thin/sheety shapes high


def plot_metrics_matrix(records, path, metrics=("iou", "vol_err"),
                        props=("faces", "area", "watertight", "thinness", "compactness")):
    """Scatter every quality ``metric`` (rows) against every mesh ``property`` (cols) over ``records``
    (each a merged metrics+properties+``name`` dict).  Reveals what mesh traits the model struggles on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, C = len(metrics), len(props)
    fig, ax = plt.subplots(R, C, figsize=(2.7 * C, 2.6 * R), squeeze=False)
    for r, mkey in enumerate(metrics):
        for c, pkey in enumerate(props):
            x = np.array([rec.get(pkey, np.nan) for rec in records], float)
            y = np.array([rec.get(mkey, np.nan) for rec in records], float)
            a = ax[r][c]; a.scatter(x, y, s=22, c="C0", alpha=0.8)
            if r == R - 1:
                a.set_xlabel(pkey, fontsize=9)
            if c == 0:
                a.set_ylabel(mkey, fontsize=9)
            a.tick_params(labelsize=7)
    fig.suptitle(f"reconstruction quality vs mesh properties  (n={len(records)} val meshes)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path
