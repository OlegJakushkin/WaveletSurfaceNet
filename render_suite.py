"""PerceiverWaveNet RENDER SUITE -- the after-each-epoch eyeball test on the four regimes we care about:

    1. favourites   -- clean canonical shapes (bunny, teapot, sphere, torus, cube, knurl)   -> IoU vs GT
    2. thin objects  -- open / thin-walled MN40 shapes (airplane, guitar, bottle, chair)     -> IoU vs GT
    3. noise         -- the bunny reconstructed from 0/2/5/10/15/20% Gaussian-noised clouds  -> IoU vs clean GT
    4. scenes        -- real indoor + outdoor scans (points+estimated normals, NO GT mesh)   -> #parts

The RESTORED paper2 model: the resolution-free PerceiverWaveNet (mixed base, smax corner head + far-field
clamp), loaded at EVAL_RES=128 via WV.load_at_res so the 42^3-trained net is always meshed at 128^3.  GPU only,
so it runs inside the project CUDA docker on the 4090 box (never on the Windows node).  Self-contained: mesh/
sampling/rendering helpers are inlined so it needs only waveshape + trimesh + skimage + matplotlib (no compare/).
Writes renders/suite_{favourites,thin,noise,scenes}.png from CKPT (default: assets/waveshape.pt).

    docker exec -e MPLBACKEND=Agg <container> python -u render_suite.py     # env CKPT=..., EVAL_RES=..., N_PTS=...
"""
import os, glob
import numpy as np, torch, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from waveshape import wavelet as WV, eval3d as E, shapes as S
from waveshape.bunny import load_bunny

BOUND, TRUNC, DEV = 1.1, 0.1, "cuda"
N_PTS = int(os.environ.get("N_PTS", "4096"))
CK = os.environ.get("CKPT", os.environ.get("VDB_CKPT", "assets/waveshape.pt"))
RES = int(os.environ.get("EVAL_RES", "128"))                # ALWAYS mesh the res-free net at 128^3
FAV    = ["bunny", "teapot", "sphere", "torus", "cube", "knurl"]
THIN   = ["airplane", "guitar", "bottle", "chair"]          # thin / open-surface MN40 classes present on the box
SCENES = ["indoor", "outdoor"]
NOISE  = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

ck = torch.load(CK, weights_only=False)
FM = ck.get("field_mode", "mixed"); EP = ck.get("epoch")
net = WV.load_at_res(ck, res=RES, bound=BOUND).to(DEV).eval()   # res-free: 42^3-trained net queried at 128^3
print(f"loaded {CK} | {ck.get('model')} train-res {ck.get('res')} -> eval {RES} | field {FM} epoch {EP} "
      f"val {ck.get('val_sdferr')}", flush=True)

_L = np.array([0.4, -0.6, 0.72]); _L /= np.linalg.norm(_L); _B = np.array([0.42, 0.55, 0.66])


# ------------------------------------------------------------------ shapes & clouds (inlined from compare/core)
def get_mesh(c):
    if c == "cube":   return S.normalize_to_unit_cube(trimesh.creation.box(extents=[1, 1, 1]))
    if c == "sphere": return S.normalize_to_unit_cube(trimesh.creation.uv_sphere(radius=0.7, count=[48, 48]))
    if c == "torus":  return S.normalize_to_unit_cube(trimesh.creation.torus(major_radius=0.5, minor_radius=0.2))
    if c == "knurl":  return S.normalize_to_unit_cube(E._knurl_mesh())
    if c == "bunny":  return S.normalize_to_unit_cube(load_bunny(normalize=True))
    if c == "teapot":
        tp = trimesh.load("assets/teapot.obj", force="mesh")
        tp.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
        return S.normalize_to_unit_cube(tp)
    m = trimesh.load(sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))[0], force="mesh"); m.fix_normals()
    return S.normalize_to_unit_cube(m)


def sample(c, n=N_PTS, seed=0):
    m = get_mesh(c)
    if c not in ("cube", "sphere", "knurl", "torus", "teapot"):
        m.fix_normals()
    sc = 1.0 / max(np.abs(m.vertices).max(), 1e-6)
    P, N = E.sample_cloud(m, n=n, noise=0.0, seed=seed)
    P = (P * sc).astype(np.float64); N = N.astype(np.float64)
    return trimesh.Trimesh(m.vertices * sc, m.faces, process=False), P, N


def draw3d(ax, v=None, f=None, pts=None, view=(20, -55)):
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c="#3a4a66", depthshade=True)
    elif v is not None and len(f):
        tri = v[f]; fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9, None)
        sh = np.clip(np.abs(fn @ _L) * 0.5 + 0.5, 0.32, 1.0)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(_B[None] * sh[:, None], 0, 1), edgecolor="none"))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(*view)


# ------------------------------------------------------------------ PerceiverWaveNet recon + metrics
def recon(P, N):
    """PerceiverWaveNet reconstruction @RES -> (verts, faces, smoothed occupancy grid) in [-1,1]."""
    Pt = torch.tensor(P[None]).float().to(DEV); Nt = torch.tensor(N[None]).float().to(DEV)
    with torch.no_grad():
        raw = net(Pt, Nt)[0][0, 0].cpu().numpy() * TRUNC
    v, f = WV.mesh_field(raw, FM, bound=BOUND, trunc=TRUNC)          # canonical mode-aware meshing
    return v, f, WV._smooth_grid(raw, 0.5)


def clean_occ(P, N):
    """Ground-truth inside-mask = sign of the clean direct TSDF at the model's resolution."""
    g = WV.tsdf_from_clouds(torch.tensor(P[None]).float().to(DEV), torch.tensor(N[None]).float().to(DEV),
                            RES, TRUNC, BOUND, DEV, mode=FM)[0, 0].cpu().numpy()
    return g < 0


def iou(gt_in, g):
    ri = g < 0
    return float((gt_in & ri).sum()) / max(float((gt_in | ri).sum()), 1)


def nparts(v, f):
    if v is None or not len(f):
        return 0
    try: return max(1, len(trimesh.Trimesh(v, f, process=False).split(only_watertight=False)))
    except Exception: return 1


# ------------------------------------------------------------------ panels
def object_panel(names, title, out):
    cols = len(names); fig = plt.figure(figsize=(2.1 * cols, 4.4))
    for j, nm in enumerate(names):
        gt, P, N = sample(nm)
        v, f, g = recon(P, N); i = iou(clean_occ(P, N), g)
        ax = fig.add_subplot(2, cols, j + 1, projection="3d"); draw3d(ax, gt.vertices, gt.faces)
        ax.set_title(nm, fontsize=11, weight="bold")
        if j == 0: ax.text2D(-0.08, 0.5, "GT", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
        ax2 = fig.add_subplot(2, cols, cols + j + 1, projection="3d"); draw3d(ax2, v, f)
        ax2.text2D(0.5, -0.04, ("fail" if v is None else f"IoU {i:.2f} ({len(f)}f)"),
                   transform=ax2.transAxes, ha="center", fontsize=8.5, color="#207020")
        if j == 0: ax2.text2D(-0.08, 0.5, "ours", transform=ax2.transAxes, rotation=90, va="center", ha="center", fontsize=10)
        print(f"  {nm}: IoU {i:.3f} | {0 if f is None else len(f)}f", flush=True)
    fig.suptitle(f"{title}  (epoch {EP}, res {RES}, {FM})", fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.05, wspace=0.0, hspace=0.06)
    os.makedirs("renders", exist_ok=True); fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out, flush=True)


def noise_panel(shape="bunny", out="renders/suite_noise.png"):
    gt, P0, N0 = sample(shape); gt_in = clean_occ(P0, N0); rng = np.random.default_rng(0)
    cols = len(NOISE); fig = plt.figure(figsize=(2.1 * cols, 4.5))
    for j, pct in enumerate(NOISE):
        Pn = P0 + pct * rng.normal(size=P0.shape)
        v, f, g = recon(Pn, N0); i = iou(gt_in, g)                   # recon-from-noisy vs CLEAN gt
        ax = fig.add_subplot(2, cols, j + 1, projection="3d"); draw3d(ax, pts=Pn)
        ax.set_title(f"{int(pct*100)}% noise", fontsize=11, weight="bold")
        if j == 0: ax.text2D(-0.08, 0.5, "input", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
        ax2 = fig.add_subplot(2, cols, cols + j + 1, projection="3d"); draw3d(ax2, v, f)
        ax2.text2D(0.5, -0.04, ("fail" if v is None else f"IoU {i:.2f} ({len(f)}f)"),
                   transform=ax2.transAxes, ha="center", fontsize=8.5, color="#207020")
        if j == 0: ax2.text2D(-0.08, 0.5, "ours", transform=ax2.transAxes, rotation=90, va="center", ha="center", fontsize=10)
        print(f"  {int(pct*100)}% noise: IoU {i:.3f} | {0 if f is None else len(f)}f", flush=True)
    fig.suptitle(f"{shape} under noise (epoch {EP}, res {RES}, {FM})", fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.05, wspace=0.0, hspace=0.06)
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out, flush=True)


def scene_panel(names, out="renders/suite_scenes.png"):
    cols = len(names); fig = plt.figure(figsize=(2.7 * cols, 5.0))
    for j, nm in enumerate(names):
        d = np.load(f"baselines_ext/scans/{nm}.npz")
        P = d["points"].astype(np.float64); N = d["normals"].astype(np.float64)
        v, f, g = recon(P, N); parts = nparts(v, f)
        ax = fig.add_subplot(2, cols, j + 1, projection="3d"); draw3d(ax, pts=P)
        ax.set_title(nm, fontsize=11, weight="bold")
        if j == 0: ax.text2D(-0.06, 0.5, "input scan", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
        ax2 = fig.add_subplot(2, cols, cols + j + 1, projection="3d"); draw3d(ax2, v, f)
        ax2.text2D(0.5, -0.04, ("fail" if v is None else f"{parts} parts ({len(f)}f)"),
                   transform=ax2.transAxes, ha="center", fontsize=8.5, color="#444")
        if j == 0: ax2.text2D(-0.06, 0.5, "ours", transform=ax2.transAxes, rotation=90, va="center", ha="center", fontsize=10)
        print(f"  {nm}: {parts} parts | {0 if f is None else len(f)}f", flush=True)
    fig.suptitle(f"real indoor/outdoor scans (epoch {EP}, res {RES}, {FM})", fontsize=12)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.05, wspace=0.0, hspace=0.06)
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out, flush=True)


if __name__ == "__main__":
    print("== favourites =="); object_panel(FAV, "favourites", "renders/suite_favourites.png")
    print("== thin =="); object_panel(THIN, "thin / open objects", "renders/suite_thin.png")
    print("== noise =="); noise_panel("bunny")
    have = [s for s in SCENES if os.path.exists(f"baselines_ext/scans/{s}.npz")]
    if have:                                     # scans are optional local inputs (untracked) -- skip if absent
        print("== scenes =="); scene_panel(have)
    else:
        print("== scenes SKIPPED (no baselines_ext/scans/*.npz in this checkout) ==")
    print("SUITE DONE", flush=True)
