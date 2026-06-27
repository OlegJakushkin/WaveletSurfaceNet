"""Comparison harness core: our mixed model + the tori baseline, cloud sampling (noise / sparsity), and
Chamfer.  Baseline runners that need external libraries live in baselines.py and are imported guardedly."""
import sys, os, time, glob
sys.path.insert(0, "tori")                                     # tori package (CoeffNet / PAT)
import numpy as np, torch, trimesh
from scipy.spatial import cKDTree
from skimage import measure
from waveshape import wavelet as WV, eval3d as E, shapes as S
from waveshape.bunny import load_bunny

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOUND, TRUNC = 1.1, 0.1

_mixed = None
_tori = None


def mixed_net():
    global _mixed
    if _mixed is None:
        ck = torch.load("assets/waveshape_mixed.pt", weights_only=False)
        _mixed = WV.load_at_res(ck, res=128, bound=BOUND).to(DEV).eval()   # resolution-free: query fine, fair to mesh baselines
    return _mixed


def tori_net():
    global _tori
    if _tori is None:
        from pat.model import CoeffNet
        tck = torch.load("tori/assets/pat_torus.pt", weights_only=False)
        cfg = tck.get("config", {}); ctor = {k: cfg[k] for k in ("d_embed", "n_layers", "n_heads", "d_ff", "supertoroid", "p_max") if k in cfg}
        net = CoeffNet(**ctor).to(DEV); net.load_state_dict(tck["state_dict"]); net.eval(); _tori = net
    return _tori


# ----------------------------------------------------------------- shapes & clouds
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


def sample(c, n=8000, noise=0.0, seed=0):
    """Return (mesh_gt_scaled, P, N) with P scaled into the [-1,1] training frame; noise is a fraction of the
    bounding-box diagonal added to point positions (normals kept), matching how we report robustness."""
    m = get_mesh(c)
    if c not in ("cube", "sphere", "knurl", "torus", "teapot"):
        m.fix_normals()
    sc = 1.0 / max(np.abs(m.vertices).max(), 1e-6)
    P, N = E.sample_cloud(m, n=n, noise=0.0, seed=seed)
    P = (P * sc).astype(np.float64); N = N.astype(np.float64)
    if noise > 0:
        rng = np.random.default_rng(seed)
        P = P + noise * rng.normal(size=P.shape)                # noise in the unit frame
    gt = trimesh.Trimesh(m.vertices * sc, m.faces, process=False)
    return gt, P, N


# ----------------------------------------------------------------- our methods
def recon_ours(P, N):
    net = mixed_net()
    Pt = torch.tensor(P[None]).float().to(DEV); Nt = torch.tensor(N[None]).float().to(DEV)
    t = time.time()
    with torch.no_grad():
        g = WV._smooth_grid(net(Pt, Nt)[0][0, 0].cpu().numpy() * TRUNC, 0.5)
    if not (g.min() < 0 < g.max()):
        return None, None, time.time() - t
    v, f, _, _ = measure.marching_cubes(g.astype(np.float64), 0.0)
    return v / (g.shape[0] - 1) * (2 * BOUND) - BOUND, f, time.time() - t


def recon_tori(P, N):
    from pat.pat import PAT
    t = time.time()
    v, f = PAT(P, N, model=tori_net(), k=16, C=64.0, device=DEV).reconstruct(res=96, bound=BOUND, neighbors=64)
    return v, f, time.time() - t


# ----------------------------------------------------------------- metrics
def chamfer(v, f, gt, n=30000):
    if v is None or not len(f):
        return float("nan")
    a = trimesh.Trimesh(v, f, process=False).sample(n); b, _ = trimesh.sample.sample_surface(gt, n)
    da, _ = cKDTree(b).query(a); db, _ = cKDTree(a).query(b); return float((da.mean() + db.mean()) / 2 * 100)


def fscore(v, f, gt, tau=0.05, n=30000):
    """F-score at distance threshold tau (the standard surface-reconstruction metric; higher=better, %).
    precision = fraction of RECON surface within tau of GT (penalises over-fill / spurious surface);
    recall    = fraction of GT surface within tau of RECON (penalises holes / missing surface).
    Unlike raw Chamfer it does not reward holey point-interpolation or over-filled blobs."""
    if v is None or not len(f):
        return 0.0
    a = trimesh.Trimesh(v, f, process=False).sample(n); b, _ = trimesh.sample.sample_surface(gt, n)
    da, _ = cKDTree(b).query(a)                                # recon -> GT  (accuracy)
    db, _ = cKDTree(a).query(b)                                # GT -> recon  (completeness)
    prec = float((da < tau).mean()); rec = float((db < tau).mean())
    return 2 * prec * rec / (prec + rec + 1e-9) * 100


def ncomp(v, f):
    if v is None or not len(f):
        return 0
    try: return int(trimesh.Trimesh(v, f, process=False).body_count)
    except Exception: return -1


# ----------------------------------------------------------------- rendering + one-call benchmark
_LIGHT = np.array([0.4, -0.6, 0.72]); _LIGHT /= np.linalg.norm(_LIGHT); _BASE = np.array([0.42, 0.55, 0.66])


def draw3d(ax, v, f, view=(20, -55)):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if v is not None and len(f):
        tri = v[f]; fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9, None)
        sh = np.clip(np.abs(fn @ _LIGHT) * 0.5 + 0.5, 0.32, 1.0)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(_BASE[None] * sh[:, None], 0, 1), edgecolor="none"))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(*view)


# canonical method order for figures/charts
ORDER = ["GT", "SPSR", "BPA", "tori", "ours"]


def run_all(shape, n=2048, noise=0.0, seed=0):
    """Run GT + every available public baseline + tori + ours on one cloud.
    Returns (gt, P, N, results) where results[name] = (v, f, seconds, chamfer, n_parts)."""
    import baselines as B
    gt, P, N = sample(shape, n=n, noise=noise, seed=seed)

    def pack(v, f, sec):
        return (v, f, sec, chamfer(v, f, gt), fscore(v, f, gt), ncomp(v, f))   # (v,f,seconds,chamfer,fscore,parts)

    out = {"GT": pack(gt.vertices, gt.faces, 0.0)}
    for name, fn in B.available().items():
        try:
            v, f, sec = fn(P, N); out[name] = pack(v, f, sec)
        except Exception as e:
            print(f"  [{name}] failed on {shape}: {e}", flush=True); out[name] = (None, None, float("nan"), float("nan"), 0.0, 0)
    vt, ft, st = recon_tori(P, N); out["tori"] = pack(vt, ft, st)
    vo, fo, so = recon_ours(P, N); out["ours"] = pack(vo, fo, so)
    return gt, P, N, out
