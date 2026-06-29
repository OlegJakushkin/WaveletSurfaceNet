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


MIXED_CKPT = os.environ.get("MIXED_CKPT", "assets/waveshape_mixed.pt")     # set to ..._v2.pt to benchmark the retrained model


def mixed_net():
    global _mixed
    if _mixed is None:
        ck = torch.load(MIXED_CKPT, weights_only=False)
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
def _cleanup_mesh(v, f, min_faces=40, min_frac=0.005):
    """Mesh-time hygiene for the metrics ours is weak on: drop tiny FLOATING components (spurious surface that
    inflates #components and Chamfer).  Removes any component smaller than max(min_faces, min_frac*#faces),
    keeps everything else (legitimate thin parts survive) and the largest if all are tiny.  Opt-in via the
    OURS_CLEANUP env var; does NOT change default behaviour."""
    if v is None or not len(f):
        return v, f
    comps = trimesh.Trimesh(v, f, process=False).split(only_watertight=False)
    if len(comps) <= 1:
        return v, f
    thr = max(min_faces, min_frac * len(f))
    keep = [c for c in comps if len(c.faces) >= thr] or [max(comps, key=lambda c: len(c.faces))]
    m2 = trimesh.util.concatenate(keep)
    return np.asarray(m2.vertices), np.asarray(m2.faces)


def recon_ours(P, N):
    net = mixed_net()
    Pt = torch.tensor(P[None]).float().to(DEV); Nt = torch.tensor(N[None]).float().to(DEV)
    t = time.time()
    with torch.no_grad():
        g = WV._smooth_grid(net(Pt, Nt)[0][0, 0].cpu().numpy() * TRUNC, 0.5)
    if not (g.min() < 0 < g.max()):
        return None, None, time.time() - t
    v, f, _, _ = measure.marching_cubes(g.astype(np.float64), 0.0)
    v = v / (g.shape[0] - 1) * (2 * BOUND) - BOUND
    if os.environ.get("OURS_CLEANUP"):
        v, f = _cleanup_mesh(v, f)
    return v, f, time.time() - t


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


def sdf_error(v, f, gt, n=4096, band=0.1, seed=0):
    """Mean |signed-distance error| to GT (x100), clamped to +/-band -- the metric our training optimises.
    Only defined where signed distance is (watertight GT); returns nan otherwise.  Computed mesh-to-mesh for
    every method (signed distance to the reconstructed mesh vs to GT) for an apples-to-apples comparison."""
    if v is None or not len(f) or not gt.is_watertight:
        return float("nan")
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, (n, 3))
    try:
        gd = np.clip(trimesh.proximity.signed_distance(gt, q), -band, band)
        rd = np.clip(trimesh.proximity.signed_distance(trimesh.Trimesh(v, f, process=False), q), -band, band)
    except Exception:
        return float("nan")
    return float(np.abs(gd - rd).mean() * 100)


def normal_consistency(v, f, gt, n=30000, seed=0):
    """Mean |cos| between recon-surface normal and nearest-GT-surface normal, both directions (1.0 = perfect).
    Sign-agnostic so a flipped winding is not penalised."""
    if v is None or not len(f):
        return float("nan")
    rec = trimesh.Trimesh(v, f, process=False)
    a, fa = trimesh.sample.sample_surface(rec, n); na = rec.face_normals[fa]
    b, fb = trimesh.sample.sample_surface(gt, n);  nb = gt.face_normals[fb]
    _, ia = cKDTree(b).query(a); _, ib = cKDTree(a).query(b)
    nc_ag = np.abs(np.einsum("ij,ij->i", na, nb[ia])).mean()
    nc_ga = np.abs(np.einsum("ij,ij->i", nb, na[ib])).mean()
    return float((nc_ag + nc_ga) / 2)


def fscore_curve(v, f, gt, taus=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1), n=30000, seed=0):
    """F-score across a sweep of thresholds (one NN-distance pair reused for all taus), plus a tau-independent
    AUC summary -- answers the reviewer's 'single tau is fragile' point."""
    if v is None or not len(f):
        return {float(t): 0.0 for t in taus}, 0.0
    a = trimesh.Trimesh(v, f, process=False).sample(n); b, _ = trimesh.sample.sample_surface(gt, n)
    da, _ = cKDTree(b).query(a); db, _ = cKDTree(a).query(b)
    curve = {}
    for t in taus:
        p = float((da < t).mean()); r = float((db < t).mean())
        curve[float(t)] = 2 * p * r / (p + r + 1e-9) * 100
    ta = np.array(sorted(curve)); vals = np.array([curve[t] for t in ta])
    auc = float(np.trapz(vals, ta) / (ta[-1] - ta[0]))
    return curve, auc


def mesh_defects(v, f):
    """Mesh-quality defects (reviewer's 'component count is not a watertightness proxy' point): boundary/open
    edges, non-manifold edges, self-intersections, degenerate faces, watertight flag, components, Euler char."""
    if v is None or not len(f):
        return dict(boundary_edges=-1, nonmanifold_edges=-1, self_intersections=-1, degenerate=-1,
                    watertight=False, components=0, euler=0)
    m = trimesh.Trimesh(v, f, process=False)
    cnt = np.bincount(m.edges_unique_inverse, minlength=len(m.edges_unique))
    boundary = int((cnt == 1).sum()); nonmanifold = int((cnt > 2).sum())
    try: degenerate = int((~m.nondegenerate_faces()).sum())
    except Exception: degenerate = -1
    self_x = -1
    try:
        import pymeshlab as ml
        ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(m.vertices, m.faces))
        ms.apply_filter("compute_selection_by_self_intersections_per_face")
        self_x = int(ms.current_mesh().selected_face_number())
    except Exception:
        pass
    return dict(boundary_edges=boundary, nonmanifold_edges=nonmanifold, self_intersections=self_x,
                degenerate=degenerate, watertight=bool(m.is_watertight), components=int(m.body_count),
                euler=int(m.euler_number))


def agg_ci95(rows, method, key):
    """Mean and t-based 95% CI half-width over per-shape rows (for error bars in the charts)."""
    xs = np.array([r["methods"][method][key] for r in rows if method in r["methods"]
                   and isinstance(r["methods"][method].get(key), (int, float))
                   and np.isfinite(r["methods"][method][key])], float)
    if xs.size == 0:
        return float("nan"), float("nan"), 0
    if xs.size < 2:
        return float(xs.mean()), 0.0, int(xs.size)
    from scipy import stats
    half = float(stats.t.ppf(0.975, xs.size - 1) * xs.std(ddof=1) / np.sqrt(xs.size))
    return float(xs.mean()), half, int(xs.size)


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
ORDER = ["GT", "SPSR", "BPA", "APSS", "RIMLS", "tori", "ours"]


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


def sample_path(path, n=4096, noise=0.0, seed=0):
    """Sample a cloud from a specific mesh file (for the ModelNet40 validation sweep)."""
    m = trimesh.load(path, force="mesh"); m.fix_normals()
    m = S.normalize_to_unit_cube(m)
    sc = 1.0 / max(np.abs(m.vertices).max(), 1e-6)
    P, N = E.sample_cloud(m, n=n, noise=0.0, seed=seed)
    P = (P * sc).astype(np.float64); N = N.astype(np.float64)
    if noise > 0:
        P = P + noise * np.random.default_rng(seed).normal(size=P.shape)
    return trimesh.Trimesh(m.vertices * sc, m.faces, process=False), P, N


def run_all_cloud(gt, P, N):
    """Run GT + every available baseline + tori + ours on an explicit (gt,P,N).  Same packing as run_all."""
    import baselines as B

    def pack(v, f, sec):
        return (v, f, sec, chamfer(v, f, gt), fscore(v, f, gt), ncomp(v, f))

    out = {"GT": pack(gt.vertices, gt.faces, 0.0)}
    for name, fn in B.available().items():
        try:
            v, f, sec = fn(P, N); out[name] = pack(v, f, sec)
        except Exception:
            out[name] = (None, None, float("nan"), float("nan"), 0.0, 0)
    vt, ft, st = recon_tori(P, N); out["tori"] = pack(vt, ft, st)
    vo, fo, so = recon_ours(P, N); out["ours"] = pack(vo, fo, so)
    return out


def thin_fraction(P, N):
    """Fraction of points the analytic gate flags as thin/open (our own closed-vs-open label)."""
    import torch as _t
    return float(WV.point_thinness(_t.tensor(P[None]).float(), _t.tensor(N[None]).float()).mean())
