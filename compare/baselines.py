"""Public-library reconstruction baselines, run honestly (not reimplemented):
  - SPSR  : Screened Poisson [Kazhdan & Hoppe 2013] via Open3D [Zhou et al. 2018]
  - BPA   : Ball-Pivoting    [Bernardini et al. 1999] via Open3D
  - GWN   : (Fast) Generalized Winding Number [Barill et al. 2018; Jacobson et al. 2013] via libigl
Each returns (verts, faces, seconds).  Missing libraries -> the method is skipped by the caller."""
import time, numpy as np
from skimage import measure

BOUND = 1.1


def _o3d_pcd(P, N):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(P, np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(N, np.float64))
    return pcd, o3d


def recon_spsr(P, N, depth=8, density_q=0.04):
    pcd, o3d = _o3d_pcd(P, N)
    t = time.time()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_q))   # trim the Poisson bubble
    dt = time.time() - t
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), dt


def recon_bpa(P, N):
    pcd, o3d = _o3d_pcd(P, N)
    d = np.asarray(pcd.compute_nearest_neighbor_distance()); r = float(np.mean(d))
    t = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([r * 1.5, r * 3.0, r * 6.0]))
    dt = time.time() - t
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), dt


def recon_gwn(P, N, res=96):
    import igl
    P = np.asarray(P, np.float64); N = np.asarray(N, np.float64)
    lin = np.linspace(-BOUND, BOUND, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    Q = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float64)
    t = time.time()
    try:
        w = igl.fast_winding_number(P, N, Q)                       # (P, N, Q)
    except Exception:
        A = np.ones(len(P))
        w = igl.fast_winding_number(P, N, A, Q)                    # (P, N, A, Q)
    dt = time.time() - t
    g = w.reshape(res, res, res) - 0.5                             # surface at winding number 0.5
    if not (g.min() < 0 < g.max()):
        return None, None, dt
    v, f, _, _ = measure.marching_cubes(g, 0.0)
    return v / (res - 1) * (2 * BOUND) - BOUND, f, dt


# name -> (callable, needs-import-module) so the driver can probe availability.
# NOTE: GWN (point-cloud fast winding number, Barill 2018) is intentionally NOT included: the libigl python
# binding here only exposes the *mesh* winding number fast_winding_number(V,F,Q), not the point-cloud variant,
# and we do not reimplement methods.  We compare against the public-library baselines that actually run.
METHODS = {
    "SPSR": (recon_spsr, "open3d"),
    "BPA":  (recon_bpa,  "open3d"),
}


def available():
    import importlib
    out = {}
    for name, (fn, mod) in METHODS.items():
        try:
            importlib.import_module(mod); out[name] = fn
        except Exception:
            pass
    return out
