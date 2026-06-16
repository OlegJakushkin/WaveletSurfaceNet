"""Real-dataset training pipeline *with noise* for the paper's network (CoeffNet).

The paper trains its coefficient predictor on ModelNet40 point clouds, crucially
*in the presence of noise* (Sec. 5): each training cloud is a noisy, finite sample
of a mesh, while the supervised signed-distance target is taken against the *clean*
surface.  This module turns a folder of ``.off`` meshes into exactly that kind of
training example -- noisy cloud + kNN neighborhoods + ground-truth signed distance
at query points -- in the tensor layout that :func:`pat.model.CoeffNet.forward` and
:func:`pat.train.pat_loss` consume.

Nothing here downloads anything.  A separate process populates ``data/`` and writes
``data/modelnet40_index.txt`` (one ``.off`` path per line); :func:`modelnet_index`
simply reads that index when it appears and returns ``[]`` otherwise, so the trainer
can fall back to the bundled bunny mesh while the download is still running.

Ground-truth signed distance is computed analytically here (vectorized point-to-
triangle distance + a face-normal sign) rather than via ``trimesh.proximity``: the
latter needs the optional ``rtree``/``embree`` backends, which are not assumed to be
installed, and it is unreliable on the many non-watertight ModelNet meshes anyway.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import trimesh

from . import shapes
from .neighbors import knn_neighborhoods


# --------------------------------------------------------------------------- #
#  ModelNet40 index (written by the downloader; we only ever read it)
# --------------------------------------------------------------------------- #
def modelnet_index(root: str = "data") -> list[str]:
    """Return the list of ``.off`` mesh paths under ``<root>/ModelNet40``.

    We walk the extracted ``ModelNet40`` directory directly so the paths are valid
    regardless of where the dataset is mounted (this matters in Docker, where the
    host paths in ``modelnet40_index.txt`` would be wrong).  If the directory is
    absent we fall back to the cached index file, and finally to an empty list so
    a trainer can keep going on the bundled stand-in until the real data lands.
    """
    mn_dir = os.path.join(root, "ModelNet40")
    if os.path.isdir(mn_dir):
        paths = []
        for dp, _, fs in os.walk(mn_dir):
            paths.extend(os.path.join(dp, f) for f in fs if f.endswith(".off"))
        if paths:
            return sorted(paths)
    index_path = os.path.join(root, "modelnet40_index.txt")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            cached = [line.strip() for line in fh if line.strip()]
        if cached and all(os.path.isfile(p) for p in cached[:5]):
            return cached
    return []


# Any mesh container trimesh can load (force='mesh' collapses scenes).  This is
# the union of the formats the supported real-mesh corpora ship in: ModelNet40
# is .off, the ABC CAD dataset is .obj, Objaverse is .glb, Thingi10K is .stl, ...
MESH_EXTS = (".off", ".obj", ".ply", ".stl", ".glb", ".gltf")


def mesh_index(root: str = "data", exts=MESH_EXTS,
               cache_name: str = "mesh_index.txt") -> list[str]:
    """Return every trimesh-loadable mesh found anywhere under ``root``.

    The generalization of :func:`modelnet_index` beyond ModelNet40, so the
    trainer can consume **any** real-mesh corpus the notebook downloads into
    ``data/`` -- in particular the **ABC CAD dataset** (≥50k ``.obj`` files), and
    also Objaverse ``.glb`` or Thingi10K ``.stl`` as fallbacks.  We walk the
    directory directly (paths valid wherever the data is mounted), falling back
    to a cached index file and finally to ``[]`` so a trainer can keep going on
    the bundled analytic shapes until the real data lands.
    """
    exts = tuple(e.lower() for e in exts)
    if os.path.isdir(root):
        paths = []
        for dp, _, fs in os.walk(root):
            paths.extend(os.path.join(dp, f) for f in fs if f.lower().endswith(exts))
        if paths:
            return sorted(paths)
    index_path = os.path.join(root, cache_name)
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            cached = [line.strip() for line in fh if line.strip()]
        if cached and all(os.path.isfile(p) for p in cached[:5]):
            return cached
    return []


# --------------------------------------------------------------------------- #
#  Mesh loading / normalization
# --------------------------------------------------------------------------- #
class DegenerateMeshError(ValueError):
    """Raised when a mesh has no usable geometry (no vertices / faces / area)."""


def load_mesh_normalized(path: str, max_faces: int | None = None) -> trimesh.Trimesh:
    """Load ``path`` as a single mesh and rescale it into ``[-1, 1]^3``.

    Uses ``trimesh.load(force='mesh')`` (collapsing scenes / multi-geometry files
    into one mesh) followed by :func:`pat.shapes.normalize_to_unit_cube`.  Raises
    :class:`DegenerateMeshError` on empty or zero-area meshes so callers can skip
    them cleanly instead of producing NaNs downstream.

    ``max_faces`` optionally skips meshes above a face budget (raising
    :class:`DegenerateMeshError`): real CAD / scanned corpora (ABC, Objaverse)
    contain the occasional multi-million-triangle model that would dominate
    cache-build time for no extra training signal.
    """
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0 \
            or mesh.vertices.shape[0] == 0:
        raise DegenerateMeshError(f"{path}: no triangular geometry")
    if max_faces is not None and mesh.faces.shape[0] > max_faces:
        raise DegenerateMeshError(f"{path}: {mesh.faces.shape[0]} faces > max_faces={max_faces}")
    if not np.isfinite(mesh.vertices).all() or float(mesh.area) <= 0.0:
        raise DegenerateMeshError(f"{path}: degenerate (zero area / non-finite verts)")
    mesh = shapes.normalize_to_unit_cube(mesh)
    if not np.isfinite(mesh.bounds).all():
        raise DegenerateMeshError(f"{path}: non-finite bounds after normalization")
    return mesh


# --------------------------------------------------------------------------- #
#  Noisy point cloud (the "training in presence of noise" piece)
# --------------------------------------------------------------------------- #
def noisy_point_cloud(mesh, n, rng, noise_std=0.01, outlier_frac=0.0,
                      return_clean=False):
    """Sample ``n`` surface points and corrupt them with sensor-like noise.

    Two perturbations are applied to the *positions* (the returned normals are the
    clean surface normals, as a real scanner would estimate them):

    * **structured + isotropic jitter** -- a displacement that is mostly along the
      surface normal (the dominant error mode of a depth sensor) plus a smaller
      isotropic component.  Total displacement has standard deviation ``noise_std``
      in unit-cube units, matching the scale used for the ground-truth band.
    * **outliers** -- a fraction ``outlier_frac`` of the points are replaced by
      points drawn uniformly in ``[-1, 1]^3`` (gross errors / background returns).

    Args:
        return_clean: if True, also return the un-noised surface points (the same
            base points the noise was added to), for diagnostics / self-tests that
            want to measure the actual per-point displacement.

    Returns ``(points (n,3), normals (n,3))`` as float64 numpy arrays, or
    ``(points, normals, clean_points)`` when ``return_clean`` is True.
    """
    pts, nrm = shapes.sample_mesh(mesh, n, rng)
    clean = np.asarray(pts, dtype=np.float64).copy()
    pts = clean.copy()
    nrm = np.asarray(nrm, dtype=np.float64).copy()
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12

    if noise_std > 0:
        # Split the variance so the total per-point displacement std == noise_std:
        # 80% of the variance along the normal, 20% isotropic in 3D.
        normal_std = noise_std * np.sqrt(0.8)
        iso_std = noise_std * np.sqrt(0.2 / 3.0)
        along = rng.normal(scale=normal_std, size=(n, 1)) * nrm
        iso = rng.normal(scale=iso_std, size=(n, 3))
        pts = pts + along + iso

    if outlier_frac > 0:
        n_out = int(round(outlier_frac * n))
        if n_out > 0:
            idx = rng.choice(n, size=n_out, replace=False)
            pts[idx] = rng.uniform(-1.0, 1.0, size=(n_out, 3))

    if return_clean:
        return pts, nrm, clean
    return pts, nrm


# --------------------------------------------------------------------------- #
#  Ground-truth signed distance to a (clean) mesh, without rtree/embree
# --------------------------------------------------------------------------- #
def mesh_signed_distance(mesh, queries):
    """Signed distance from ``queries (Q,3)`` to the surface of ``mesh``.

    Unsigned distance is the exact point-to-mesh distance (vectorized over queries,
    looping over triangles via ``trimesh``'s closest-point routine).  The sign is
    taken from the dot product of ``(query - closest_point)`` with the normal of the
    closest face: positive (outside) when the query is on the outward side.  This is
    the standard surface-pseudonormal test and is robust on non-watertight meshes,
    where a winding-number / ray-parity test would be ill-defined.

    Returns ``phi (Q,)`` float64 (negative inside, positive outside).
    """
    from trimesh.proximity import closest_point_naive

    queries = np.asarray(queries, dtype=np.float64)
    closest, dist, tri_id = closest_point_naive(mesh, queries)
    fn = mesh.face_normals[tri_id]
    sign = np.einsum("ij,ij->i", queries - closest, fn)
    sign = np.where(sign >= 0.0, 1.0, -1.0)
    return (sign * dist).astype(np.float64)


def _sample_queries(mesh, rng, n_band, n_cube, noise_std, bound):
    """Query points: a narrow band hugging the surface + a uniform bulk fill.

    Mirrors the train-time split of Sec. 4.3 but drops the on-surface third (those
    targets are ~0 and add little signal under noise); we use a band whose width is
    a few times the noise level so the network is supervised exactly where the noisy
    cloud is most misleading.
    """
    band_std = max(3.0 * noise_std, 0.02)
    surf, _ = shapes.sample_mesh(mesh, n_band, rng)
    band = np.asarray(surf, dtype=np.float64) + rng.normal(scale=band_std,
                                                           size=(n_band, 3))
    cube = rng.uniform(-bound, bound, size=(n_cube, 3))
    return np.concatenate([band, cube], axis=0)


# --------------------------------------------------------------------------- #
#  One full training example
# --------------------------------------------------------------------------- #
def make_training_example(path_or_mesh, rng, *, n_points=256, k=16, n_query=256,
                          noise_std=0.01, outlier_frac=0.0, bound=1.0):
    """Build one noisy training example for :class:`pat.model.CoeffNet`.

    Pipeline: load + normalize the mesh -> noisy point cloud (positions perturbed,
    normals clean) -> kNN neighborhoods -> query points (1/3 in a narrow surface
    band, the rest uniform in the cube) -> ground-truth signed distance to the
    *clean* surface (never to the noisy cloud).

    Args:
        path_or_mesh: a path to a mesh file or an already-loaded ``trimesh.Trimesh``.
        rng: a ``numpy.random.Generator``.
        n_points: number of cloud points ``N`` (one fitted torus / token per point).
        k: neighbors per point; neighborhoods have ``k + 1`` members (self first).
        n_query: number of query points ``Q``.
        noise_std / outlier_frac: cloud corruption (see :func:`noisy_point_cloud`).
        bound: half-extent of the bulk query cube.

    Returns a dict of CPU float32 / long tensors:
        ``P (N,3)``, ``Nn (N,3)``, ``nbr_pos (N,k+1,3)``, ``nbr_nrm (N,k+1,3)``,
        ``q (Q,3)``, ``phi (Q,)``.
    """
    mesh = path_or_mesh if isinstance(path_or_mesh, trimesh.Trimesh) \
        else load_mesh_normalized(path_or_mesh)

    # Noisy cloud (training input) -- positions corrupted, normals from the surface.
    P, Nn = noisy_point_cloud(mesh, n_points, rng, noise_std=noise_std,
                              outlier_frac=outlier_frac)

    # kNN neighborhoods over the *noisy* positions (what the network really sees).
    idx = knn_neighborhoods(P, k)                    # (N, k+1)
    nbr_pos = P[idx]                                  # (N, k+1, 3)
    nbr_nrm = Nn[idx]                                 # (N, k+1, 3)

    # Query points + ground-truth signed distance to the CLEAN surface.
    n_band = n_query // 3
    n_cube = n_query - n_band
    q = _sample_queries(mesh, rng, n_band, n_cube, noise_std, bound)
    phi = mesh_signed_distance(mesh, q)

    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return {
        "P": f32(P),
        "Nn": f32(Nn),
        "nbr_pos": f32(nbr_pos),
        "nbr_nrm": f32(nbr_nrm),
        "q": f32(q),
        "phi": f32(phi),
    }


# --------------------------------------------------------------------------- #
#  Streaming many examples for the trainer
# --------------------------------------------------------------------------- #
def iter_training_examples(paths, rng, *, skip_errors=True, **kw):
    """Yield :func:`make_training_example` dicts for random meshes from ``paths``.

    This is how the trainer consumes the full ModelNet40 corpus (>=10000 models)
    under noise: an infinite stream of independent noisy examples drawn (with
    replacement) from the index.  Meshes that fail to load / are degenerate are
    skipped when ``skip_errors`` (the default), so one bad ``.off`` never stops a
    run.  Yields forever -- the caller decides how many examples to pull.
    """
    paths = list(paths)
    if not paths:
        return
    while True:
        path = paths[rng.integers(0, len(paths))]
        try:
            yield make_training_example(path, rng, **kw)
        except (DegenerateMeshError, ValueError, OSError) as exc:
            if not skip_errors:
                raise
            # Skip this mesh and try another draw on the next iteration.
            _ = exc
            continue


# --------------------------------------------------------------------------- #
#  Dense per-mesh cache (the GPU trainer's "cache once, reuse" representation)
# --------------------------------------------------------------------------- #
def surface_band_queries(rng, surf, n_query, bound):
    """Query points: half in a thin band hugging ``surf``, half uniform in the cube."""
    nb = n_query // 2
    band = surf[:nb] + rng.normal(scale=0.04, size=(nb, 3))
    bulk = rng.uniform(-bound, bound, size=(n_query - nb, 3))
    return np.concatenate([band, bulk], 0)


def mesh_dense_example(path, dense, n_query, rng, bound=1.0, dense_surf=50000,
                       max_faces=None):
    """One dense ``(cloud, normals, queries, GT SDF)`` tuple from a real mesh.

    Ground-truth signed distance uses a KD-tree over a dense surface sample (fast
    and accurate to the surface spacing).  ``max_faces`` skips pathologically heavy
    meshes (see :func:`load_mesh_normalized`).  Returns four float32 numpy arrays.
    """
    from scipy.spatial import cKDTree
    mesh = load_mesh_normalized(path, max_faces=max_faces)
    pts, nrm = shapes.sample_mesh(mesh, dense, rng)
    # Real meshes have duplicate/degenerate faces -> coincident points and the
    # occasional zero-area-face normal.  Jitter the cloud (breaks coincidences,
    # which otherwise make a neighborhood's median distance 0 and blow up the
    # features) and replace degenerate normals with a fallback.
    pts = pts + rng.normal(scale=1e-4, size=pts.shape)
    nn = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(nn < 1e-6, np.array([0.0, 0.0, 1.0]), nrm / (nn + 1e-9))
    surf, _ = shapes.sample_mesh(mesh, n_query, rng)
    q = surface_band_queries(rng, surf, n_query, bound)
    ds, dn = shapes.sample_mesh(mesh, dense_surf, rng)
    tree = cKDTree(ds)
    d, idx = tree.query(q)
    sign = np.einsum("ij,ij->i", q - ds[idx], dn[idx])
    phi = np.where(sign >= 0, d, -d)
    return (pts.astype(np.float32), nrm.astype(np.float32),
            q.astype(np.float32), phi.astype(np.float32))


def build_mesh_cache(paths, dense, n_query, bound=1.0, max_faces=None, seed=0,
                     limit=None, log_every=1000):
    """Stack ``(P, N, Q, PHI)`` tensors for the loadable meshes in ``paths``.

    The incremental building block the Colab notebook calls **per ABC chunk**, so a
    chunk's meshes can be cached and then its extracted ``.obj`` files deleted
    before the next chunk is fetched -- keeping peak disk to one chunk instead of
    the whole corpus.  Degenerate / over-budget / unreadable meshes are skipped;
    returns a dict of CPU ``torch`` tensors ``{P, N, Q, PHI}`` (shapes
    ``(M, dense, 3)``/``(M, dense, 3)``/``(M, n_query, 3)``/``(M, n_query)``), or
    ``None`` if nothing usable was found.
    """
    import time
    rng = np.random.default_rng(seed)
    P, N, Q, PHI = [], [], [], []
    tried, t0 = 0, time.time()
    for path in paths:
        if limit is not None and len(P) >= limit:
            break
        tried += 1
        try:
            ex = mesh_dense_example(path, dense, n_query, rng, bound, max_faces=max_faces)
        except Exception:
            continue
        if not all(np.isfinite(a).all() for a in ex):       # drop degenerate meshes
            continue
        P.append(ex[0]); N.append(ex[1]); Q.append(ex[2]); PHI.append(ex[3])
        if log_every and len(P) % log_every == 0:
            print(f"  cached {len(P)} meshes (kept {len(P)}/{tried}; "
                  f"{len(P)/(time.time()-t0):.0f}/s)", flush=True)
    if not P:
        return None
    return {"P": torch.from_numpy(np.stack(P)), "N": torch.from_numpy(np.stack(N)),
            "Q": torch.from_numpy(np.stack(Q)), "PHI": torch.from_numpy(np.stack(PHI))}
