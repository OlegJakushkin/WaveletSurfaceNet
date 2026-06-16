"""Stanford bunny as a ground-truth mesh :class:`~pat.shapes.Shape`.

The bunny is the canonical "real" test case for the PAT pipeline: an organic,
non-convex, *non-watertight* scan with ~35k vertices.  Because it is not a clean
analytic primitive we cannot write its SDF in closed form, so :class:`MeshShape`
provides the **exact** signed distance to the triangle soup instead -- this is the
ground truth against which the learned/blended PAT field is compared.

Two robustness wrinkles are handled here:

* The mesh is **open** (``fill_holes`` cannot fully close it), so naive inside/outside
  tests are unreliable.  We sign the distance with the *generalized winding number*
  (Jacobson et al. 2013), which degrades gracefully on open meshes -- a query is
  "inside" when its winding number is near 1 rather than 0.
* ``trimesh``'s fast proximity/containment backends (``signed_distance``,
  ``mesh.ray``, ``mesh.contains``) require the optional ``rtree`` spatial index,
  which may be absent.  We *prefer* :func:`trimesh.proximity.signed_distance` when it
  works and otherwise fall back to a self-contained, fully-vectorized computation
  (kNN-pruned point-to-triangle distance + winding-number sign) that needs only
  numpy + scipy.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import shapes
from .shapes import Shape, normalize_to_unit_cube, sample_mesh

_BUNNY_PATH = "assets/stanford-bunny.obj"


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_bunny(normalize: bool = True) -> trimesh.Trimesh:
    """Load the Stanford bunny from ``assets/stanford-bunny.obj``.

    Args:
        normalize: if True, center the mesh at its centroid and scale it into
            ``[-1, 1]^3`` (via :func:`pat.shapes.normalize_to_unit_cube`) so it lines
            up with the SDF grid and renderer used everywhere else in the package.

    The loader also attempts ``mesh.fill_holes()`` to make signing more reliable,
    but the scan stays open in general -- this is *not* treated as an error.
    """
    mesh = trimesh.load(_BUNNY_PATH, force="mesh")
    try:
        mesh.fill_holes()                       # best-effort; bunny stays open
    except Exception:
        pass
    if normalize:
        mesh = normalize_to_unit_cube(mesh)
    # The OBJ is y-up (ears along +y); rotate +90 deg about x so the bunny stands
    # upright with ears along +z, matching the renderer's z-up convention (so a view
    # of the bunny shows its face and ears).  A 90 deg axis rotation just permutes the
    # coordinates, so the unit-cube fit is preserved: (x, y, z) -> (x, -z, y).
    v = mesh.vertices
    mesh.vertices = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    return mesh


# --------------------------------------------------------------------------- #
#  Vectorized point-to-triangle distance (Ericson) + winding-number sign
# --------------------------------------------------------------------------- #
def _closest_point_on_tris(p: np.ndarray, tris: np.ndarray):
    """Closest point on each query's candidate triangles ``tris[m, k, 3, 3]``.

    Vectorized over both the query axis ``m`` and the candidate axis ``k`` using the
    branchless region test from Ericson, *Real-Time Collision Detection* (Sec. 5.1.5):
    the closest point lies in the triangle's interior, on one of the three edges, or at
    one of the three vertices, selected by the barycentric/edge-projection signs.

    Returns ``(dist (m,), closest (m, 3), arg (m,))`` where ``arg`` is the index *within
    each row's candidate set* of the winning triangle (so ``idx[row, arg]`` recovers the
    global face id).
    """
    a = tris[..., 0, :]; b = tris[..., 1, :]; c = tris[..., 2, :]
    pq = p[:, None, :]
    ab = b - a; ac = c - a; ap = pq - a
    d1 = np.einsum("mki,mki->mk", ab, ap)
    d2 = np.einsum("mki,mki->mk", ac, ap)
    bp = pq - b
    d3 = np.einsum("mki,mki->mk", ab, bp)
    d4 = np.einsum("mki,mki->mk", ac, bp)
    cp = pq - c
    d5 = np.einsum("mki,mki->mk", ab, cp)
    d6 = np.einsum("mki,mki->mk", ac, cp)

    # interior of the face via barycentric coordinates
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = 1.0 / (va + vb + vc + 1e-30)
    v = vb * denom; w = vc * denom
    res = a + v[..., None] * ab + w[..., None] * ac

    # vertex regions
    mA = (d1 <= 0) & (d2 <= 0);          res = np.where(mA[..., None], a, res)
    mB = (d3 >= 0) & (d4 <= d3);         res = np.where(mB[..., None], b, res)
    mC = (d6 >= 0) & (d5 <= d6);         res = np.where(mC[..., None], c, res)
    # edge regions (clamped projection onto the edge segment)
    den = d1 - d3; mAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.clip(d1 / np.where(den == 0, 1, den), 0, 1)
    res = np.where(mAB[..., None], a + t[..., None] * ab, res)
    den = d2 - d6; mAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.clip(d2 / np.where(den == 0, 1, den), 0, 1)
    res = np.where(mAC[..., None], a + t[..., None] * ac, res)
    den = (d4 - d3) + (d5 - d6); mBC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = np.clip((d4 - d3) / np.where(den == 0, 1, den), 0, 1)
    res = np.where(mBC[..., None], b + t[..., None] * (c - b), res)

    dist = np.linalg.norm(pq - res, axis=-1)
    arg = np.argmin(dist, axis=1)
    rows = np.arange(len(p))
    return dist[rows, arg], res[rows, arg], arg


def _winding_number(q: np.ndarray, tris: np.ndarray, chunk: int = 128) -> np.ndarray:
    """Generalized winding number of points ``q`` w.r.t. the triangle soup ``tris``.

    Sums the signed solid angle each triangle subtends at the query (Van Oosterom &
    Strackee's robust ``atan2`` form) and divides by ``4*pi``.  For a closed mesh this
    is ~1 strictly inside and ~0 outside; for an open mesh it varies smoothly, so
    thresholding at 1/2 gives a robust inside/outside test.  Chunked over queries to
    bound the ``(chunk, n_faces, 3)`` working set -- with ~70k faces the chunk must
    stay small (the default keeps each temporary to a few tens of MB).
    """
    a = tris[:, 0]; b = tris[:, 1]; c = tris[:, 2]
    out = np.empty(len(q))
    for i in range(0, len(q), chunk):
        Q = q[i:i + chunk][:, None, :]               # (m,1,3)
        A = a[None] - Q; B = b[None] - Q; C = c[None] - Q
        la = np.linalg.norm(A, axis=-1)
        lb = np.linalg.norm(B, axis=-1)
        lc = np.linalg.norm(C, axis=-1)
        numer = np.einsum("mfi,mfi->mf", A, np.cross(B, C))
        denom = (la * lb * lc
                 + np.einsum("mfi,mfi->mf", A, B) * lc
                 + np.einsum("mfi,mfi->mf", B, C) * la
                 + np.einsum("mfi,mfi->mf", C, A) * lb)
        out[i:i + chunk] = np.arctan2(numer, denom).sum(1) / (2.0 * np.pi)
    return out


# --------------------------------------------------------------------------- #
#  Mesh shape
# --------------------------------------------------------------------------- #
class MeshShape(Shape):
    """Wrap an arbitrary ``trimesh.Trimesh`` as a ground-truth-SDF :class:`Shape`.

    The unsigned distance is *exact* (distance to the nearest triangle), so this is the
    reference any approximate field is judged against.  Three signing strategies are
    supported, tried/selected via ``sign_method``:

    * ``"trimesh"`` -- :func:`trimesh.proximity.signed_distance` (fast & exact, but
      needs the optional ``rtree`` index; silently skipped if unavailable);
    * ``"normal"``  -- the angle-weighted *pseudonormal* test: sign by the dot product
      between ``query - closest_point`` and the nearest face normal.  O(1) per query and
      reliable near the surface, which is exactly where the SDF must be accurate;
    * ``"winding"`` -- the *generalized winding number* (Jacobson et al. 2013), the most
      robust choice for open meshes but ``O(n_queries * n_faces)``.

    ``"auto"`` (the default) uses ``"trimesh"`` when its backend works and otherwise
    ``"normal"`` -- a fully self-contained, numpy + scipy path that needs no native
    spatial-index libraries.

    Args:
        mesh: the triangle mesh.  It is **not** required to be watertight.
        knn:  number of nearest candidate triangles considered per query (pruned by a
            centroid KD-tree before the exact point-to-triangle distance).  32 is
            plenty for a well-tessellated mesh; raise it for very irregular meshes.
        sign_method: one of ``"auto"``, ``"trimesh"``, ``"normal"``, ``"winding"``.
    """

    def __init__(self, mesh: trimesh.Trimesh, knn: int = 32, sign_method: str = "auto"):
        self.mesh = mesh
        self.knn = int(knn)
        self.sign_method = sign_method
        # cached geometry for the self-contained paths
        self._tris = mesh.triangles.view(np.ndarray).astype(np.float64)        # (nf,3,3)
        self._fnrm = mesh.face_normals.view(np.ndarray).astype(np.float64)     # (nf,3)
        self._tree = cKDTree(self._tris.mean(axis=1))                          # centroids
        # probe the trimesh backend once so "auto" can decide without retrying per call
        self._trimesh_ok = sign_method in ("auto", "trimesh") and self._probe_trimesh()

    # ------------------------------------------------------------------ #
    def _probe_trimesh(self) -> bool:
        try:
            trimesh.proximity.signed_distance(self.mesh, self._tris[:1, 0])
            return True
        except Exception:
            return False

    def _sdf_trimesh(self, pts: np.ndarray) -> np.ndarray:
        """trimesh signed distance (positive inside) -> negated to our convention."""
        sd = trimesh.proximity.signed_distance(self.mesh, pts)     # +inside
        return -np.asarray(sd, dtype=np.float64)

    def _closest(self, pts: np.ndarray):
        """kNN-pruned exact closest point: ``(dist, closest, face_id)``."""
        k = min(self.knn, len(self._tris))
        _, idx = self._tree.query(pts, k=k)
        idx = np.atleast_2d(idx)
        dist, closest, arg = _closest_point_on_tris(pts, self._tris[idx])
        face_id = idx[np.arange(len(pts)), arg]
        return dist, closest, face_id

    def _sdf_normal(self, pts: np.ndarray) -> np.ndarray:
        """Exact distance signed by the nearest-face pseudonormal (negative inside)."""
        dist, closest, face_id = self._closest(pts)
        outward = np.einsum("mi,mi->m", pts - closest, self._fnrm[face_id])
        sign = np.where(outward >= 0, 1.0, -1.0)                    # behind face -> inside
        return sign * dist

    def _sdf_winding(self, pts: np.ndarray) -> np.ndarray:
        """Exact distance signed by the generalized winding number (negative inside)."""
        dist, _, _ = self._closest(pts)
        wn = _winding_number(pts, self._tris)
        sign = np.where(wn > 0.5, -1.0, 1.0)                        # inside -> negative
        return sign * dist

    def sdf(self, x: np.ndarray) -> np.ndarray:
        """Signed distance to the mesh (negative inside) for ``x`` of shape ``(..., 3)``."""
        x = np.asarray(x, dtype=np.float64)
        flat = x.reshape(-1, 3)
        method = self.sign_method
        if method == "auto":
            method = "trimesh" if self._trimesh_ok else "normal"
        if method == "trimesh":
            out = self._sdf_trimesh(flat)
        elif method == "winding":
            out = self._sdf_winding(flat)
        elif method == "normal":
            out = self._sdf_normal(flat)
        else:
            raise ValueError("unknown sign_method %r" % self.sign_method)
        return out.reshape(x.shape[:-1])

    # ------------------------------------------------------------------ #
    def sample_surface(self, n: int, rng: np.random.Generator):
        """Exact surface samples ``(points (n,3), normals (n,3))`` via face sampling."""
        return sample_mesh(self.mesh, n, rng)


# --------------------------------------------------------------------------- #
#  Convenience
# --------------------------------------------------------------------------- #
def bunny_shape() -> MeshShape:
    """The normalized Stanford bunny as a ready-to-use :class:`MeshShape`."""
    return MeshShape(load_bunny(normalize=True))
