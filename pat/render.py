"""Paper-style comparison renderer (Fig. 8 / Fig. 17 layout).

Produces a 2-row figure per asset, saved as a PNG:

* **top row** -- the 3D surface: ground truth, then each method's reconstructed zero
  level set (marching cubes), Lambert-shaded;
* **bottom row** -- a 2D slice of the signed-distance field: red distance isolines,
  the zero level set highlighted in blue, and the input point cloud as black dots.

Uses the Agg backend so it is fully headless (no display needed).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

GT_COLOR = (0.80, 0.68, 0.42)        # clay / tan, like the paper's ground truth
OURS_COLOR = (0.55, 0.64, 0.72)      # steel blue-gray reconstructions
_REDS = LinearSegmentedColormap.from_list(
    "pat_reds", [(1, 1, 1), (0.99, 0.86, 0.80), (0.94, 0.58, 0.46), (0.78, 0.18, 0.16)])


def _mc(sdf_fn, res, bound, level=0.0, chunk=131072):
    """Marching cubes of an SDF callable over ``[-bound, bound]^3`` -> (verts, faces).

    The grid SDF is evaluated in chunks so a ground-truth callable that allocates per
    query (e.g. the bunny's mesh SDF) doesn't blow up memory on a ``res^3`` grid.
    """
    lin = np.linspace(-bound, bound, res)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    vol = np.empty(grid.shape[0], dtype=np.float32)
    for a in range(0, len(grid), chunk):
        vol[a:a + chunk] = np.asarray(sdf_fn(grid[a:a + chunk])).ravel()
    vol = vol.reshape(res, res, res)
    if not (vol.min() < level < vol.max()):
        return None, None
    v, f, _, _ = measure.marching_cubes(vol, level=level)
    v = v / (res - 1) * (2 * bound) - bound
    return v, f


def _smooth(verts, faces, iters):
    """Volume-preserving Taubin smoothing of a mesh (cosmetic, for the figure only)."""
    if verts is None or iters <= 0:
        return verts, faces
    try:
        import trimesh
        m = trimesh.Trimesh(verts, faces, process=False)
        trimesh.smoothing.filter_taubin(m, iterations=iters)
        return np.asarray(m.vertices), np.asarray(m.faces)
    except Exception:
        return verts, faces


def _render_mesh(ax, verts, faces, base_color, view=(22, -62), light=(0.4, 0.3, 1.0)):
    """Lambert-shade a triangle mesh into a 3D axis with no decorations."""
    ax.set_axis_off()
    if verts is None or len(faces) == 0:
        ax.text2D(0.5, 0.5, "(no surface)", ha="center", va="center",
                  transform=ax.transAxes, color="gray")
        return
    tris = verts[faces]                                   # (F,3,3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    shade = 0.35 + 0.65 * np.clip(np.abs(n @ L), 0, 1)    # ambient + diffuse
    fc = np.clip(np.asarray(base_color)[None, :] * shade[:, None], 0, 1)
    coll = Poly3DCollection(tris, facecolors=fc, edgecolors="none")
    ax.add_collection3d(coll)
    lim = np.abs(verts).max() * 1.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view[0], azim=view[1])


def _render_slice(ax, sdf_fn, points, axis=2, value=0.0, extent=1.2, res=220,
                  band=0.04, neighbors=64):
    """Draw a 2D SDF slice: red distance isolines + blue zero set + black points."""
    lin = np.linspace(-extent, extent, res)
    ga, gb = np.meshgrid(lin, lin, indexing="ij")
    flat = np.stack([ga.ravel(), gb.ravel()], 1)
    grid = np.insert(flat, axis, value, axis=1)
    try:
        vals = np.asarray(sdf_fn(grid)).reshape(res, res)
    except TypeError:                                    # PAT.sdf needs neighbors kw
        vals = np.asarray(sdf_fn(grid, neighbors=neighbors)).reshape(res, res)
    A, B = ga, gb
    vmax = np.percentile(np.abs(vals), 98) + 1e-6
    ax.contourf(A, B, np.abs(vals), levels=24, cmap=_REDS, vmin=0, vmax=vmax)
    ax.contour(A, B, np.abs(vals), levels=18, colors="white", linewidths=0.5, alpha=0.7)
    ax.contourf(A, B, np.abs(vals), levels=[0, band], colors=[(0.16, 0.40, 0.78)])
    ax.contour(A, B, vals, levels=[0], colors=[(0.10, 0.28, 0.62)], linewidths=1.2)
    if points is not None and len(points):
        near = np.abs(points[:, axis] - value) < 0.12
        proj = points[near][:, [i for i in range(3) if i != axis]]
        ax.scatter(proj[:, 0], proj[:, 1], s=2, c="k", alpha=0.7, linewidths=0)
    ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render_comparison(shape, pats, points, out_path, *, slice_axis=2, slice_value=0.0,
                      recon_res=80, recon_bound=1.2, slice_extent=1.2, slice_res=220,
                      neighbors=64, view=(22, -62), suptitle=None, npoints_label=None,
                      smooth_iters=12, dpi=130):
    """Render a paper-style comparison and save it to ``out_path``.

    Args:
        shape:  the ground-truth :class:`pat.shapes.Shape` (its ``.sdf`` is GT).
        pats:   ordered dict ``{label: PAT}`` of fitted methods (e.g. torus, supertoroid).
        points: the input point cloud ``(N,3)`` to overlay on the slices.
        out_path: PNG path.
    """
    cols = ["ground truth"] + list(pats.keys())
    ncol = len(cols)
    fig = plt.figure(figsize=(3.0 * ncol, 6.2))

    # ground-truth surface + slice
    gv, gf = _mc(shape.sdf, recon_res, recon_bound)
    ax = fig.add_subplot(2, ncol, 1, projection="3d"); _render_mesh(ax, gv, gf, GT_COLOR, view)
    ax.set_title(cols[0], fontsize=14)
    axs = fig.add_subplot(2, ncol, ncol + 1)
    _render_slice(axs, shape.sdf, points, slice_axis, slice_value, slice_extent, slice_res)

    for j, (label, pat) in enumerate(pats.items(), start=1):
        v, f = pat.reconstruct(res=recon_res, bound=recon_bound, neighbors=neighbors)
        v, f = _smooth(v, f, smooth_iters)
        ax = fig.add_subplot(2, ncol, 1 + j, projection="3d")
        _render_mesh(ax, v, f, OURS_COLOR, view)
        ax.set_title(label, fontsize=14)
        axs = fig.add_subplot(2, ncol, ncol + 1 + j)
        _render_slice(axs, lambda x, neighbors=neighbors, _p=pat: _p.sdf(x, neighbors=neighbors),
                      points, slice_axis, slice_value, slice_extent, slice_res, neighbors=neighbors)

    if npoints_label is not None:
        fig.text(0.012, 0.04, f"# points: {npoints_label}", fontsize=13, style="italic")
    if suptitle:
        fig.suptitle(suptitle, fontsize=15, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.05, hspace=0.02)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
