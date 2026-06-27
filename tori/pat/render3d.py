"""Proper offscreen 3D rendering of SDF fields.

``matplotlib``'s ``plot_trisurf`` has no real lighting and broken depth-sorting -> ugly, flat, aliased
surfaces.  This renders the marching-cubes surface with a REAL engine:

* **pyvista / VTK** when available -- smooth-shaded, lit, anti-aliased; headless-safe on Colab via
  ``start_xvfb`` (install with ``pip install pyvista`` + ``apt-get install xvfb libgl1-mesa-glx``);
* otherwise a **LightSource-shaded matplotlib ``Poly3DCollection``** fallback -- still far better than
  ``plot_trisurf`` (true per-face diffuse shading) and dependency-free, so it always works.

Same call shape as :func:`pat.splat.render_comparison`, so the notebook just swaps the import.
"""

from __future__ import annotations

import numpy as np


def _march(sdf_fn, res=96, bound=1.0):
    """Marching-cubes a callable SDF on a ``res^3`` grid -> ``(verts, faces)`` (or ``None`` if no surface)."""
    from skimage import measure
    lin = np.linspace(-bound, bound, res)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    vol = np.asarray(sdf_fn(grid)).reshape(res, res, res)
    if not (vol.min() < 0 < vol.max()):
        return None
    v, f, _, _ = measure.marching_cubes(vol, level=0.0)
    return v / (res - 1) * (2 * bound) - bound, f


_PV_READY = [False]


def _pyvista():
    """Import pyvista and one-time-init headless rendering (xvfb + SSAA); ``None`` if unavailable."""
    try:
        import pyvista as pv
    except Exception:
        return None
    if not _PV_READY[0]:
        try:
            pv.start_xvfb()                                        # Colab headless display
        except Exception:
            pass
        try:
            pv.global_theme.anti_aliasing = "ssaa"
        except Exception:
            pass
        _PV_READY[0] = True
    return pv


def render_meshes(named_meshes, path, *, title="", color="#7f9bb8", size=(1120, 560)):
    """Render N named ``(label, verts, faces)`` meshes side by side (pyvista else matplotlib).  Use this
    to render a KNOWN mesh DIRECTLY (exact, fast) -- no marching-cubes-of-an-SDF voxelization."""
    return _render(list(named_meshes), path, title, color, size)


def render_fields(named_fns, path, *, res=96, bound=1.0, title="", color="#7f9bb8", size=(1120, 560)):
    """Render N named SDFs ``[(label, sdf_fn), ...]`` side by side to ``path`` (pyvista else matplotlib)."""
    surfs = []
    for label, fn in named_fns:
        m = _march(fn, res, bound)
        surfs.append((label, m[0] if m else None, m[1] if m else None))
    return _render(surfs, path, title, color, size)


def _render(surfs, path, title, color, size):
    pv = _pyvista()
    if pv is not None:
        try:
            pl = pv.Plotter(shape=(1, len(surfs)), off_screen=True, window_size=size, border=False)
            pl.set_background("white")
            for i, (label, v, f) in enumerate(surfs):
                pl.subplot(0, i)
                if v is not None:
                    faces = np.hstack([np.full((len(f), 1), 3, np.int64), f.astype(np.int64)]).ravel()
                    pl.add_mesh(pv.PolyData(v, faces), color=color, smooth_shading=True,
                                specular=0.25, specular_power=12, ambient=0.35, diffuse=0.75)
                pl.add_text(label, font_size=10, color="black", position="upper_edge")
                pl.view_isometric(); pl.camera.zoom(1.3)
            if title:
                pl.subplot(0, 0); pl.add_text(title, font_size=9, color="gray", position="lower_left")
            pl.screenshot(path); pl.close()
            return path
        except Exception:
            pass                                                  # fall through to matplotlib

    _render_matplotlib(surfs, path, title, size)
    return path


def _render_matplotlib(surfs, path, title, size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(size[0] / 110, size[1] / 110))
    light = np.array([0.4, -0.6, 0.72]); light /= np.linalg.norm(light)
    base = np.array([0.42, 0.55, 0.66])
    for i, (label, v, f) in enumerate(surfs):
        ax = fig.add_subplot(1, len(surfs), i + 1, projection="3d")
        if v is not None:
            tris = v[f]                                            # (F,3,3)
            fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9, None)
            shade = np.clip((fn @ light) * 0.5 + 0.5, 0.28, 1.0)   # per-face diffuse term
            cols = np.clip(base[None] * shade[:, None], 0, 1)
            pc = Poly3DCollection(tris, facecolors=cols, edgecolor="none")
            ax.add_collection3d(pc)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_title(label, fontsize=10); ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1)); ax.view_init(20, -55)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def render_comparison(shape, splat, path, title="", res=96, edge_gamma=0.0):
    """Ground-truth vs splat reconstruction, properly shaded.  Drop-in for
    :func:`pat.splat.render_comparison` (same signature)."""
    M = getattr(splat, "M", "")
    return render_fields(
        [("ground truth", shape.sdf),
         (f"supertoroid splats ({M} splats)", lambda g: splat.sdf(g, edge_gamma=edge_gamma))],
        path, res=res, title=title)
