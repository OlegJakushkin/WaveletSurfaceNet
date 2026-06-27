"""Polyscope visualization helpers (Nicholas Sharp's ``polyscope`` -- "nmwsharp").

Everything here is import-safe: polyscope is imported lazily and no window opens
until you explicitly call :func:`show`.  This keeps the unit tests headless.

Typical use::

    from pat import PAT, viz
    from pat.shapes import Torus
    import numpy as np
    pts, nrm = Torus().sample_surface(2000, np.random.default_rng(0))
    pat = PAT(pts, nrm)
    viz.init()
    viz.register_point_cloud("input", pts, nrm)
    viz.register_reconstruction("recon", pat)
    viz.register_sdf_slice("slice", pat)
    viz.show()
"""

from __future__ import annotations

import numpy as np


def init():
    import polyscope as ps
    ps.init()
    ps.set_up_dir("z_up")
    return ps


def register_point_cloud(name, points, normals=None, color=(0.1, 0.5, 0.9)):
    import polyscope as ps
    pc = ps.register_point_cloud(name, np.asarray(points), color=color,
                                 radius=0.005)
    if normals is not None:
        pc.add_vector_quantity("normals", np.asarray(normals), enabled=False)
    return pc


def register_reconstruction(name, pat, res=64, bound=1.2, neighbors=64,
                            color=(0.9, 0.6, 0.2)):
    """Marching-cubes the PAT zero level set and register it as a surface mesh."""
    import polyscope as ps
    verts, faces = pat.reconstruct(res=res, bound=bound, neighbors=neighbors)
    return ps.register_surface_mesh(name, verts, faces, color=color)


def register_sdf_slice(name, pat, axis=2, value=0.0, extent=1.2, res=128,
                       neighbors=64):
    """Register an axis-aligned plane colored by the PAT SDF (a slice of the field)."""
    import polyscope as ps
    lin = np.linspace(-extent, extent, res)
    gx, gy = np.meshgrid(lin, lin, indexing="ij")
    flat = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid = np.insert(flat, axis, value, axis=1)
    vals = pat.sdf(grid, neighbors=neighbors).reshape(res, res)

    # build a flat quad mesh in the slicing plane
    V = np.zeros((res * res, 3))
    idx = [0, 1, 2]
    a, b = [i for i in idx if i != axis]
    V[:, a] = gx.ravel()
    V[:, b] = gy.ravel()
    V[:, axis] = value
    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            p0 = i * res + j
            faces.append([p0, p0 + 1, p0 + res])
            faces.append([p0 + 1, p0 + res + 1, p0 + res])
    mesh = ps.register_surface_mesh(name, V, np.asarray(faces))
    mesh.add_scalar_quantity("sdf", vals.ravel(), enabled=True, cmap="coolwarm",
                             isolines_enabled=True)
    return mesh


def register_tori(name, pat, max_tori=400, samples=24):
    """Draw the fitted tori (or supertoroids) as curve networks of their equators."""
    import polyscope as ps
    P = pat.params
    n = min(max_tori, pat.N)
    sel = np.linspace(0, pat.N - 1, n).astype(int)
    c = P["center"][sel].numpy()
    u = P["axis"][sel].numpy()
    ea = P["ea"][sel].numpy()
    R = P["R"][sel].numpy()
    nodes, edges = [], []
    th = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    off = 0
    for i in range(n):
        eb = np.cross(u[i], ea[i])
        ring = c[i][None] + R[i] * (np.cos(th)[:, None] * ea[i][None]
                                    + np.sin(th)[:, None] * eb[None])
        nodes.append(ring)
        e = np.stack([np.arange(samples), (np.arange(samples) + 1) % samples], axis=1) + off
        edges.append(e)
        off += samples
    return ps.register_curve_network(name, np.concatenate(nodes),
                                     np.concatenate(edges), radius=0.0015)


def show():
    import polyscope as ps
    ps.show()
