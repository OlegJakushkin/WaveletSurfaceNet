"""Smoke/invariant tests for the teacher (Stage A) + student (Stage B) amortized splat optimizer.

These lock in the hard-won correctness facts validated during the build:
* ``param_rows`` <-> ``from_rows`` round-trip and ``single_splat_sdf`` == ``_g_splat``;
* ``CloudShape`` GT occupancy respects holes (torus hole center is OUTSIDE);
* ``md_filled_volume`` (blend-sign occupancy) is ~0 for a self-fit and rises as splats are removed;
* GroupNet / FitNet shapes + the end-to-end ``reconstruct_amortized`` plumbing.
"""

import numpy as np
import torch

from pat import splat as S
from pat import teacher as T
from pat import student as ST


def _torus(P_n=800, R=0.55, r=0.20, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, P_n); ph = rng.uniform(0, 2 * np.pi, P_n)
    P = np.stack([(R + r * np.cos(ph)) * np.cos(th), (R + r * np.cos(ph)) * np.sin(th),
                  r * np.sin(ph)], 1).astype(np.float32)
    N = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], 1).astype(np.float32)
    return P, N


class _Torus:
    def __init__(self, R=0.55, r=0.20): self.R, self.r = R, r
    def sdf(self, q):
        q = np.asarray(q, float); xy = np.linalg.norm(q[:, :2], axis=1) - self.R
        return np.sqrt(xy ** 2 + q[:, 2] ** 2) - self.r


def test_param_rows_roundtrip_union_sdf():
    sp = S.SuperToroidSplats.from_rows(torch.randn(5, S.ROW_W) * 0.3)
    q = torch.randn(64, 3) * 0.5
    sp2 = S.SuperToroidSplats.from_rows(sp.param_rows())
    assert torch.allclose(sp.union_sdf(q), sp2.union_sdf(q), atol=1e-4)


def test_single_splat_sdf_matches_module():
    sp = S.SuperToroidSplats.from_rows(torch.randn(3, S.ROW_W) * 0.3)
    q = torch.randn(32, 3) * 0.5
    u, ea, eb, R, r, pt, pr, b = sp._params(); boxc = sp.center + sp.box_offset.clamp(-1, 1)
    g_mod = sp._g_splat(q[:, None, :], u, ea, eb, R, r, pt, pr, b, boxc).T   # (M,Q)
    g_fn = ST.single_splat_sdf(sp.param_rows(), q[None].expand(3, -1, -1))    # (M,Q)
    assert torch.allclose(g_mod, g_fn, atol=1e-4)


def test_cloudshape_respects_holes():
    P, N = _torus()
    cs = T.CloudShape(P, N, k_dense=20000)
    # the torus hole centre must be OUTSIDE the solid
    assert cs.sdf(np.zeros((1, 3)))[0] > 0
    occ = T.gt_occupancy(cs, res=40)
    # central voxel (the hole) must be empty in every antithetic offset
    assert not occ.reshape(4, 40, 40, 40)[:, 20, 20, 20].any()


def test_md_self_small_and_rises_on_deletion():
    P, N = _torus()
    cs = T.CloudShape(P, N, k_dense=20000)
    occ = T.gt_occupancy(cs, res=40)
    sp = S.fit_shape(_Torus(), P, N, n_init=24, steps=120, n_query=2000, prune_every=0, device="cpu")
    md_full = T.md_filled_volume(sp, occ, res=40, device="cpu")
    assert md_full < 0.05                                     # a converged fit is near the GT floor
    keep = torch.topk(sp.surface_ownership(P).sum(0), max(4, sp.M - 10)).indices
    sp_small = S.SuperToroidSplats.from_rows(sp.param_rows()[keep])
    md_small = T.md_filled_volume(sp_small, occ, res=40, device="cpu")
    assert md_small >= md_full - 1e-6                        # removing splats cannot improve coverage


def test_student_shapes_and_reconstruct():
    P, N = _torus(P_n=300)
    gn = ST.GroupNet(d_embed=32, n_layers=1, d_ff=64, d_g=8)
    fn = ST.FitNet(d_embed=32, n_layers=1, d_ff=64)
    npos, nnrm = ST.build_neighborhoods(P, N, k=12)
    seed, emb = gn(npos, nnrm)
    assert seed.shape == (300,) and emb.shape == (300, 8)
    assert torch.allclose(emb.norm(dim=1), torch.ones(300), atol=1e-4)
    chosen, assign = ST.group_points(seed, emb, npos[:, 0, :], nms_radius=0.15)
    assert len(chosen) >= 1 and assign.shape == (300,)
    sp, K = ST.reconstruct_amortized(P, N, gn, fn, k=12, min_group=2, device="cpu")
    assert K >= 1 and sp.M == K


def test_batch_blend_matches_single():
    from pat.teacher_batch import BatchSplats
    sp = S.SuperToroidSplats.from_rows(torch.randn(6, S.ROW_W) * 0.25)
    bs = BatchSplats(sp.param_rows()[None], torch.ones(1, 6, dtype=torch.bool), device="cpu")
    q = torch.randn(40, 3) * 0.6
    assert torch.allclose(sp.sdf_torch(q), bs.blend_sdf(q[None])[0], atol=1e-4)
    bs.alive[0, 2] = False                                   # killing a splat changes the blend
    assert (sp.sdf_torch(q) - bs.blend_sdf(q[None])[0]).abs().max() > 1e-5


def test_fit_teacher_batch_runs():
    from pat import teacher_batch as TB
    Ps = [_torus(seed=0)[0], _torus(R=0.5, r=0.18, seed=1)[0]]
    Ns = [_torus(seed=0)[1], _torus(R=0.5, r=0.18, seed=1)[1]]
    res = TB.fit_teacher_batch(Ps, Ns, m_init=20, md_target=0.02, res=36, steps_warm=40,
                               steps_refit=20, keep_schedule=(0.6, 0.4), min_keep=4,
                               k_dense=15000, device="cpu")
    assert len(res) == 2
    for sp, md, iou, status in res:
        assert sp.M >= 4 and np.isfinite(md) and status in ("ok", "hard")


def test_eval3d_voxel_free_metrics(tmp_path):
    from pat import eval3d as E
    shapes = E.canonical_shapes(bunny=False)                  # skip bunny (slow mesh) for the unit test
    assert len(shapes) == 4 and {s[0] for s in shapes} >= {"teapot", "hole+bolts plate", "diamond knurl"}
    name, gt, mesh = shapes[0]                                # cube+cylinder (analytic, exact GT)
    P, N = E.sample_cloud(mesh, n=400, seed=0)
    sp = S.fit_shape(_SDFShapeAdapter(gt), P, N, n_init=12, steps=60, n_query=1500, prune_every=0, device="cpu")
    m = E.proper_metrics(gt, sp, n=6000)                      # continuous, no voxel grid
    assert 0.0 <= m["iou"] <= 1.0 and np.isfinite(m["vol_err"])
    props = E.mesh_properties(mesh)
    assert props["faces"] > 0 and "thinness" in props
    out = tmp_path / "matrix.png"
    E.plot_metrics_matrix([dict(name=name, **m, **props)], str(out))
    assert out.exists()


class _SDFShapeAdapter:                                       # expose gt.sdf as a fit_shape `shape`
    def __init__(self, gt): self.gt = gt
    def sdf(self, q): return self.gt.sdf(q)


def test_render3d_produces_image(tmp_path):
    from pat import render3d as R3
    sp = S.fit_shape(_Torus(), *_torus(P_n=400), n_init=10, steps=40, n_query=800, prune_every=0, device="cpu")
    out = tmp_path / "r.png"
    R3.render_comparison(_Torus(), sp, str(out), title="t", res=40)   # pyvista if present else shaded mpl
    assert out.exists() and out.stat().st_size > 1000


def test_shard_version_autoregen(tmp_path):
    from pat import teacher_batch as TB
    P, N = _torus(P_n=600)
    out = str(tmp_path)
    kw = dict(m_init=10, m_max=20, grow_add=6, max_grow=1, res=32, steps_warm=20, steps_refit=10,
              keep_schedule=(0.6,), min_keep=4, k_dense=12000, device="cpu")
    r = TB.fit_and_cache_batch([P], [N], [3], out, **kw)
    assert r[0][1] in ("ok", "hard")                                # fresh -> generated
    p = T.shard_path(out, 3)
    assert torch.load(p, weights_only=False)["version"] == T.TEACHER_VERSION
    assert TB.fit_and_cache_batch([P], [N], [3], out, **kw)[0][1] == "cached"   # current -> skipped
    a = torch.load(p, weights_only=False); a["version"] = 0; torch.save(a, p)   # simulate OLD shard
    assert not T.shard_is_current(p) and T.count_stale_shards(out, [3]) == 1
    assert TB.fit_and_cache_batch([P], [N], [3], out, **kw)[0][1] != "cached"    # stale -> regenerated
    assert torch.load(p, weights_only=False)["version"] == T.TEACHER_VERSION


def test_groupnet_loss_decreases():
    P, N = _torus(P_n=300)
    # a synthetic 2-cluster owner label the position-aware net can learn
    owner = (P[:, 0] > 0).astype(np.int64)
    gn = ST.GroupNet(d_embed=32, n_layers=1, d_ff=64, d_g=8)
    opt = torch.optim.Adam(gn.parameters(), lr=2e-3)
    npos, nnrm = ST.build_neighborhoods(P, N, k=12)
    first = last = None
    for it in range(15):
        seed, emb = gn(npos, nnrm)
        loss, _ = ST.groupnet_loss(seed, emb, owner)
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 0: first = float(loss)
        last = float(loss)
    assert last < first                                      # position-aware grouping is learnable
