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
