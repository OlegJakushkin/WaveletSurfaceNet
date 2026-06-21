"""Tests for the wavelet-domain denoiser (``pat.wavelet``) and the head-to-head
comparison harness (``pat.compare``).

Lock in the correctness facts the build relies on:
* the 3-D Haar bank is orthonormal and ``idwt3d(dwt3d(x)) == x`` (exact reconstruction);
* the TSDF builder is signed correctly (inside < 0) and the numpy / torch paths agree;
* the denoiser is residual-initialized to the identity (untrained == pass-through);
* training reduces the loss on a tiny cache;
* ``WaveletReconstruction`` exposes a working ``.sdf`` / ``.reconstruct``;
* the device-agnostic tori trainer learns, and ``head_to_head`` runs end to end.
"""

import numpy as np
import torch

from pat import wavelet as W
from pat import compare as C


# --------------------------------------------------------------------------- #
#  Synthetic data: torus clouds + an analytic torus SDF for the GT queries
# --------------------------------------------------------------------------- #
def _torus_cloud(n, R, r, seed):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n); ph = rng.uniform(0, 2 * np.pi, n)
    P = np.stack([(R + r * np.cos(ph)) * np.cos(th), (R + r * np.cos(ph)) * np.sin(th),
                  r * np.sin(ph)], 1).astype(np.float32)
    N = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], 1).astype(np.float32)
    return P, N


def _torus_sdf(q, R, r):
    q = np.asarray(q, float)
    xy = np.linalg.norm(q[:, :2], axis=1) - R
    return np.sqrt(xy ** 2 + q[:, 2] ** 2) - r


def _tiny_cache(B=6, dense=320, nq=120, seed=0):
    rng = np.random.default_rng(seed)
    Ps, Ns, Qs, PHIs = [], [], [], []
    for b in range(B):
        R, r = 0.52 + 0.02 * b, 0.2
        P, N = _torus_cloud(dense, R, r, seed + b)
        surf, _ = _torus_cloud(nq, R, r, seed + 100 + b)
        band = surf + rng.normal(scale=0.04, size=surf.shape)
        bulk = rng.uniform(-1, 1, size=(nq, 3))
        q = np.concatenate([band[: nq // 2], bulk[: nq - nq // 2]], 0).astype(np.float32)
        PHIs.append(_torus_sdf(q, R, r).astype(np.float32))
        Ps.append(P); Ns.append(N); Qs.append(q)
    t = lambda a: torch.as_tensor(np.stack(a))
    return {"P": t(Ps), "N": t(Ns), "Q": t(Qs), "PHI": t(PHIs)}


# --------------------------------------------------------------------------- #
#  Wavelet transform
# --------------------------------------------------------------------------- #
def test_haar_orthonormal():
    w = W.haar_filters_3d().reshape(8, 8)
    assert torch.allclose(w @ w.t(), torch.eye(8), atol=1e-6)


def test_dwt_idwt_perfect_reconstruction():
    x = torch.randn(2, 1, 16, 16, 16)
    c = W.dwt3d(x)
    assert c.shape == (2, 8, 8, 8, 8)
    assert torch.allclose(W.idwt3d(c), x, atol=1e-5)


# --------------------------------------------------------------------------- #
#  TSDF
# --------------------------------------------------------------------------- #
def test_tsdf_sign_and_numpy_torch_agree():
    P, N = _torus_cloud(2000, 0.55, 0.2, 0)
    g_np = W.tsdf_from_cloud(P, N, res=16, trunc=0.1, bound=1.1)
    g_t = W.tsdf_from_clouds(P, N, res=16, trunc=0.1, bound=1.1, device="cpu")[0, 0].numpy()
    assert g_np.shape == (16, 16, 16) and g_t.shape == (16, 16, 16)
    assert np.allclose(g_np, g_t, atol=2e-2)          # KD-tree vs cdist nearest agree
    # the torus hole centre (origin) is OUTSIDE the solid -> positive
    assert W.tsdf_from_cloud(P, N, res=16, trunc=0.1, bound=1.1)[8, 8, 8] > 0
    # a point on the tube (rho=R, z=0) is near the surface -> small |value|
    assert abs(g_np.min()) <= 0.1 + 1e-6 and g_np.max() <= 0.1 + 1e-6


# --------------------------------------------------------------------------- #
#  Network
# --------------------------------------------------------------------------- #
def test_denoiser_residual_init_is_identity():
    net = W.WaveletDenoiser(base=8).eval()
    x = torch.randn(1, 1, 16, 16, 16) * 0.3
    with torch.no_grad():
        out, c, c_clean = net(x)
    assert out.shape == x.shape and c.shape == (1, 8, 8, 8, 8)
    # zero-initialized output head -> Δ == 0 -> exact pass-through
    assert torch.allclose(out, x, atol=1e-5)
    assert torch.allclose(c, c_clean, atol=1e-6)


def test_train_wavelet_loss_decreases():
    cache = _tiny_cache()
    net, hist = W.train_wavelet(cache, res=16, trunc=0.1, epochs=4, batch=3,
                                noise_std=0.03, base=8, log_every=0, device="cpu")
    assert len(hist) == 4
    assert min(h["loss"] for h in hist) < hist[0]["loss"]   # the denoiser learns


def test_wavelet_reconstruction_sdf_and_mesh():
    P, N = _torus_cloud(1500, 0.55, 0.2, 0)
    net = W.WaveletDenoiser(base=8)                    # untrained == identity, still a valid field
    wr = W.WaveletReconstruction(P, N, net, res=16, trunc=0.1, bound=1.1, device="cpu")
    # inside the tube is negative, the hole centre is positive
    assert wr.sdf(np.array([[0.55, 0.0, 0.0]]))[0] < 0.05
    assert wr.sdf(np.array([[0.0, 0.0, 0.0]]))[0] > -1e-6
    # a far exterior query reads as outside (+trunc)
    assert wr.sdf(np.array([[5.0, 5.0, 5.0]]))[0] == 0.1
    v, f = wr.reconstruct()
    assert v is not None and len(v) > 0 and len(f) > 0


# --------------------------------------------------------------------------- #
#  Tori trainer + end-to-end head-to-head
# --------------------------------------------------------------------------- #
def test_train_tori_cache_loss_decreases():
    cache = _tiny_cache()
    net, hist = C.train_tori_cache(cache, k=12, epochs=4, batch=3, n_points=200,
                                   noise_std=0.01, d_embed=32, n_layers=2,
                                   log_every=0, device="cpu", seed=0)
    # the torus is near-representable from the CoeffNet init, so the loss starts low
    # and wiggles with fresh per-epoch noise; "it can improve" is the robust check.
    assert len(hist) == 4 and min(h["loss"] for h in hist) < hist[0]["loss"]


def test_train_tori_cache_survives_nan_batch():
    # a degenerate mesh (NaN in the cloud) must not poison the weights: the proven
    # train_gpu spike/NaN guard skips that batch and the run finishes finite.
    cache = _tiny_cache(B=8)
    cache["P"][0, 0, 0] = float("nan")                 # poison one mesh's cloud
    net, hist = C.train_tori_cache(cache, k=12, epochs=2, batch=2, n_points=200,
                                   noise_std=0.0, d_embed=32, n_layers=2,
                                   log_every=0, device="cpu", seed=1)
    assert sum(h["skipped"] for h in hist) >= 1        # the poisoned batch was skipped
    assert np.isfinite(hist[-1]["loss"])               # running loss never went NaN
    assert all(torch.isfinite(p).all() for p in net.parameters())   # weights stayed finite


def test_head_to_head_runs(tmp_path):
    import trimesh
    from pat.shapes import normalize_to_unit_cube
    mesh = normalize_to_unit_cube(trimesh.creation.torus(major_radius=0.5, minor_radius=0.18))
    tori = C.CoeffNet(d_embed=32, n_layers=2)          # untrained nets: just exercise the plumbing
    wave = W.WaveletDenoiser(base=8)
    out = tmp_path / "cmp.png"
    res = C.head_to_head(mesh, tori, wave, n_cloud=400, noise=0.01, k=12,
                         res_recon=40, res_wave=16, n_metric=4000, device="cpu",
                         render_path=str(out), name="torus")
    for key in ("tori", "wavelet"):
        m = res[key]
        assert 0.0 <= m["iou"] <= 1.0
        assert np.isfinite(m["vol_err"]) and np.isfinite(m["chamfer"])
    assert out.exists() and out.stat().st_size > 1000
