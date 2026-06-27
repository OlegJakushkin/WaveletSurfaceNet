"""Core differentiable math for *Points as Tori* (PAT) and its supertoroid extension.

Everything here is written once, in PyTorch, so the *exact same code* is used at
inference time (tests, reconstruction, visualization) and at training time
(the Colab notebook).  All functions are vectorized over a leading batch and are
fully differentiable, which is what lets us back-propagate the L1 + eikonal loss
of Equation 27 of the paper.

References (equation numbers refer to Feng, Gkioulekas & Crane, "Points as Tori:
Fast Pointwise Signed Distance for Point Clouds", ACM TOG 2026):

* Torus SDF ....................... Eq. 23 (Quilez's circle-SDF-applied-twice).
* Per-point torus fitting ......... Sec. 4.1 + Appendix C (Monge-patch curvatures).
* Self-normalized blending ........ Eq. 1 / Eq. 25 with auto lambda from Eq. 26.

The *supertoroid* extension (our addition, the paper only uses tori) replaces the
two circular cross-sections of the torus by L^p "super-ellipse" cross-sections,
controlled by two squareness exponents.  At p = 2 (e = 1) it reduces *exactly* to
the paper's torus, so the trained torus model is a strict special case.
"""

from __future__ import annotations

import torch

EPS = 1e-9

# Index convention for the six polynomial coefficients a_{n,m} (n = power of s,
# m = power of t) of the local height function Q_i(s, t) = sum_{n,m} a_{n,m} s^n t^m.
# We pack them in the fixed order used throughout the code and the network output:
A00, A01, A10, A11, A02, A20 = 0, 1, 2, 3, 4, 5
COEFF_NAMES = ["a00", "a01", "a10", "a11", "a02", "a20"]


# --------------------------------------------------------------------------- #
#  Local orthonormal frames
# --------------------------------------------------------------------------- #
def local_basis(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic orthonormal tangent basis ``(s, t)`` for unit normals ``n``.

    Uses the branchless orthonormal-basis construction of Duff et al. (2017),
    which is continuous away from ``n = -z`` and, crucially, *deterministic*:
    the network is trained with the same fixed basis it sees at inference time
    (Sec. 4.3, "We use a fixed orthonormal basis for each point").

    ``n`` has shape ``(..., 3)``; returns ``s, t`` of the same shape.
    """
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    sign = torch.where(nz >= 0, torch.ones_like(nz), -torch.ones_like(nz))
    a = -1.0 / (sign + nz)
    b = nx * ny * a
    s = torch.stack([1.0 + sign * nx * nx * a, sign * b, -sign * nx], dim=-1)
    t = torch.stack([b, sign + ny * ny * a, -ny], dim=-1)
    return s, t


def _normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return v / (v.norm(dim=dim, keepdim=True) + EPS)


# --------------------------------------------------------------------------- #
#  L^p (super-ellipse) norm
# --------------------------------------------------------------------------- #
def lp_norm2(x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Numerically-stable 2D L^p norm ``(|x|^p + |y|^p)^(1/p)``.

    ``p`` may be a scalar or broadcastable tensor.  ``p = 2`` gives the ordinary
    Euclidean norm, recovering the circular cross-section of a torus.  ``p -> inf``
    gives the Chebyshev (square) cross-section.  We factor out the max magnitude
    so the power never overflows.
    """
    ax, ay = x.abs(), y.abs()
    m = torch.maximum(ax, ay).clamp_min(EPS)
    # Clamp the ratios away from exactly 0: ``base ** p`` has a NaN gradient w.r.t.
    # ``p`` at ``base == 0`` (it is ``base**p * log(base)`` = ``0 * -inf``).  A tiny
    # positive floor makes ``log(base)`` finite while changing the value negligibly.
    rx = (ax / m).clamp_min(1e-12)
    ry = (ay / m).clamp_min(1e-12)
    return m * (rx ** p + ry ** p) ** (1.0 / p)


# --------------------------------------------------------------------------- #
#  Torus & supertoroid signed distance (Eq. 23 and our generalization)
# --------------------------------------------------------------------------- #
def torus_sdf(x, c, u, R, r):
    """SDF of a single torus, Eq. 23.

    ``x``  query points ``(..., 3)``;
    ``c``  center ``(3,)`` or broadcastable; ``u`` unit axis of revolution;
    ``R``  major radius (>= r); ``r`` minor radius.
    """
    rel = x - c
    axial = (rel * u).sum(-1)
    perp = rel - axial.unsqueeze(-1) * u
    perp_radial = perp.norm(dim=-1)
    dx = perp_radial - R
    return torch.sqrt(dx * dx + axial * axial + EPS) - r


def supertoroid_sdf(x, c, u, ea, R, r, p_tube, p_ring):
    """Approximate SDF of a supertoroid (our extension).

    The torus is generalized in two independent ways:

    * ``p_ring`` controls the cross-section of the *ring* (the donut hole), in the
      plane perpendicular to ``u`` with squareness axis ``ea``;
    * ``p_tube`` controls the cross-section of the *tube*.

    Both default to ``p = 2`` (circle), which makes this identical to
    :func:`torus_sdf`.  For ``p != 2`` this is the radial / "inside-outside" L^p
    approximation of distance, exact for ``p = 2`` and a smooth, differentiable
    proxy otherwise -- which is all the blending framework (itself an
    approximation) requires.
    """
    rel = x - c
    axial = (rel * u).sum(-1)
    eb = torch.cross(u, ea, dim=-1)
    a = (rel * ea).sum(-1)
    b = (rel * eb).sum(-1)
    ring_radius = lp_norm2(a, b, p_ring)
    dx = ring_radius - R
    return lp_norm2(dx, axial, p_tube) - r


def superellipsoid_sdf(x, c, u, ea, ha, hb, hc, e):
    """Approximate SDF of a **superellipsoid** (superquadric) — NO torus, NO cutout.

    A single solid primitive whose squareness exponent ``e`` sweeps the whole family the
    project cares about, from one knob:

    * ``e < 1``  -> **pinched** / star (concave faces);
    * ``e = 1``  -> octahedron (**diamond**);
    * ``e = 2``  -> ellipsoid (**circle** cross-section / sphere);
    * ``2 < e``  -> **rounded square / box**;
    * ``e -> inf`` -> **cube**.

    ``c`` center, ``(u, ea, eb=u x ea)`` an orthonormal frame, ``(ha, hb, hc)`` half-extents
    along ``(ea, eb, u)``, ``e`` the exponent (scalar or broadcastable).  The field is the
    radial inside--outside value ``F = (|x/ha|^e + |y/hb|^e + |z/hc|^e)^{1/e}`` rescaled to a
    smooth, monotone distance proxy ``(F - 1) * min(ha,hb,hc)`` — exact enough for the blend
    (Eq. 25), which is itself an approximation.  ``e=2`` is the exact ellipsoid radial form.
    """
    rel = x - c
    eb = torch.cross(u, ea, dim=-1)
    px = (rel * ea).sum(-1).abs() / ha.clamp_min(EPS)
    py = (rel * eb).sum(-1).abs() / hb.clamp_min(EPS)
    pz = (rel * u).sum(-1).abs() / hc.clamp_min(EPS)
    # factor out the max so ``base ** e`` never overflows and has a finite gradient at 0
    m = torch.stack([px, py, pz], dim=-1).amax(dim=-1).clamp_min(EPS)
    rx = (px / m).clamp_min(1e-12); ry = (py / m).clamp_min(1e-12); rz = (pz / m).clamp_min(1e-12)
    F = m * (rx ** e + ry ** e + rz ** e) ** (1.0 / e)
    scale = torch.minimum(torch.minimum(ha, hb), hc)
    return (F - 1.0) * scale


# --------------------------------------------------------------------------- #
#  Polynomial coefficients  ->  fitted torus / supertoroid parameters (Sec. 4.1)
# --------------------------------------------------------------------------- #
def coeffs_to_torus(p, n, a, kappa_floor: float = 0.05):
    """Map the six polynomial coefficients to torus parameters (Sec. 4.1, App. C).

    Args:
        p: central point positions ``(..., 3)``.
        n: unit normals at ``p`` ``(..., 3)``.
        a: coefficients ``(..., 6)`` ordered ``[a00, a01, a10, a11, a02, a20]``.
        kappa_floor: minimum principal-curvature magnitude.  Curvatures below this
            are treated as flat: the radii are capped at ``1/kappa_floor`` and a
            near-flat umbilic point falls back to a large tangent *sphere*, which
            is the correct planar limit (Sec. 4.1, Fig. 2).  Without this, a true
            plane produces a degenerate, arbitrarily-oriented giant torus.

    Returns dict with ``center (...,3)``, ``axis (...,3)``, ``ea (...,3)``
    (in-plane squareness axis = minor principal direction), ``R (...,)``,
    ``r (...,)`` and ``sign (...,)`` such that the signed per-point function is
    ``g_i = sign * torus_sdf(x, center, axis, R, r)`` (Eq. 24).
    """
    s, t = local_basis(n)
    a00 = a[..., A00]
    a01 = a[..., A01]
    a10 = a[..., A10]
    a11 = a[..., A11]
    a02 = a[..., A02]
    a20 = a[..., A20]

    A = torch.sqrt(1.0 + a01 * a01 + a10 * a10 + EPS)        # App. C
    H = (a02 * (1 + a10 * a10) + a20 * (1 + a01 * a01) - a11 * a10 * a01) / A**3
    K = (4 * a02 * a20 - a11 * a11) / A**4
    # Clamp the argument to a small positive floor (not 0): at umbilic points
    # (sphere) ``H^2 - K == 0`` and ``d sqrt/dx`` is infinite, producing NaN
    # gradients during training.  The floor (1e-6 -> disc ~ 1e-3) is far below the
    # curvature floor, so flat detection and the radii are unaffected.
    disc = torch.sqrt((H * H - K).clamp_min(1e-6))
    kappa_p = H + disc                                       # kappa_+
    kappa_m = H - disc                                       # kappa_-

    # Principal directions w_+- (App. C), lifted to 3D as v_+-.
    def vdir(kappa):
        wx = kappa * a10 * a01 * A - a11
        wy = 2 * a20 - kappa * (1 + a10 * a10) * A
        v = wx.unsqueeze(-1) * (s + a10.unsqueeze(-1) * n) \
            + wy.unsqueeze(-1) * (t + a01.unsqueeze(-1) * n)
        return v

    v_p = vdir(kappa_p)
    v_m = vdir(kappa_m)

    # n*(0,0): unit normal of the polynomial surface at the central point (Eq. 21).
    n_star = _normalize(n - a10.unsqueeze(-1) * s - a01.unsqueeze(-1) * t)
    q_star = p + a00.unsqueeze(-1) * n                       # Q*(0,0) = p + a00 n

    # kappa_min = smaller |kappa|, kappa_max = larger |kappa|; pick matching dir.
    plus_is_min = kappa_p.abs() < kappa_m.abs()
    kappa_min = torch.where(plus_is_min, kappa_p, kappa_m)
    kappa_max = torch.where(plus_is_min, kappa_m, kappa_p)
    v_min = torch.where(plus_is_min.unsqueeze(-1), v_p, v_m)
    v_min = _normalize(v_min)

    # Floor the curvature magnitudes so radii stay bounded by 1/kappa_floor.
    amax = kappa_max.abs().clamp_min(kappa_floor)
    amin = kappa_min.abs().clamp_min(kappa_floor)
    r = 1.0 / amax
    sign_kk = torch.sign(kappa_p * kappa_m)
    R = (1.0 / amin - sign_kk * r).clamp_min(0.0)            # major radius >= 0

    # sign(T): +1 iff kappa_max < 0 (Eq. 22).
    signT = torch.where(kappa_max < 0, torch.ones_like(r), -torch.ones_like(r))

    center = q_star - (signT / amin).unsqueeze(-1) * n_star
    axis = _normalize(torch.cross(n_star, v_min, dim=-1))

    # Planar limit: a near-flat umbilic point (both curvatures below the floor) has
    # an ill-defined principal frame, so the torus above would be arbitrarily
    # oriented.  Replace it by a large tangent sphere (R=0), which is the correct
    # planar limit and is orientation-free.
    flat = (kappa_max.abs() < kappa_floor)
    rad = torch.full_like(r, 1.0 / kappa_floor)
    sphere_center = q_star - n_star / kappa_floor
    fu = flat.unsqueeze(-1)
    center = torch.where(fu, sphere_center, center)
    R = torch.where(flat, torch.zeros_like(R), R)
    r = torch.where(flat, rad, r)
    signT = torch.where(flat, torch.ones_like(signT), signT)

    return {
        "center": center, "axis": axis, "ea": v_min,
        "R": R, "r": r, "sign": signT, "q_star": q_star,
        "kappa_min": kappa_min, "kappa_max": kappa_max, "n_star": n_star,
    }


def g_torus(x, params):
    """Signed per-point function ``g_i`` for a fitted torus (Eq. 24).

    ``x`` may carry extra leading query dims; ``params`` tensors carry the
    per-point dims.  Broadcasting is the caller's responsibility (see
    :func:`blend`).
    """
    sdf = torus_sdf(x, params["center"], params["axis"], params["R"], params["r"])
    return params["sign"] * sdf


def g_supertoroid(x, params, p_tube, p_ring):
    """Signed per-point function for a fitted supertoroid (our extension)."""
    sdf = supertoroid_sdf(x, params["center"], params["axis"], params["ea"],
                          params["R"], params["r"], p_tube, p_ring)
    return params["sign"] * sdf


# --------------------------------------------------------------------------- #
#  Squareness parameterization: raw network output  ->  exponent p >= 1
# --------------------------------------------------------------------------- #
def raw_to_p(raw: torch.Tensor, p_max: float | None = None) -> torch.Tensor:
    """Map an unconstrained scalar to a super-ellipse exponent ``p = 1 + softplus(raw)``.

    This keeps ``p >= 1`` (a convex, valid cross-section) and is smooth.  The
    bias :data:`P2_RAW` below makes ``p = 2`` (an ordinary torus) the
    initialization, mirroring the paper's "init = sphere" choice.

    ``p_max`` (e.g. ``6``) optionally *caps* the exponent for training stability.
    The L^p SDF's sensitivity to ``p`` **saturates** as ``p`` grows -- a cross
    section that is already nearly square barely changes when it gets boxier --
    so an *unbounded* ``p`` is a near-dead direction the data loss cannot pull
    back: on boxy assets it only ratchets up and destabilizes training (the
    supertoroid's epoch-4 val spike).  Capping it (paired with an early
    ``p -> 2`` square-regularizer in the trainer) keeps ``p`` in a stable,
    well-conditioned range.  ``p > 6`` is already visually a rounded box, so the
    cap costs no useful expressiveness.  ``p_max=None`` (the default) preserves
    the original unbounded behavior, so pre-cap checkpoints and the training-free
    optimizer are unaffected.
    """
    p = 1.0 + torch.nn.functional.softplus(raw)
    if p_max is not None:
        p = p.clamp(max=float(p_max))
    return p


# raw value for which raw_to_p(raw) == 2  (softplus(raw) == 1).
P2_RAW = float(torch.log(torch.expm1(torch.tensor(1.0))))


# --------------------------------------------------------------------------- #
#  Self-normalized convolutional distance / blending (Eq. 1, 25, 26)
# --------------------------------------------------------------------------- #
def blend(x, points, g_vals, C: float = 64.0):
    """Blend per-point signed functions into a global SDF (Eq. 25 with Eq. 26).

    Args:
        x:       query points ``(Q, 3)``.
        points:  point-cloud positions ``(N, 3)``.
        g_vals:  per-point signed values at the queries, ``(Q, N)`` -- i.e.
                 ``g_vals[q, i] = g_i(x_q)``.
        C:       precision constant; ``exp(-C)`` must stay above machine eps.
                 The paper uses ``C = 64`` for single precision (Eq. 26).

    Returns the blended SDF ``phi(x)`` of shape ``(Q,)``.

    The shift ``sigma_x`` and screening ``lambda_x`` are chosen *per query* from
    Eq. 26 so that the exponentials never under/overflow:
        sigma_x = 1/2 max_i ||x - p_i||,   lambda_x = C / sigma_x.
    """
    d = torch.cdist(x, points)                       # (Q, N)  Euclidean distances
    sigma = 0.5 * d.max(dim=1, keepdim=True).values  # (Q, 1)  Eq. 26
    lam = C / (sigma + EPS)                           # screening, Eq. 26
    # The self-normalized ratio (Eq. 25) is invariant to a per-query shift of the
    # exponent, so we shift by the *nearest* distance: the largest weight is then
    # exactly 1 and the denominator never underflows, even for far queries.
    dmin = d.min(dim=1, keepdim=True).values
    w = torch.exp(-lam * (d - dmin))                 # max weight == 1 per query
    num = (w * g_vals).sum(dim=1)
    den = w.sum(dim=1).clamp_min(EPS)
    return num / den


def blend_batched(x, points, g_vals, C: float = 64.0, wmul=None):
    """Batched version of :func:`blend` for training many clouds at once on the GPU.

    Args:
        x:       query points ``(B, Q, 3)``.
        points:  point-cloud positions ``(B, N, 3)``.
        g_vals:  per-point signed values ``(B, Q, N)``.
        wmul:    optional per-(query, point) weight multiplier ``(B, Q, N)`` folded
                 into the blend weights -- used by the **one-sided** superellipse prior
                 to down-weight each primitive's back side (so a closed solid behaves as
                 a one-sided plane/corner patch) without altering ``g`` (no CSG faces).

    Returns ``(B, Q)``.  Same self-normalized, dmin-stabilized formula as
    :func:`blend`, vectorized over the leading cloud dimension ``B``.
    """
    d = torch.cdist(x, points)                       # (B, Q, N)
    sigma = 0.5 * d.amax(dim=2, keepdim=True)        # (B, Q, 1)
    lam = C / (sigma + EPS)
    dmin = d.amin(dim=2, keepdim=True)
    w = torch.exp(-lam * (d - dmin))
    if wmul is not None:
        w = w * wmul
    num = (w * g_vals).sum(dim=2)
    den = w.sum(dim=2).clamp_min(EPS)
    return num / den
