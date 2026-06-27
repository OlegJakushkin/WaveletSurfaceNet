"""Tests for the analytic SDF primitives used as exact ground truth."""

import numpy as np
import pytest

from pat.shapes import Plane, RoundedBox, Sphere, Torus


@pytest.mark.parametrize("shape", [
    Sphere(0.7), Torus(0.6, 0.22), Plane(extent=1.0), RoundedBox(),
])
def test_surface_samples_lie_on_zero_level_set(shape):
    rng = np.random.default_rng(0)
    pts, nrm = shape.sample_surface(500, rng)
    assert np.abs(shape.sdf(pts)).max() < 1e-3
    # sampled normals agree with the SDF gradient
    g = shape.normal(pts)
    cos = np.sum(g * nrm, axis=1)
    assert np.median(cos) > 0.95


@pytest.mark.parametrize("shape", [Sphere(0.7), Torus(0.6, 0.22), Plane()])
def test_sdf_is_eikonal(shape):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.2, 1.2, (400, 3))
    g = shape.normal(x)                         # = grad(sdf)/|grad(sdf)|
    # finite-difference gradient magnitude ~ 1 for a true SDF
    eps = 1e-3
    grad = np.zeros_like(x)
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        grad[:, k] = (shape.sdf(x + d) - shape.sdf(x - d)) / (2 * eps)
    assert np.abs(np.linalg.norm(grad, axis=1) - 1.0).mean() < 0.05


def test_sphere_sign_convention():
    sph = Sphere(1.0)
    assert sph.sdf(np.array([[0.0, 0, 0]]))[0] < 0      # inside negative
    assert sph.sdf(np.array([[2.0, 0, 0]]))[0] > 0      # outside positive
    assert abs(sph.sdf(np.array([[1.0, 0, 0]]))[0]) < 1e-9
