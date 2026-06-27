"""Points as (Super)Tori -- a from-scratch reimplementation.

A Python reimplementation of Feng, Gkioulekas & Crane, *"Points as Tori: Fast
Pointwise Signed Distance for Point Clouds"* (ACM TOG 2026), extended from tori to
**supertoroids** (L^p super-ellipse cross-sections) for sharper local fits.

Public surface:

* :class:`pat.pat.PAT`           -- point cloud + normals -> callable SDF / mesh.
* :class:`pat.model.CoeffNet`    -- the learned coefficient predictor (Sec. 4.3).
* :mod:`pat.core`                -- the differentiable torus/supertoroid math.
* :mod:`pat.shapes`              -- analytic SDF primitives (exact ground truth).
* :mod:`pat.baselines`           -- SSPD / signed Hopf-Cole comparisons.
* :mod:`pat.optimize`            -- torus-vs-supertoroid fitting and comparison.
* :mod:`pat.viz`                 -- polyscope visualization (lazy, headless-safe).
"""

from . import baselines, core, neighbors, shapes, train  # noqa: F401
from .model import CoeffNet  # noqa: F401
from .pat import PAT  # noqa: F401

__all__ = ["PAT", "CoeffNet", "core", "shapes", "baselines", "neighbors", "train"]
__version__ = "0.1.0"
