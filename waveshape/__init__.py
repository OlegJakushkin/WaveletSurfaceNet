"""WaveletSurfaceNet -- a unified mixed-base surface field from a point cloud.

A template-free, grid-free, resolution-free point transformer that *emits* the multi-scale (Haar wavelet)
coefficients of a distance field directly from a point cloud, and a single **mixed-base** model whose
analytic per-point gate selects a *signed* field for closed solids and an *unsigned* band for thin/open
shells -- one model, one forward, both bases (see the paper in ``paper/``).

Entry points:

* :mod:`waveshape.wavelet`         -- the model: ``PerceiverWaveNet``, ``load_at_res``, meshing helpers.
* :mod:`waveshape.eval3d`          -- sampling clouds from meshes and the procedural example shapes.
* :mod:`waveshape.shapes`          -- analytic primitives + ``normalize_to_unit_cube``.

For end-to-end "points/mesh in -> mesh out", use the ``generate.py`` CLI at the repo root.
"""
from . import wavelet, eval3d, shapes  # noqa: F401

__all__ = ["wavelet", "eval3d", "shapes"]
__version__ = "1.0.0"
