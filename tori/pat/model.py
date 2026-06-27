"""The learned coefficient predictor (Sec. 4.3, Fig. 15).

A small Transformer is applied to each k-neighborhood and predicts the six
polynomial coefficients of the local height function (and, in the supertoroid
variant, two extra squareness logits).  Self-attention lets the network learn
which neighbors matter for the local fit -- itself a form of kernel regression.

The *same* class is imported by the Colab training notebook and by the inference
pipeline (:class:`pat.pat.PAT`), so a checkpoint trained in Colab plugs straight
into every code path and test.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import core
from .neighbors import neighborhood_features, rescale_coeffs


class CoeffNet(nn.Module):
    """Predict torus (or supertoroid) coefficients from point neighborhoods.

    Args:
        d_embed:   token embedding width (paper: 128).
        n_layers:  number of transformer encoder layers (paper: 8).
        n_heads:   attention heads (paper: 8).
        d_ff:      transformer feed-forward width (paper: 512).
        supertoroid: if True, also output two squareness logits -> ``p_tube``,
                     ``p_ring``, initialized at ``p = 2`` (an ordinary torus), so
                     training starts from the exact paper model and can only
                     specialize away from it.
        p_max:       optional cap on the squareness exponent (e.g. ``6``) for
                     training stability; see :func:`pat.core.raw_to_p`.  ``None``
                     (default) keeps the original unbounded ``p``, so older
                     checkpoints reconstruct identically from their saved config.
    """

    def __init__(self, d_embed=128, n_layers=8, n_heads=8, d_ff=512,
                 supertoroid=False, dropout=0.0, p_max=None):
        super().__init__()
        self.supertoroid = supertoroid
        self.p_max = p_max
        self.embed = nn.Linear(6, d_embed)
        layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        n_out = 8 if supertoroid else 6
        self.head = nn.Linear(d_embed, n_out)
        self._init_head()

    def _init_head(self):
        """Initialize so the central point starts as a tangent sphere / torus.

        Paper: ``a00=a01=a10=a11=0, a02=a20=-0.5`` (a sphere tangent to the point).
        For the supertoroid we additionally bias the two squareness logits to
        ``p = 2`` (a circular cross-section), i.e. an ordinary torus.
        """
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            b = torch.zeros(self.head.out_features)
            b[core.A02] = -0.5
            b[core.A20] = -0.5
            if self.supertoroid:
                b[6] = core.P2_RAW
                b[7] = core.P2_RAW
            self.head.bias.copy_(b)

    # ------------------------------------------------------------------ #
    def forward(self, nbr_pos, nbr_nrm):
        """Map neighborhoods to physical coefficients (and squareness exponents).

        Args:
            nbr_pos: ``(B, M, 3)`` neighbor positions (column 0 = central point).
            nbr_nrm: ``(B, M, 3)`` neighbor normals.

        Returns:
            coeffs: ``(B, 6)`` polynomial coefficients in physical coordinates.
            sigma:  ``(B,)`` neighborhood scale (median neighbor distance).
            sq:     ``(B, 2)`` exponents ``[p_tube, p_ring]`` if supertoroid else None.
        """
        feats, sigma, _, _ = neighborhood_features(nbr_pos, nbr_nrm)
        tok = self.embed(feats)                      # (B,M,d)
        enc = self.encoder(tok)                      # (B,M,d)
        out = self.head(enc[:, 0, :])                # central token -> coeffs
        a_raw = out[:, :6]
        coeffs = rescale_coeffs(a_raw, sigma)
        sq = None
        if self.supertoroid:
            sq = core.raw_to_p(out[:, 6:8], p_max=self.p_max)   # (B,2) -> p in [1, p_max]
        return coeffs, sigma, sq


def predict_params(model, nbr_pos, nbr_nrm):
    """Run the model and fit torus params -- convenience for the notebook/loss."""
    coeffs, sigma, sq = model(nbr_pos, nbr_nrm)
    center = nbr_pos[:, 0, :]
    nrm = nbr_nrm[:, 0, :]
    params = core.coeffs_to_torus(center, nrm, coeffs)
    return params, sq
