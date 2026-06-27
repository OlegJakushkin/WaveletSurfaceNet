# Honest comparison harness

Runs our unified mixed-base model against **real public-library reconstruction baselines** (no
reimplementations) and produces the comparison figures in `paper/figs/`. All methods receive the *same*
oriented point clouds; metrics and runtimes are measured in one process.

## Baselines (implementation → method → citation)

| name | implementation | method | citation |
|------|----------------|--------|----------|
| **SPSR** | Open3D `create_from_point_cloud_poisson` | Screened Poisson | Kazhdan & Hoppe 2013; Open3D: Zhou et al. 2018 |
| **BPA**  | Open3D `create_from_point_cloud_ball_pivoting` | Ball-Pivoting | Bernardini et al. 1999 |
| **tori** | `tori/pat` (this repo) | per-point torus blend | Feng et al. 2026 |
| **ours** | `waveshape` (this repo) | unified mixed-base field | — |

**Not included (honestly):** point-cloud generalized/fast **winding number** (Jacobson 2013; Barill 2018) — the
`libigl` Python binding here exposes only the *mesh* winding number `fast_winding_number(V,F,Q)`, not the
point-cloud variant, and we do not reimplement methods. The signed-heat method, NN-VIPSS, SSPD, etc. have no
drop-in public Python library and are likewise omitted rather than faked.

## Metric

We report the **F-score at distance threshold τ = 0.05** (`core.fscore`), the standard surface-reconstruction
metric: *precision* penalises spurious/over-filled surface, *recall* penalises holes. We switched to it because
raw Chamfer is misleading here — it ranks BPA's holey point-interpolation "best". τ = 0.05 matches our R-grid /
ε-band scale (below that, F-score measures *resolution*, not correctness); our resolution-free field is queried
at R = 128 for a fair sub-voxel comparison to the mesh baselines. We also report connected-component count
(watertightness) and runtime.

## Run it (GPU, Docker)

```bash
docker build -t waveshape-compare -f compare/Dockerfile .   # bakes open3d + libigl + pymeshlab onto waveshape:latest
docker compose run --rm compare compare/fig_baselines.py    # qualitative grid + compare/metrics.json
docker compose run --rm compare compare/fig_charts.py        # quality / watertightness / time bar charts
docker compose run --rm compare compare/fig_noise.py         # noise robustness
docker compose run --rm compare compare/fig_sparsity.py      # sparsity robustness
```

## Honest summary of results (4096 points, F-score @ τ=0.05)

- **Closed solids:** ours competitive with SPSR (≈93 vs 99), far above tori (69).
- **Open shells:** ours best among *usable* methods (≈81); SPSR over-fills into blobs (≈59); BPA scores high but
  produces non-watertight ~2200-component fragment soups.
- **Speed:** ours ≈2× faster than SPSR (2.0 s vs 3.5 s) and emits a signed field directly (baselines emit only a mesh).
- **Robustness — where we do _not_ win:** SPSR's global Poisson smoothing is more robust to heavy noise (≥5%) and
  extreme sparsity (≤256 pts); our anchored start trusts the raw points and our gate needs density.
