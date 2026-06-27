# WaveletSurfaceNet — a unified mixed-base surface field from a point cloud

Turn a point cloud (points **+** outward normals) into a clean mesh with a single **unified mixed-base**
model. A resolution-free point transformer *emits* the multi-scale (Haar-wavelet) coefficients of a distance
field directly from the points — **no analytic primitive, no input grid, no shape template** — and an
analytic **per-point gate** picks the right base everywhere:

- **closed solids** → a **signed** field → meshed at level 0 → a crisp watertight **solid**;
- **thin / open shells** → an **unsigned** band → meshed at level 0 → a clean **shell**.

One model, one forward, both bases — the base chosen where it matters, per point, at **zero added
parameters**. A bunny comes back a solid body with thin shell ears; a chair comes back a clean shell.

![the principle](paper/figs/principle.png)

> *Neither single base suffices: an unsigned field can't make a solid (holey hollow cube); a signed field
> over-fills an open shell (exploded chair). The mixed model gets both right.*

The full write-up is **[`paper/paper2.pdf`](paper/paper2.pdf)**.

---

## Quickstart (Docker + GPU)

Requires Docker with the **NVIDIA Container Toolkit** (a CUDA GPU). No host Python/CUDA setup needed.

```bash
docker compose build                                              # one-time
docker compose run --rm generate --shape bunny --out out/bunny.obj
```

The released checkpoint is `assets/waveshape_mixed.pt`; output meshes land in `out/` on the host.

### The examples

```bash
# closed solids  (signed path -> crisp solids)
docker compose run --rm generate --shape bunny  --out out/bunny.obj
docker compose run --rm generate --shape teapot --out out/teapot.obj

# open shell     (unsigned path -> clean shell)
docker compose run --rm generate --shape chair  --out out/chair.obj

# context + dense SUPER-RESOLUTION on the knurled cylinder
#   --region   : reconstruct one surface box as part of the full pass (coarse)
#   --superres : same box, box-normalised with the whole shape as global context (several x the detail)
docker compose run --rm generate --shape knurl --region   --out out/knurl_region.obj
docker compose run --rm generate --shape knurl --superres --out out/knurl_superres.obj
```

### Your own input

```bash
# from any mesh (a cloud is sampled from its surface)
docker compose run --rm generate --mesh path/to/shape.obj --out out/shape.obj

# from a raw point cloud: a .npy of shape (N, 6) = xyz + outward unit normal per row
docker compose run --rm generate --points my_cloud.npy --out out/mine.obj
```

`generate.py` flags: `--shape {cube,sphere,torus,bunny,teapot,chair,knurl}` | `--mesh PATH` | `--points PATH`,
plus `--out PATH`, `--region`, `--superres`, `--res N` (output lattice, default 64), `--ckpt PATH`. Run
`docker compose run --rm generate --help` for all options. (To run on a bare host instead of Compose:
`python generate.py --shape bunny --out out/bunny.obj`.)

---

## Repository layout

| path | what |
|------|------|
| `waveshape/`  | the model package — `wavelet.py` (the `PerceiverWaveNet` + `load_at_res` + meshing), plus the geometry support it needs. Self-contained: numpy + torch. |
| `generate.py` | the CLI above: points / mesh (+ optional region box) → mesh. |
| `train.py`    | train the model on ModelNet40 (see below). |
| `assets/`     | the released checkpoint `waveshape_mixed.pt` + the example inputs (bunny, teapot, one ModelNet chair). |
| `paper/`      | the paper — `paper2.tex` / `paper2.pdf` (+ figures). Build with `sh paper/render.sh`. |
| `tori/`       | the original **Points-as-Tori** model (the baseline we compare against), isolated and runnable — see [`tori/README.md`](tori/README.md). |
| `wavescene/`  | a separate monocular-image → 3D-scene model (its own project; kept here for convenience). |

---

## Training (optional — the checkpoint already ships)

Training needs `data/ModelNet40/` on the host. The unified mixed-base model:

```bash
docker compose run --rm train --base mixed --out waveshape_mixed --region --epochs 6
```

`--base {signed,unsigned,mixed}` selects the field the model is anchored on and trained toward; `--region`
trains the context+dense super-resolution path. Checkpoints are written to `assets/<out>.pt` (best by
validation) and `assets/<out>_latest.pt`.

---

## The tori baseline

The `tori/` folder holds the from-scratch reimplementation of *Points as Tori* (Feng, Gkioulekas & Crane,
ACM TOG 2026) extended to supertoroids — the fixed-primitive baseline. It is fully self-contained; see
[`tori/README.md`](tori/README.md) to run it.

## License

See [`LICENSE`](LICENSE).
