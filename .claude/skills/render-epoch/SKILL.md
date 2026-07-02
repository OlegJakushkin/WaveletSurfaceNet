---
name: render-epoch
description: After a training epoch (or anytime the waveshape checkpoint advances), render and show the PerceiverWaveNet surface model on the four regimes we track — favourites, thin/open objects, noise robustness (bunny 0–20%), and real indoor/outdoor scans. Runs render_suite.py on the 4090 box against the latest checkpoint (meshed at 128³), pulls the four PNG panels back, and displays them. Use when the user asks to see progress / renders / "how's it doing" after an epoch.
---

# render-epoch — after-epoch render suite

Produces four panels from the **latest** `assets/waveshape_latest.pt` (the restored paper2
**PerceiverWaveNet**: mixed base + smax corner head + far-field clamp, trained at 42³ and **always meshed at
128³** by `render_suite.py` via `WV.load_at_res(ck, res=128)`):

1. **favourites** — bunny, teapot, sphere, torus, cube, knurl — IoU vs GT
2. **thin / open** — airplane, guitar, bottle, chair (MN40) — IoU vs GT
3. **noise** — bunny reconstructed from 0/2/5/10/15/20 % Gaussian-noised clouds — IoU vs clean GT
4. **scenes** — real indoor + outdoor scans (no GT) — connected-component (#parts) count

GPU-only, so this **runs on the box**, inside the running training container (shares the GPU), never on the
Windows node. The container is `wstrain` from the **`pat-fvdb` image + `fvdb` conda env** (has torch + skimage +
trimesh + matplotlib pip-installed at launch) — so render via `conda run -n fvdb python`, NOT bare `python`.
Connection / reachability troubleshooting: see the **rdp-4090** skill (`.env` has key + host). `$BOXREPO` below =
the repo checkout on the box mounted at `/work` (currently `C:/fvdb/pat`).

## Run it
```
K=C:/Users/olegj/.ssh/pat_4090; H=Administrator@192.168.0.11
C=wstrain                                  # the running waveshape training container (name it this at launch)
# 1. render on the box inside the training container (shares the GPU; 128^3 recon is B=1, a few GB headroom)
ssh -i $K -o StrictHostKeyChecking=no $H \
  "docker exec -w /work -e MPLBACKEND=Agg $C conda run --no-capture-output -n fvdb python -u render_suite.py"
# 2. pull the four panels back to the scratchpad and Read/show them
for p in favourites thin noise scenes; do
  scp -i $K -o StrictHostKeyChecking=no "$H:C:/fvdb/pat/renders/suite_$p.png" "<scratchpad>/suite_$p.png"
done
```
Then **Read** each `suite_*.png` and show it to the user. The suite prints per-shape IoU / #parts to stdout —
relay the headline numbers. A specific checkpoint / eval-res can be rendered with
`-e CKPT=assets/<name>.pt -e EVAL_RES=128`.

## If the box is a training-only checkout (files missing)
`render_suite.py` is self-contained but needs, on the box under `$BOXREPO` (`C:/fvdb/pat`):
- `render_suite.py`, `assets/teapot.obj`, the MN40 classes it lists (`airplane bottle chair guitar` present),
  and `baselines_ext/scans/{indoor,outdoor}.npz` for the scenes panel.
- If any are absent, `scp` them from the local repo (same relative paths). The scan npz are ~96 KB each.
  Trim `THIN`/`FAV` in `render_suite.py` to only shapes present on the box (MN40 there is a subset).

## Notes
- Heavier than the built-in per-epoch `wsn_fav`/`wsn_val` renders, so it's an **on-demand** check, not wired
  into every epoch of `train.py`.
- Reads whatever `assets/waveshape_latest.pt` currently holds — run it right after an `epoch N/…` line in
  `docker logs wstrain` to snapshot that epoch. For best-by-val weights use `-e CKPT=assets/waveshape.pt`
  (its stored `epoch` may be an EARLIER epoch since selection freezes at best val).
- All compute on the GPU; never render on CPU (project rule).
