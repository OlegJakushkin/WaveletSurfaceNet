---
name: latest-renderings
description: Answer "show me latest renderings" (also "latest renders", "show me the latest model output", "render the latest model") for the PerceiverWaveNet surface model. Fetches the LATEST trained checkpoint on the 4090 box, runs the 4-panel render suite (favourites / thin / noise sweep / indoor-outdoor scans) against it at 128³, pulls the PNGs to a stable local folder, shows them inline, AND opens that folder in Windows Explorer. Use whenever the user asks to see the latest renderings / renders / model output.
---

# latest-renderings — render the latest model and open the folder

End-to-end answer to "show me latest renderings": **fetch latest checkpoint → render suite → show inline → open Explorer.**

"Latest" = the most recent WEIGHTS = `assets/waveshape_latest.pt` (the restored paper2 **PerceiverWaveNet**:
mixed base + smax corner head + far-field clamp). `render_suite.py` loads it with `WV.load_at_res(ck, res=128)`
so the 42³-trained net is **always meshed at 128³**. This is the suite's default `CKPT`, so no override is
needed. (For best-by-val weights instead, pass `-e CKPT=assets/waveshape.pt` — note its stored `epoch` may be
an EARLIER epoch than the last, since selection freezes it at the best val.)

GPU-only → this renders ON THE BOX, inside the running training container (`wstrain`, `pat-fvdb` image + `fvdb`
conda env with matplotlib pip-installed at launch) — render via `conda run -n fvdb python`, NOT bare `python`.
Connection (SSH key + host) lives in the **rdp-4090** skill's `.env`. Do the four steps below IN ORDER; do not
skip the Explorer step — it is the point of this skill.

## Steps
```
K=C:/Users/olegj/.ssh/pat_4090; H=Administrator@192.168.0.11
C=wstrain                                                  # running waveshape training container
LOCAL="C:/work/Points_as_supertoroids/renders/latest"     # stable local view folder

# 0. container up? (the training container also serves render via `docker exec`; if down, start it per rdp-4090)
ssh -i $K -o StrictHostKeyChecking=no $H "docker ps --format \"{{.Names}}\"" | findstr $C

# 1. render the LATEST checkpoint on the box (default CKPT = assets/waveshape_latest.pt, meshed at 128^3)
ssh -i $K -o StrictHostKeyChecking=no $H \
  "docker exec -w /work -e MPLBACKEND=Agg $C conda run --no-capture-output -n fvdb python -u render_suite.py"

# 2. pull the four panels to the local view folder (create it first)
mkdir -p "$LOCAL"
for p in favourites thin noise scenes; do
  scp -i $K -o StrictHostKeyChecking=no "$H:C:/fvdb/pat/renders/suite_$p.png" "$LOCAL/suite_$p.png"
done
```
Then, in the agent:
3. **Read** each `renders/latest/suite_*.png` and show it inline. Relay the headline numbers the suite printed to
   stdout — the `loaded ... epoch E val V` line (say WHICH checkpoint/epoch the user is seeing) and the per-shape
   IoU / #parts.
4. **Open Explorer** at the folder (PowerShell):
   `Start-Process explorer.exe "C:\work\Points_as_supertoroids\renders\latest"`

## Notes
- Always state the `epoch`/`val` from the `loaded ...` line so the user knows which checkpoint they're viewing.
- ~1–2 min on an idle GPU; a bit more sharing the GPU with a live training run (128³ recon is B=1).
- Box is a training-only checkout — needs `render_suite.py`, `assets/teapot.obj`, the MN40 subset
  (airplane/bottle/car/chair/guitar) and `baselines_ext/scans/{indoor,outdoor}.npz`; `scp` any missing from the
  local repo (see the **render-epoch** skill, which produces the same four panels without opening Explorer).
- `renders/latest/` is a derived view folder — safe to gitignore. All compute on the GPU; never render on CPU (project rule).
