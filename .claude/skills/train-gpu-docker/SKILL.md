---
name: train-gpu-docker
description: Train / retrain / fine-tune the Points-as-(Super)Tori models (the torus and supertoroid CoeffNets). Training is ALWAYS done inside Docker and ALWAYS on the GPU — never directly on the host, never on CPU. Use whenever the user asks to train, retrain, continue training, fine-tune, or rebuild the models.
---

# Train PAT models — Docker + GPU only

**Hard policy. No exceptions.**

1. **Always train inside Docker.** Never run `python train_gpu.py` (or any training) directly
   on the host. The only sanctioned entry point is the Compose `train` service.
2. **Always train on the GPU.** `train_gpu.py` aborts if CUDA is unavailable, and the Compose
   service reserves the NVIDIA GPU. If the GPU is not visible inside the container, **stop and
   report it** — do not fall back to CPU.
3. **Two models, one dataset.** Training always fits BOTH the plain-torus model
   (`assets/pat_torus.pt`) and the supertoroid model (`assets/pat_supertoroid.pt`) on the same
   cached dataset, so they are comparable.

## How to run

```bash
# 1) Build the CUDA image once (or after Dockerfile/deps change):
docker compose build

# 2) Confirm the GPU is visible inside the container (must print CUDA: True and the device):
docker compose run --rm train python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3) Train (writes assets/pat_torus.pt and assets/pat_supertoroid.pt via the bind mount):
docker compose run --rm train
#   override knobs, e.g. more assets / epochs / bigger batch:
docker compose run --rm train python train_gpu.py --assets 20000 --epochs 8 --batch 16
```

The trainer uses **>=10k assets**, **>=5 epochs**, **per-epoch point + noise
re-randomization** (a random 50% of each cloud's points are noised, re-rolled every
epoch; the rest stay noiseless), and a **50/50 noisy/clean eval split**.  It is
batched on the GPU (GPU kNN + batched blend) so the run is feasible on one laptop GPU.

The bind mount in `docker-compose.yml` makes outputs land in the host `./assets` folder.

## Progress logging

`train_gpu.py` streams progress to stdout (and Docker shows it live), with `PYTHONUNBUFFERED=1`:

- a startup line with total steps, epochs × examples, and the GPU name;
- an **intra-epoch** line every `--log-every` steps: `step k/N`, running torus/supertoroid
  loss, throughput (`it/s`) and ETA;
- an **end-of-epoch** line: mean train loss for each model and the **val-torus-err** (mean abs
  SDF error reconstructing a default torus) for each model — this is the acceptance metric;
- best checkpoints are saved whenever val-torus-err improves.

To watch a backgrounded run: `docker compose run --rm train 2>&1 | tee renders/train.log`, then
`tail -f renders/train.log`.

## Acceptance

Training is "good enough" when **val-torus-err is below ~0.01 for both models** (error invisible
to the naked eye on a default torus). Verify afterwards with:

```bash
docker compose run --rm test pytest -q tests/test_validation.py
```

## Do NOT

- Do not run training on the host Python, or in the Colab notebook, when the user asked to train
  *here* — those are CPU-or-different-GPU paths and violate this skill.
- Do not silently switch to CPU if CUDA is missing — surface the problem instead.
