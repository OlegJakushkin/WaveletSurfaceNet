# Points as (Super)Tori — the baseline

A from-scratch reimplementation of **Feng, Gkioulekas & Crane, *"Points as Tori: Fast Pointwise Signed
Distance for Point Clouds"*** (ACM TOG 2026), extended from tori to **supertoroids** (Lᵖ super-ellipse
cross-sections) for sharper local fits. This is the **fixed-primitive baseline** that WaveletSurfaceNet (the
main model in the repo root) is compared against in the paper — it is kept here, isolated and self-contained,
so the release repo stays about the unified mixed-base model.

## Layout

| path | what |
|------|------|
| `pat/`     | the package — `pat.PAT` (point cloud + normals → callable SDF / mesh), `pat.model.CoeffNet` (the learned coefficient predictor), `pat.core` (the torus / supertoroid math), `pat.shapes`, `pat.baselines`. |
| `demo.py`, `demo_wavetori.py` | reconstruction demos. |
| `make_renders.py` | reconstruct meshes + render the torus-vs-supertoroid comparison figures. |
| `train_gpu.py`    | train the torus and supertoroid `CoeffNet`s. |
| `tests/`          | the test suite (`pytest -q`). |
| `assets/`         | pretrained checkpoints (`pat_torus.pt`, `pat_supertoroid.pt`, …) + example inputs. |

## Run it (GPU, via the repo's Docker image)

Build the image once from the repo root (`docker compose build`), then run tori scripts with **this folder**
mounted as the working directory:

```bash
# from the repo root, on Linux/macOS (PowerShell: replace ${PWD} with the absolute path):
docker run --rm --gpus all -v "${PWD}/tori:/workspace" -w /workspace -e PYTHONPATH=/workspace \
  waveshape:latest python make_renders.py --model assets/pat_supertoroid.pt --res 112

# reconstruct one shape to an OBJ
docker run --rm --gpus all -v "${PWD}/tori:/workspace" -w /workspace -e PYTHONPATH=/workspace \
  waveshape:latest python -m pat.meshgen --shape torus --model assets/pat_supertoroid.pt --out torus.obj

# tests
docker run --rm --gpus all -v "${PWD}/tori:/workspace" -w /workspace -e PYTHONPATH=/workspace \
  waveshape:latest pytest -q
```

Interactive `polyscope` visualization (`pat.viz`) is optional and not in the base image; `pip install
polyscope` if you want it. Headless rendering (`make_renders.py`) uses matplotlib and needs nothing extra.
