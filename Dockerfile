# CUDA-enabled image for WaveletSurfaceNet (the unified mixed-base surface model):
# generate meshes from point clouds and train, with no host-side Python/CUDA setup.
#
# Base already ships GPU PyTorch (cu124) matching the released checkpoint.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# System libs trimesh/scikit-image occasionally want for mesh IO.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# Python deps not in the base image (torch is already present with CUDA).
RUN pip install --no-cache-dir \
        "numpy>=1.26" "scipy>=1.11" "trimesh>=4.0" "scikit-image>=0.22" \
        "matplotlib>=3.7" "pytest>=7.0"

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace

WORKDIR /workspace
COPY . /workspace

# Default: show GPU + generate.py usage. Override per service in docker-compose (generate / train / shell).
CMD ["bash", "-lc", "python -c 'import torch;print(\"CUDA:\", torch.cuda.is_available())' && python generate.py --help"]
