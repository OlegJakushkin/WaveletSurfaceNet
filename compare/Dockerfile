# Comparison image: our CUDA stack + the public reconstruction libraries used as honest baselines.
#   docker build -t waveshape-compare -f compare/Dockerfile .
FROM waveshape:latest
# pymeshlab's meshing plugins (APSS / RIMLS / Poisson) need these GL/X libs to load:
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopengl0 libegl1 libglu1-mesa libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir open3d libigl pymeshlab
