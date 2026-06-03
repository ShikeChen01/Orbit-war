# Vertex AI custom-training image for Orbit Wars PPO.
# Build context = repo ROOT:  docker build -f deploy/vertex_ai/Dockerfile -t <IMAGE> .
#
# Base already includes CUDA + cuDNN + a matching torch, so we don't reinstall torch
# from the CUDA index inside the container.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime deps first for better layer caching (skip torch: it's in the base).
COPY pyproject.toml ./
RUN pip install "kaggle-environments>=1.21" "gymnasium>=1.0" "numpy>=1.26"

# App code.
COPY orbit_wars_rl ./orbit_wars_rl
COPY scripts ./scripts

# Vertex passes training args after the entrypoint; --save-dir should be a gs:// path
# (GCS FUSE is mounted automatically) or use a bucket via gcsfuse.
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--total-steps", "5000000", "--device", "cuda", "--save-dir", "/gcs/CHANGE_ME/runs"]
