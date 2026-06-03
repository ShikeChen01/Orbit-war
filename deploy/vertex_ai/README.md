# Vertex AI training deployment (for later)

> Status: **scaffold only**. We train locally for now. This folder is wired up so we
> can move PPO training to a Vertex AI custom job (single GPU to start, scalable later)
> with minimal changes. Nothing here runs as part of local development.

## How it will work

Vertex AI "custom training" runs a **container** you provide on Google-managed GPUs.
The flow:

1. Package this repo into a Docker image whose entrypoint is `scripts/train.py`.
2. Push the image to Artifact Registry.
3. Submit a Vertex `CustomJob` that runs the image on a GPU machine
   (e.g. `n1-standard-8` + 1× `NVIDIA_TESLA_T4`/`L4`).
4. Checkpoints are written to a GCS bucket (`--save-dir gs://<bucket>/runs`).

## Prerequisites (when we're ready)

- gcloud CLI — **installed** (SDK 569.0.0). Authenticate:
  - `! gcloud auth login`
  - `! gcloud config set project <PROJECT_ID>`
  - `! gcloud auth application-default login`
- A GCP project with the **Vertex AI** and **Artifact Registry** APIs enabled.
- A GCS bucket for checkpoints.
- To build the image you need **either**:
  - **Docker** (not currently installed locally), **or**
  - **Cloud Build** (no local Docker needed) — recommended here:
    `gcloud builds submit --tag <IMAGE_URI> .`

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | CUDA + torch image that runs `scripts/train.py`. |
| `build_and_push.sh` | Build via Cloud Build (or Docker) and push to Artifact Registry. |
| `submit_job.sh` | Submit the Vertex `CustomJob`. |
| `job_config.yaml` | Worker pool spec (machine type, GPU, image, args). |
| `.gcloudignore` | Keep `.venv/`, `runs/`, reference dumps out of the build context. |

## Quick start (later)

```bash
# 0. one-time
export PROJECT_ID=my-project REGION=us-central1 BUCKET=gs://my-bucket
export REPO=orbit-wars IMAGE=us-central1-docker.pkg.dev/$PROJECT_ID/$REPO/trainer:latest
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION

# 1. build + push (Cloud Build; run from repo ROOT)
bash deploy/vertex_ai/build_and_push.sh

# 2. launch training
bash deploy/vertex_ai/submit_job.sh
```

## Notes / decisions to make later

- **GPU**: T4 (cheap) is fine for this small policy; L4/A10 if we vectorize envs.
- **Throughput**: the bottleneck is the pure-Python kaggle simulation, not the GPU.
  Before scaling to expensive GPUs, parallelize env stepping (multiprocessing /
  multiple worker replicas writing to a shared buffer) — that's a code change in
  `train/ppo.py`, the abstraction layer doesn't change.
- **Hyperparameter tuning**: Vertex Vizier can sweep `TrainConfig` fields.
