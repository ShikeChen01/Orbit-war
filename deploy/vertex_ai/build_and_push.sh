#!/usr/bin/env bash
# Build the trainer image and push to Artifact Registry. Run from the REPO ROOT.
# Uses Cloud Build (no local Docker needed). Set env vars first (see README).
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${IMAGE:?set IMAGE, e.g. us-central1-docker.pkg.dev/\$PROJECT_ID/orbit-wars/trainer:latest}"

echo "Building $IMAGE via Cloud Build..."
gcloud builds submit \
  --project "$PROJECT_ID" \
  --tag "$IMAGE" \
  --gcs-log-dir "${BUILD_LOG_DIR:-}" \
  -f deploy/vertex_ai/Dockerfile \
  .

echo "Pushed $IMAGE"
# Local Docker alternative (if Docker is installed):
#   docker build -f deploy/vertex_ai/Dockerfile -t "$IMAGE" .
#   docker push "$IMAGE"
