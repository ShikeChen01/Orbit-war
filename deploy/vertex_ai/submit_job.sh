#!/usr/bin/env bash
# Submit a Vertex AI custom training job. Run from the REPO ROOT after build_and_push.sh.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${REGION:=us-central1}"
: "${IMAGE:?set IMAGE}"
JOB_NAME="${JOB_NAME:-orbitwars-ppo-$(date +%Y%m%d-%H%M%S)}"

# Render the worker-pool spec with the image substituted in.
TMP_SPEC="$(mktemp)"
sed "s#__IMAGE__#${IMAGE}#g" deploy/vertex_ai/job_config.yaml > "$TMP_SPEC"

echo "Submitting Vertex CustomJob: $JOB_NAME (region $REGION)"
gcloud ai custom-jobs create \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --display-name "$JOB_NAME" \
  --config "$TMP_SPEC"

echo "Track: gcloud ai custom-jobs list --region $REGION --project $PROJECT_ID"
