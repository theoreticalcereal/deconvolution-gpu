#!/usr/bin/env bash
set -euo pipefail

readonly REGISTRY=${REGISTRY:-git.biohpc.swmed.edu:5050/dean-lab}
readonly IMAGE_NAME=${IMAGE_NAME:-ctaslm2-deconvolution}
readonly TAG=${TAG:-0.1.0}
readonly IMAGE=${REGISTRY}/${IMAGE_NAME}:${TAG}
readonly CONTAINER_URI="docker://${IMAGE}"

# Default Astrocyte workflow_containers URI:
# docker://git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.0

if ! id -un >/dev/null 2>&1; then
  echo "ERROR: current UID is not resolvable by id -un." >&2
  echo "Singularity fails before registry access when the UID has no passwd/SSSD entry." >&2
  exit 3
fi

if command -v module >/dev/null 2>&1; then
  module load singularity/3.9.9 || true
fi

if ! command -v singularity >/dev/null 2>&1; then
  echo "ERROR: singularity is not available on PATH." >&2
  exit 4
fi

if [[ -n ${REGISTRY_USERNAME:-} && -n ${REGISTRY_PASSWORD:-} ]]; then
  export SINGULARITY_DOCKER_USERNAME="${REGISTRY_USERNAME}"
  export SINGULARITY_DOCKER_PASSWORD="${REGISTRY_PASSWORD}"
else
  echo "WARNING: REGISTRY_USERNAME/REGISTRY_PASSWORD are not set." >&2
  echo "Private registry images may fail with unauthorized/not found errors." >&2
fi

echo "Checking ${CONTAINER_URI}"
singularity inspect "${CONTAINER_URI}"
echo "Singularity can inspect ${CONTAINER_URI}"
