#!/usr/bin/env bash
set -euo pipefail

readonly REGISTRY=${REGISTRY:-git.biohpc.swmed.edu:5050/dean-lab}
readonly IMAGE_NAME=${IMAGE_NAME:-ctaslm2-deconvolution}
readonly TAG=${TAG:-0.1.0}
readonly IMAGE=${REGISTRY}/${IMAGE_NAME}:${TAG}
readonly CONTAINER_URI="docker://${IMAGE}"
readonly MATLAB_BIND=${MATLAB_BIND:-/home1/apps/MATLAB:/home1/apps/MATLAB}
readonly MATLAB_FALLBACK=${MATLAB_FALLBACK:-/home1/apps/MATLAB/R2024a/bin/matlab}

# Default Astrocyte workflow_containers URI:
# docker://git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.0

if ! id -un >/dev/null 2>&1; then
  echo "ERROR: current UID is not resolvable by id -un." >&2
  echo "Singularity fails before registry access when the UID has no passwd/SSSD entry." >&2
  exit 3
fi

if command -v module >/dev/null 2>&1; then
  module load singularity/3.9.9 || true
  module load matlab/2024a || true
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

echo "Checking MATLAB visibility inside ${CONTAINER_URI}"
singularity exec --bind "${MATLAB_BIND}" "${CONTAINER_URI}" sh -lc "
  set -e
  resolved_matlab_bin=''
  for candidate in matlab '${MATLAB_FALLBACK}'; do
    if [ -n \"\${candidate}\" ] && command -v \"\${candidate}\" >/dev/null 2>&1; then
      resolved_matlab_bin=\"\$(command -v \"\${candidate}\")\"
      break
    elif [ -n \"\${candidate}\" ] && [ -x \"\${candidate}\" ]; then
      resolved_matlab_bin=\"\${candidate}\"
      break
    fi
  done
  if [ -z \"\${resolved_matlab_bin}\" ]; then
    echo 'ERROR: MATLAB executable not visible inside container.' >&2
    echo 'Checked command -v matlab and ${MATLAB_FALLBACK}.' >&2
    exit 5
  fi
  echo \"MATLAB resolved inside container: \${resolved_matlab_bin}\"
  \"\${resolved_matlab_bin}\" -batch \"disp('matlab -batch container check ok')\"
"
echo "MATLAB is visible inside ${CONTAINER_URI}"
