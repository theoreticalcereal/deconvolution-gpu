#!/usr/bin/env bash
set -euo pipefail

readonly REGISTRY=${REGISTRY:-git.biohpc.swmed.edu:5050/dean-lab}
readonly IMAGE_NAME=${IMAGE_NAME:-ctaslm2-deconvolution}
readonly TAG=${TAG:-0.1.2}
readonly REGISTRY_HOST=${REGISTRY%%/*}
readonly IMAGE_ROOT=$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
readonly REPOSITORY_ROOT=$(cd -P "${IMAGE_ROOT}/../../.." >/dev/null 2>&1 && pwd)
readonly SOURCE_ENV_PREFIX=${SOURCE_ENV_PREFIX:-${REPOSITORY_ROOT}/../decon_env}
readonly -a PODMAN_GLOBAL_ARGS=(--cgroup-manager=cgroupfs --events-backend=file)
readonly IMAGE=${REGISTRY}/${IMAGE_NAME}:${TAG}
readonly HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-30}
BUILD_ARGS=(--tag "${IMAGE}")

if ! [[ ${HEARTBEAT_SECONDS} =~ ^[1-9][0-9]*$ ]]; then
  echo "HEARTBEAT_SECONDS must be a positive integer." >&2
  exit 2
fi

run_with_heartbeat() {
  local label=$1
  shift

  printf '[%s] %s started\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${label}"
  "$@" &
  local command_pid=$!

  (
    while kill -0 "${command_pid}" 2>/dev/null; do
      sleep "${HEARTBEAT_SECONDS}"
      if kill -0 "${command_pid}" 2>/dev/null; then
        printf '[%s] %s still running (pid=%s)\n' \
          "$(date '+%Y-%m-%d %H:%M:%S')" "${label}" "${command_pid}"
      fi
    done
  ) &
  local heartbeat_pid=$!

  local status=0
  if wait "${command_pid}"; then
    status=0
  else
    status=$?
  fi
  kill "${heartbeat_pid}" 2>/dev/null || true
  wait "${heartbeat_pid}" 2>/dev/null || true

  if (( status == 0 )); then
    printf '[%s] %s completed\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${label}"
  else
    printf '[%s] %s failed with exit status %s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${label}" "${status}" >&2
  fi
  return "${status}"
}

if [[ ${NO_CACHE:-0} == 1 || ${NO_CACHE:-false} == true ]]; then
  BUILD_ARGS=(--no-cache "${BUILD_ARGS[@]}")
fi

# Default published image: git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.2

if [[ -z ${REGISTRY_USERNAME:-} || -z ${REGISTRY_PASSWORD:-} ]]; then
  echo "REGISTRY_USERNAME and REGISTRY_PASSWORD are required to publish ${IMAGE}" >&2
  exit 2
fi

if command -v module >/dev/null 2>&1; then
  module load singularity/3.9.9 || true
  module load mamba/2.3.0
fi

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba/2.3.0 is required to resolve the container environment." >&2
  exit 3
fi

if [[ $(mamba --version) != 2.3.0 ]]; then
  echo "Expected mamba 2.3.0 from Lmod, found $(mamba --version)." >&2
  exit 3
fi

BUILD_CONTEXT=$(mktemp -d "${TMPDIR:-/tmp}/deconvolution-image-context.XXXXXX")
cleanup() {
  rm -rf -- "${BUILD_CONTEXT}"
}
trap cleanup EXIT

cp -a "${IMAGE_ROOT}/." "${BUILD_CONTEXT}/"

if [[ ! -x ${SOURCE_ENV_PREFIX}/bin/python ]]; then
  echo "Validated source environment is missing: ${SOURCE_ENV_PREFIX}" >&2
  echo "Set SOURCE_ENV_PREFIX to a compatible Conda environment." >&2
  exit 4
fi

echo "Validating source environment: ${SOURCE_ENV_PREFIX}"
"${SOURCE_ENV_PREFIX}/bin/python" -c "
import cucim, cupy, numba, numpy, scipy, tifffile, zarr
actual = {
    'numpy': numpy.__version__,
    'numba': numba.__version__,
    'scipy': scipy.__version__,
    'tifffile': tifffile.__version__,
    'zarr': zarr.__version__,
    'cupy': cupy.__version__,
    'cucim': cucim.__version__,
}
expected = {
    'numpy': '1.26.4',
    'numba': '0.59.1',
    'scipy': '1.15.2',
    'tifffile': '2022.10.10',
    'zarr': '2.18.3',
    'cupy': '13.6.0',
    'cucim': '23.06.00',
}
mismatches = [
    f'{name}: expected {expected[name]}, found {actual[name]}'
    for name in expected
    if actual[name] != expected[name]
]
if mismatches:
    raise SystemExit('source environment version mismatch: ' + '; '.join(mismatches))
print('source environment versions verified:', actual)
"

readonly LOCK_PATH="${BUILD_CONTEXT}/conda-linux-64.lock"
echo "Exporting explicit lock with Lmod mamba $(mamba --version)"
{
  printf '@EXPLICIT\n'
  mamba list \
    --prefix "${SOURCE_ENV_PREFIX}" \
    --explicit \
    --md5 \
    | awk '/^(https?|file):\/\// && $0 !~ /\/(pycudadecon|cudadecon)-/ { print }'
} > "${LOCK_PATH}"

if [[ $(head -n 1 "${LOCK_PATH}") != "@EXPLICIT" ]] \
  || [[ $(wc -l < "${LOCK_PATH}") -le 1 ]] \
  || ! grep -q '/cupy-' "${LOCK_PATH}" \
  || ! grep -q '/cucim-' "${LOCK_PATH}" \
  || grep -Eq '/(pycudadecon|cudadecon)-' "${LOCK_PATH}"; then
  echo "Failed to generate a valid explicit Conda lock file." >&2
  exit 4
fi

echo "Explicit lock contains $(( $(wc -l < "${LOCK_PATH}") - 1 )) packages"

printf '%s' "${REGISTRY_PASSWORD}" \
  | podman "${PODMAN_GLOBAL_ARGS[@]}" login "${REGISTRY_HOST}" \
      --username "${REGISTRY_USERNAME}" \
      --password-stdin

run_with_heartbeat "Podman image build" \
  podman "${PODMAN_GLOBAL_ARGS[@]}" build "${BUILD_ARGS[@]}" "${BUILD_CONTEXT}"
podman "${PODMAN_GLOBAL_ARGS[@]}" push "${IMAGE}"

export SINGULARITY_DOCKER_USERNAME="${REGISTRY_USERNAME}"
export SINGULARITY_DOCKER_PASSWORD="${REGISTRY_PASSWORD}"

if id -un >/dev/null 2>&1; then
  run_with_heartbeat "Singularity image verification" \
    singularity exec --nv "docker://${IMAGE}" sh -lc '
    export PATH=/opt/conda/envs/app/bin:$PATH
    python -c "import numpy, scipy, numba, zarr, tifffile, dask, pandas, psfmodels, cupy, cucim; from cupyx.scipy.signal import fftconvolve; from cucim.skimage.restoration import richardson_lucy; assert numpy.__version__ == \"1.26.4\", numpy.__version__; print(\"deconvolution image imports ok\", \"numpy=\" + numpy.__version__, \"cupy=\" + cupy.__version__, \"cucim=\" + cucim.__version__)"
  '
else
  echo "WARNING: pushed ${IMAGE}, but skipped local Singularity verification because the current UID is not resolvable." >&2
  echo "Run check-deployment-container.sh from the deployment environment before deploying in Astrocyte." >&2
fi

echo "Pushed ${IMAGE}"
echo "Astrocyte container URI: docker://${IMAGE}"
