#!/usr/bin/env bash
set -euo pipefail

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
    DIR=$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
DIR=$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)
PACKAGE_ROOT=$(cd "${DIR}/.." >/dev/null 2>&1 && pwd)

REPO="vizapp-neuroglancer"
TAG="1.0"
SINGULARITY_IMAGE="${DIR}/${REPO}-${TAG}.img"
PORT="${VIZAPP_PORT:?VIZAPP_PORT is required}"

if [ ! -f "${SINGULARITY_IMAGE}" ]; then
    echo "ERROR: VizApp Singularity image is missing: ${SINGULARITY_IMAGE}" >&2
    echo "Expected Astrocyte to materialize docker://git.biohpc.swmed.edu:5050/dean-lab/ctASLM2-deconvolution/${REPO}:${TAG}" >&2
    exit 1
fi

MANIFEST="${PACKAGE_ROOT}/workflow/output/neuroglancer/layers.json"
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Neuroglancer manifest is missing: ${MANIFEST}" >&2
    echo "Run the workflow before launching the VizApp." >&2
    exit 1
fi

echo "Launching ctASLM2 Neuroglancer VizApp"
echo "Image: ${SINGULARITY_IMAGE}"
echo "Package root: ${PACKAGE_ROOT}"
echo "Manifest: ${MANIFEST}"
echo "Port: ${PORT}"

exec singularity run \
    --no-home \
    --bind "${PACKAGE_ROOT}:${PACKAGE_ROOT}" \
    --pwd "${DIR}" \
    "${SINGULARITY_IMAGE}" \
    python3 "${DIR}/neuroloader.py" "${PORT}"
