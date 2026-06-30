#!/bin/bash
set -euo pipefail

echo -e "\nLoading neuroglancer module ...\n"
module load neuroglancer/2.40.1

echo "Port is ${VIZAPP_PORT:?VIZAPP_PORT is required}"
python neuroloader.py "${VIZAPP_PORT}"
