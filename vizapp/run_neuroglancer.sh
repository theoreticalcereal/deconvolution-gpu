#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "${script_dir}/.." && pwd)"
port="${VIZAPP_PORT:?VIZAPP_PORT is required}"

python_bin="$(command -v python3)"
if ! "${python_bin}" -c "import neuroglancer" >/dev/null 2>&1; then
    python_bin=""
    newest_env=""
    if [ -d "${package_root}/work" ]; then
        newest_env="$(
            find "${package_root}/work" -path "*/decon_runtime/decon_env/bin/python3" \
                -type f -executable -printf "%T@ %p\n" 2>/dev/null \
            | sort -nr \
            | head -n 1 \
            | cut -d' ' -f2-
        )"
    fi
    if [ -z "${newest_env}" ]; then
        echo "ERROR: base python3 cannot import neuroglancer and no decon_runtime Python was found." >&2
        exit 1
    fi
    python_bin="${newest_env}"
    conda_prefix="$(cd "$(dirname "${python_bin}")/.." && pwd)"
    export CONDA_PREFIX="${conda_prefix}"
    export CONDA_DEFAULT_ENV=decon_env
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

cd "${script_dir}"
exec "${python_bin}" neuroloader.py "${port}"
