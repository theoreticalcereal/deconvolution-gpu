#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKFLOW_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
REPO_ROOT=$(cd "${WORKFLOW_DIR}/.." && pwd -P)
LAUNCH_DIR=$(pwd -P)

IMAGE_ROOT=${IMAGE_ROOT:-/archive/bioinformatics/Danuser_lab/Fiolka/MicroscopeDevelopment/omniOPM/Oil/U2OS/CLC/250417/Cell19}
CELL_NAME=${CELL_NAME:-Top_Cell}
CELL_INDEX=${CELL_INDEX:-19}
CHANNEL=${CHANNEL:-0}
TIMEPOINT=${TIMEPOINT:-0}

PSF_ROOT=${PSF_ROOT:-/archive/bioinformatics/Danuser_lab/Fiolka/MicroscopeDevelopment/SyntheticPSF/omniOPM/oil}
PSF_FILE=${PSF_FILE:-NA0.2_ill_561_det_610_NA1_40degree_0.118umxyz_BottomtoTop.tif}

ITER=${ITER:-10}
WF_DECON_ITERS=${WF_DECON_ITERS:-${ITER}}
BACKGROUND=${BACKGROUND:-0}
MATLAB_PAD_XY=${MATLAB_PAD_XY:-20}
MATLAB_PAD_Z=${MATLAB_PAD_Z:-20}
MATLAB_THREADS=${MATLAB_THREADS:-8}
MATLAB_BIN=${MATLAB_BIN:-matlab}
NEXTFLOW_MODULE=${NEXTFLOW_MODULE:-nextflow/24.10.0}

WF_PROFILE=${WF_PROFILE:-light_sheet}
WF_BLIND_BACKEND=${WF_BLIND_BACKEND:-cupy}
WF_CUPY_FFT_ENGINE=${WF_CUPY_FFT_ENGINE:-scout}
WF_PSF_MODE=${WF_PSF_MODE:-light_sheet}
WF_PSF_MODEL=${WF_PSF_MODEL:-vectorial}
WF_PSF_SEED_PATH=${WF_PSF_SEED_PATH:-}
WF_FIXED_PSF_PATH=${WF_FIXED_PSF_PATH:-}
WF_WAVELENGTH=${WF_WAVELENGTH:-0.610}
WF_DETECTION_NA=${WF_DETECTION_NA:-1.1}
WF_NA=${WF_NA:-${WF_DETECTION_NA}}
WF_ILLUMINATION_NA=${WF_ILLUMINATION_NA:-0.19}
WF_NI=${WF_NI:-1.515}
WF_NS=${WF_NS:-1.515}
WF_DXY=${WF_DXY:-0.104}
WF_DZ=${WF_DZ:-0.3}
WF_LIGHT_SHEET_ANGLE=${WF_LIGHT_SHEET_ANGLE:-40}
WF_PSF_SIZE_Z=${WF_PSF_SIZE_Z:-101}
WF_PSF_SIZE_XY=${WF_PSF_SIZE_XY:-61}
WF_BLIND_ITERS=${WF_BLIND_ITERS:-8}
WF_BLIND_WORKERS=${WF_BLIND_WORKERS:-1}
WF_MATLAB_WORKERS=${WF_MATLAB_WORKERS:-1}
WF_MATLAB_THREADS=${WF_MATLAB_THREADS:-1}
WF_DECON_WORKERS=${WF_DECON_WORKERS:-1}
WF_GPU_QUEUE=${WF_GPU_QUEUE:-GPUp40}
WF_PYRAMID_MAX_DOWNSAMPLE=${WF_PYRAMID_MAX_DOWNSAMPLE:-1}
WF_NO_PSF_CACHE=${WF_NO_PSF_CACHE:-1}
NEXTFLOW_EXTRA_ARGS=${NEXTFLOW_EXTRA_ARGS:-}

PETAKIT_ROOT=${PETAKIT_ROOT:-${REPO_ROOT}/../petakit}
PETAKIT_RL_METHOD=${PETAKIT_RL_METHOD:-simplified}
PETAKIT_DECON_ITERS=${PETAKIT_DECON_ITERS:-${WF_DECON_ITERS}}
PETAKIT_GPU_JOB=${PETAKIT_GPU_JOB:-true}
PETAKIT_PSF_GEN=${PETAKIT_PSF_GEN:-false}
PETAKIT_EDGE_EROSION=${PETAKIT_EDGE_EROSION:-0}
PETAKIT_SAVE16BIT=${PETAKIT_SAVE16BIT:-true}
PETAKIT_LARGE_FILE=${PETAKIT_LARGE_FILE:-false}
PETAKIT_LARGE_METHOD=${PETAKIT_LARGE_METHOD:-inmemory}
PETAKIT_BATCH_SIZE=${PETAKIT_BATCH_SIZE:-[1024,1024,1024]}
PETAKIT_BLOCK_SIZE=${PETAKIT_BLOCK_SIZE:-[256,256,256]}

if [[ -z "${PYTHON_BIN:-}" && -x "${REPO_ROOT}/../decon_env/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/../decon_env/bin/python"
else
    PYTHON_BIN=${PYTHON_BIN:-python3}
fi

RUN_WORKFLOW=${RUN_WORKFLOW:-1}
RUN_MATLAB_PSF=${RUN_MATLAB_PSF:-}
RUN_PETAKIT=${RUN_PETAKIT:-1}
RUN_CUSTOM=${RUN_CUSTOM:-1}
RUN_COMPARE=${RUN_COMPARE:-1}
ENFORCE_GATES=${ENFORCE_GATES:-1}
CUSTOM_EXECUTOR=${CUSTOM_EXECUTOR:-slurm}
CUSTOM_GPU_MEMORY=${CUSTOM_GPU_MEMORY:-64G}
WORKFLOW_SECONDS=${WORKFLOW_SECONDS:-}
MATLAB_REFERENCE_PSF_SECONDS=${MATLAB_REFERENCE_PSF_SECONDS:-}
REFERENCE_RUN_DIR=${REFERENCE_RUN_DIR:-}
MATLAB_PSF_PATH=${MATLAB_PSF_PATH:-}
WORKFLOW_VOLUME=${WORKFLOW_VOLUME:-}
WORKFLOW_PSF=${WORKFLOW_PSF:-}
COMPARE_ROOT=${COMPARE_ROOT:-${REPO_ROOT}/comparison_runs}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

channel_label=$(printf "CH%02d" "${CHANNEL}")
timepoint_label=$(printf "%06d" "${TIMEPOINT}")
sample_name="${CELL_NAME}${CELL_INDEX}"
input_filename="1_${channel_label}_${timepoint_label}.tif"
launch_input_tiff="${LAUNCH_DIR}/${input_filename}"
repo_input_tiff="${REPO_ROOT}/${input_filename}"
archive_input_tiff="${IMAGE_ROOT}/${sample_name}/${input_filename}"
if [[ -n "${INPUT_TIFF:-}" ]]; then
    source_input_tiff="${INPUT_TIFF}"
elif [[ -f "${repo_input_tiff}" ]]; then
    source_input_tiff="${repo_input_tiff}"
elif [[ -f "${launch_input_tiff}" ]]; then
    source_input_tiff="${launch_input_tiff}"
else
    source_input_tiff="${archive_input_tiff}"
fi
psf_tiff="${PSF_ROOT}/${PSF_FILE}"
workflow_psf_seed_path=${WF_PSF_SEED_PATH:-${psf_tiff}}

run_dir="${COMPARE_ROOT}/petakit_reference_psf_${sample_name}_${channel_label}_${timepoint_label}_${RUN_ID}"
workflow_out="${run_dir}/workflow"
matlab_out="${run_dir}/matlab_reference"
petakit_matlab_psf="${run_dir}/petakit_matlab_psf"
petakit_workflow_psf="${run_dir}/petakit_workflow_psf"
custom_matlab_psf="${run_dir}/custom_matlab_psf"
metrics_out="${run_dir}/metrics"
stage1_psf_effect="${metrics_out}/stage1_psf_effect"
stage2_application="${metrics_out}/stage2_application"
timing_out="${metrics_out}/timing"
tmp_dir="${run_dir}/tmp"
staged_image_root="${run_dir}/staged_input"
staged_sample_dir="${staged_image_root}/${sample_name}"
staged_input_tiff="${staged_sample_dir}/${input_filename}"
mkdir -p \
    "${workflow_out}" "${matlab_out}" \
    "${petakit_matlab_psf}" "${petakit_workflow_psf}" "${custom_matlab_psf}" \
    "${stage1_psf_effect}" "${stage2_application}" "${timing_out}" \
    "${tmp_dir}" "${staged_sample_dir}"

absolute_path() {
    local path=$1
    local dir
    local base
    dir=$(cd "$(dirname "${path}")" && pwd -P)
    base=$(basename "${path}")
    printf '%s/%s\n' "${dir}" "${base}"
}

if [[ -n "${REFERENCE_RUN_DIR}" ]]; then
    reference_root=$(cd "${REFERENCE_RUN_DIR%/}" && pwd -P)
    MATLAB_PSF_PATH=${MATLAB_PSF_PATH:-${reference_root}/matlab_reference/PSFr_${sample_name}/${sample_name}psfr1.tif}
    WORKFLOW_PSF=${WORKFLOW_PSF:-${reference_root}/workflow/estimated_psf.tif}
    if [[ -z "${WORKFLOW_VOLUME}" ]]; then
        expected="${reference_root}/workflow/deconvolved_tiff/DB2_${input_filename}"
        if [[ -f "${expected}" ]]; then
            WORKFLOW_VOLUME="${expected}"
        else
            WORKFLOW_VOLUME=$(find "${reference_root}/workflow" -type f \( -name "DB2_${input_filename}" -o -name "DB2_*.tif" -o -name "DB2_*.tiff" \) | sort | head -n 1)
        fi
    fi
    RUN_MATLAB_PSF=${RUN_MATLAB_PSF:-0}
fi
RUN_MATLAB_PSF=${RUN_MATLAB_PSF:-1}

if [[ -n "${MATLAB_PSF_PATH}" ]]; then
    MATLAB_PSF_PATH=$(absolute_path "${MATLAB_PSF_PATH}")
fi
if [[ -n "${WORKFLOW_PSF}" ]]; then
    WORKFLOW_PSF=$(absolute_path "${WORKFLOW_PSF}")
fi
if [[ -n "${WORKFLOW_VOLUME}" ]]; then
    WORKFLOW_VOLUME=$(absolute_path "${WORKFLOW_VOLUME}")
fi

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

timestamp_seconds() {
    date +%s.%N
}

elapsed_seconds() {
    awk -v start="$1" -v finish="$2" 'BEGIN { printf "%.6f", finish - start }'
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

matlab_literal() {
    local value=${1//\'/\'\'}
    printf "'%s'" "${value}"
}

require_file() {
    local path=$1
    [[ -f "${path}" ]] || die "Required file not found: ${path}"
}

require_dir() {
    local path=$1
    [[ -d "${path}" ]] || die "Required directory not found: ${path}"
}

resolve_matlab_bin() {
    local candidate
    for candidate in "${MATLAB_BIN}" matlab /home1/apps/MATLAB/R2024a/bin/matlab; do
        if [[ -n "${candidate}" ]] && command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return 0
        fi
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

find_workflow_volume() {
    if [[ -n "${WORKFLOW_VOLUME}" && -f "${WORKFLOW_VOLUME}" ]]; then
        printf '%s\n' "${WORKFLOW_VOLUME}"
        return 0
    fi
    local expected="${workflow_out}/deconvolved_tiff/DB2_${input_filename}"
    if [[ -f "${expected}" ]]; then
        printf '%s\n' "${expected}"
        return 0
    fi
    find "${workflow_out}" -type f \( -name "DB2_${input_filename}" -o -name "DB2_*.tif" -o -name "DB2_*.tiff" \) | sort | head -n 1
}

require_file "${source_input_tiff}"
require_file "${psf_tiff}"
require_dir "${PETAKIT_ROOT}"
require_file "${SCRIPT_DIR}/readtiffstack.m"
require_file "${SCRIPT_DIR}/writetiffstack.m"
ln -sfn "${source_input_tiff}" "${staged_input_tiff}"

log "Comparison run directory: ${run_dir}"
log "Source input TIFF: ${source_input_tiff}"
log "Staged input TIFF: ${staged_input_tiff}"
log "MATLAB reference PSF seed: ${psf_tiff}"
log "Petakit root: ${PETAKIT_ROOT}"

if [[ "${RUN_WORKFLOW}" == "1" ]]; then
    log "Running current Nextflow workflow with ${WF_BLIND_BACKEND} blind PSF backend."
    if ! command -v nextflow >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load "${NEXTFLOW_MODULE}"
    fi
    nextflow_bin=$(command -v nextflow) || die "Nextflow executable not found after loading ${NEXTFLOW_MODULE}."
    gpu_queue_config="${tmp_dir}/gpu_queue.config"
    cat > "${gpu_queue_config}" <<NEXTFLOW_CONFIG
process {
    withName: DECON {
        queue = '${WF_GPU_QUEUE}'
    }
}
NEXTFLOW_CONFIG
    no_cache_flag=()
    if [[ "${WF_NO_PSF_CACHE}" == "1" ]]; then
        no_cache_flag+=(--no_psf_cache)
    fi
    fixed_psf_flag=()
    if [[ -n "${WF_FIXED_PSF_PATH}" ]]; then
        fixed_psf_flag+=(--fixed_psf_path "${WF_FIXED_PSF_PATH}")
    fi
    nextflow_extra=()
    if [[ -n "${NEXTFLOW_EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        nextflow_extra=(${NEXTFLOW_EXTRA_ARGS})
    fi
    workflow_start=$(timestamp_seconds)
    (
        cd "${WORKFLOW_DIR}"
        "${nextflow_bin}" run main.nf \
            -c configs/biohpc.config \
            -c "${gpu_queue_config}" \
            -profile "${WF_PROFILE}" \
            --input "${staged_input_tiff}" \
            --output_dir "${workflow_out}" \
            --output_formats tiff \
            --background "${BACKGROUND}" \
            --iter "${WF_DECON_ITERS}" \
            --blind_iters "${WF_BLIND_ITERS}" \
            --blind_backend "${WF_BLIND_BACKEND}" \
            --cupy_fft_engine "${WF_CUPY_FFT_ENGINE}" \
            --blind_workers "${WF_BLIND_WORKERS}" \
            --matlab_workers "${WF_MATLAB_WORKERS}" \
            --matlab_threads "${WF_MATLAB_THREADS}" \
            --decon_workers "${WF_DECON_WORKERS}" \
            --psf_mode "${WF_PSF_MODE}" \
            --psf_model "${WF_PSF_MODEL}" \
            --psf_seed_path "${workflow_psf_seed_path}" \
            "${fixed_psf_flag[@]}" \
            --wavelength "${WF_WAVELENGTH}" \
            --na "${WF_NA}" \
            --detection_na "${WF_DETECTION_NA}" \
            --illumination_na "${WF_ILLUMINATION_NA}" \
            --ni "${WF_NI}" \
            --ns "${WF_NS}" \
            --dxy "${WF_DXY}" \
            --dz "${WF_DZ}" \
            --light_sheet_angle "${WF_LIGHT_SHEET_ANGLE}" \
            --psf_size_z "${WF_PSF_SIZE_Z}" \
            --psf_size_xy "${WF_PSF_SIZE_XY}" \
            --pyramid_max_downsample "${WF_PYRAMID_MAX_DOWNSAMPLE}" \
            "${no_cache_flag[@]}" \
            "${nextflow_extra[@]}"
    )
    WORKFLOW_SECONDS=$(elapsed_seconds "${workflow_start}" "$(timestamp_seconds)")
    printf '%s\n' "${WORKFLOW_SECONDS}" > "${timing_out}/workflow_total_seconds.txt"
fi

matlab_psf="${MATLAB_PSF_PATH:-${matlab_out}/PSFr_${sample_name}/${sample_name}psfr1.tif}"
matlab_psf_seconds_path="${timing_out}/matlab_psf_seconds.txt"
matlab_psf_script="${tmp_dir}/generate_matlab_reference_psf.m"
cat > "${matlab_psf_script}" <<MATLAB
clc; clear;
addpath($(matlab_literal "${SCRIPT_DIR}"));
maxNumCompThreads(${MATLAB_THREADS});
fprintf('MATLAB maxNumCompThreads=%d\\n', maxNumCompThreads);

inputPath = $(matlab_literal "${staged_input_tiff}");
seedPsfPath = $(matlab_literal "${psf_tiff}");
outputPsfPath = $(matlab_literal "${matlab_psf}");
background = ${BACKGROUND};
iter = ${ITER};

if exist(inputPath, 'file') ~= 2
    error('Input file not found: %s', inputPath);
end
if exist(seedPsfPath, 'file') ~= 2
    error('PSF seed file not found: %s', seedPsfPath);
end

FinalImage = readtiffstack(inputPath);
mImage = size(FinalImage, 1);
nImage = size(FinalImage, 2);
NumberImages = size(FinalImage, 3);
E1 = padarray(single(FinalImage), [${MATLAB_PAD_XY} ${MATLAB_PAD_XY} ${MATLAB_PAD_Z}], 'symmetric');
PSFimage = double(readtiffstack(seedPsfPath));
PSFimage = abs(PSFimage - background);
psfi = single(PSFimage);
if any(size(psfi) > size(E1))
    warning('PSF shape [%s] exceeds padded input shape [%s]; center-cropping PSF to fit deconvblind OUTSIZE.', num2str(size(psfi)), num2str(size(E1)));
    targetSize = min(size(psfi), size(E1));
    startIdx = floor((size(psfi) - targetSize) / 2) + 1;
    endIdx = startIdx + targetSize - 1;
    psfi = psfi(startIdx(1):endIdx(1), startIdx(2):endIdx(2), startIdx(3):endIdx(3));
end

fprintf('Starting deconvblind PSF generation: input size [%s], psf seed size [%s], iter=%d\\n', num2str(size(E1)), num2str(size(psfi)), iter);
deconvblindStart = tic;
[~, psfr] = deconvblind(E1, psfi, iter);
deconvblindSeconds = toc(deconvblindStart);
fprintf('Finished deconvblind in %.2f seconds\\n', deconvblindSeconds);
writematrix(deconvblindSeconds, $(matlab_literal "${matlab_psf_seconds_path}"), 'FileType', 'text');

psfr = psfr ./ max(psfr(:));
psfr2 = uint16(60000 * psfr);
outputFolder = fileparts(outputPsfPath);
if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end
writetiffstack(psfr2, outputPsfPath);
fprintf('Wrote MATLAB reference PSF: %s\\n', outputPsfPath);
MATLAB

if [[ "${RUN_MATLAB_PSF}" == "1" ]]; then
    if command -v module >/dev/null 2>&1; then
        module load matlab/2024a >/dev/null 2>&1 || true
    fi
    resolved_matlab_bin=$(resolve_matlab_bin) || die "MATLAB executable not found. Set MATLAB_BIN=/path/to/matlab."
    export OMP_NUM_THREADS="${MATLAB_THREADS}"
    export MKL_NUM_THREADS="${MATLAB_THREADS}"
    export OPENBLAS_NUM_THREADS="${MATLAB_THREADS}"
    export NUMEXPR_NUM_THREADS="${MATLAB_THREADS}"
    log "Generating MATLAB reference PSF with ${resolved_matlab_bin}; MATLAB_THREADS=${MATLAB_THREADS}."
    "${resolved_matlab_bin}" -batch "run('${matlab_psf_script}')"
fi
require_file "${matlab_psf}"

if [[ -z "${MATLAB_REFERENCE_PSF_SECONDS}" && -f "${matlab_psf_seconds_path}" ]]; then
    MATLAB_REFERENCE_PSF_SECONDS=$(tr -d '[:space:]' < "${matlab_psf_seconds_path}")
fi

workflow_volume=$(find_workflow_volume)
workflow_psf="${WORKFLOW_PSF:-${workflow_out}/estimated_psf.tif}"
require_file "${workflow_psf}"
workflow_psf=$(absolute_path "${workflow_psf}")

petakit_matlab_seconds_path="${timing_out}/petakit_matlab_psf_seconds.txt"
petakit_workflow_seconds_path="${timing_out}/petakit_workflow_psf_seconds.txt"
petakit_script="${tmp_dir}/run_petakit_two_psfs.m"
cat > "${petakit_script}" <<MATLAB
clc; clear;
petakitRoot = $(matlab_literal "${PETAKIT_ROOT}");
if exist(fullfile(petakitRoot, 'setup.m'), 'file') ~= 2
    error('PETAKIT_ROOT does not contain setup.m: %s', petakitRoot);
end
cd(petakitRoot);
run(fullfile(petakitRoot, 'setup.m'));
maxNumCompThreads(${MATLAB_THREADS});
fprintf('MATLAB maxNumCompThreads=%d\\n', maxNumCompThreads);

frameFullpath = $(matlab_literal "${staged_input_tiff}");

psfFullpath = $(matlab_literal "${matlab_psf}");
deconPath = $(matlab_literal "${petakit_matlab_psf}");
if ~exist(deconPath, 'dir'); mkdir(deconPath); end

fprintf('Running Petakit with MATLAB reference PSF.\\n');
petakitStart = tic;
XR_RLdeconFrame3D(frameFullpath, ${WF_DXY}, ${WF_DZ}, deconPath, ...
    'PSFfile', psfFullpath, ...
    'Background', ${BACKGROUND}, ...
    'DeconIter', ${PETAKIT_DECON_ITERS}, ...
    'RLMethod', $(matlab_literal "${PETAKIT_RL_METHOD}"), ...
    'psfGen', ${PETAKIT_PSF_GEN}, ...
    'GPUJob', ${PETAKIT_GPU_JOB}, ...
    'save16bit', ${PETAKIT_SAVE16BIT}, ...
    'largeFile', ${PETAKIT_LARGE_FILE}, ...
    'largeMethod', $(matlab_literal "${PETAKIT_LARGE_METHOD}"), ...
    'batchSize', ${PETAKIT_BATCH_SIZE}, ...
    'blockSize', ${PETAKIT_BLOCK_SIZE}, ...
    'parseCluster', false, ...
    'EdgeErosion', ${PETAKIT_EDGE_EROSION}, ...
    'mipAxis', [0, 0, 0], ...
    'Overwrite', true);
petakitSeconds = toc(petakitStart);
writematrix(petakitSeconds, $(matlab_literal "${petakit_matlab_seconds_path}"), 'FileType', 'text');

psfFullpath = $(matlab_literal "${workflow_psf}");
deconPath = $(matlab_literal "${petakit_workflow_psf}");
if ~exist(deconPath, 'dir'); mkdir(deconPath); end

fprintf('Running Petakit with workflow PSF.\\n');
petakitStart = tic;
XR_RLdeconFrame3D(frameFullpath, ${WF_DXY}, ${WF_DZ}, deconPath, ...
    'PSFfile', psfFullpath, ...
    'Background', ${BACKGROUND}, ...
    'DeconIter', ${PETAKIT_DECON_ITERS}, ...
    'RLMethod', $(matlab_literal "${PETAKIT_RL_METHOD}"), ...
    'psfGen', ${PETAKIT_PSF_GEN}, ...
    'GPUJob', ${PETAKIT_GPU_JOB}, ...
    'save16bit', ${PETAKIT_SAVE16BIT}, ...
    'largeFile', ${PETAKIT_LARGE_FILE}, ...
    'largeMethod', $(matlab_literal "${PETAKIT_LARGE_METHOD}"), ...
    'batchSize', ${PETAKIT_BATCH_SIZE}, ...
    'blockSize', ${PETAKIT_BLOCK_SIZE}, ...
    'parseCluster', false, ...
    'EdgeErosion', ${PETAKIT_EDGE_EROSION}, ...
    'mipAxis', [0, 0, 0], ...
    'Overwrite', true);
petakitSeconds = toc(petakitStart);
writematrix(petakitSeconds, $(matlab_literal "${petakit_workflow_seconds_path}"), 'FileType', 'text');

fprintf('Both Petakit validation deconvolutions complete.\\n');
MATLAB

if [[ "${RUN_PETAKIT}" == "1" ]]; then
    resolved_matlab_bin=$(resolve_matlab_bin) || die "MATLAB executable not found. Set MATLAB_BIN=/path/to/matlab."
    export OMP_NUM_THREADS="${MATLAB_THREADS}"
    export MKL_NUM_THREADS="${MATLAB_THREADS}"
    export OPENBLAS_NUM_THREADS="${MATLAB_THREADS}"
    export NUMEXPR_NUM_THREADS="${MATLAB_THREADS}"
    log "Running Petakit with the MATLAB and workflow PSFs."
    "${resolved_matlab_bin}" -batch "run('${petakit_script}')"
fi

petakit_matlab_volume="${petakit_matlab_psf}/${input_filename}"
petakit_workflow_volume="${petakit_workflow_psf}/${input_filename}"
custom_matlab_volume="${custom_matlab_psf}/DB2_${input_filename}"
custom_timing_seconds="${timing_out}/custom_production_seconds.txt"

if [[ "${RUN_CUSTOM}" == "1" ]]; then
    read -r stage2_psf_z stage2_psf_y stage2_psf_x < <(
        "${PYTHON_BIN}" -c \
            "from tifffile import imread; print(*imread(r'${staged_input_tiff}').shape)"
    )
    [[ "${stage2_psf_y}" == "${stage2_psf_x}" ]] \
        || die "Stage 2 requires square XY input, got ${stage2_psf_y}x${stage2_psf_x}"
    log "Running the production CuPy workflow path with the MATLAB reference PSF."
    custom_command=(
        /usr/bin/time -f %e -o "${custom_timing_seconds}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/decon_wrapper.py"
        --image_path "${staged_sample_dir}" \
        --output_format tiff \
        --fixed_psf_path "${matlab_psf}" \
        --iter "${PETAKIT_DECON_ITERS}" \
        --background "${BACKGROUND}" \
        --wavelength "${WF_WAVELENGTH}" \
        --na "${WF_NA}" \
        --detection_na "${WF_DETECTION_NA}" \
        --illumination_na "${WF_ILLUMINATION_NA}" \
        --ni "${WF_NI}" \
        --ns "${WF_NS}" \
        --dxy "${WF_DXY}" \
        --dz "${WF_DZ}" \
        --psf_mode "${WF_PSF_MODE}" \
        --psf_size_z "${stage2_psf_z}" \
        --psf_size_xy "${stage2_psf_x}" \
        --decon_workers 1 \
        --pyramid_max_downsample 1
    )
    if [[ "${CUSTOM_EXECUTOR}" == "slurm" ]]; then
        printf -v custom_command_text '%q ' "${custom_command[@]}"
        sbatch \
            --wait \
            --parsable \
            --partition="${WF_GPU_QUEUE}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem="${CUSTOM_GPU_MEMORY}" \
            --chdir="${custom_matlab_psf}" \
            --output="${timing_out}/custom_production.slurm.log" \
            --error="${timing_out}/custom_production.slurm.err" \
            --wrap="${custom_command_text}"
    elif [[ "${CUSTOM_EXECUTOR}" == "local" ]]; then
        (
            cd "${custom_matlab_psf}"
            "${custom_command[@]}"
        )
    else
        die "CUSTOM_EXECUTOR must be 'slurm' or 'local', got: ${CUSTOM_EXECUTOR}"
    fi
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
    require_file "${petakit_matlab_volume}"
    require_file "${petakit_workflow_volume}"
    require_file "${custom_matlab_volume}"
    require_file "${workflow_volume}"

    log "Stage 1: comparing PSFs and their effect through Petakit."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_psfs.py" \
        "${matlab_psf}" \
        "${workflow_psf}" \
        --spacing "${WF_DZ}" "${WF_DXY}" "${WF_DXY}" \
        --csv "${stage1_psf_effect}/stage1_psf_comparison.csv" \
        --json "${stage1_psf_effect}/stage1_psf_comparison.json" \
        > "${stage1_psf_effect}/stage1_psf_comparison.stdout.csv"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_volumes.py" \
        "${petakit_matlab_volume}" \
        "${petakit_workflow_volume}" \
        --spacing "${WF_DZ}" "${WF_DXY}" "${WF_DXY}" \
        --csv "${stage1_psf_effect}/stage1_volume_comparison.csv" \
        --json "${stage1_psf_effect}/stage1_volume_comparison.json" \
        > "${stage1_psf_effect}/stage1_volume_comparison.stdout.csv"

    log "Stage 2: comparing the production workflow and Petakit PSF application."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_volumes.py" \
        "${petakit_matlab_volume}" \
        "${custom_matlab_volume}" \
        --spacing "${WF_DZ}" "${WF_DXY}" "${WF_DXY}" \
        --csv "${stage2_application}/stage2_volume_comparison.csv" \
        --json "${stage2_application}/stage2_volume_comparison.json" \
        > "${stage2_application}/stage2_volume_comparison.stdout.csv"

    log "Comparing end-to-end workflow output against Petakit reference."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_volumes.py" \
        "${petakit_matlab_volume}" \
        "${workflow_volume}" \
        --spacing "${WF_DZ}" "${WF_DXY}" "${WF_DXY}" \
        --csv "${metrics_out}/end_to_end_volume_comparison.csv" \
        --json "${metrics_out}/end_to_end_volume_comparison.json" \
        > "${metrics_out}/end_to_end_volume_comparison.stdout.csv"

    timing_summary="${timing_out}/timing_summary.json"
    "${PYTHON_BIN}" - \
        "${timing_summary}" \
        "${WORKFLOW_SECONDS}" \
        "${MATLAB_REFERENCE_PSF_SECONDS}" \
        "${petakit_matlab_seconds_path}" \
        "${petakit_workflow_seconds_path}" \
        "${custom_timing_seconds}" <<'PYTHON'
import json
from pathlib import Path
import sys

output, workflow_text, matlab_text, petakit_matlab_path, petakit_workflow_path, custom_path = sys.argv[1:]

def optional_number(value):
    value = value.strip()
    return float(value) if value else None

def timing_file(path):
    file_path = Path(path)
    return float(file_path.read_text().strip()) if file_path.is_file() else None

workflow = optional_number(workflow_text)
matlab = optional_number(matlab_text)
petakit_matlab = timing_file(petakit_matlab_path)
petakit_workflow = timing_file(petakit_workflow_path)
custom = timing_file(custom_path)
reference = matlab + petakit_matlab if matlab is not None and petakit_matlab is not None else None
payload = {
    "workflow_total_seconds": workflow,
    "reference_total_seconds": reference,
    "matlab_psf_seconds": matlab,
    "petakit_matlab_psf_seconds": petakit_matlab,
    "petakit_workflow_psf_seconds": petakit_workflow,
    "custom_production_total_seconds": custom,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n")
PYTHON

    validation_args=()
    if [[ "${ENFORCE_GATES}" == "1" ]]; then
        validation_args+=(--enforce)
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_petakit_validation.py" \
        --stage1-psf "${stage1_psf_effect}/stage1_psf_comparison.json" \
        --stage1-volume "${stage1_psf_effect}/stage1_volume_comparison.json" \
        --stage2-volume "${stage2_application}/stage2_volume_comparison.json" \
        --timing-summary "${timing_summary}" \
        --output "${metrics_out}/validation_summary.json" \
        "${validation_args[@]}"
fi

log "Comparison complete."
printf 'Run directory: %s\n' "${run_dir}"
printf 'MATLAB reference PSF: %s\n' "${matlab_psf}"
printf 'Petakit MATLAB-PSF volume: %s\n' "${petakit_matlab_volume}"
printf 'Petakit workflow-PSF volume: %s\n' "${petakit_workflow_volume}"
printf 'Production MATLAB-PSF volume: %s\n' "${custom_matlab_volume}"
printf 'Workflow PSF: %s\n' "${workflow_psf}"
printf 'Workflow volume: %s\n' "${workflow_volume}"
printf 'Validation summary: %s\n' "${metrics_out}/validation_summary.json"
