#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKFLOW_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
REPO_ROOT=$(cd "${WORKFLOW_DIR}/.." && pwd -P)
LAUNCH_DIR=$(pwd -P)

# Dataset from the provided MATLAB reference. Override any value by exporting it
# before running this script, or by prefixing the command with NAME=value.
IMAGE_ROOT=${IMAGE_ROOT:-/archive/bioinformatics/Danuser_lab/Fiolka/MicroscopeDevelopment/omniOPM/Oil/U2OS/CLC/250417/Cell19}
CELL_NAME=${CELL_NAME:-Top_Cell}
CELL_INDEX=${CELL_INDEX:-19}
CHANNEL=${CHANNEL:-0}
TIMEPOINT=${TIMEPOINT:-0}

PSF_ROOT=${PSF_ROOT:-/archive/bioinformatics/Danuser_lab/Fiolka/MicroscopeDevelopment/SyntheticPSF/omniOPM/oil}
PSF_FILE=${PSF_FILE:-NA0.2_ill_561_det_610_NA1_40degree_0.118umxyz_BottomtoTop.tif}

ITER=${ITER:-10}
WF_DECON_ITERS=${WF_DECON_ITERS:-$((ITER * 2))}
BACKGROUND=${BACKGROUND:-0}
MATLAB_PAD_XY=${MATLAB_PAD_XY:-20}
MATLAB_PAD_Z=${MATLAB_PAD_Z:-20}
MATLAB_THREADS=${MATLAB_THREADS:-8}
MATLAB_BIN=${MATLAB_BIN:-matlab}

# Workflow optical/acquisition parameters. These defaults are inferred from the
# provided synthetic PSF filename and should be adjusted if the workflow run you
# want to diagnose used different parameters.
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

if [[ -z "${PYTHON_BIN:-}" && -x "${REPO_ROOT}/../decon_env/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/../decon_env/bin/python"
else
    PYTHON_BIN=${PYTHON_BIN:-python3}
fi
RUN_WORKFLOW=${RUN_WORKFLOW:-1}
RUN_MATLAB=${RUN_MATLAB:-1}
RUN_COMPARE=${RUN_COMPARE:-1}
REFERENCE_RUN_DIR=${REFERENCE_RUN_DIR:-}
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

run_dir="${COMPARE_ROOT}/matlab_reference_${sample_name}_${channel_label}_${timepoint_label}_${RUN_ID}"
workflow_out="${run_dir}/workflow"
matlab_out="${run_dir}/matlab_reference"
if [[ -n "${REFERENCE_RUN_DIR}" ]]; then
    matlab_out="${REFERENCE_RUN_DIR%/}/matlab_reference"
fi
metrics_out="${run_dir}/metrics"
tmp_dir="${run_dir}/tmp"
staged_image_root="${run_dir}/staged_input"
staged_sample_dir="${staged_image_root}/${sample_name}"
staged_input_tiff="${staged_sample_dir}/${input_filename}"
mkdir -p "${workflow_out}" "${metrics_out}" "${tmp_dir}" "${staged_sample_dir}"
if [[ -z "${REFERENCE_RUN_DIR}" ]]; then
    mkdir -p "${matlab_out}"
fi

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
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
    local expected="${workflow_out}/deconvolved_tiff/DB2_${input_filename}"
    if [[ -f "${expected}" ]]; then
        printf '%s\n' "${expected}"
        return 0
    fi
    find "${workflow_out}" -type f \( -name "DB2_${input_filename}" -o -name "DB2_*.tif" -o -name "DB2_*.tiff" \) | sort | head -n 1
}

require_file "${source_input_tiff}"
require_file "${psf_tiff}"
require_file "${SCRIPT_DIR}/readtiffstack.m"
require_file "${SCRIPT_DIR}/writetiffstack.m"
if [[ -n "${REFERENCE_RUN_DIR}" && "${RUN_MATLAB}" == "1" ]]; then
    die "REFERENCE_RUN_DIR reuses cached MATLAB outputs and requires RUN_MATLAB=0"
fi
ln -sfn "${source_input_tiff}" "${staged_input_tiff}"
input_tiff="${staged_input_tiff}"

log "Comparison run directory: ${run_dir}"
log "Source input TIFF: ${source_input_tiff}"
log "Staged input TIFF: ${input_tiff}"
log "MATLAB reference PSF seed: ${psf_tiff}"
log "Workflow deconvolution iterations: ${WF_DECON_ITERS} (MATLAB reference: ${ITER})"

if [[ "${RUN_WORKFLOW}" == "1" ]]; then
    log "Running current Nextflow workflow with ${WF_BLIND_BACKEND} blind PSF backend."
    gpu_queue_config="${tmp_dir}/gpu_queue.config"
    cat > "${gpu_queue_config}" <<NEXTFLOW_CONFIG
process {
    withName: DECON {
        queue = '${WF_GPU_QUEUE}'
    }
}
NEXTFLOW_CONFIG
    log "Using Nextflow DECON queue: ${WF_GPU_QUEUE}"
    no_cache_flag=()
    if [[ "${WF_NO_PSF_CACHE}" == "1" ]]; then
        no_cache_flag+=(--no_psf_cache)
    fi
    nextflow_extra=()
    if [[ -n "${NEXTFLOW_EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        nextflow_extra=(${NEXTFLOW_EXTRA_ARGS})
    fi
    (
        cd "${WORKFLOW_DIR}"
        nextflow run main.nf \
            -c configs/biohpc.config \
            -c "${gpu_queue_config}" \
            -profile "${WF_PROFILE}" \
            --input "${input_tiff}" \
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
            --fixed_psf_path "${WF_FIXED_PSF_PATH}" \
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
fi

matlab_script="${tmp_dir}/run_standalone_matlab_reference.m"
cat > "${matlab_script}" <<MATLAB
clc; clear;
addpath($(matlab_literal "${SCRIPT_DIR}"));
maxNumCompThreads(${MATLAB_THREADS});
fprintf('MATLAB maxNumCompThreads=%d\\n', maxNumCompThreads);

referenceInputPath = $(matlab_literal "${staged_input_tiff}");
referenceInputName = $(matlab_literal "${input_filename}");
imagePath = $(matlab_literal "${staged_image_root}");
Cell_name = $(matlab_literal "${CELL_NAME}");
Cell_index = [${CELL_INDEX}];
ChannelstoProcess = [${CHANNEL}];
timepoint = [${TIMEPOINT}];

psfPath = $(matlab_literal "${PSF_ROOT}");
psf{1} = $(matlab_literal "${PSF_FILE}");
background = ${BACKGROUND};
iter = ${ITER};
dir_Dec = $(matlab_literal "${matlab_out}");
if ~exist(dir_Dec, 'dir'); mkdir(dir_Dec); end

for p = 1:size(psf, 2)
    filepath = fullfile(psfPath, psf{p});
    if exist(filepath, 'file') ~= 2
        error('PSF file not found: %s', filepath);
    end
    PSFimage = readtiffstack(filepath);
    PSFimage = double(PSFimage);
    PSFimage = PSFimage - background;
    PSFimage = abs(PSFimage);
    PSF{p} = PSFimage;
end
clear PSFimage

numfolder = size(Cell_index, 2);
ch_number = size(ChannelstoProcess, 2);

for c = 1:numfolder
    names2 = strcat(Cell_name, num2str(Cell_index(c)));
    if size(timepoint, 2) == 0
        numImages = size(dir(fullfile(imagePath, names2)), 1) - 3;
        t_st = 0;
        t_end = round(numImages / ch_number) - 1;
    else
        t_st = min(timepoint);
        t_end = max(timepoint);
    end

    for t = t_st:t_end
        for ch = 1:ch_number
            tic
            filename = referenceInputName;
            filepath = referenceInputPath;
            if exist(filepath, 'file') ~= 2
                error('Input file not found: %s', filepath);
            end

            FinalImage = readtiffstack(filepath);
            mImage = size(FinalImage, 1);
            nImage = size(FinalImage, 2);
            NumberImages = size(FinalImage, 3);

            E1 = padarray(single(FinalImage), [${MATLAB_PAD_XY} ${MATLAB_PAD_XY} ${MATLAB_PAD_Z}], 'symmetric');
            maxE1 = max(E1(:));
            minE1 = min(E1(:));
            psfi = single(PSF{ch});
            if any(size(psfi) > size(E1))
                warning('PSF shape [%s] exceeds padded input shape [%s]; center-cropping PSF to fit deconvblind OUTSIZE.', num2str(size(psfi)), num2str(size(E1)));
                targetSize = min(size(psfi), size(E1));
                startIdx = floor((size(psfi) - targetSize) / 2) + 1;
                endIdx = startIdx + targetSize - 1;
                psfi = psfi(startIdx(1):endIdx(1), startIdx(2):endIdx(2), startIdx(3):endIdx(3));
            end

            fprintf('Starting deconvblind: input size [%s], psf size [%s], iter=%d\\n', num2str(size(E1)), num2str(size(psfi)), iter);
            deconvblindStart = tic;
            [Dec, psfr] = deconvblind(E1, psfi, iter);
            fprintf('Finished deconvblind in %.2f seconds\\n', toc(deconvblindStart));
            Dec = Dec((${MATLAB_PAD_XY} + 1):(${MATLAB_PAD_XY} + mImage), (${MATLAB_PAD_XY} + 1):(${MATLAB_PAD_XY} + nImage), (${MATLAB_PAD_Z} + 1):(${MATLAB_PAD_Z} + NumberImages));
            Dec = (Dec - min(Dec(:))) / (max(Dec(:) - min(Dec(:))));
            Dec = Dec .* (maxE1 - minE1) + minE1;
            Dec = uint16(Dec);

            psfr = psfr ./ max(psfr(:));
            psfr2 = uint16(60000 * psfr);
            PSFfolder = fullfile(dir_Dec, strcat('PSFr_', names2));
            if ~exist(PSFfolder, 'dir'); mkdir(PSFfolder); end
            PSFname = fullfile(PSFfolder, strcat(names2, 'psfr', num2str(ch), '.tif'));
            writetiffstack(psfr2, PSFname);

            fprintf('Starting deconvlucy: input size [%s], psf size [%s], iter=%d\\n', num2str(size(E1)), num2str(size(psfr)), iter);
            deconvlucyStart = tic;
            Dec2 = deconvlucy(E1, psfr, iter);
            fprintf('Finished deconvlucy in %.2f seconds\\n', toc(deconvlucyStart));
            Dec2 = Dec2((${MATLAB_PAD_XY} + 1):(${MATLAB_PAD_XY} + mImage), (${MATLAB_PAD_XY} + 1):(${MATLAB_PAD_XY} + nImage), (${MATLAB_PAD_Z} + 1):(${MATLAB_PAD_Z} + NumberImages));
            Dec2 = (Dec2 - min(Dec2(:))) / (max(Dec2(:) - min(Dec2(:))));
            Dec2 = Dec2 .* (maxE1 - minE1) + minE1;
            Dec2 = uint16(Dec2);

            finalPath2 = fullfile(dir_Dec, strcat('DB2_', names2));
            if ~exist(finalPath2, 'dir'); mkdir(finalPath2); end
            Decname2 = fullfile(finalPath2, filename);
            writetiffstack(Dec2, Decname2);
            toc, disp('Done')
        end
    end
end

disp('All Done')
MATLAB

if [[ "${RUN_MATLAB}" == "1" ]]; then
    if command -v module >/dev/null 2>&1; then
        module load matlab/2024a >/dev/null 2>&1 || true
    fi
    resolved_matlab_bin=$(resolve_matlab_bin) || die "MATLAB executable not found. Set MATLAB_BIN=/path/to/matlab."
    export OMP_NUM_THREADS="${MATLAB_THREADS}"
    export MKL_NUM_THREADS="${MATLAB_THREADS}"
    export OPENBLAS_NUM_THREADS="${MATLAB_THREADS}"
    export NUMEXPR_NUM_THREADS="${MATLAB_THREADS}"
    log "Running standalone MATLAB reference with ${resolved_matlab_bin}; MATLAB_THREADS=${MATLAB_THREADS}."
    "${resolved_matlab_bin}" -batch "run('${matlab_script}')"
fi

workflow_psf="${workflow_out}/estimated_psf.tif"
matlab_psf="${matlab_out}/PSFr_${sample_name}/${sample_name}psfr1.tif"
matlab_volume="${matlab_out}/DB2_${sample_name}/${input_filename}"
workflow_volume=$(find_workflow_volume)

if [[ "${RUN_COMPARE}" == "1" ]]; then
    require_file "${workflow_psf}"
    require_file "${matlab_psf}"
    require_file "${matlab_volume}"
    require_file "${workflow_volume}"

    log "Comparing generated PSFs: MATLAB reference vs workflow."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_psfs.py" \
        "${matlab_psf}" \
        "${workflow_psf}" \
        --spacing "${WF_DZ}" "${WF_DXY}" "${WF_DXY}" \
        --csv "${metrics_out}/psf_comparison.csv" \
        --json "${metrics_out}/psf_comparison.json" \
        > "${metrics_out}/psf_comparison.stdout.csv"

    log "Comparing full deconvolved volumes: MATLAB reference vs workflow."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_volumes.py" \
        "${matlab_volume}" \
        "${workflow_volume}" \
        --spacing "${WF_DZ}" "${WF_DXY}" "${WF_DXY}" \
        --csv "${metrics_out}/volume_comparison.csv" \
        --json "${metrics_out}/volume_comparison.json" \
        > "${metrics_out}/volume_comparison.stdout.csv"
fi

log "Comparison complete."
printf 'Run directory: %s\n' "${run_dir}"
printf 'Workflow PSF: %s\n' "${workflow_psf}"
printf 'MATLAB PSF: %s\n' "${matlab_psf}"
printf 'Workflow volume: %s\n' "${workflow_volume}"
printf 'MATLAB volume: %s\n' "${matlab_volume}"
printf 'PSF metrics: %s\n' "${metrics_out}/psf_comparison.csv"
printf 'Volume metrics: %s\n' "${metrics_out}/volume_comparison.csv"
