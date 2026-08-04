def WORKFLOW_CONTAINER_IMAGE = 'git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.2'
def CONTAINER_ENV_PREFIX = '/opt/conda/envs/app'

process STAGE_DECON_INPUT {
    tag "decon_input"
    module 'singularity/3.9.9'
    container WORKFLOW_CONTAINER_IMAGE
    scratch true

    input:
    path input_tiffs

    output:
    path "input_zarr", emit: decon_input_dir

    script:
    def shell_quote = { value -> "'${value.toString().replace("'", "'\\''")}'" }
    def link_commands = input_tiffs.collect { tiff ->
        "ln -s \"\$PWD/${tiff}\" ${shell_quote("decon_input/${tiff.name}")}"
    }.join('\n')
    def metadata_commands = input_tiffs.collect { tiff ->
        "printf '%s\\t%s\\n' ${shell_quote(tiff.name)} ${shell_quote(tiff.name)} >> decon_input/original_filenames.tsv"
    }.join('\n')

    """
    mkdir -p decon_input
    : > decon_input/original_filenames.tsv
    ${link_commands}
    ${metadata_commands}

    export CONDA_PREFIX="${CONTAINER_ENV_PREFIX}"
    export CONDA_DEFAULT_ENV="app"
    export PATH="${CONTAINER_ENV_PREFIX}/bin:\${PATH}"
    export LD_LIBRARY_PATH="${CONTAINER_ENV_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

    python3 ${projectDir}/scripts/normalize_input_to_ome_zarr.py \\
        --input decon_input \\
        --output input_zarr
    """
}

process STAGE_DECON_TIFF_INPUT {
    tag "decon_tiff_input"
    scratch true

    input:
    path input_tiffs

    output:
    path "input_tiff", emit: decon_input_dir

    script:
    def shell_quote = { value -> "'${value.toString().replace("'", "'\\''")}'" }
    def link_commands = input_tiffs.collect { tiff ->
        "ln -s \"\$PWD/${tiff}\" ${shell_quote("input_tiff/${tiff.name}")}"
    }.join('\n')
    def metadata_commands = input_tiffs.collect { tiff ->
        "printf '%s\\t%s\\n' ${shell_quote(tiff.name)} ${shell_quote(tiff.name)} >> input_tiff/original_filenames.tsv"
    }.join('\n')

    """
    mkdir -p input_tiff
    : > input_tiff/original_filenames.tsv
    ${link_commands}
    ${metadata_commands}
    """
}

process DECON {
    tag "decon"
    module 'singularity/3.9.9:matlab/2024a'
    container WORKFLOW_CONTAINER_IMAGE
    containerOptions = '--nv -B /home1/apps/MATLAB:/home1/apps/MATLAB'

    publishDir "${params.output_dir}", mode: 'copy', pattern: 'estimated_psf.tif'
    publishDir "${params.output_dir}", mode: 'copy', pattern: 'DB2_*.ozx'

    maxForks 8
    cpus 72
    memory '256 GB'
    clusterOptions '--gres=gpu:1'
    scratch true

    input:
    path deskewed_dir
    val  background
    val  iter
    val  output_dir
    val  output_format

    output:
    path "DB2_*.{ozx,tif,tiff}", emit: decon_output
    path "estimated_psf.tif", emit: psf_output

    script:
    // Build parameter flags only when supplied. decon_wrapper.py validates
    // required optical/acquisition parameters and fails fast if any are missing.
    def is_supplied = { value ->
        if (value == null) {
            return false
        }
        def text = value.toString().trim()
        return text && text != '-1' && text != '-1.0'
    }
    def flag = { name, value -> is_supplied(value) ? "--${name} ${value}" : "" }

    def background_flag  = flag('background', background)
    def iter_flag        = flag('iter', iter)
    def output_format_flag = flag('output_format', output_format)
    def na_flag          = flag('na', params.na)
    def detection_na_flag = flag('detection_na', params.detection_na)
    def illumination_na_flag = flag('illumination_na', params.illumination_na)
    def wavelength_flag  = flag('wavelength', params.wavelength)
    def ni_flag          = flag('ni', params.ni)
    def ns_flag          = flag('ns', params.ns)
    def ni0_flag         = flag('ni0', params.ni0)
    def tg_flag          = flag('tg', params.tg)
    def tg0_flag         = flag('tg0', params.tg0)
    def ng_flag          = flag('ng', params.ng)
    def ng0_flag         = flag('ng0', params.ng0)
    def ti0_flag         = flag('ti0', params.ti0)
    def oversample_factor_flag = flag('oversample_factor', params.oversample_factor)
    def psf_model_flag   = flag('psf_model', params.psf_model)
    def psf_mode_flag    = flag('psf_mode', params.psf_mode)
    def light_sheet_angle_flag = flag('light_sheet_angle', params.light_sheet_angle)
    def camera_pixel_size_flag = flag('camera_pixel_size', params.camera_pixel_size)
    def magnification_flag = flag('magnification', params.magnification)
    def dxy_flag         = flag('dxy', params.dxy)
    def dz_flag          = flag('dz', params.dz)
    def psf_size_z_flag  = flag('psf_size_z', params.psf_size_z)
    def psf_size_xy_flag = flag('psf_size_xy', params.psf_size_xy)
    def blind_iters_flag = flag('blind_iters', params.blind_iters)
    def blind_backend_flag = flag('blind_backend', params.blind_backend)
    def chunk_xy_flag    = flag('chunk_xy', params.chunk_xy)
    def blind_max_tiles_flag = flag('blind_max_tiles', params.blind_max_tiles)
    def cupy_fft_engine_flag = flag('cupy_fft_engine', params.cupy_fft_engine)
    def adaptive_scout_iters_flag = flag('adaptive_scout_iters', params.adaptive_scout_iters)
    def adaptive_keep_tiles_flag = flag('adaptive_keep_tiles', params.adaptive_keep_tiles)
    def tile_selection_strategy_flag = flag('tile_selection_strategy', params.tile_selection_strategy)
    def coarse_region_rows_flag = flag('coarse_region_rows', params.coarse_region_rows)
    def coarse_region_columns_flag = flag('coarse_region_columns', params.coarse_region_columns)
    def coarse_region_limit_flag = flag('coarse_region_limit', params.coarse_region_limit)
    def decon_chunk_xy_flag = flag('decon_chunk_xy', params.decon_chunk_xy)
    def pad_xy_flag      = flag('pad_xy', params.pad_xy)
    def pad_z_flag       = flag('pad_z', params.pad_z)
    def blind_peak_normalization_flag = flag('blind_peak_normalization', params.blind_peak_normalization)
    def blind_peak_gamma_max_flag = flag('blind_peak_gamma_max', params.blind_peak_gamma_max)
    def blind_latent_update_period_flag = flag('blind_latent_update_period', params.blind_latent_update_period)
    def blind_workers_flag = flag('blind_workers', params.blind_workers)
    def matlab_workers_flag = flag('matlab_workers', params.matlab_workers)
    def matlab_threads_flag = flag('matlab_threads', params.matlab_threads)
    def matlab_timeout_flag = flag('matlab_timeout', params.matlab_timeout)
    def blind_z_slices_flag = flag('blind_z_slices', params.blind_z_slices)
    def snr_weight_cap_flag = flag('snr_weight_cap', params.snr_weight_cap)
    def prefetch_chunks_flag = flag('prefetch_chunks', params.prefetch_chunks)
    def decon_workers_flag = flag('decon_workers', params.decon_workers)
    def overlap_xy_flag  = flag('overlap_xy', params.overlap_xy)
    def vram_gb_flag     = flag('vram_gb', params.vram_gb)
    def pyramid_max_downsample_flag = flag('pyramid_max_downsample', params.pyramid_max_downsample)
    def cache_dir_flag   = flag('cache_dir', params.psf_cache_dir)
    def no_psf_cache_flag = params.no_psf_cache ? "--no_psf_cache"                    : ""
    def shell_quote = { value -> "'${value.toString().replace("'", "'\\''")}'" }
    def outputRoot = output_dir.toString()
    def publishRoot = outputRoot.startsWith('/') ? outputRoot : "${workflow.launchDir}/${outputRoot}"

    """
    mkdir -p ${shell_quote(publishRoot)}

    export CONDA_PREFIX="${CONTAINER_ENV_PREFIX}"
    export CONDA_DEFAULT_ENV="app"
    export PATH="${CONTAINER_ENV_PREFIX}/bin:\${PATH}"
    export LD_LIBRARY_PATH="${CONTAINER_ENV_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

    matlab_bin="${params.matlab_bin ?: 'matlab'}"
    resolved_matlab_bin=""
    for candidate in "\${matlab_bin}" matlab /home1/apps/MATLAB/R2024a/bin/matlab; do
        if [ -n "\$candidate" ] && command -v "\$candidate" >/dev/null 2>&1; then
            resolved_matlab_bin="\$(command -v "\$candidate")"
            break
        elif [ -n "\$candidate" ] && [ -x "\$candidate" ]; then
            resolved_matlab_bin="\$candidate"
            break
        fi
    done
    if [ -z "\$resolved_matlab_bin" ]; then
        echo "ERROR: MATLAB executable not found. Checked requested matlab_bin='\$matlab_bin', PATH, and /home1/apps/MATLAB/R2024a/bin/matlab." >&2
        exit 127
    fi
    matlab_bin="\$resolved_matlab_bin"
    
    echo "=== GPU Check ==="
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
    else
        echo "nvidia-smi not found; continuing."
    fi
    echo "================="

    python3 ${projectDir}/scripts/decon_wrapper.py \\
        --image_path "${deskewed_dir}" \\
        --script_dir "${projectDir}/scripts" \\
        ${output_format_flag} \\
        ${background_flag} \\
        ${iter_flag} \\
        ${na_flag} \\
        ${detection_na_flag} \\
        ${illumination_na_flag} \\
        ${wavelength_flag} \\
        ${ni_flag} \\
        ${ns_flag} \\
        ${ni0_flag} \\
        ${tg_flag} \\
        ${tg0_flag} \\
        ${ng_flag} \\
        ${ng0_flag} \\
        ${ti0_flag} \\
        ${oversample_factor_flag} \\
        ${psf_model_flag} \\
        ${psf_mode_flag} \\
        ${light_sheet_angle_flag} \\
        ${camera_pixel_size_flag} \\
        ${magnification_flag} \\
        ${dxy_flag} \\
        ${dz_flag} \\
        ${psf_size_z_flag} \\
        ${psf_size_xy_flag} \\
        ${blind_iters_flag} \\
        ${blind_backend_flag} \\
        ${chunk_xy_flag} \\
        ${blind_max_tiles_flag} \\
        ${cupy_fft_engine_flag} \\
        ${adaptive_scout_iters_flag} \\
        ${adaptive_keep_tiles_flag} \\
        ${tile_selection_strategy_flag} \\
        ${coarse_region_rows_flag} \\
        ${coarse_region_columns_flag} \\
        ${coarse_region_limit_flag} \\
        ${decon_chunk_xy_flag} \\
        ${pad_xy_flag} \\
        ${pad_z_flag} \\
        ${blind_peak_normalization_flag} \\
        ${blind_peak_gamma_max_flag} \\
        ${blind_latent_update_period_flag} \\
        ${blind_workers_flag} \\
        ${matlab_workers_flag} \\
        ${matlab_threads_flag} \\
        --matlab_bin "\$matlab_bin" \\
        ${matlab_timeout_flag} \\
        ${blind_z_slices_flag} \\
        ${snr_weight_cap_flag} \\
        ${prefetch_chunks_flag} \\
        ${decon_workers_flag} \\
        ${overlap_xy_flag} \\
        ${vram_gb_flag} \\
        ${pyramid_max_downsample_flag} \\
        ${cache_dir_flag} \\
        ${no_psf_cache_flag}

    PYTHONPATH="${projectDir}/scripts:\${PYTHONPATH:-}" python3 -c "from pathlib import Path; from ome_zarr_io import zip_ome_zarr_to_ozx; [zip_ome_zarr_to_ozx(path, path.with_suffix('').with_suffix('.ozx')) for path in sorted(Path('.').glob('DB2_*.ome.zarr'))]"
    if [ -f "estimated_psf.tif" ]; then
        cp -f "estimated_psf.tif" ${shell_quote(publishRoot)}/
    fi
    for output_archive in DB2_*.ozx; do
        [ -e "\$output_archive" ] || continue
        cp -f "\$output_archive" ${shell_quote(publishRoot)}/
    done
    rm -rf DB2_*.ome.zarr
    """
}

process EXPORT_OUTPUT_FORMAT {
    tag "${output_format}"
    module 'singularity/3.9.9'
    container WORKFLOW_CONTAINER_IMAGE
    scratch true

    input:
    path decon_outputs
    val output_format

    script:
    def outputRoot = params.output_dir.toString()
    def outputPrefix = outputRoot.startsWith('/') ? outputRoot : "${workflow.launchDir}/${outputRoot}"
    """
    export CONDA_PREFIX="${CONTAINER_ENV_PREFIX}"
    export CONDA_DEFAULT_ENV="app"
    export PATH="${CONTAINER_ENV_PREFIX}/bin:\${PATH}"
    export LD_LIBRARY_PATH="${CONTAINER_ENV_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

    python3 ${projectDir}/scripts/export_ome_zarr_to_tiff.py \\
        --input "\$PWD" \\
        --output "${outputPrefix}/deconvolved_tiff" \\
        --output-format "${output_format}"
    """
}
