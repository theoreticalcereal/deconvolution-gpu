process DESKEW {
    tag "${cell_name ?: 'deskew'}"

    publishDir "${params.output_dir}", mode: 'copy'

    input:
    val image_path
    val cell_name
    val cell_index
    val dx
    val dz
    val angle
    val flip
    val output_dir
    path decon_runtime

    output:
    path "Top_shear", emit: deskewed_path
    path "shear", optional: true, emit: shear_output

    script:
    """
    if [ ! -x "${decon_runtime}/decon_env/bin/python3" ] && [ ! -x "${decon_runtime}/decon_env/bin/python" ]; then
        echo "ERROR: no supported decon runtime found at ${decon_runtime}" >&2
        exit 1
    fi
    export CONDA_PREFIX="${decon_runtime}/decon_env"
    export CONDA_DEFAULT_ENV=decon_env
    export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}

    python3 ${projectDir}/scripts/chunked_deskew.py \\
        --image_path "${image_path}" \\
        --cell_name "${cell_name}" \\
        --cell_index "${cell_index}" \\
        --dx ${dx} \\
        --dz ${dz} \\
        --angle ${angle} \\
        --flip ${flip} \\
        --output_dir .
    """
}

process BUILD_DECON_CONTAINER {
    tag "decon_env"

    cpus 2
    memory '8 GB'
    queue 'super'

    output:
    path "decon_runtime", emit: image

    script:
    """
    set -euo pipefail
    mkdir -p decon_runtime

    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda is required to build the deconvolution environment." >&2
        exit 127
    fi

    export CONDA_PKGS_DIRS="\$PWD/.conda_pkgs"
    conda create -y -p .conda_libmamba -c conda-forge "conda>=23.7" conda-libmamba-solver
    .conda_libmamba/bin/python -m conda create -y --solver=libmamba \\
        -p decon_runtime/decon_env \\
        -c conda-forge \\
        -c bioconda \\
        --file "${projectDir}/envs/decon-conda.txt"
    decon_runtime/decon_env/bin/python -m pip install -r "${projectDir}/envs/decon-pip-requirements.txt"

    if [ ! -x decon_runtime/decon_env/bin/python3 ] && [ ! -x decon_runtime/decon_env/bin/python ]; then
        echo "ERROR: failed to build a usable decon conda environment." >&2
        exit 1
    fi
    """
}

process STAGE_DECON_INPUT {
    tag "decon_input"

    input:
    path input_tiffs

    output:
    path "decon_input", emit: decon_input_dir

    script:
    def shell_quote = { value -> "'${value.toString().replace("'", "'\\''")}'" }
    def link_commands = input_tiffs.collect { tiff ->
        "ln -s \"\$PWD/${tiff}\" ${shell_quote("decon_input/${tiff.name}")}"
    }.join('\n')

    """
    mkdir -p decon_input
    ${link_commands}
    """
}

process DECON {
    tag "decon"

    publishDir "${params.output_dir}/deconvolved", mode: 'copy', pattern: 'DB2_*'
    publishDir "${params.output_dir}", mode: 'copy', pattern: 'estimated_psf.tif'

    maxForks 8
    cpus 72
    memory '256 GB'
    clusterOptions '--gres=gpu:1'

    input:
    path deskewed_dir
    val  background
    val  iter
    val  output_dir
    path decon_runtime

    output:
    path "DB2_*", emit: decon_output
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
    def chunk_xy_flag    = flag('chunk_xy', params.chunk_xy)
    def decon_chunk_xy_flag = flag('decon_chunk_xy', params.decon_chunk_xy)
    def pad_xy_flag      = flag('pad_xy', params.pad_xy)
    def pad_z_flag       = flag('pad_z', params.pad_z)
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
    def cache_dir_flag   = flag('cache_dir', params.psf_cache_dir)
    def no_psf_cache_flag = params.no_psf_cache ? "--no_psf_cache"                    : ""

    """
    if [ ! -x "${decon_runtime}/decon_env/bin/python3" ] && [ ! -x "${decon_runtime}/decon_env/bin/python" ]; then
        echo "ERROR: no supported decon runtime found at ${decon_runtime}" >&2
        exit 1
    fi
    export CONDA_PREFIX="${decon_runtime}/decon_env"
    export CONDA_DEFAULT_ENV=decon_env
    export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}

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
        ${chunk_xy_flag} \\
        ${decon_chunk_xy_flag} \\
        ${pad_xy_flag} \\
        ${pad_z_flag} \\
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
        ${cache_dir_flag} \\
        ${no_psf_cache_flag}
    """
}
