process DESKEW {
    tag "${cell_name ?: 'deskew'}"
    module 'matlab/2024a'

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

    output:
    path "Top_shear", emit: deskewed_path
    path "shear", optional: true, emit: shear_output

    script:
    """
    python3 ${projectDir}/scripts/deskew_wrapper.py \\
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
    module 'singularityce/4.1.0'

    cpus 2
    memory '8 GB'
    queue 'super'

    input:
    val container_image
    val conda_env_archive

    output:
    path "decon_runtime", emit: image

    script:
    """
    set -euo pipefail
    mkdir -p decon_runtime

    if command -v module >/dev/null 2>&1; then
        module load singularity/3.9.9 || module load singularity || true
    fi

    packaged_sif="${projectDir}/images/decon_env.sif"
    conda_env_archive="${conda_env_archive}"
    is_usable_sif() {
        command -v singularity >/dev/null 2>&1 || return 1
        [ -s "\$1" ] || return 1
        ! head -n 1 "\$1" | grep -q 'git-lfs.github.com/spec' || return 1
        singularity sif list "\$1" >/dev/null 2>&1
    }
    container_image="${container_image}"
    if is_usable_sif "\$container_image"; then
        ln -s "\$container_image" decon_runtime/decon_env.sif
    elif is_usable_sif "\$packaged_sif"; then
        ln -s "\$packaged_sif" decon_runtime/decon_env.sif
    else
        magic_offset="\$(LC_ALL=C grep -abo -m 1 'SIF_MAGIC' "\$packaged_sif" | cut -d: -f1 || true)"
        if [ -n "\$magic_offset" ] && [ "\$magic_offset" -gt 1 ]; then
            start_byte="\$((magic_offset + 1))"
            tail -c "+\$start_byte" "\$packaged_sif" > decon_runtime/decon_env.sif
        fi
    fi

    if ! is_usable_sif decon_runtime/decon_env.sif; then
        rm -f decon_runtime/decon_env.sif
        if [ -s "\$conda_env_archive" ] && ! head -n 1 "\$conda_env_archive" | grep -q 'git-lfs.github.com/spec'; then
            mkdir -p decon_runtime/decon_env
            tar -xzf "\$conda_env_archive" -C decon_runtime/decon_env
            if [ -x decon_runtime/decon_env/bin/conda-unpack ]; then
                decon_runtime/decon_env/bin/conda-unpack || true
            fi
        fi
    fi

    if ! is_usable_sif decon_runtime/decon_env.sif && [ ! -x decon_runtime/decon_env/bin/python3 ] && [ ! -x decon_runtime/decon_env/bin/python ]; then
        echo "ERROR: failed to prepare a usable decon runtime." >&2
        echo "Expected a valid SIF at \$container_image or \$packaged_sif, or a packed conda env at \$conda_env_archive." >&2
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
    module 'singularity/3.9.9:matlab/2024a'

    publishDir "${params.output_dir}/deconvolved", mode: 'copy', pattern: 'DB2_*'
    publishDir "${params.output_dir}", mode: 'copy', pattern: 'estimated_psf.tif'

    maxForks 8
    cpus 8
    memory '32 GB'
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
    def matlab_bin_flag = flag('matlab_bin', params.matlab_bin)
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
    runtime_prefix=()
    if [ -f "${decon_runtime}/decon_env.sif" ]; then
        runtime_prefix=(singularity exec --nv "${decon_runtime}/decon_env.sif")
        export LD_LIBRARY_PATH=\${CONDA_PREFIX:-/opt/conda/envs/decon_env}/lib:\${LD_LIBRARY_PATH:-}
    elif [ -f "${decon_runtime}" ]; then
        runtime_prefix=(singularity exec --nv "${decon_runtime}")
        export LD_LIBRARY_PATH=\${CONDA_PREFIX:-/opt/conda/envs/decon_env}/lib:\${LD_LIBRARY_PATH:-}
    elif [ -x "${decon_runtime}/decon_env/bin/python3" ]; then
        source "${decon_runtime}/decon_env/bin/activate"
        export LD_LIBRARY_PATH=\${CONDA_PREFIX:-${decon_runtime}/decon_env}/lib:\${LD_LIBRARY_PATH:-}
    else
        echo "ERROR: no supported decon runtime found at ${decon_runtime}" >&2
        exit 1
    fi
    
    echo "=== GPU Check ==="
    if [ "\${#runtime_prefix[@]}" -gt 0 ]; then
        "\${runtime_prefix[@]}" nvidia-smi || echo "nvidia-smi not found inside container; continuing."
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
    else
        echo "nvidia-smi not found; continuing."
    fi
    echo "================="

    "\${runtime_prefix[@]}" python3 ${projectDir}/scripts/decon_wrapper.py \\
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
        ${matlab_bin_flag} \\
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
