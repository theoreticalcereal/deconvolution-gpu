process DECON {
    conda "${projectDir}/environment.yml"
    tag "${cell_name}"

    publishDir "${params.output_dir}/deconvolved", mode: 'copy'

    maxForks 8
    cpus 8
    memory '32 GB'
    clusterOptions '--gres=gpu:1'

    input:
    val  deskewed_dir
    val  cell_name
    val  background
    val  iter
    val  output_dir

    output:
    path "DB2_*", emit: decon_output

    script:
    // Build optional optical-parameter flags only if the user supplied them.
    // All have defaults in decon_wrapper.py so omitting is safe but not recommended
    def na_flag          = params.na          ? "--na ${params.na}"                   : ""
    def wavelength_flag  = params.wavelength  ? "--wavelength ${params.wavelength}"   : ""
    def ni_flag          = params.ni          ? "--ni ${params.ni}"                   : ""
    def dxy_flag         = params.dxy         ? "--dxy ${params.dxy}"                 : ""
    def dz_flag          = params.dz          ? "--dz ${params.dz}"                   : ""
    def psf_size_z_flag  = params.psf_size_z  ? "--psf_size_z ${params.psf_size_z}"   : ""
    def psf_size_xy_flag = params.psf_size_xy ? "--psf_size_xy ${params.psf_size_xy}" : ""
    def blind_iters_flag = params.blind_iters ? "--blind_iters ${params.blind_iters}" : ""
    def chunk_xy_flag    = params.chunk_xy    ? "--chunk_xy ${params.chunk_xy}"       : ""
    def decon_chunk_xy_flag = params.decon_chunk_xy ? "--decon_chunk_xy ${params.decon_chunk_xy}" : ""
    def pad_xy_flag      = params.pad_xy      ? "--pad_xy ${params.pad_xy}"           : ""
    def blind_workers_flag = params.blind_workers ? "--blind_workers ${params.blind_workers}" : ""
    def matlab_threads_flag = params.matlab_threads ? "--matlab_threads ${params.matlab_threads}" : ""
    def matlab_timeout_flag = params.matlab_timeout ? "--matlab_timeout ${params.matlab_timeout}" : ""
    def snr_weight_cap_flag = params.snr_weight_cap != null ? "--snr_weight_cap ${params.snr_weight_cap}" : ""
    def prefetch_chunks_flag = params.prefetch_chunks ? "--prefetch_chunks ${params.prefetch_chunks}" : ""
    def decon_workers_flag = params.decon_workers ? "--decon_workers ${params.decon_workers}" : ""
    def overlap_xy_flag  = params.overlap_xy  ? "--overlap_xy ${params.overlap_xy}"   : ""
    def vram_gb_flag     = params.vram_gb     ? "--vram_gb ${params.vram_gb}"         : ""
    def cache_dir_flag   = params.psf_cache_dir ? "--cache_dir ${params.psf_cache_dir}" : ""
    def no_psf_cache_flag = params.no_psf_cache ? "--no_psf_cache"                    : ""
    def no_blind_flag    = params.no_blind    ? "--no_blind"                          : ""

    """
    module load cuda/11.8
    module load matlab/2024a
    export LD_LIBRARY_PATH=\${CUDA_HOME:-}/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
    
    echo "=== GPU Check ==="
    nvidia-smi
    echo "================="

    python3 ${projectDir}/scripts/decon_wrapper.py \\
        --image_path "${deskewed_dir}" \\
        --background ${background} \\
        --iter ${iter} \\
        --script_dir "${projectDir}/scripts" \\
        ${na_flag} \\
        ${wavelength_flag} \\
        ${ni_flag} \\
        ${dxy_flag} \\
        ${dz_flag} \\
        ${psf_size_z_flag} \\
        ${psf_size_xy_flag} \\
        ${blind_iters_flag} \\
        ${chunk_xy_flag} \\
        ${decon_chunk_xy_flag} \\
        ${pad_xy_flag} \\
        ${blind_workers_flag} \\
        ${matlab_threads_flag} \\
        ${matlab_timeout_flag} \\
        ${snr_weight_cap_flag} \\
        ${prefetch_chunks_flag} \\
        ${decon_workers_flag} \\
        ${overlap_xy_flag} \\
        ${vram_gb_flag} \\
        ${cache_dir_flag} \\
        ${no_psf_cache_flag} \\
        ${no_blind_flag}
    """
}
