process PSF_SANITY_CHECK {
    conda "${projectDir}/environment.yml"
    tag "psf_sanity"

    publishDir "${params.output_dir}/psf_sanity", mode: 'copy'

    cpus 4
    memory '32 GB'
    queue 'super'

    input:
    val deskewed_dir
    val output_dir

    output:
    path "psf_sanity", emit: psf_sanity_output

    script:
    def na_flag          = params.na          ? "--na ${params.na}"                   : ""
    def wavelength_flag  = params.wavelength  ? "--wavelength ${params.wavelength}"   : ""
    def ni_flag          = params.ni          ? "--ni ${params.ni}"                   : ""
    def dxy_flag         = params.dxy         ? "--dxy ${params.dxy}"                 : ""
    def dz_flag          = params.dz          ? "--dz ${params.dz}"                   : ""
    def psf_size_z_flag  = params.psf_size_z  ? "--psf_size_z ${params.psf_size_z}"   : ""
    def psf_size_xy_flag = params.psf_size_xy ? "--psf_size_xy ${params.psf_size_xy}" : ""
    def blind_iters_flag = params.blind_iters ? "--blind_iters ${params.blind_iters}" : ""
    def chunk_xy_flag    = params.chunk_xy    ? "--chunk_xy ${params.chunk_xy}"       : ""
    def pad_xy_flag      = params.pad_xy      ? "--pad_xy ${params.pad_xy}"           : ""
    def pad_z_flag       = params.pad_z != null ? "--pad_z ${params.pad_z}"           : ""
    def blind_workers_flag = params.blind_workers ? "--blind_workers ${params.blind_workers}" : ""
    def matlab_workers_flag = params.matlab_workers ? "--matlab_workers ${params.matlab_workers}" : ""
    def matlab_threads_flag = params.matlab_threads ? "--matlab_threads ${params.matlab_threads}" : ""
    def matlab_timeout_flag = params.matlab_timeout ? "--matlab_timeout ${params.matlab_timeout}" : ""
    def blind_z_slices_flag = params.blind_z_slices ? "--blind_z_slices ${params.blind_z_slices}" : ""
    def snr_weight_cap_flag = params.snr_weight_cap != null ? "--snr_weight_cap ${params.snr_weight_cap}" : ""
    def prefetch_chunks_flag = params.prefetch_chunks ? "--prefetch_chunks ${params.prefetch_chunks}" : ""
    def vram_gb_flag     = params.vram_gb     ? "--vram_gb ${params.vram_gb}"         : ""
    def tiff_index_flag  = params.psf_sanity_tiff_index != null ? "--tiff_index ${params.psf_sanity_tiff_index}" : ""
    def sanity_xy_flag   = params.psf_sanity_xy != null ? "--sanity_xy ${params.psf_sanity_xy}" : ""

    """
    module load matlab/2024a

    mkdir -p psf_sanity

    python3 ${projectDir}/scripts/compare_psf.py \\
        --image_path "${deskewed_dir}" \\
        --output_dir psf_sanity \\
        --script_dir "${projectDir}/scripts" \\
        ${tiff_index_flag} \\
        ${sanity_xy_flag} \\
        ${na_flag} \\
        ${wavelength_flag} \\
        ${ni_flag} \\
        ${dxy_flag} \\
        ${dz_flag} \\
        ${psf_size_z_flag} \\
        ${psf_size_xy_flag} \\
        ${blind_iters_flag} \\
        ${chunk_xy_flag} \\
        ${pad_xy_flag} \\
        ${pad_z_flag} \\
        ${blind_workers_flag} \\
        ${matlab_workers_flag} \\
        ${matlab_threads_flag} \\
        ${matlab_timeout_flag} \\
        ${blind_z_slices_flag} \\
        ${snr_weight_cap_flag} \\
        ${prefetch_chunks_flag} \\
        ${vram_gb_flag}
    """
}
