process DECON {
    module 'mamba/2.3.0'
    tag "${cell_name}"

    publishDir "${params.output_dir}/deconvolved", mode: 'copy'

    maxForks 8
    cpus 4
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
    def pad_xy_flag      = params.pad_xy      ? "--pad_xy ${params.pad_xy}"           : ""
    def no_blind_flag    = params.no_blind    ? "--no_blind"                           : ""

    """
    module load cuda/11.8
    export LD_LIBRARY_PATH=\${CUDA_HOME:-}/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
    echo "=== GPU Check ==="
    nvidia-smi
    echo "================="

    if [ ! -d "${projectDir}/.conda_env" ]; then
        echo "Building conda environment via Mamba..."
        mamba env create -p ${projectDir}/.conda_env -f ${projectDir}/environment.yml -y --quiet
    fi

    mamba run -p ${projectDir}/.conda_env ${projectDir}/.conda_env/bin/python3 ${projectDir}/scripts/decon_wrapper.py \\
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
        ${pad_xy_flag} \\
        ${no_blind_flag}
    """
}
