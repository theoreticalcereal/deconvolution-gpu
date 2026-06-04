process BLIND_DECON {
    publishDir { "${output_dir}/devoluted" }, mode: 'copy'

    input:
    path deskewed_path
    val background
    val iter
    path output_dir

    output:
    path "DB2_*", emit: decon_output

    script:
    """
    module load matlab/2024a

    # Copy PSF file to work directory first
    cp ${projectDir}/scripts/ctASLM2_510nm.tif .

    python3 ${projectDir}/scripts/decon_wrapper.py \\
        --image_path ${deskewed_path} \\
        --psf_path . \\
        --psf_file ctASLM2_510nm.tif \\
        --background ${background} \\
        --iter ${iter} \\
        --output_dir ${output_dir}
    """
}