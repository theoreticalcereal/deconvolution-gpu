process BLIND_DECON {
    publishDir { "${output_dir}/devoluted" }, mode: 'copy'

    input:
    path deskewed_path
    path psf_path
    val psf_file
    val background
    val iter
    path output_dir

    output:
    path "DB2_*", emit: decon_output

    script:
    """
    module load matlab/2024a

    python3 ${projectDir}/scripts/decon_wrapper.py \
        --image_path ${deskewed_path} \
        --psf_path ${psf_path} \
        --psf_file ${psf_file} \
        --background ${background} \
        --iterations ${iter} \
        --output_dir ${output_dir}
    """
}