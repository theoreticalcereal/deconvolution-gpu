process BLIND_DECON {
    input:
    val image_path
    val psf_path
    val psf_file
    val background
    val iter
    val output_dir

    output:
    val "${image_path}/DBv8_synPSFOPM_${iter}_chop${background}", emit: decon_path

    script:
    """
    module load matlab/2024a

    python3 ${projectDir}/scripts/decon_wrapper.py \
        --image_path ${image_path} \
        --psf_path ${psf_path} \
        --psf_file ${psf_file} \
        --background ${background} \
        --iter ${iter}
    """
}