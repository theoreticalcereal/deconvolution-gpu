// modules/static_deconvolution.nf
process STATIC_DECON {
    tag "${cell_name}"
    publishDir "${output_dir}/deconvolved", mode: 'copy'
    maxForks 8

    input:
    val  deskewed_dir  
    val  cell_name
    val  psf_path  
    val  psf_file
    val  background
    val  iter
    val  output_dir 

    output:
    path "DB2_*.tif", emit: decon_output

    script:
    """
    module load cuda/11.8

    # Search the deskewed directory for the generated file
    TARGET_IMAGE=\$(ls ${deskewed_dir}/CH*_registered_consistent.tif ${deskewed_dir}/CH*_registered_consistent.tiff 2>/dev/null | head -n 1)

    if [ -z "\$TARGET_IMAGE" ]; then
        echo "Error: No deskewed images found in ${deskewed_dir}"
        exit 1
    fi

    python3 ${projectDir}/scripts/decon_wrapper.py \\
        --image_path "\$TARGET_IMAGE" \\
        --psf_path ${psf_path} \\
        --psf_file ${psf_file} \\
        --background ${background} \\
        --iter ${iter}
    """
}