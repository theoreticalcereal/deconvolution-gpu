process DECON {
    tag "${cell_name}"
    
    publishDir "${params.output_dir}/deconvolved", mode: 'copy'
    
    maxForks 8
    cpus 4
    memory '32 GB'

    clusterOptions '--gres=gpu:1'

    input:
    val  deskewed_dir  
    val  cell_name
    val  psf_path  
    val  psf_file
    val  background
    val  iter
    val  output_dir 

    output:
    path "DB2_*", emit: decon_output

    script:
    """
    module load cuda/11.8
    export LD_LIBRARY_PATH=\${CUDA_HOME:-}/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
    echo "=== GPU Check ==="
    nvidia-smi
    echo "================="

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