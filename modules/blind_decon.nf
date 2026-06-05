process BLIND_DECON {
    tag "${deskewed_path.baseName}"
    publishDir "${output_dir}/deconvolved", mode: 'copy'
    maxForks 8
    cpus 4
    memory '32 GB'

    clusterOptions '--gres=gpu:1'

    input:
    val  deskewed_path  
    val  psf_path  
    val  psf_file
    val  background
    val  iter
    val  output_dir 

    output:
    path "DB2_*.tif", emit: decon_output

    script:
    """
    module load matlab/2024a
    module load cuda/11.8

    python3 ${projectDir}/scripts/decon_wrapper.py \
        --image_path ${deskewed_path} \
        --psf_path ${psf_path} \
        --psf_file ${psf_file} \
        --background ${background} \
        --iter ${iter} \
        --output_dir ${output_dir}
    """
}
