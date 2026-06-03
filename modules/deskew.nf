process DESKEW {
    input:
    val image_path
    val cell_name
    val cell_index
    val channels
    val timepoints
    val dx
    val dz
    val angle
    val flip

    output:
    val "${image_path}/Top_shear${angle}_mlv2_${cell_name}${cell_index}", emit: deskewed_path

    script:
    """
    module load matlab/2022b
    python3 ${projectDir}/scripts/deskew_wrapper.py \
        --image_path ${image_path} \
        --cell_name ${cell_name} \
        --cell_index ${cell_index} \
        --channels ${channels} \
        --timepoints ${timepoints} \
        --dx ${dx} \
        --dz ${dz} \
        --angle ${angle} \
        --flip ${flip}
    """

}