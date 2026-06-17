#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules'
include { BUILD_DECON_CONTAINER } from './modules'
include { DECON }  from './modules'

workflow {
    if (params.build_decon_container) {
        BUILD_DECON_CONTAINER()
        decon_container_ch = BUILD_DECON_CONTAINER.out.image
    } else {
        decon_container_ch = Channel.value("${projectDir}/images/decon_env.sif")
    }

    if (params.decon_only) {
        DECON(
            params.decon_input_dir ?: params.image_path,
            params.cell_name,
            params.background,
            params.iter,
            params.output_dir,
            decon_container_ch
        )
    } else {
        DESKEW(
            params.image_path,
            params.cell_name,
            params.cell_index,
            params.channels,
            params.timepoints,
            params.dx,
            params.dz,
            params.angle,
            params.flip,
            params.output_dir
        )

        DECON(
            DESKEW.out.deskewed_path,
            params.cell_name,
            params.background,
            params.iter,
            params.output_dir,
            decon_container_ch
        )
    }
}
