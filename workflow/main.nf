#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules'
include { BUILD_DECON_CONTAINER } from './modules'
include { STAGE_DECON_INPUT } from './modules'
include { DECON }  from './modules'

workflow {
    if (params.input) {
        log.info "Selected input TIFF(s): ${params.input}"
        input_tiffs_ch = Channel.fromPath(params.input, checkIfExists: true).collect()
        STAGE_DECON_INPUT(input_tiffs_ch)
        selected_input_dir_ch = STAGE_DECON_INPUT.out.decon_input_dir
    } else {
        selected_input_dir_ch = Channel.empty()
    }

    if (params.build_decon_container) {
        BUILD_DECON_CONTAINER()
        decon_container_ch = BUILD_DECON_CONTAINER.out.image
    } else {
        decon_container_ch = Channel.value("${projectDir}/images/decon_env.sif")
    }

    if (params.decon_only) {
        if (params.input) {
            decon_input_ch = selected_input_dir_ch
        } else {
            decon_input_ch = Channel.value(params.decon_input_dir ?: params.image_path)
        }

        DECON(
            decon_input_ch,
            params.background,
            params.iter,
            params.output_dir,
            decon_container_ch
        )
    } else {
        deskew_image_path_ch = params.input ? selected_input_dir_ch : Channel.value(params.image_path)
        deskew_cell_name = params.input ? '' : params.cell_name

        DESKEW(
            deskew_image_path_ch,
            deskew_cell_name,
            params.cell_index,
            params.dx,
            params.dz,
            params.angle,
            params.flip,
            params.output_dir
        )

        DECON(
            DESKEW.out.deskewed_path,
            params.background,
            params.iter,
            params.output_dir,
            decon_container_ch
        )
    }
}
