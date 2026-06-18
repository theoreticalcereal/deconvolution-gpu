#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules'
include { BUILD_DECON_CONTAINER } from './modules'
include { STAGE_DECON_INPUT } from './modules'
include { DECON }  from './modules'

workflow {
    if (params.build_decon_container) {
        BUILD_DECON_CONTAINER()
        decon_container_ch = BUILD_DECON_CONTAINER.out.image
    } else {
        decon_container_ch = Channel.value("${projectDir}/images/decon_env.sif")
    }

    if (params.decon_only) {
        if (params.input) {
            input_patterns = params.input
            if (!(input_patterns instanceof List)) {
                input_text = input_patterns.toString().trim()
                if (input_text.startsWith('[') && input_text.endsWith(']')) {
                    input_text = input_text.substring(1, input_text.length() - 1).trim()
                    input_patterns = input_text ? input_text.split(/\s*,\s*/) as List : []
                } else {
                    input_patterns = [input_text]
                }
            }
            input_tiffs_ch = Channel
                .fromList(input_patterns)
                .map { input_pattern ->
                    input_text = input_pattern.toString().trim()
                    input_text = input_text.replaceAll(/^['"]|['"]$/, '')
                    file(input_text, checkIfExists: true)
                }
                .collect()
            STAGE_DECON_INPUT(input_tiffs_ch)
            decon_input_ch = STAGE_DECON_INPUT.out.decon_input_dir
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
            params.background,
            params.iter,
            params.output_dir,
            decon_container_ch
        )
    }
}
