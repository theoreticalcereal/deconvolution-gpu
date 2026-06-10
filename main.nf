#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules/deskew'
include { DECON }  from './modules/deconvolution'

workflow {
    if (params.decon_only) {
        DECON(
            params.decon_input_dir ?: params.image_path,
            params.cell_name,
            params.background,
            params.iter,
            params.output_dir
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
            params.output_dir
        )
    }
}
