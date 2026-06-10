#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules/deskew'
include { DECON }  from './modules/deconvolution'
include { PSF_SANITY_CHECK } from './modules/psf_sanity_check'

workflow {
    if (params.decon_only) {
        DECON(
            params.decon_input_dir ?: params.image_path,
            params.cell_name,
            params.background,
            params.iter,
            params.output_dir
        )
    } else if (params.psf_sanity_check && params.psf_sanity_input_dir) {
        PSF_SANITY_CHECK(
            params.psf_sanity_input_dir,
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

        if (params.psf_sanity_check) {
            PSF_SANITY_CHECK(
                DESKEW.out.deskewed_path,
                params.output_dir
            )
        } else {
            DECON(
                DESKEW.out.deskewed_path,
                params.cell_name,
                params.background,
                params.iter,
                params.output_dir
            )
        }
    }
}
