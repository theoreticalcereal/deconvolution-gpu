#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules/deskew'
include { STATIC_DECON } from './modules/static_decon'

workflow {
    // Run the MATLAB deskew step
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

    // Run the GPU Deconvolution step using your provided PSF
    STATIC_DECON(
        DESKEW.out.deskewed_path,
        params.cell_name,
        params.psf_path,
        params.psf_file,
        params.background,
        params.iter,
        params.output_dir
    )
}