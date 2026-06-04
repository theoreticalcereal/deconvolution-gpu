#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules/deskew'
include { BLIND_DECON } from './modules/blind_decon'

params.image_path = ''
params.cell_name = 'Cell'
params.cell_index = ''
params.channels = '0'
params.timepoints = '0'
params.dx = 0.167
params.dz = 0.200
params.angle = 45
params.flip = 1
params.psf_path = ''
params.psf_file = ''
params.background = 0
params.iter = 10
params.output_dir = '/work/bioinformatics/s249154/deconvolution-skeleton/output'

workflow {
    DESKEW(
        params.image_path,
        params.cell_name,
        params.cell_index,
        params.channels,
        params.timepoints,
        params.dx,
        params.dz,
        params.angle,
        params.flip
    )

    BLIND_DECON(
        DESKEW.out.deskewed_path,
        params.psf_path,
        params.psf_file,
        params.background,
        params.iter,
        params.output_dir
    )
}