#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules/deskew'
include { BLIND_DECON } from './modules/blind_decon'

params.image_path = '/work/bioinformatics/s249154/deconvolution-skeleton'
params.cell_name = '2026-05-18-vasculature'
params.cell_index = ''
params.channels = '1'
params.timepoints = '0'
params.dx = 0.15
params.dz = 0.318
params.angle = 45
params.flip = 1
params.psf_path = '/work/bioinformatics/s249154/deconvolution-skeleton/scripts'
params.psf_file = 'ctASLM2-510nm.tif'  
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
        params.flip,
        params.output_dir
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
