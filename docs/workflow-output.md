# workflow/output

`workflow/output` is the default published output directory for this workflow.

Nextflow runs tasks in isolated work directories so parallel processes do not
contend for the same files. Those work directories are useful for debugging,
but final products are published into `workflow/output` with `publishDir`.

Astrocyte can clean up Nextflow work directories after a run to reduce disk
usage. Files in `workflow/output` are not removed by that cleanup.

## Published Files

A full light-sheet run publishes:

```text
workflow/output/
|-- Top_shear/
|   |-- <sample>.ome.zarr/
|   `-- note.txt
|-- estimated_psf.tif
|-- deconvolved/
|   `-- DB2_<sample>.ome.zarr/
`-- neuroglancer/
    `-- layers.json
```

A decon-only run publishes:

```text
workflow/output/
|-- estimated_psf.tif
|-- deconvolved/
|   `-- DB2_<sample>.ome.zarr/
`-- neuroglancer/
    `-- layers.json
```

The process-local `input_zarr/` directory is an intermediate staging product.
It is normally kept in the Nextflow work directory rather than published.

## Process Publishing

`STAGE_DECON_INPUT` converts supported selected files into normalized
OME-Zarr volumes for downstream processes.

`DESKEW` publishes deskewed OME-Zarr volumes and `note.txt` to
`workflow/output/Top_shear/`.

`DECON` publishes final deconvolved OME-Zarr volumes to
`workflow/output/deconvolved/`. It also publishes the merged blind PSF as
`workflow/output/estimated_psf.tif`.

`CONVERT_TIFFS_TO_NEUROGLANCER` now writes a Neuroglancer layer manifest that
points at the OME-Zarr outputs. Legacy TIFF conversion support remains in the
script for compatibility, but native OME-Zarr output does not need a
separate visualization data conversion step.
