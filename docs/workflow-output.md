# workflow/output

`workflow/output` is the default published output directory for this workflow.

Nextflow runs tasks in isolated work directories so parallel processes do not
contend for the same files. Those work directories are useful for debugging,
but they are not the place users should look for final results. Every final
workflow product should be copied into `workflow/output` with `publishDir`.

Astrocyte can clean up Nextflow work directories after a run to reduce disk
usage. Files in `workflow/output` are not removed by that cleanup. They remain
available unless the entire workflow run is removed.

## Published Files

A full light-sheet run publishes:

```text
workflow/output/
|-- shear/
|   `-- CH##_######.tif
|-- Top_shear/
|   |-- CH##_######.tif
|   `-- note.txt
|-- estimated_psf.tif
`-- deconvolved/
    `-- DB2_CH##_######.tif
```

A decon-only run publishes:

```text
workflow/output/
|-- estimated_psf.tif
`-- deconvolved/
    `-- DB2_<input_stem>.tif
```

## Process Publishing

`DESKEW` publishes `shear/` and `Top_shear/` to `workflow/output`.

`DECON` publishes final deconvolved TIFFs to `workflow/output/deconvolved/`.
It also publishes the merged blind PSF as `workflow/output/estimated_psf.tif`.

The PSF may also be written beside the deconvolution input directory when that
location is writable, but the published copy in `workflow/output` is the stable
workflow result.
