# Workflow Output

```text
workflow/output/
|-- DB2_<sample>.ome.zarr/
|-- estimated_psf.tif
`-- deconvolved_tiff/
    `-- DB2_<sample>.tif    # only when output_formats = tiff
```

`estimated_psf.tif` is copied to the configured output directory. Native
deconvolved image volumes are written as `DB2_*.ome.zarr`. TIFF stacks are only
written when requested with `output_formats = tiff`.
