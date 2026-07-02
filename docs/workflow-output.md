# Workflow Output

```text
workflow/output/
|-- DB2_<sample>.ome.zarr/
|   |-- 0/
|   |-- 1/
|   |-- 2/
|   |-- 3/
|   `-- 4/
|-- estimated_psf.tif
`-- deconvolved_tiff/
    `-- DB2_<sample>.tif    # only when output_formats = tiff
```

`estimated_psf.tif` is copied to the configured output directory. Native
deconvolved image volumes are written as `DB2_*.ome.zarr`. Levels `1` through
`4` are XY downsampled from level `0` by row/column stride slicing at
`2x, 4x, 8x, 16x`; Z is not downsampled. TIFF stacks are only written when
requested with `output_formats = tiff`.
