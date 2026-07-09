# Workflow Output

```text
workflow/output/
|-- DB2_<sample>.ozx
|-- estimated_psf.tif
`-- deconvolved_tiff/
    `-- DB2_<sample>.tif    # only when output_formats = tiff
```

`estimated_psf.tif` is copied to the configured output directory. Native
deconvolved image volumes are written as `DB2_*.ozx`, a zipped OME-Zarr archive.
Inside the archive, levels `1` through `4` are XY downsampled from level `0` by
row/column stride slicing at `2x, 4x, 8x, 16x`; Z is not downsampled. TIFF
stacks are only written when requested with `output_formats = tiff`.
