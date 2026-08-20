# Workflow Output

```text
workflow/output/
|-- DB2_<sample>.ozx
|-- estimated_psf.tif
`-- deconvolved_tiff/
    `-- DB2_<sample>.tif    # only when Output = TIFF (1x)
```

`estimated_psf.tif` is copied to the configured output directory. Non-TIFF
inputs are normalized to OME-Zarr and their native deconvolution output is
written as `DB2_*.ozx`, a zipped OME-Zarr archive.
The final form dropdown controls the output. OZX output defaults to `1x` and
can include XY downsampled levels through `2x`, `4x`, `8x`, or `16x` by
row/column stride slicing; Z is not downsampled. **TIFF (1x)** writes a TIFF
only, without OME-Zarr pyramid levels. TIFF inputs bypass normalization: a TIFF
request writes the restored array directly, while an OZX request writes it to
OME-Zarr before archiving.
