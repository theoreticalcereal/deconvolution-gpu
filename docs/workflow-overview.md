# Workflow Overview

The deconvolution package contains only the deconvolution half of the original
combined workflow.

```text
BUILD_DECON_CONTAINER
STAGE_DECON_INPUT
DECON
EXPORT_OUTPUT_FORMAT  # only when output_formats = tiff
```

`STAGE_DECON_INPUT` normalizes selected input files to OME-Zarr and preserves
original filenames in `original_filenames.tsv`. `DECON` reads those volumes,
estimates one blind PSF from the first selected volume, and applies the resulting
PSF to all selected volumes in the run.

Use profile `light_sheet` to preserve light-sheet PSF behavior for data that has
already been deskewed by `deskew-gpu`.
