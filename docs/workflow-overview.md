# Workflow Overview

The deconvolution package contains only the deconvolution half of the original
combined workflow.

```text
BUILD_DECON_CONTAINER  # skipped when decon_runtime_dir is supplied
STAGE_DECON_INPUT
DECON
EXPORT_OUTPUT_FORMAT  # only when output_formats = tiff
```

`STAGE_DECON_INPUT` normalizes selected input files to OME-Zarr and preserves
original filenames in `original_filenames.tsv`. OZX inputs are unzipped into
the same internal OME-Zarr layout. `DECON` reads those volumes, estimates one
blind PSF from the first selected volume, and applies the resulting PSF to all
selected volumes in the run.

OME-Zarr outputs are written as multiscale pyramids in task scratch space, then
zipped to `DB2_*.ozx` for publication and removed from the task directory.
Level `0` is full resolution; levels `1` through `4` are generated with
`[:, ::2, ::2]`, `[:, ::4, ::4]`, `[:, ::8, ::8]`, and `[:, ::16, ::16]`.

Use profile `light_sheet` to preserve light-sheet PSF behavior for data that has
already been deskewed by `deskew-gpu`.

Set `decon_runtime_dir` to reuse a runtime built upstream by `deskew-gpu`. The
workflow accepts a path containing `decon_env/` or `deskew_env/`, or a direct
conda environment path. The sentinel value `-1` means no external runtime was
supplied and the workflow should build its own runtime.
