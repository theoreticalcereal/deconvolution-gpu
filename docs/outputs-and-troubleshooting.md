# Outputs and Troubleshooting

This page summarizes the files produced by each process and the most common
places to check when a run fails.

## Published Output Layout

For a full light-sheet run:

```text
<output_dir>/
|-- Top_shear/
|   |-- <sample>.ome.zarr/
|   `-- note.txt
|-- estimated_psf.tif
|-- deconvolved/
|   `-- DB2_<sample>.ome.zarr/
`-- neuroglancer/
    `-- layers.json
```

For decon-only runs, `Top_shear` is not created by the workflow.
`DB2_*` OME-Zarr outputs are published to `deconvolved`, and the merged blind
PSF is published as `<output_dir>/estimated_psf.tif`.

## Process Work Directories

Nextflow writes detailed logs in process work directories. For a failed task,
check:

```text
.command.sh
.command.out
.command.err
.command.log
```

The process stdout is useful because the Python scripts print volume shape,
chunk sizes, selected workers, MATLAB chunk progress, and deconvolution chunk
timing.

## Input Staging Checks

Expected staging product:

```text
input_zarr/<sample>.ome.zarr/
```

If staging fails, confirm the selected file extension is supported and that the
runtime includes the optional reader needed for that format. TIFF and existing
OME-Zarr use the core path. CZI, ND2, LIF, and HDF5 use optional dependencies
declared in the decon runtime environment.

For existing OME-Zarr inputs, the selected path should be the `.ome.zarr`
directory itself.

## Deskew Output Checks

Expected deskew products:

```text
Top_shear/<sample>.ome.zarr/
Top_shear/note.txt
```

If staging succeeds but deskew does not write `Top_shear`, check the resolved
geometry parameters and the input volume shape in `.command.out`.

## Deconvolution Output Checks

Expected deconvolution products:

```text
estimated_psf.tif
deconvolved/DB2_<sample>.ome.zarr/
```

If `estimated_psf.tif` exists but no `DB2_*` OME-Zarr output exists, PSF
estimation completed and the failure likely happened during OTF creation, GPU
context creation, Dask chunk execution, or OME-Zarr writing.

If no PSF exists, inspect the blind PSF estimation messages first.

## Common Errors

### Mixed Optical Configurations

Select inputs from one channel and optical configuration unless you
intentionally want one estimated PSF applied across multiple wavelengths. The
workflow estimates one PSF from the first sorted input volume and applies it to
all inputs in the same `DECON` call.

### Unsupported Input Type

`STAGE_DECON_INPUT` accepts TIFF, OME-Zarr, CZI, ND2, LIF, and HDF5. Other
formats must be converted before running the workflow.

### Optional Reader Missing

Reader-specific import errors during staging mean the current runtime is
missing the optional package for the selected format. Check
`workflow/envs/decon-pip-requirements.txt` and the `BUILD_DECON_CONTAINER`
output.

### `dxy` Resolution Fails

If `dxy` is omitted, both `camera_pixel_size` and `magnification` must be
provided. Otherwise, set `dxy` directly.

### MATLAB `deconvblind` Chunks Timeout

Reduce one or more of:

- `chunk_xy`
- `blind_z_slices`
- `blind_workers`
- `matlab_workers`
- `blind_iters`

You can also increase `matlab_timeout` if chunks are progressing but slow.

### First Three PSF Chunks Fail

The PSF estimator aborts after the first three failed chunks when no chunk has
successfully produced a PSF. This usually points to MATLAB availability,
`deconvblind` availability, temporary TIFF compatibility, an all-zero seed PSF,
or invalid optical parameters.

### GPU Memory Errors

Reduce `decon_chunk_xy`, reduce `decon_workers`, or provide a conservative
`vram_gb` value. The deconvolution chunks are full-Z, so deep stacks need
smaller XY tiles.

### Tile Boundary Artifacts

Increase `overlap_xy` if boundaries are visible in the deconvolved result.
The default is derived from PSF XY size and capped at 48 pixels.

### PSF Cache Reuse Is Unexpected

Use:

```text
--no_psf_cache true
```

or change `psf_cache_dir`. The cache key includes input file metadata and the
main PSF parameters, but disabling cache is the simplest way to force a fresh
blind estimate.

## Performance Tuning Order

Tune blind PSF estimation first:

1. Start with conservative `matlab_workers = 1`.
2. Use `blind_z_slices` to keep MATLAB tile sizes bounded.
3. Increase `chunk_xy` only if MATLAB and RAM are stable.
4. Increase `matlab_workers` only after single-worker behavior is reliable.

Then tune GPU deconvolution:

1. Let `decon_chunk_xy <= 0` auto-size once.
2. If GPU memory fails, set a smaller explicit `decon_chunk_xy`.
3. Increase `decon_workers` only if the GPU has headroom.
4. Increase `overlap_xy` only when visual artifacts justify the extra work.
