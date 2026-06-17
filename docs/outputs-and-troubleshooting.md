# Outputs and Troubleshooting

This page summarizes the files produced by each process and the most common
places to check when a run fails.

## Published Output Layout

For a full light-sheet run:

```text
<output_dir>/
|-- shear/
|   `-- CH00_000000.tif
|-- Top_shear/
|   |-- CH00_000000.tif
|   |-- note.txt
|   `-- ...
|-- estimated_psf.tif
`-- deconvolved/
    |-- DB2_CH00_000000.tif
    `-- ...
```

For decon-only runs, `shear` and `Top_shear` are not created by the workflow.
`DB2_*` outputs are published to `deconvolved`, and the merged blind PSF is
published as `<output_dir>/estimated_psf.tif`.

## Process Work Directories

Nextflow writes detailed logs in process work directories. For a failed task,
check:

```text
.command.sh
.command.out
.command.err
.command.log
```

The process stdout is especially useful because the Python scripts print volume
shape, chunk sizes, selected workers, MATLAB chunk progress, and deconvolution
chunk timing.

## Deskew Output Checks

Expected deskew products:

```text
shear/CH##_######.tif
Top_shear/CH##_######.tif
Top_shear/note.txt
```

If `shear` exists but `Top_shear` does not, the failure likely happened during
resize, rotation, permutation, or final TIFF writing.

If neither output exists, check input discovery and MATLAB startup first.

## Deconvolution Output Checks

Expected deconvolution products:

```text
estimated_psf.tif
DB2_<input_stem>.tif
```

If `estimated_psf.tif` exists but no `DB2_*` files exist, PSF estimation
completed and the failure likely happened during OTF creation, GPU context
creation, Dask chunk execution, or TIFF output writing.

If no PSF exists, inspect the blind PSF estimation messages first.

## Common Errors

### Multiple Channels Selected

Select TIFFs from one channel unless you intentionally want one estimated PSF
applied across multiple wavelengths. The workflow estimates one PSF from the
first selected TIFF and applies it to all selected TIFFs.

### No TIFFs Found During Deskew

The deskew process looks in:

```text
<image_path>/<cell_name>/
```

Make sure `image_path` is the parent directory and `cell_name` is the folder
inside it.

### No Matching Deconvolution TIFFs

The deconvolution wrapper accepts `CH*.tif` and `CH*.tiff`, but the stem must
match:

```text
CH<channel>_<timepoint>
CH<channel>_<timepoint>_registered_consistent
```

Examples that match:

```text
CH0_0.tif
CH00_000000.tif
CH1_12_registered_consistent.tiff
```

If filters are supplied, confirm that `channels` and `timepoints` match the
numbers parsed from the filename.

### `dxy` Resolution Fails

If `dxy <= 0`, both `camera_pixel_size` and `magnification` must be provided.
Otherwise, set `dxy` directly.

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
`deconvblind` availability, TIFF compatibility, an all-zero seed PSF, or invalid
optical parameters.

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
