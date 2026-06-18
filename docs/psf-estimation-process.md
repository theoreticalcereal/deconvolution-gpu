# PSF Estimation Process

The deconvolution process always resolves a blind PSF before running CUDA
deconvolution. The theoretical PSF is only the starting seed for MATLAB
`deconvblind`; it is not used directly as the final deconvolution PSF.

The implementation is split across:

- `workflow/scripts/decon_wrapper.py`
- `workflow/scripts/psf_modes.py`
- `workflow/scripts/psf_estimation.py`

## Input Selection

`decon_wrapper.py` sorts and filters deconvolution input TIFFs first. The first
selected TIFF is used for blind PSF estimation.

Deconvolution input can be any TIFF stack when no channel/timepoint filters are
requested. Supported filterable stems are:

```text
CH0_0
CH1_0
CH0_0_registered_consistent
```

The filter regex is:

```text
^CH(?P<channel>\d+)_(?P<timepoint>\d+)(?:_registered_consistent)?$
```

If `channels` or `timepoints` are supplied, only files matching this pattern
contribute to both PSF estimation and deconvolution.

## Lateral Pixel Size Resolution

The PSF code needs `dxy`.

If `--dxy` is greater than zero, that value is used. If `--dxy <= 0`, both
`--camera_pixel_size` and `--magnification` must be supplied, and:

```text
dxy = camera_pixel_size / magnification
```

## Seed PSF Generation

The seed is generated with `psfmodels.make_psf` using optical parameters such
as:

- `dxy`
- `dz`
- `detection_na` or fallback `na`
- `wavelength`
- `ni`
- optional refractive-index and coverslip settings
- `psf_model`
- `oversample_factor`
- `psf_size_z`
- `psf_size_xy`

The generated seed is background-subtracted, clipped to non-negative values,
and normalized to sum to one.

## PSF Modes

`psf_mode = single` returns the normalized detection PSF seed.

`psf_mode = light_sheet` generates two PSFs:

1. Detection PSF.
2. Illumination PSF, using `illumination_na` when supplied.

The illumination PSF is rotated in the Z/X plane by `light_sheet_angle`. The
effective seed is:

```text
normalise(detection_psf * rotated_illumination_psf)
```

Right-angle rotations use exact `np.rot90`; arbitrary angles use interpolated
`scipy.ndimage.rotate`.

## Blind Z Window

For large Z stacks, blind estimation can use a subset of planes.

If `blind_z_slices <= 0`, or the volume is no deeper than `blind_z_slices`, the
full Z range is used.

Otherwise, the code samples up to 64 planes, scores each plane by the 99.9th
intensity percentile, centers a Z window around the brightest sampled plane,
and uses that window for chunked blind estimation.

## XY Tiling

The first TIFF is memory-mapped where possible. It is tiled into XY chunks with
full Z from the selected Z window.

Positive `chunk_xy` is used directly. A non-positive value triggers
VRAM-aware sizing through `resolve_chunk_xy`.

Tiles smaller than half the requested tile size on either edge are skipped.
Each tile is read with an XY halo of `pad_xy`. Border halos are filled with
reflect padding.

## MATLAB `deconvblind`

Each tile is written to a temporary TIFF and processed by MATLAB:

1. Read chunk with `readtiffstack`.
2. Read seed PSF with `readtiffstack`.
3. Normalize the seed.
4. Optionally pad Z symmetrically by `pad_z`.
5. Run `[~, psf_est] = deconvblind(chunk, psf_seed, blind_iters)`.
6. Normalize the estimated PSF.
7. Write the tile PSF with `writetiffstack`.

`matlab_threads` is clamped to one or two threads per MATLAB process.
`matlab_workers` controls concurrent MATLAB calls and is capped by the resolved
I/O worker count. `matlab_timeout` kills a stuck MATLAB chunk when positive.

If the first three chunks fail and no PSF estimate has succeeded, the process
aborts instead of submitting every tile.

## SNR-Weighted Merge

Each tile gets a weight from its core image region. The code estimates signal
from percentile spread and noise from MAD/standard deviation:

```text
snr = max(0, p99 - p50) / noise
weight = snr * snr
```

`snr_weight_cap` limits how much any single bright tile can dominate the merge.
All successful tile PSFs are normalized, stacked, and merged by weighted mean.
The final merged PSF is normalized to sum to one.

## Cache Behavior

When caching is enabled, the code writes cache files named:

```text
estimated_psf_<cache-key>.tif
```

The cache root is resolved in this order:

1. Explicit `--cache_dir`.
2. `.psf_cache` next to the input TIFF.
3. `.psf_cache` in the process working directory.

The cache key includes the input file path, size, modification time, seed
content hash, iteration count, chunking, padding, script path, merge mode,
SNR cap, and Z window. Use `--no_psf_cache` to force re-estimation.

The active PSF is also saved as:

```text
<decon input directory>/estimated_psf.tif
```

If that directory is not writable, the wrapper writes `estimated_psf.tif` in
the current process directory instead. In both cases, Nextflow publishes the
process-local copy to:

```text
<output_dir>/estimated_psf.tif
```
