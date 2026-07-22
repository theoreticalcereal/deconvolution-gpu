# PSF Estimation Process

## Backend Selection

`blind_backend = cupy` is the default workflow path. It is a native blind
Richardson-Lucy implementation and no longer routes through MATLAB. Set
`blind_backend = matlab` only for a MATLAB `deconvblind` comparison or
compatibility run.

Each CuPy XY tile runs in an isolated spawned worker process. CUDA initializes
inside the child and inherits the Slurm allocation through
`CUDA_VISIBLE_DEVICES`; device `0` is the first logical GPU visible inside
the job. Use `blind_workers = 1` for a normal one-GPU allocation.

## Native CuPy Calculation

The CuPy implementation and SciPy CPU reference use the same linear convolution
operator and exact adjoints. Each iteration performs:

1. Forward convolution of the current image and PSF.
2. A dampable observed/model likelihood ratio.
3. Image back-projection through the flipped PSF with boundary sensitivity.
4. A second forward model using the updated image.
5. A fixed-support PSF adjoint update with boundary sensitivity.
6. Non-negativity enforcement and PSF sum normalization.

Z padding is symmetric. FFT convolution remains linear rather than circular,
and centered adjoint cropping handles odd and even PSF dimensions. Damping uses
the MATLAB-compatible default of zero and is not yet a workflow parameter.

`blind_peak_normalization = none` most closely matches unscaled MATLAB input.
`unit` scales the observed tile to `[0, 1]`; `gamma` additionally applies
the positive `blind_peak_gamma_max` transform.

The cache key includes the backend, normalization mode, gamma setting, and Z
window, so native CuPy results cannot reuse former MATLAB-fallback entries.

The runtime image contains SciPy, CuPy, cupyx FFT convolution, and cuCIM. The
blind-RL module exposes a cuCIM adapter using the normalized PSF with clipping
disabled, and full-volume production restoration uses that same adapter.

The deconvolution process always resolves a blind PSF before running CUDA
deconvolution. The theoretical PSF is only the starting seed for MATLAB
`deconvblind`; it is not used directly as the final deconvolution PSF.

The implementation is split across:

- `workflow/scripts/decon_wrapper.py`
- `workflow/scripts/psf_modes.py`
- `workflow/scripts/psf_estimation.py`

## Input Selection

`decon_wrapper.py` sorts the selected deconvolution input volumes first. The
first sorted volume is used for blind PSF estimation.

Package runs pass TIFF volumes linked by `STAGE_DECON_TIFF_INPUT` or OME-Zarr
volumes created by `STAGE_DECON_INPUT`. For light-sheet data, run `deskew-gpu`
first and select its `Top_shear/` outputs here. OME-Zarr inputs are opened
directly at level 0 for PSF tile selection, so the full volume is not first
materialized as a temporary TIFF. Channel and timepoint filtering is not
performed by the wrapper, so select only inputs intended for one optical
configuration.

Example normalized names:

```text
CH00_000000.ome.zarr
sample_a.ome.zarr
DB_input.ome.zarr
```

## Lateral Pixel Size Resolution

The PSF code needs `dxy`.

If `--dxy` is supplied and greater than zero, that value is used. If `--dxy`
is omitted, both `--camera_pixel_size` and `--magnification` must be supplied,
and:

```text
dxy = camera_pixel_size / magnification
```

## Seed PSF Generation

The seed is generated with `psfmodels.make_psf` using optical parameters such
as:

- `dxy`
- `dz`
- `detection_na` or backward-compatible `na`
- `wavelength`
- `ni`
- `ns`
- optional coverslip settings
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

The first selected volume is opened as a Dask-backed image when possible. It is
tiled into XY chunks with full Z from the selected Z window.

For the CuPy backend, positive `chunk_xy` is an upper bound. The workflow
models the padded 3-D FFT dimensions, reads free VRAM from the allocated visible
GPU, and chooses the largest 32-pixel-aligned core that fits its safety budget.
A 24 GB GPU can retain the configured 256-pixel core while smaller GPUs select
smaller tiles automatically.

If cuFFT still raises an out-of-memory error, the partial estimation pass is
discarded and every tile is restarted at the next smaller aligned core size.
This preserves complete image coverage instead of cropping the failed tile.
CuPy blind estimation is limited to one spawned worker per allocated GPU.

Tiles smaller than half the requested tile size on either edge are skipped.
Each tile is read with an XY halo of `pad_xy`. Border halos are filled with
reflect padding.

By default, `blind_max_tiles = 16` limits blind RL to a representative subset
of the candidate grid. Every candidate core is scored with the same SNR weight
used by the final merge. The grid is divided into balanced spatial regions,
the strongest candidate in each region is retained, and any remaining slots
are filled by the strongest unselected candidates. Coordinate ordering makes
selection deterministic when scores are equal.

The log records the candidate and selected counts, estimated work reduction,
coordinates, and SNR weights. Set `blind_max_tiles = 0` to process the complete
grid for comparison runs. Tile selection does not change the selected tiles'
halos, Z window, blind iterations, or merge weights.

## MATLAB `deconvblind`

MATLAB still operates on temporary TIFF chunks. For OME-Zarr inputs, Python
reads the selected tile from the level-0 Zarr array before this TIFF handoff.
For each tile, the Python code:

1. Writes the chunk to a temporary TIFF.
2. Writes the seed PSF to a temporary TIFF.
3. Calls MATLAB `deconvblind`.
4. Reads the estimated tile PSF back from TIFF.

The MATLAB side:

1. Reads the chunk with `readtiffstack`.
2. Reads the seed PSF with `readtiffstack`.
3. Normalizes the seed.
4. Optionally pads Z symmetrically by `pad_z`.
5. Runs `[~, psf_est] = deconvblind(chunk, psf_seed, blind_iters)`.
6. Normalizes the estimated PSF.
7. Writes the tile PSF with `writetiffstack`.

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
2. `.psf_cache` next to the input volume when possible.
3. `.psf_cache` in the process working directory.

The cache key includes the input path, size, modification time, seed content
hash, iteration count, chunking, padding, script path, merge mode, SNR cap, Z
window, tile limit, and tile-selection strategy. Representative and full-grid
PSFs therefore cannot reuse one another's cache entries. Use `--no_psf_cache`
to force re-estimation.

The active PSF is saved as:

```text
estimated_psf.tif
```

Nextflow publishes the process-local copy to:

```text
<output_dir>/estimated_psf.tif
```
