# Profiles and Parameters

## Profiles

| Profile | Purpose |
| --- | --- |
| `wide_frame` | Uses a single-detection PSF seed. |
| `light_sheet` | Uses light-sheet PSF seed settings for already deskewed light-sheet data. |

## Required Inputs

| Parameter | Description |
| --- | --- |
| `input` | Selected ready-to-deconvolve image volumes. |
| `pyramid_max_downsample` | Required OME-Zarr pyramid depth selection. Default `16` preserves the full `1x, 2x, 4x, 8x, 16x` multiscale output. |
| `wavelength` | Emission wavelength in microns. |
| `na` | Detection numerical aperture fallback. |
| `ni` | Immersion refractive index. |
| `ns` | Specimen/sample refractive index. |
| `dz` | Z pixel size in microns. |

Use `dxy` directly or provide `camera_pixel_size` and `magnification` so the
wrapper can derive X/Y pixel size.

## Optional Outputs

`output_formats = ozx` writes native zipped OME-Zarr outputs. Set
`output_formats = tiff` to also publish TIFF stacks under `deconvolved_tiff/`.

`pyramid_max_downsample` controls the maximum XY pyramid level written for
the OME-Zarr data inside each OZX archive. Lower values reduce pyramid
generation time and disk usage; Z is preserved at full resolution for every
level.

## Blind PSF Backend

| Parameter | Default | Description |
| --- | --- | --- |
| `blind_backend` | `cupy` | Native CuPy blind RL. Use `matlab` for compatibility comparisons. |
| `blind_iters` | `20` | Alternating image/PSF update count per tile. |
| `chunk_xy` | `256` | Maximum CuPy blind-estimation XY tile size; FFT-aware VRAM sizing and OOM recovery may reduce it. |
| `blind_max_tiles` | `16` | Spatially distributed high-SNR tile limit; use `0` for a full-grid comparison. |
| `cupy_fft_engine` | `scout` | CuPy PSF estimation mode. `scout` runs a short filtering pass and refines the most consistent tile PSFs; `cupyx` runs every selected tile directly. |
| `adaptive_scout_iters` | `2` | Blind-RL iterations used by the scout pass before tile filtering. |
| `adaptive_keep_tiles` | `4` | Number of scout-approved tile PSFs kept for the final CuPy refinement. |
| `tile_selection_strategy` | `spatial_snr_v1` | Preselection strategy before blind estimation. Use `coarse_to_fine_snr` to search high-signal coarse regions first. |
| `coarse_region_rows` | `4` | Row bands used by `coarse_to_fine_snr`. |
| `coarse_region_columns` | `4` | Column bands used by `coarse_to_fine_snr`. |
| `coarse_region_limit` | `8` | Maximum coarse regions considered by `coarse_to_fine_snr`. |
| `blind_z_slices` | `128` | Centered Z planes used for estimation; nonpositive values use full Z. |
| `blind_workers` | `1` | Concurrent blind tile tasks; use one per allocated GPU. |
| `prefetch_chunks` | `0` | Additional tile tasks submitted ahead of processing. |
| `snr_weight_cap` | `100` | Maximum tile contribution during SNR-weighted PSF merging. |
| `blind_peak_normalization` | `none` | CuPy input scaling: `none`, `unit`, or `gamma`. |
| `blind_peak_gamma_max` | `2.5` | Positive gamma transform when gamma normalization is selected. |

Slurm controls physical GPU placement. Spawned workers use logical device zero
from `CUDA_VISIBLE_DEVICES`; do not place a physical GPU ID in `params.yml`.

## MATLAB Compatibility Controls

`matlab_workers`, `matlab_threads`, and `matlab_timeout` apply only when
`blind_backend = matlab`.

## Final Deconvolution Controls

| Parameter | Default | Description |
| --- | --- | --- |
| `decon_chunk_xy` | `0` | Full-Z core XY tile size; zero enables VRAM-based sizing. |
| `decon_workers` | `1` | Requested cuCIM workers; clamped to one per allocated GPU process. |
| `vram_gb` | `-1` | Manual VRAM budget; `-1` uses runtime GPU memory detection. |
