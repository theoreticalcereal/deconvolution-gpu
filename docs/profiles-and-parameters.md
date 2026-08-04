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
| `cupy_fft_engine` | `scout` | CuPy PSF estimation mode. `scout` runs a short filtering pass and refines the most consistent tile PSFs; `cupyx` runs every selected tile directly. |

Advanced defaults are kept in `workflow/configs/nextflow.config` instead of
the Astrocyte parameter form. This includes blind iteration count, tile sizing,
scout filtering details, MATLAB compatibility settings, worker counts, cache
controls, and VRAM sizing. Override them from a custom Nextflow params file only
for method-development or comparison runs.

Slurm controls physical GPU placement. Spawned workers use logical device zero
from `CUDA_VISIBLE_DEVICES`; do not place a physical GPU ID in `params.yml`.

## MATLAB Compatibility Controls

`matlab_workers`, `matlab_threads`, and `matlab_timeout` apply only when
`blind_backend = matlab`.

## Final Deconvolution Controls

| Parameter | Default | Description |
| --- | --- | --- |
| `decon_chunk_xy` | `0` | Full-Z core XY tile size; zero enables VRAM-based sizing. Hidden from the Astrocyte form by default. |
