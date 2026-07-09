# Profiles and Parameters

## Profiles

| Profile | Purpose |
| --- | --- |
| `wide_frame` | Uses a single-detection PSF seed. |
| `light_sheet` | Uses light-sheet PSF seed settings for already deskewed light-sheet data. |
| `conda_runtime` | Builds the workflow conda runtime inside the run directory. |

## Required Inputs

| Parameter | Description |
| --- | --- |
| `input` | Selected ready-to-deconvolve image volumes. |
| `decon_runtime_dir` | Optional existing decon/deskew runtime. Use `-1` to build the default runtime. |
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
