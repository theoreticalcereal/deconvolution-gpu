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
| `wavelength` | Emission wavelength in microns. |
| `na` | Detection numerical aperture fallback. |
| `ni` | Immersion refractive index. |
| `ns` | Specimen/sample refractive index. |
| `dz` | Z pixel size in microns. |

Use `dxy` directly or provide `camera_pixel_size` and `magnification` so the
wrapper can derive X/Y pixel size.

## Optional Outputs

`output_formats = ome_zarr` writes native OME-Zarr outputs. Set
`output_formats = tiff` to also publish TIFF stacks under `deconvolved_tiff/`.
