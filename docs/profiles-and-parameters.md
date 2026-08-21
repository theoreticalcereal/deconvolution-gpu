# Profiles and Parameters

## Profiles

| Profile | Purpose |
| --- | --- |
| `wide_frame` | Uses a single-detection PSF seed. |
| `light_sheet` | Uses light-sheet PSF seed settings for already deskewed light-sheet data. |

## Astrocyte Inputs

The launch form has seven fields: image file(s), a complete microscope profile,
an optional acquisition YAML, optional wavelength and Z-spacing inputs, and an
`image-aggressiveness` mode, plus an output selector. Selecting a profile sets detection and illumination
NA, magnification, lateral pixel size, refractive index, and light-sheet
angle. No separate optical-parameter form entry is required.

The optional acquisition file is normally a Navigate `.yml`/`.yaml`
configuration. When a value was not entered in the form, it
can infer the profile from `MicroscopeState.microscope_name`, `Saving.prefix`,
and `Saving.solvent`; wavelength from the one selected laser channel; and Z
spacing from the absolute `MicroscopeState.step_size`. `Nanoscale` maps to
`Multiscale - High Res`; `Macroscale` maps to `Multiscale - Low Res`. BABB is
recognized as RI 1.56. Explicit profile, wavelength, and Z-spacing entries
always take precedence over inferred values.

Choose **Infer from optional acquisition YAML** only when providing a supported
Navigate YAML. Otherwise select a named profile and enter the experiment's
wavelength and Z spacing. The workflow fails before processing when either of
those two values cannot be resolved.

Choose **Custom — provide deconvolution parameters YAML** (the final dropdown
option) to use the legacy flat YAML schema instead of a microscope profile.
That YAML can provide optical, acquisition, and advanced tuning values,
such as `wavelength`, `dxy`, `dz`, `detection_na`, `ni`, and `iter`.
`output_format`, `image_aggressiveness`, `blind_backend`, `cupy_fft_engine`,
and `decon_backend` remain controlled by form selections.

The final **Output** dropdown selects either an OZX output with a maximum XY
pyramid level of `1x`, `2x`, `4x`, `8x`, or `16x`, or **TIFF (1x)**. Its default
is `1x`; selecting TIFF produces a TIFF only and does not generate pyramids.

| Profile | Detection NA | Illumination NA | Magnification | Pixel size (µm) | RI | Light-sheet angle | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Upright ASLM | 0.643 | 0.643 | 36x | 0.181 | 1.33 | 45 | |
| BenchTop MesoSPIM | 0.25 | 0.1 | 4x | 1.609 | 1.56 | 0 | |
| BenchTop MesoSPIM | 0.431 | 0.1 | 10x | 0.65 | 1.56 | 0 | Immersion objective |
| BenchTop MesoSPIM | 0.42 | 0.1 | 10x | 0.65 | 1.52 | 0 | Immersion objective |
| ctASLM v3 | 1.2 | 0.7 | 50x | 0.128 | 1.56 | 0 | |
| ctASLM v3 | 1.2 | 0.7 | 50x | 0.128 | 1.52 | 0 | |
| Multiscale - Low Res | 0.25 | 0.1 | 0.63x | 9.7 | 1.56 | 0 | |
| Multiscale - Low Res | 0.25 | 0.1 | 1x | 6.38 | 1.56 | 0 | |
| Multiscale - Low Res | 0.25 | 0.1 | 2x | 3.14 | 1.56 | 0 | |
| Multiscale - Low Res | 0.25 | 0.1 | 3x | 2.12 | 1.56 | 0 | |
| Multiscale - Low Res | 0.25 | 0.1 | 4x | 1.609 | 1.56 | 0 | |
| Multiscale - Low Res | 0.25 | 0.1 | 5x | 1.255 | 1.56 | 0 | |
| Multiscale - Low Res | 0.25 | 0.1 | 6x | 1.044 | 1.56 | 0 | |
| Multiscale - High Res | 0.753 | 0.753 | 38x | 0.171 | 1.56 | 0 | |
| Multiscale - High Res | 0.734 | 0.734 | 37x | 0.171 | 1.52 | 0 | |

The supplied table leaves NA blank for the 1x–6x Multiscale low-resolution
rows. Those profiles inherit the stated Multiscale low-resolution 0.25
detection NA and 0.1 illumination NA values.

## Image Aggressiveness

| Mode | Queue | PSF estimation | Deconvolution |
| --- | --- | --- | --- |
| `low` | `GPUp40` | CuPy scout mode | CuPy |
| `medium` | `GPUp40` | CuPy direct mode over every blind chunk | CuPy |
| `high` | `256GBv1` | MATLAB `deconvblind` | MATLAB `deconvlucy` |

The `high` preset deliberately receives no GPU allocation. The lightweight
input-staging and export processes continue to use the general queue.
