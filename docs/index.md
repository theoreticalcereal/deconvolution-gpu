# 3D GPU Deconvolution

This package runs blind PSF estimation and GPU Richardson-Lucy deconvolution on
ready-to-deconvolve image volumes. It no longer performs deskewing and no longer
launches visualization.

Use `deskew-gpu` first when ctASLM/light-sheet data needs geometric correction.
Then run this package on the deskewed `Top_shear/` output.

## Workflow

```text
selected images -> STAGE_DECON_TIFF_INPUT or STAGE_DECON_INPUT -> DECON -> optional TIFF export
```

The native output is OZX, a zipped OME-Zarr archive. Set
`output_formats = tiff` to publish TIFF stacks in addition to native OZX
output.

## Documentation

| Page | Purpose |
| --- | --- |
| [Workflow Overview](workflow-overview.md) | Process order and data flow. |
| [PSF Estimation Process](psf-estimation-process.md) | Blind PSF generation and merge behavior. |
| [GPU Deconvolution Process](gpu-deconvolution-process.md) | GPU deconvolution parameters and outputs. |
| [Profiles and Parameters](profiles-and-parameters.md) | BioHPC profiles and Astrocyte parameters. |
| [Workflow Output](workflow-output.md) | Published output layout. |
| [Outputs and Troubleshooting](outputs-and-troubleshooting.md) | Common runtime and data issues. |
| [Creating an Astrocyte Project](astrocyte-project-setup.md) | Project creation and input upload steps. |
