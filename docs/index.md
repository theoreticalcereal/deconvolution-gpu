# ctASLM2 Deskew/Deconvolution Workflow for Astrocyte

![Astrocyte version](https://img.shields.io/badge/astrocyte-%E2%89%A52.1.0-blue)
![Nextflow version](https://img.shields.io/badge/nextflow-%E2%89%A523.04.3-brightgreen)

This Astrocyte workflow runs a Nextflow DSL2 pipeline for ctASLM and related 3-D TIFF microscopy data. It can deskew raw oblique light-sheet TIFFs, estimate a blind point-spread function (PSF), and run GPU-accelerated Richardson-Lucy deconvolution with `pycudadecon`.

The workflow is designed for BioHPC SLURM execution. Python and CUDA deconvolution dependencies are packaged in the workflow container, while cluster modules provide workflow/runtime tools such as Nextflow, Java, CUDA, and MATLAB.

## Input Data

Provide an input directory containing TIFF stacks for one acquisition. For full light-sheet processing, use `image_path` to point at the raw TIFF directory and set `cell_name` to the dataset/cell prefix used by the deskew code.

For data that has already been deskewed or is already stored as 3-D stacks, set `decon_only` to `true` and provide `decon_input_dir`. The deconvolution input directory should contain TIFF files named like `CH0_0.tif`, `CH1_0.tif`, or `CH0_0_registered_consistent.tif`.

# Project Data

## Example Parameter Table

| Mode | image_path | decon_input_dir | cell_name | channels | timepoints | decon_only |
|------|------------|-----------------|-----------|----------|------------|------------|
| Light-sheet deskew + decon | `/project/app/astrocyte/astrocyte_incoming/YOUR_ID/raw_tiffs` | | `Cell001` | | | `false` |
| Light-sheet decon only | | `/project/app/astrocyte/astrocyte_incoming/YOUR_ID/Top_shear` | `Cell001` | `0` | `0` | `true` |
| Wide-frame decon only | | `/project/app/astrocyte/astrocyte_incoming/YOUR_ID/stack_tiffs` | `Sample001` | `0` | `0` | `true` |

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `image_path` | Directory containing raw TIFFs for deskewing. |
| `cell_name` | Dataset or cell prefix used to locate and label input files. |
| `output_dir` | Directory where `shear`, `Top_shear`, and `deconvolved` outputs are published. |
| `decon_only` | Skip deskewing and run only deconvolution. |
| `decon_input_dir` | Directory containing already deskewed or stacked TIFFs for decon-only runs. |
| `channels` | Optional channel filter, for example `0` or `0,1`. Empty means all discovered channels. |
| `timepoints` | Optional timepoint filter, for example `0` or `0,1`. Empty means all discovered timepoints. |
| `dx` | Deskew lateral pixel size in microns. |
| `dz` | Deskew/deconvolution axial spacing in microns. |
| `angle` | Light-sheet acquisition angle in degrees. |
| `flip` | Deskew orientation flag, usually `1` or `-1`. |
| `iter` | Richardson-Lucy iteration count. |
| `background` | Constant background subtraction value. |

## Pipeline Summary

1. Deskew raw ctASLM data with MATLAB.
   1. Read the raw TIFF folder selected by `image_path` and `cell_name`.
   2. Apply the requested shear transform using `dx`, `dz`, `angle`, and `flip`.
   3. Write intermediate sheared volumes to `shear`.
   4. Write final deskewed stacks to `Top_shear`.
2. Select deconvolution inputs.
   1. Use `Top_shear` from the deskew step for full light-sheet runs.
   2. Use `decon_input_dir` directly when `decon_only` is true.
   3. Filter TIFFs by `channels` and `timepoints` when those parameters are supplied.
3. Generate a theoretical PSF seed with `psfmodels`.
   1. Use optical parameters such as `dxy`, `dz`, `wavelength`, `na`, `detection_na`, `ni`, and refractive-index settings.
   2. Use `psf_mode` to choose single detection PSF or light-sheet effective PSF.
   3. Use the theoretical PSF only as the blind-estimation starting point.
4. Estimate the active PSF using MATLAB blind deconvolution.
   1. Tile the first input TIFF into XY chunks.
   2. Run MATLAB `deconvblind` on each chunk.
   3. Merge per-chunk PSFs with SNR weighting.
   4. Save the merged blind PSF as `estimated_psf.tif`.
   5. Optionally reuse cached PSFs with `psf_cache_dir`.
5. Run GPU deconvolution with `pycudadecon`.
   1. Convert the estimated PSF to a temporary OTF.
   2. Process each TIFF using full-Z XY chunks.
   3. Use overlap padding to reduce tile-boundary artifacts.
   4. Write final files named `DB2_<input_stem>.tif`.
6. Publish workflow outputs.
   1. `shear` contains intermediate deskewed/sheared files.
   2. `Top_shear` contains deskewed stacks and the estimated PSF.
   3. `deconvolved` contains final deconvolved TIFFs.

## Output Structure

```text
output_dir/
|-- shear/
|-- Top_shear/
|   |-- CH0_0.tif
|   |-- estimated_psf.tif
|   `-- ...
`-- deconvolved/
    |-- DB2_CH0_0.tif
    `-- ...
```

## Documentation

This workflow supports three main run modes:

1. `light_sheet`: deskew raw light-sheet data, then run deconvolution.
2. `wide_frame`: run wide-frame deconvolution settings; use `decon_only=true` when the input is already stacked.
3. `light_sheet_decon`: skip deskewing and run light-sheet deconvolution from an existing stack directory.

Optical metadata should be set per dataset. The most commonly adjusted values are `camera_pixel_size`, `magnification`, `dxy`, `dz`, `wavelength`, `detection_na`, `illumination_na`, `ni`, and `ns`. More information on parameters is available in documentation.

Large runs can generate substantial intermediate data in Nextflow work directories. After confirming the published outputs, clean old work directories according to local BioHPC guidance.

## Citations

If you run on Astrocyte, please acknowledge in publications:

> This research was supported in part by the computational resources provided by the BioHPC supercomputing facility located in the Lyda Hill Department of Bioinformatics, UT Southwestern Medical Center.

If you use this workflow, also cite the relevant tools and methods used in your analysis, including Nextflow, MATLAB, `psfmodels`, `pycudadecon`, CUDA/NVIDIA software, Dask, NumPy, and tifffile as appropriate for your publication.
