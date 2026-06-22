# ctASLM Deskew/Deconvolution Documentation

This Astrocyte workflow runs a Nextflow DSL2 pipeline for ctASLM and related
3-D TIFF microscopy data. It can deskew raw oblique light-sheet TIFFs, estimate
a blind point-spread function (PSF), and run GPU-accelerated
Richardson-Lucy deconvolution with `pycudadecon`.

The workflow is designed for BioHPC SLURM execution. Python and CUDA
deconvolution dependencies are packaged in the workflow Singularity image,
while cluster modules provide workflow/runtime tools such as Nextflow, Java,
Singularity, CUDA, and MATLAB.

## Process Guides

| Document | What it covers |
|---|---|
| [Workflow Overview](workflow-overview.md) | How Nextflow routes full runs and decon-only runs. |
| [Deskew Process](deskew-process.md) | MATLAB deskew input discovery, shear correction, top-view output, and file naming. |
| [PSF Estimation Process](psf-estimation-process.md) | Theoretical PSF seed generation, blind MATLAB estimation, chunking, merge weights, and cache behavior. |
| [GPU Deconvolution Process](gpu-deconvolution-process.md) | TIFF selection, OTF creation, Dask chunking, `pycudadecon`, and output scaling. |
| [Profiles and Parameters](profiles-and-parameters.md) | Nextflow profiles, cluster resources, and the parameters that control each mode. |
| [workflow/output](workflow-output.md) | The stable published output directory used by Astrocyte. |
| [Outputs and Troubleshooting](outputs-and-troubleshooting.md) | Published output layout, expected filenames, common failures, and where to look first. |

## Input Data

Select the TIFF stacks for one acquisition with the Astrocyte file picker.
Astrocyte handles file ingestion for the workflow package.

Select files from only one channel at a time. The workflow estimates one PSF
from the first selected TIFF and applies it to all selected TIFFs; mixing
channels with different wavelengths can skew deconvolution results.

## Example Modes

| Mode | input files |
|------|-------------|
| Light-sheet deskew + decon | Raw `CH##_######.tif[f]` files from one channel |
| Light-sheet decon only | Already deskewed 3-D TIFF stacks from one channel |
| Wide-frame decon only | Stacked 3-D TIFF files from one channel |

## Main Run Modes

`light_sheet` runs deskew first and then deconvolution using the deskewed
`Top_shear` stacks.

`wide_frame` selects the single-detection PSF seed mode. It does not skip
deskew by itself; use `--decon_only true` and `--decon_input_dir` when the data
is already stored as 3-D stacks.

`light_sheet_decon` skips deskew and runs deconvolution with the light-sheet
effective PSF seed mode.

In Astrocyte runs, the selected `input` TIFFs are staged into the input
directory used by the selected mode. Full light-sheet runs pass those staged
files to `DESKEW`; decon-only runs pass them directly to `DECON`.
`decon_input_dir` is only needed for manual CLI runs that point at an existing
directory.

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `input` | TIFF files selected in Astrocyte. Select one channel at a time. |
| `cell_name` | Optional legacy dataset folder under `image_path` for deskew runs. Unused by decon-only runs. |
| `output_dir` | Directory where final outputs are published; defaults to `./workflow/output`. |
| `decon_only` | Skip deskewing and run only deconvolution. |
| `dx` | Deskew lateral pixel size in microns. |
| `dz` | Deskew/deconvolution axial spacing in microns. |
| `angle` | Light-sheet acquisition angle in degrees. |
| `flip` | Deskew orientation flag, usually `1` or `-1`. |
| `iter` | Richardson-Lucy iteration count. |
| `background` | Constant background subtraction value. |

## Output Structure

```text
workflow/output/
|-- shear/
|-- Top_shear/
|   |-- CH0_0.tif
|   `-- ...
|-- estimated_psf.tif
`-- deconvolved/
    |-- DB2_CH0_0.tif
    `-- ...
```

Files in `workflow/output` remain available after Astrocyte cleans Nextflow work
directories. See [workflow/output](workflow-output.md) for details.

## Code Map

| File | Role |
|---|---|
| `workflow/main.nf` | Connects `DESKEW` and `DECON` based on `params.decon_only`. |
| `workflow/modules.nf` | Defines process resources, scripts, published outputs, and command-line flags. |
| `workflow/configs/nextflow.config` | Defines defaults, profiles, SLURM queues, conda/mamba setup, and GPU requests. |
| `workflow/configs/astrocyte.config` | Astrocyte entry config that builds and enables the Singularity image for `DECON`. |
| `workflow/scripts/deskew_wrapper.py` | Builds the MATLAB batch command for `deskew.m`. |
| `workflow/scripts/deskew.m` | Reads raw TIFF stacks, applies shear correction, rotates top view, and writes TIFF outputs. |
| `workflow/scripts/decon_wrapper.py` | Selects TIFF inputs, resolves PSF, runs GPU deconvolution, and writes `DB2_*` outputs. |
| `workflow/scripts/psf_estimation.py` | Generates PSF seeds, runs chunked MATLAB `deconvblind`, merges PSFs, and manages the PSF cache. |
| `workflow/scripts/psf_modes.py` | Builds either a single-detection seed PSF or light-sheet effective seed PSF. |

## Citations

If you run on Astrocyte, please acknowledge in publications:

> This research was supported in part by the computational resources provided by
> the BioHPC supercomputing facility located in the Lyda Hill Department of
> Bioinformatics, UT Southwestern Medical Center.

If you use this workflow, also cite the relevant tools and methods used in your
analysis, including Nextflow, MATLAB, `psfmodels`, `pycudadecon`, CUDA/NVIDIA
software, Dask, NumPy, and tifffile as appropriate for your publication.
