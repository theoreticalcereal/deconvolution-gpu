# ctASLM Deskew/Deconvolution Documentation

This Astrocyte workflow runs a Nextflow DSL2 pipeline for ctASLM and related
3-D microscopy volumes. Inputs are normalized to OME-Zarr before image
processing, deskew and deconvolution can operate on chunked Zarr arrays, and
Neuroglancer consumes OME-Zarr outputs directly.

The workflow is designed for BioHPC SLURM execution. A per-run conda runtime is
built for Python, Zarr, Dask, pycudadecon, and reader packages. Astrocyte
provides workflow modules such as Nextflow, Java, MATLAB, and Anaconda.

## Process Guides

| Document | What it covers |
|---|---|
| [Workflow Overview](workflow-overview.md) | Nextflow process order, full runs, decon-only runs, and publishing. |
| [Deskew Process](deskew-process.md) | Chunked deskew from normalized OME-Zarr and deskew output layout. |
| [PSF Estimation Process](psf-estimation-process.md) | PSF seed generation, blind MATLAB estimation, chunking, merge weights, and cache behavior. |
| [GPU Deconvolution Process](gpu-deconvolution-process.md) | OME-Zarr/TIFF input discovery, Dask chunking, `pycudadecon`, and native outputs. |
| [Neuroglancer VizApp](neuroglancer.md) | Launch instructions, display defaults, and VizApp troubleshooting. |
| [Profiles and Parameters](profiles-and-parameters.md) | Nextflow profiles, cluster resources, and user parameters. |
| [workflow/output](workflow-output.md) | Published output directory and stable products. |
| [Outputs and Troubleshooting](outputs-and-troubleshooting.md) | Expected files, common failures, and tuning order. |

## Supported Input Data

Astrocyte `input` accepts TIFF, OME-Zarr, CZI, ND2, LIF, and HDF5. The
`STAGE_DECON_INPUT` process converts supported image files into
`input_zarr/*.ome.zarr` before any deskew or deconvolution process runs.

Select one channel/optical configuration per run. The workflow estimates one
blind PSF from the first selected volume and applies that PSF to all selected
volumes.

## Main Modes

| Mode | Use |
|---|---|
| `light_sheet` | Raw oblique light-sheet data that needs deskew and deconvolution. |
| `wide_frame` | Wide-field data using a single-detection PSF seed. Add `--decon_only true` for already-stacked data. |
| `light_sheet_decon` | Already deskewed/stacked light-sheet data that should use the light-sheet PSF seed. |

## Common Parameters

| Parameter | Description |
|---|---|
| `input` | File-picker inputs. Supported formats are normalized to OME-Zarr. |
| `output_dir` | Directory where final outputs are published. |
| `output_formats` | Requested leaf export formats. Default: `ome_zarr`. |
| `decon_only` | Skip deskewing and run deconvolution directly. |
| `dx`, `dz`, `angle`, `flip` | Deskew geometry. |
| `dxy`, `wavelength`, `detection_na`, `illumination_na`, `ni`, `ns` | Optical/acquisition values. |
| `iter` | Richardson-Lucy iteration count. |

## Published Output Summary

```text
workflow/output/
|-- Top_shear/
|   `-- <sample>.ome.zarr/
|-- estimated_psf.tif
|-- deconvolved/
|   `-- DB2_<sample>.ome.zarr/
`-- neuroglancer/
    `-- layers.json
```

## Code Map

| File | Role |
|---|---|
| `workflow/main.nf` | Connects build, normalization, deskew, deconvolution, and visualization steps. |
| `workflow/modules.nf` | Defines process resources, command lines, and published outputs. |
| `workflow/scripts/normalize_input_to_ome_zarr.py` | Converts supported inputs to OME-Zarr. |
| `workflow/scripts/chunked_deskew.py` | Reads OME-Zarr/TIFF volumes and writes chunked deskew outputs. |
| `workflow/scripts/decon_wrapper.py` | Resolves PSF, runs chunked GPU deconvolution, and writes `DB2_*` outputs. |
| `workflow/scripts/convert_tiff_to_ome_zarr.py` | Converts legacy TIFF outputs or writes Neuroglancer manifests for existing OME-Zarr. |
| `vizapp/neuroloader.py` | Validates OME-Zarr layers and rewrites local sources for browser serving. |

## Citation

If you run on Astrocyte, please acknowledge:

> This research was supported in part by the computational resources provided by
> the BioHPC supercomputing facility located in the Lyda Hill Department of
> Bioinformatics, UT Southwestern Medical Center.
