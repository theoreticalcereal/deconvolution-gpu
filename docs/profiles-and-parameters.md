# Profiles and Parameters

Runtime defaults and profiles live in `workflow/configs/nextflow.config`.

## Profiles

| Profile | Behavior |
|---|---|
| `my_cluster` | Enables conda and mamba for cluster execution. |
| `light_sheet` | Runs deskew plus deconvolution with light-sheet PSF seed settings. |
| `wide_frame` | Uses single-detection PSF seed settings. It does not skip deskew automatically. |
| `light_sheet_decon` | Enables decon-only mode and light-sheet PSF seed settings. |
| `docker` | Enables Docker. |
| `singularity` | Enables Singularity and uses `workflow/images/decon_env.sif` for `DECON`. |

## Light-Sheet Profile

`light_sheet` sets:

```text
decon_only = false
dx = 0.118
dxy = 0.118
dz = 0.118
angle = 40
flip = 1
psf_mode = light_sheet
light_sheet_angle = 90
```

Use this profile when the input is raw oblique light-sheet data and should pass
through `DESKEW`.

## Wide-Frame Profile

`wide_frame` sets:

```text
psf_mode = single
```

Use `--decon_only true` and `--decon_input_dir` when wide-frame input is already
a 3-D TIFF stack directory. Also provide dataset-specific optical parameters
such as `camera_pixel_size`, `magnification`, `dxy`, `dz`, `wavelength`,
`detection_na`, `ni`, and `ns`.

## Light-Sheet Decon Profile

`light_sheet_decon` sets:

```text
decon_only = true
psf_mode = light_sheet
light_sheet_angle = 90
```

Use this when the data is already deskewed or already stacked but should use
the light-sheet effective PSF seed.

## Cluster Resources

The global executor is SLURM.

`DESKEW` runs on queue `super` with:

```text
cpus = 4
memory = 32 GB
```

`DECON` runs on queue `GPU` with:

```text
cpus = 8
memory = 32 GB
clusterOptions = --gres=gpu:1
```

`DECON` also sets `maxForks 8` in `workflow/modules.nf`, so multiple
deconvolution process instances can be submitted by Nextflow if the workflow is
expanded to produce multiple independent calls.

## Environment Setup

The cluster profile enables conda and mamba. For `DECON`, the config loads
`mamba/2.3.0` and creates a small activation shim so Nextflow can activate the
environment reliably on compute nodes.

The process script then loads:

```text
cuda/11.8
matlab/2024a
```

It also prepends CUDA library paths to `LD_LIBRARY_PATH` and runs `nvidia-smi`
before launching Python.

## Parameter Groups

### Input and Mode

| Parameter | Purpose |
|---|---|
| `image_path` | Raw input parent directory, or fallback decon-only directory. |
| `cell_name` | Dataset folder name used by deskew. |
| `output_dir` | Published output root. |
| `decon_only` | Skip `DESKEW` and run `DECON` directly. |
| `decon_input_dir` | Directory of already deskewed or stacked TIFFs. |

### Selection

| Parameter | Purpose |
|---|---|
| `cell_index` | Optional MATLAB `CellIndex` injection. |
| `channels` | Channel filter such as `0` or `0,1`; empty means all discovered channels. |
| `timepoints` | Timepoint filter such as `0` or `0,1`; empty means all discovered timepoints. |

### Deskew Geometry

| Parameter | Purpose |
|---|---|
| `dx` | Deskew lateral pixel size in microns. |
| `dz` | Deskew and deconvolution axial spacing in microns. |
| `angle` | Acquisition angle in degrees. |
| `flip` | Shear/rotation orientation flag. |

### Deconvolution

| Parameter | Purpose |
|---|---|
| `iter` | Richardson-Lucy iterations for CUDA deconvolution. |
| `background` | Background subtraction for PSF seed generation. |
| `decon_chunk_xy` | Core XY chunk size for CUDA deconvolution; `<=0` auto-sizes. |
| `overlap_xy` | XY overlap for deconvolution; `<=0` derives from PSF size. |
| `decon_workers` | Dask workers for deconvolution chunks. |
| `vram_gb` | Manual VRAM budget for auto chunk sizing. |

### Blind PSF Estimation

| Parameter | Purpose |
|---|---|
| `blind_iters` | MATLAB `deconvblind` iterations per tile. |
| `chunk_xy` | XY tile size for blind PSF estimation; `<=0` auto-sizes. |
| `pad_xy` | XY halo for blind-estimation tiles. |
| `pad_z` | Symmetric Z padding applied before `deconvblind`. |
| `blind_z_slices` | Number of Z planes used for blind estimation; `<=0` uses full Z. |
| `blind_workers` | Python tile extraction/submission workers. |
| `matlab_workers` | Concurrent MATLAB `deconvblind` processes. |
| `matlab_threads` | MATLAB threads per process, clamped to one or two. |
| `matlab_timeout` | Per-tile MATLAB timeout in seconds; `<=0` disables. |
| `snr_weight_cap` | Maximum SNR merge weight per tile. |
| `prefetch_chunks` | Number of blind tiles submitted/read ahead. |
| `psf_cache_dir` | Optional reusable PSF cache directory. |
| `no_psf_cache` | Disable PSF cache reads and writes. |

### Optical Model

| Parameter | Purpose |
|---|---|
| `na` | Backward-compatible numerical aperture fallback. |
| `detection_na` | Detection NA; overrides `na` when supplied. |
| `illumination_na` | Illumination NA for light-sheet seed generation. |
| `wavelength` | Emission wavelength in microns. |
| `ni` | Immersion refractive index. |
| `ns`, `ni0`, `tg`, `tg0`, `ng`, `ng0`, `ti0` | Optional `psfmodels` optical-path settings. |
| `dxy` | Lateral sample pixel size in microns. |
| `camera_pixel_size`, `magnification` | Used to derive `dxy` when `dxy <= 0`. |
| `psf_size_z`, `psf_size_xy` | Seed PSF dimensions. |
| `psf_model` | `vectorial`, `scalar`, or `gaussian`. |
| `psf_mode` | `single` or `light_sheet`. |
| `light_sheet_angle` | Illumination PSF rotation angle for light-sheet mode. |
| `oversample_factor` | PSF model oversampling factor. |
