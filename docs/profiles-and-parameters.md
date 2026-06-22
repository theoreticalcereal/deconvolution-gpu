# Profiles and Parameters

Runtime defaults and profiles live in `workflow/configs/nextflow.config`.
Astrocyte uses `workflow/configs/astrocyte.config`, which includes the standard
config and enables the Singularity image for `DECON`.

## Profiles

| Profile | Behavior |
|---|---|
| `my_cluster` | Enables conda and mamba for cluster execution. |
| `light_sheet` | Runs deskew plus deconvolution with light-sheet PSF seed settings. |
| `wide_frame` | Uses single-detection PSF seed settings. It does not skip deskew automatically. |
| `light_sheet_decon` | Enables decon-only mode and light-sheet PSF seed settings. |
| `docker` | Enables Docker. |
| `singularity` | Enables Singularity and builds or uses `workflow/images/decon_env.sif` for `DECON`. |

## Astrocyte Container Config

`astrocyte_pkg.yml` points Astrocyte at `astrocyte.config`. That config includes
`nextflow.config`, disables conda for the containerized path, enables
Singularity, and sets `build_decon_container = true`.

When `build_decon_container` is true, the workflow runs `BUILD_DECON_CONTAINER`
before `DECON`. That process reuses `workflow/images/decon_env.sif` when a real
Git LFS image is present; otherwise it builds `decon_env.sif` from
`workflow/images/decon_env.def` in the Nextflow work directory, then passes the
image path to `DECON`.

`DESKEW` still runs on the host with the MATLAB module because the container is
the Python/CUDA deconvolution environment.

## Light-Sheet Profile

`light_sheet` sets:

```text
decon_only = false
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

The Astrocyte profile disables conda for `DECON` and enables Singularity. The
container is built from `environment.yml` through `workflow/images/decon_env.def`
or supplied as a Git LFS-managed `workflow/images/decon_env.sif`.

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
| `input` | TIFF files selected in Astrocyte. Select one channel at a time. |
| `image_path` | Backward-compatible raw input parent directory for manual CLI runs without `input`. |
| `cell_name` | Optional legacy dataset folder under `image_path` for manual deskew runs. Ignored when `input` files are selected. |
| `output_dir` | Published output root. |
| `decon_only` | Skip `DESKEW` and run `DECON` directly. |
| `decon_input_dir` | Backward-compatible directory of already deskewed or stacked TIFFs. |

Selected TIFFs should come from the same channel and optical configuration. The
workflow estimates one PSF and applies it to the selected files, so mixed
wavelengths can skew deconvolution results.

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
| `ns` | Sample refractive index. |
| `ni0`, `tg`, `tg0`, `ng`, `ng0`, `ti0` | Optional `psfmodels` coverslip/design settings. |
| `dxy` | Lateral sample pixel size in microns. |
| `camera_pixel_size`, `magnification` | Used to derive `dxy` when `dxy <= 0`. |
| `psf_size_z`, `psf_size_xy` | Seed PSF dimensions. |
| `psf_model` | `vectorial`, `scalar`, or `gaussian`. |
| `psf_mode` | `single` or `light_sheet`. |
| `light_sheet_angle` | Illumination PSF rotation angle for light-sheet mode. |
| `oversample_factor` | PSF model oversampling factor. |
