# deconvolution-gpu

A Nextflow DSL2 pipeline for GPU-accelerated deskewing and deconvolution of light-sheet microscopy (ctASLM) TIFF volumes. The pipeline runs on SLURM and uses a conda environment managed by **mamba**. It requires **Java 17+** and **mamba** to be available on the cluster. Made for BioHPC @ UTSouthwestern. 

---

## Pipeline Process

Pipeline has two sequential stages:

### 1. DESKEW

Calls MATLAB via a Python wrapper to correct the oblique acquisition angle of ctASLM data. It applies a 3-D shear transform to each selected channel/timepoint TIFF and writes output into two folders:

- `<output_dir>/shear/` — intermediate sheared volumes
- `<output_dir>/Top_shear/` — final deskewed volumes (passed to DECON)

### 2. DECON

Reads the deskewed `CH*.tif` files from `Top_shear/` and runs Richardson–Lucy GPU deconvolution via `pycudadecon`. Volumes are processed as full-Z XY tiles using Dask `map_overlap` with reflect-padded boundaries to suppress edge artifacts. Output files are named `DB2_<original_stem>.tif` and written to `<output_dir>/deconvolved/`.

**PSF resolution always uses blind estimation.** The pipeline generates a theoretical Gibson-Lanni PSF from the optical parameters you supply, but uses it only as the starting guess for MATLAB `deconvblind`. It then estimates the PSF from the first TIFF by tiling it into XY chunks, running `deconvblind` on each tile, and merging the per-tile PSFs with SNR weighting. The merged blind PSF is published as `<output_dir>/estimated_psf.tif` and cached for reuse.

---

## Running the Pipeline

For Astrocyte runs, select TIFF files with the file picker. Select files from
only one channel at a time. The workflow estimates one PSF from the first
selected TIFF and applies it to every selected TIFF; mixing channels with
different wavelengths can skew deconvolution results.

For manual light-sheet/ctASLM runs that still need deskewing:

```bash
nextflow run main.nf -profile light_sheet \
    --image_path /path/to/raw/tiffs \
    --cell_name MyCellName \
    --output_dir /path/to/output
```

Add `-resume` to restart from the last successful checkpoint after a failure.

For wide-frame data:

```bash
nextflow run main.nf -profile wide_frame \
    --image_path /path/to/raw/tiffs \
    --cell_name MyCellName \
    --camera_pixel_size 6.5 \
    --magnification 36 \
    --dxy 0.167 \
    --dz 1.0 \
    --ni 1.56 \
    --ns 1.56 \
    --detection_na 0.7 \
    --illumination_na 0.7 \
    --wavelength 0.525 \
    --output_dir /path/to/output
```

For already-stacked wide-frame 3-D TIFFs that should not be deskewed, add
`--decon_only true` and use `--decon_input_dir`:

```bash
nextflow run main.nf -profile wide_frame \
    --decon_only true \
    --decon_input_dir /path/to/stack_tiffs \
    --camera_pixel_size 6.5 \
    --magnification 36 \
    --dxy 0.167 \
    --dz 1.0 \
    --ni 1.56 \
    --ns 1.56 \
    --detection_na 0.7 \
    --illumination_na 0.7 \
    --wavelength 0.525 \
    --output_dir /path/to/output
```

In Astrocyte, selected TIFFs from the file picker are staged into the input
directory used by the selected mode. Full light-sheet runs pass the staged
files to `DESKEW`; decon-only runs pass the same staged files directly to
`DECON`. `decon_input_dir` is only needed for manual command-line runs.

The `wide_frame` profile only selects `--psf_mode single`; pass voxel size,
refractive index, NA, wavelength, and camera calibration values for each run.
It does not skip deskew unless `--decon_only true` is passed.

For already-stacked light-sheet data that should not be deskewed:

```bash
nextflow run main.nf -profile light_sheet_decon \
    --decon_input_dir /path/to/renamed_stack_tiffs \
    --camera_pixel_size 6.5 \
    --magnification 36 \
    --dxy 0.167 \
    --dz 1.0 \
    --ni 1.56 \
    --ns 1.56 \
    --detection_na 0.7 \
    --illumination_na 0.7 \
    --wavelength 0.525 \
    --output_dir /path/to/output
```

The `light_sheet_decon` profile uses `--decon_only true` and
`--psf_mode light_sheet`, which builds the blind-estimation seed from detection
PSF times a rotated illumination PSF. Pass dataset-specific optical parameters
on every run.

---

## Parameters

All parameters can be passed on the command line as `--param_name value` or set in a custom `nextflow.config`.

### Required

| Parameter | Description |
|---|---|
| `--image_path` | Path to the directory containing raw input TIFFs |
| `--cell_name` | Optional legacy dataset folder under `image_path` for deskew runs |

### I/O

| Parameter | Default | Description |
|---|---|---|
| `--output_dir` | `./workflow/output` | Root directory for all published outputs |
| `--decon_only` | `false` | Skip deskewing; go straight to deconvolution |
| `--decon_input_dir` | `''` | Input directory for `--decon_only` mode (overrides `--image_path`) |

### Deskew

| Parameter | Default | Description |
|---|---|---|
| `--cell_index` | `''` | Integer index to select a specific cell in the dataset |
| `--dx` | required for deskew | Lateral pixel size in µm |
| `--dz` | required | Axial step size in µm |
| `--angle` | `40` | Acquisition angle in degrees |
| `--flip` | `1` | Flip direction flag (1 or -1) |

### Deconvolution

| Parameter | Default | Description |
|---|---|---|
| `--iter` | `10` | Number of Richardson–Lucy iterations |
| `--background` | `0` | Background value subtracted before deconvolution |

### Blind PSF Estimation

| Parameter | Default | Description |
|---|---|---|
| `--blind_iters` | `10` | MATLAB `deconvblind` iterations per chunk |
| `--chunk_xy` | `256` | XY tile size for blind estimation (px). `<=0` auto-sizes from VRAM |
| `--pad_xy` | `32` | XY halo added per edge before each blind chunk (px) |
| `--pad_z` | `20` | Z halo added per edge before each blind chunk (slices) |
| `--blind_z_slices` | `128` | Z planes used per blind PSF tile. `<=0` uses full Z |
| `--blind_workers` | `8` | Concurrent blind PSF chunk workers |
| `--matlab_workers` | `8` | Concurrent MATLAB `deconvblind` processes (keep `1` on SLURM) |
| `--matlab_threads` | `1` | Threads per MATLAB process (clamped to 1–2) |
| `--matlab_timeout` | `1800` | Seconds before a blind chunk is killed. `<=0` disables |
| `--snr_weight_cap` | `100` | Max per-chunk SNR weight during PSF merge; prevents bright-artifact dominance |
| `--prefetch_chunks` | `0` | PSF tile read-ahead. `<=0` = one worker batch |
| `--psf_cache_dir` | `''` | Directory to cache/reuse blind PSF estimates |
| `--no_psf_cache` | `false` | Disable PSF cache; always re-estimate |

### CUDA Deconvolution Chunking

| Parameter | Default | Description |
|---|---|---|
| `--decon_chunk_xy` | `0` | Core XY tile size for CUDA decon (px). `<=0` auto-sizes from VRAM |
| `--overlap_xy` | `0` | XY overlap between tiles (px). `<=0` = PSF-size/4, capped at 48 |
| `--decon_workers` | `1` | Dask workers for CUDA decon chunks |
| `--vram_gb` | `0` | Override detected free VRAM (GiB) for auto-sizing |

### Optical / PSF Parameters

These are used to generate the theoretical PSF seed for blind estimation. The theoretical PSF is not used directly for final deconvolution. Dataset-specific optical/acquisition values must be supplied explicitly.

| Parameter | Default | Description |
|---|---|---|
| `--na` | none | Detection numerical aperture (backward-compatible) |
| `--detection_na` | required | Detection NA; overrides `--na` when provided |
| `--illumination_na` | none | Illumination NA; required for `psf_mode=light_sheet` |
| `--wavelength` | required | Emission wavelength in µm |
| `--ni` | required | Immersion medium refractive index |
| `--ns` | required | Sample refractive index |
| `--ni0` | `''` | Design immersion refractive index |
| `--tg` | `''` | Experimental coverslip thickness (µm) |
| `--tg0` | `''` | Design coverslip thickness (µm) |
| `--ng` | `''` | Experimental coverslip refractive index |
| `--ng0` | `''` | Design coverslip refractive index |
| `--ti0` | `''` | Objective working distance (µm) |
| `--dxy` | required unless derived | Lateral pixel size used for PSF model (µm) |
| `--psf_size_z` | `101` | Z dimension of PSF volume (voxels) |
| `--psf_size_xy` | `61` | XY dimension of PSF volume (voxels) |
| `--psf_model` | `vectorial` | PSF model type: `vectorial`, `scalar`, or `gaussian` |
| `--psf_mode` | `single` | Seed mode: `single` detection PSF or `light_sheet` detection × rotated illumination PSF |
| `--light_sheet_angle` | `90` | Illumination PSF rotation angle in degrees for `--psf_mode light_sheet` |
| `--oversample_factor` | `3` | PSF model oversampling factor |
| `--camera_pixel_size` | `''` | Camera pixel size (µm); used to derive `dxy` when `--dxy <= 0` |
| `--magnification` | `''` | Total magnification; used to derive `dxy` when `--dxy <= 0` |

---

## Output Structure

```
<output_dir>/
├── shear/              # Intermediate sheared volumes (from DESKEW)
├── Top_shear/          # Deskewed volumes passed to DECON
│   ├── CH0_0.tif
│   └── ...
├── estimated_psf.tif   # Merged blind PSF used for deconvolution
└── deconvolved/        # Final deconvolved TIFFs
    ├── DB2_CH0_0.tif
    └── ...
```

---

## Notes

- Load the cluster Nextflow module before running the pipeline.
- The `light_sheet` profile runs DESKEW then DECON.
- The `wide_frame` profile skips DESKEW only when `--decon_only true` is passed.
- The `light_sheet_decon` profile skips DESKEW and expects `CH##_######.tif(f)` 3-D stacks.
- The `my_cluster` profile only enables cluster conda/mamba defaults. The `docker` profile is also available for non-HPC use.
- DESKEW runs on the `super` queue, while DECON runs on the `GPU` queue.
- The active deconvolution PSF is always the merged blind estimate. The theoretical PSF generated from optical parameters is only a starting guess for blind estimation.
- Nextflow work directories accumulate large intermediate files. Clean up with `nextflow clean -f` after a successful run.
