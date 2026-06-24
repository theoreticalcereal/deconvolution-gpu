# Workflow Overview

The pipeline is a Nextflow DSL2 workflow with these named processes:

1. `STAGE_DECON_INPUT`
2. `DESKEW`
3. `BUILD_DECON_CONTAINER`
4. `DECON`

The control path is defined in `workflow/main.nf`.

## Full Light-Sheet Path

When `params.decon_only` is false, the workflow runs:

1. `DESKEW(...)`
2. `DECON(DESKEW.out.deskewed_path, ...)`

When Astrocyte file-picker inputs are present, `STAGE_DECON_INPUT` first links
the selected TIFFs into a `decon_input/` directory. `DESKEW` reads that staged
directory directly, with `cell_name` ignored for the staged-file path.

The deskew process publishes two directories:

- `shear`: intermediate sheared volumes.
- `Top_shear`: final deskewed top-view stacks.

The `DECON` process receives the `Top_shear` path emitted by `DESKEW`, filters
the `CH*.tif` files inside it, estimates a blind PSF from the first selected
TIFF, and deconvolves every selected TIFF.

## Decon-Only Path

When `params.decon_only` is true, the workflow skips `DESKEW` and runs:

```text
STAGE_DECON_INPUT(input files)
DECON(decon_input, ...)
```

Use this mode for:

- Already deskewed light-sheet stacks.
- Wide-frame 3-D TIFF stacks.
- Any data that already matches the deconvolution input naming pattern.

When Astrocyte file-picker inputs are present, `STAGE_DECON_INPUT` links those
TIFFs into a `decon_input/` directory and passes that directory to `DECON`.
Manual CLI runs without `input` still support the backward-compatible directory
parameters `image_path` and `decon_input_dir`.

## Process Boundaries

`DESKEW` is CPU/MATLAB work. It loads `matlab/2024a`, runs
`deskew_wrapper.py`, and calls `deskew.m` through `matlab -batch`.

`BUILD_DECON_CONTAINER` prepares the runtime used by `DECON`. It loads the
`mamba` module and creates `decon_runtime/decon_env` from `environment.yml` for
each workflow run.

`DECON` is GPU work. It activates the prepared mamba runtime, checks the GPU with
`nvidia-smi`, estimates the PSF, and runs CUDA Richardson-Lucy deconvolution.

## Data Flow

```text
selected or raw TIFF folder
    |
    | full light-sheet mode
    v
STAGE_DECON_INPUT (Astrocyte file-picker runs only)
    |
    v
DESKEW
    |-- shear/
    `-- Top_shear/
            |
            v
          DECON
            |-- estimated_psf.tif
            `-- DB2_*.tif

selected or already stacked TIFF folder
    |
    | decon-only mode
    v
STAGE_DECON_INPUT (Astrocyte file-picker runs only)
    |
    v
DECON
    |-- estimated_psf.tif
    `-- DB2_*.tif
```

## Publishing Behavior

Nextflow runs each process in its own work directory. `publishDir` copies
selected outputs into the configured `output_dir`.

`DESKEW` publishes `shear` and `Top_shear` into `output_dir`.

`DECON` publishes files matching `DB2_*` into `output_dir/deconvolved` and
publishes the merged blind PSF as `output_dir/estimated_psf.tif`.

The PSF file may also be written next to the deconvolution input directory when
that location is writable, but the published copy in `output_dir` is the stable
workflow result.
