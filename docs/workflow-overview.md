# Workflow Overview

The pipeline is a Nextflow DSL2 workflow. The control path is defined in
`workflow/main.nf`; process bodies live in `workflow/modules.nf`.

## Process Order

For file-picker runs, the workflow runs:

1. `BUILD_DECON_CONTAINER`
2. `STAGE_DECON_INPUT`
3. `DESKEW` unless `decon_only = true`
4. `DECON`
5. `CONVERT_TIFFS_TO_NEUROGLANCER`

`BUILD_DECON_CONTAINER` prepares the conda runtime used by normalization,
deconvolution, and OME-Zarr/Neuroglancer conversion.

## Input Normalization Boundary

`STAGE_DECON_INPUT` links the selected inputs into a process-local
`decon_input/` directory and writes an `original_filenames.tsv` map. It then
runs `normalize_input_to_ome_zarr.py`, producing:

```text
input_zarr/<sample>.ome.zarr/
input_zarr/original_filenames.tsv
```

Supported file-picker inputs include TIFF, OME-Zarr, CZI, ND2, LIF, and HDF5.
Existing OME-Zarr inputs are copied into the normalized input directory.

Manual CLI runs without `input` still support `image_path` and
`decon_input_dir`. Those paths should already contain compatible TIFF or
OME-Zarr volumes.

## Full Light-Sheet Path

When `decon_only = false`, the workflow runs:

```text
STAGE_DECON_INPUT -> DESKEW -> DECON -> CONVERT_TIFFS_TO_NEUROGLANCER
```

`DESKEW` reads normalized OME-Zarr volumes and writes deskewed volumes in
`Top_shear/`. `DECON` receives `Top_shear/`, estimates the blind PSF from the
first volume, and deconvolves each selected volume.

## Decon-Only Path

When `decon_only = true`, the workflow skips `DESKEW`:

```text
STAGE_DECON_INPUT -> DECON -> CONVERT_TIFFS_TO_NEUROGLANCER
```

Use this mode for already deskewed light-sheet data, wide-frame 3-D stacks, or
existing OME-Zarr volumes.

## Publishing Behavior

Nextflow runs each process in an isolated work directory. `publishDir` copies
stable products into `output_dir`.

`DESKEW` publishes:

```text
<output_dir>/Top_shear/
```

`DECON` publishes:

```text
<output_dir>/estimated_psf.tif
<output_dir>/deconvolved/DB2_<sample>.ome.zarr/
```

`CONVERT_TIFFS_TO_NEUROGLANCER` publishes:

```text
<output_dir>/neuroglancer/layers.json
```

The Neuroglancer manifest points at OME-Zarr data. If `DECON` emitted native
OME-Zarr, no TIFF reconversion is needed.
