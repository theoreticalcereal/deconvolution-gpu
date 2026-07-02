# 3D GPU Deconvolution

Astrocyte/Nextflow package for blind PSF estimation and GPU Richardson-Lucy
deconvolution. Deskewing is intentionally not part of this package; run the
separate `deskew-gpu` workflow first when ctASLM/light-sheet data needs
geometric correction.

## Pipeline

1. `BUILD_DECON_CONTAINER` builds the per-run conda runtime.
2. `STAGE_DECON_INPUT` links selected files, preserves original filenames, and
   normalizes supported images to `input_zarr/*.ome.zarr`.
3. `DECON` estimates one blind PSF from the first selected volume and applies
   GPU deconvolution to all selected volumes.
4. `EXPORT_OUTPUT_FORMAT` runs only when `output_formats = 'tiff'`.

The native output is OME-Zarr. TIFF export remains available for downstream
tools that require TIFF stacks.

## Inputs

Astrocyte `input` accepts TIFF, OME-Zarr, CZI, ND2, LIF, HDF5, and H5 files.
For light-sheet deconvolution, pass already deskewed volumes from `deskew-gpu`
and set `psf_mode = light_sheet`.

## Manual Run

```bash
cd workflow
nextflow run main.nf \
  -c configs/biohpc.config \
  -profile light_sheet \
  --input '/path/to/deskewed/*.ome.zarr' \
  --output_dir ./output \
  --wavelength 0.561 \
  --na 1.0 \
  --ni 1.33 \
  --ns 1.33 \
  --dxy 0.108 \
  --dz 0.3
```

## Output

```text
workflow/output/
|-- DB2_<sample>.ome.zarr/
|-- estimated_psf.tif
`-- deconvolved_tiff/
    `-- DB2_<sample>.tif    # only when output_formats = tiff
```

## VizApp

`vizapp/` is intentionally a placeholder containing only `.keep`. Visualization
is handled by a separate workflow package.
