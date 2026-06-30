# Deskew Process

`DESKEW` corrects oblique ctASLM acquisition geometry before deconvolution. The
current Nextflow path uses `workflow/scripts/chunked_deskew.py`, which reads the
normalized OME-Zarr input created by `STAGE_DECON_INPUT` and writes chunked
OME-Zarr output.

The older MATLAB wrapper remains in the repository for legacy TIFF workflows,
but the package workflow no longer depends on TIFF stacks as the deskew
boundary.

## Inputs

`DESKEW` receives normalized OME-Zarr volumes from:

```text
input_zarr/<sample>.ome.zarr/
```

Astrocyte and Nextflow create that directory before processing. Supported
file-picker inputs include TIFF, existing OME-Zarr, CZI, ND2, LIF, and HDF5.

The process also receives the deskew geometry:

| Value | Meaning |
|---|---|
| `dx` | Lateral pixel size in microns. |
| `dz` | Axial step size in microns. |
| `angle` | Light-sheet acquisition angle in degrees. |
| `flip` | Orientation flag used in the shear direction. |

Manual legacy runs can still use `image_path` and `cell_name` with TIFF stacks,
but package runs should use `input` so normalization happens once at the
workflow boundary.

## Chunkwise Deskew

`chunked_deskew.py` loads each OME-Zarr volume through Dask. The implementation
keeps the operation chunk-aware so large volumes do not need to be materialized
as a single dense array before writing.

For each volume, the deskew transform:

1. Applies the Y shear implied by `angle`, `dz`, `dx`, and `flip`.
2. Pads the sheared volume so shifted planes are not clipped.
3. Computes the top-view scaling from `dz * sin(angle) / dx`.
4. Rotates into the top-view orientation.
5. Writes the result as OME-Zarr.

The legacy MATLAB implementation uses the same geometry and remains useful for
comparison or manual TIFF-only runs.

## Outputs

For each normalized input volume, `DESKEW` publishes:

```text
<output_dir>/Top_shear/<sample>.ome.zarr/
<output_dir>/Top_shear/note.txt
```

The `note.txt` file records the deskew geometry used for the run. Native
deconvolution consumes the OME-Zarr output directly.

## Common Failure Points

`No normalized OME-Zarr inputs found` means `STAGE_DECON_INPUT` did not create
an input volume or the deskew process was pointed at the wrong directory.

Reader import errors during staging usually mean the optional reader for that
format is missing from the runtime environment. CZI, ND2, LIF, and HDF5 support
uses optional Python dependencies in addition to the core TIFF and OME-Zarr
path.

Geometry errors usually come from missing or invalid `dx`, `dz`, `angle`, or
`flip` values. Check the process `.command.out` for the resolved parameters and
input volume shape.
