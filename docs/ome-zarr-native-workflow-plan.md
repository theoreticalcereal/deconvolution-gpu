# OME-Zarr Native Workflow

This page records the current OME-Zarr behavior that replaced the earlier
TIFF-primary workflow plan.

## Current Contract

OME-Zarr is the workflow-native image representation.

```text
selected input
  -> input_zarr/<sample>.ome.zarr
  -> Top_shear/<sample>.ome.zarr
  -> deconvolved/DB2_<sample>.ome.zarr
  -> neuroglancer/layers.json
```

TIFF remains supported as an input format, as an internal compatibility format
for MATLAB blind PSF estimation, and as the published `estimated_psf.tif`
artifact.

## Input Normalization

`STAGE_DECON_INPUT` runs before deskew or deconvolution. It accepts:

- TIFF: `.tif`, `.tiff`
- OME-Zarr: `.ome.zarr`
- CZI: `.czi`
- ND2: `.nd2`
- LIF: `.lif`
- HDF5: `.h5`, `.hdf5`

All compatible inputs are normalized to:

```text
input_zarr/<sample>.ome.zarr/
```

Existing OME-Zarr inputs are copied into the staging directory. Other formats
are read with the available Python reader and written as chunked OME-Zarr.

## Deskew

`DESKEW` reads normalized OME-Zarr volumes and writes:

```text
Top_shear/<sample>.ome.zarr/
Top_shear/note.txt
```

The current path uses `workflow/scripts/chunked_deskew.py`. The legacy MATLAB
deskew wrapper remains in the repository for manual TIFF compatibility.

## Deconvolution

`DECON` reads OME-Zarr volumes through Dask and runs GPU Richardson-Lucy
deconvolution chunkwise in XY with full-Z chunks. Native outputs are written as:

```text
deconvolved/DB2_<sample>.ome.zarr/
```

The blind PSF estimator and `pycudadecon.TemporaryOTF` still use temporary TIFF
files internally. The merged PSF is published as:

```text
estimated_psf.tif
```

## Neuroglancer

The Neuroglancer step writes a layer manifest that points at the OME-Zarr
outputs:

```text
neuroglancer/layers.json
```

The VizApp validates `.ome.zarr` directories and rewrites local paths for
browser serving.

## Export Formats

`output_formats` defaults to:

```text
ome_zarr
```

Native OME-Zarr is the stable published image output. TIFF export can be added
or used for legacy tooling, but it is not the default processing contract.

## Runtime Dependencies

The core OME-Zarr path uses `zarr`, `dask`, `numpy`, and `tifffile`. Dynamic
input support also uses optional readers from the decon runtime environment:

- `aicsimageio` for CZI and related microscopy formats.
- `nd2` for ND2.
- `readlif` for LIF.
- `h5py` for HDF5.

If one of those optional packages is unavailable, staging fails before any
deskew or GPU work starts.
