# GPU Deconvolution Process

`DECON` runs chunkwise Richardson-Lucy deconvolution on GPU with
`pycudadecon`. It is orchestrated by `workflow/scripts/decon_wrapper.py`.

The current workflow is OME-Zarr-first. Deconvolution reads normalized
OME-Zarr volumes from the staging step or deskewed OME-Zarr volumes from
`Top_shear`. TIFF input remains supported for manual and compatibility runs.

## Input Discovery

Package runs normalize selected files before deconvolution:

```text
input_zarr/<sample>.ome.zarr/
```

Full light-sheet runs pass the deskewed output into `DECON`:

```text
Top_shear/<sample>.ome.zarr/
```

Decon-only runs use the normalized selected input directly. Existing OME-Zarr
inputs are copied into the normalized input directory. TIFF, CZI, ND2, LIF, and
HDF5 inputs are converted there before any GPU work starts.

For compatibility, `decon_wrapper.py` can also scan a directory containing
`*.tif` or `*.tiff` stacks. Use that path only when intentionally bypassing the
normalization process.

Select inputs from one optical configuration at a time. The PSF is estimated
from the first sorted input volume and reused for every volume in the same
`DECON` call, so mixing wavelengths or acquisition modes can skew the result.

## PSF to OTF Conversion

Blind PSF estimation is still TIFF-backed at the MATLAB boundary because
MATLAB `deconvblind` and `pycudadecon.TemporaryOTF` consume TIFF files. OME-Zarr
inputs are opened directly at level 0 for PSF tiling; only the temporary MATLAB
tile inputs, seed PSFs, tile PSF outputs, and merged `estimated_psf.tif` are
written as TIFFs. The wrapper then builds the OTF from the merged PSF file.

The OTF is built with:

| Value | Source |
|---|---|
| `dzpsf` | `dz` |
| `dxpsf` | `dxy` |
| `wavelength` | `round(wavelength * 1000)` nanometers |
| `na` | `detection_na`, or backward-compatible `na` |
| `nimm` | `ni` |

Temporary PSF files are removed after processing. The published
`estimated_psf.tif` is retained as the reproducible PSF artifact for the run.

## Chunking Model

OME-Zarr inputs are opened as Dask arrays. TIFF compatibility inputs are
memory-mapped where possible. Every input must resolve to a 3-D volume.

The deconvolution chunks are full-Z and tiled only in XY:

```text
(z = full stack depth, y = decon_chunk_xy, x = decon_chunk_xy)
```

Z is not split. This avoids stitching artifacts along the axial dimension.

If `decon_chunk_xy > 0`, it is used as the core tile size. If
`decon_chunk_xy <= 0`, the code estimates a tile size from available VRAM, data
type, stack depth, overlap, and worker count. Shallow inputs use a conservative
per-chunk VRAM target so very wide single-slice images do not turn into a few
long-running GPU chunks. Deeper stacks remain capped at 1024 pixels in XY.

## Overlap Handling

Deconvolution uses Dask `map_overlap` with reflect boundaries.

If `overlap_xy > 0`, that value is used. Otherwise, overlap is derived from the
PSF support:

```text
overlap_xy = min(48, max(16, ceil(max(psf_y, psf_x) / 4)))
```

The overlap is capped so it cannot exceed half of the smallest image XY
dimension.

## Per-Chunk GPU Work

For each overlapped chunk, `_decon_chunk`:

1. Opens a `pycudadecon.RLContext`.
2. Runs `rl_decon` with `n_iters = iter`.
3. Clips values to the `uint16` range.
4. Returns a `uint16` block to Dask.

If `decon_workers > 1`, Dask uses the threaded scheduler. Otherwise it uses the
single-threaded scheduler. The Nextflow process requests one GPU, so increasing
`decon_workers` should be tested on the target GPU.

## Intensity Rescaling

For native OME-Zarr inputs, deconvolution is streamed through a temporary raw
Zarr store in the Nextflow work directory. This avoids holding the full
deconvolved stack in RAM. The raw store is then used to compute global output
min/max, and the final OME-Zarr is written chunk-by-chunk after linear mapping
back to the original input intensity range:

```text
scaled = (output - output_min) / (output_max - output_min)
scaled = scaled * (input_max - input_min) + input_min
```

The final array is rounded, clipped to `uint16`, and written as OME-Zarr.
Compatibility TIFF inputs still materialize the deconvolved array before TIFF
writing.

## Outputs

The merged PSF is published to:

```text
<output_dir>/estimated_psf.tif
```

For each native OME-Zarr input, the process writes:

```text
<output_dir>/DB2_<sample>.ome.zarr/
```

Compatibility TIFF runs may still emit `DB2_<input_stem>.tif`, but the package
workflow treats OME-Zarr as the primary deconvolution output. Set
`output_formats` to `tiff` to publish TIFF stack exports under
`<output_dir>/deconvolved_tiff/`.
