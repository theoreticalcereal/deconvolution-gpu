# GPU Deconvolution Process

The `DECON` process runs Richardson-Lucy deconvolution on GPU with
`pycudadecon`. It is orchestrated by `workflow/scripts/decon_wrapper.py`.

## Input Discovery

Astrocyte handles file ingestion for package runs. The wrapper scans the
selected input directory for:

```text
*.tif
*.tiff
```

When `channels` or `timepoints` are supplied, it filters names with:

```text
^CH(?P<channel>\d+)_(?P<timepoint>\d+)(?:_registered_consistent)?$
```

This means decon-only inputs can use arbitrary TIFF names when no
channel/timepoint filters are requested. If filters are requested, inputs
should be named like:

```text
CH0_0.tif
CH1_0.tif
CH0_0_registered_consistent.tif
```

The deskew process currently writes raw-style names such as
`CH00_000000.tif`. Those still match the deconvolution regex because the
channel and timepoint captures accept any number of digits.

Select files from one channel at a time. The PSF is estimated from the first
selected TIFF and reused for every selected TIFF, so mixing wavelengths/channels
can skew the result.

## PSF to OTF Conversion

Before deconvolving a TIFF, the wrapper writes the merged blind PSF to a
temporary TIFF and passes it to `pycudadecon.TemporaryOTF`.

The OTF is built with:

- `dzpsf = dz`
- `dxpsf = dxy`
- `wavelength = round(wavelength * 1000)` in nanometers
- `na = detection_na` or fallback `na`
- `nimm = ni`

The temporary PSF file is removed after processing.

## Chunking Model

Each input TIFF is memory-mapped where possible and must be a 3-D volume.

The Dask array chunks are full-Z and tiled only in XY:

```text
(z = full stack depth, y = decon_chunk_xy, x = decon_chunk_xy)
```

Z is never split. This avoids stitching artifacts along the axial dimension.

If `decon_chunk_xy > 0`, it is used as the core tile size. If
`decon_chunk_xy <= 0`, the code estimates a tile size from available VRAM, data
type, stack depth, overlap, and worker count.

## Overlap Handling

The deconvolution uses Dask `map_overlap` with reflect boundaries.

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
3. Clips values to the uint16 range.
4. Returns a `uint16` block to Dask.

If `decon_workers > 1`, Dask uses the threaded scheduler. Otherwise it uses the
single-threaded scheduler. The Nextflow process requests one GPU, so increasing
`decon_workers` should be tested carefully on the target GPU.

## Intensity Rescaling

After all chunks are merged, the output is linearly mapped back to the original
input TIFF intensity range:

```text
scaled = (output - output_min) / (output_max - output_min)
scaled = scaled * (input_max - input_min) + input_min
```

The final array is rounded, clipped to the uint16 range, and written as
`uint16`.

## Outputs

The merged PSF used for all selected TIFFs is published to:

```text
<output_dir>/estimated_psf.tif
```

For each selected input TIFF, the process writes:

```text
DB2_<input_stem>.tif
```

Examples:

```text
DB2_CH0_0.tif
DB2_CH00_000000.tif
DB2_CH0_0_registered_consistent.tif
DB2_my_stack.tif
```

Nextflow publishes these files to:

```text
<output_dir>/deconvolved/
```
