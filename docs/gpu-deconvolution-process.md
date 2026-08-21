# GPU Deconvolution Process

`DECON` runs chunkwise Richardson-Lucy deconvolution through the generic CuPy
backend for low/medium presets and MATLAB for the high preset. It is
orchestrated by `workflow/scripts/decon_wrapper.py`.

The current workflow uses OME-Zarr internally and OZX for published native
outputs. Deconvolution reads selected TIFF inputs directly, normalized OME-Zarr
volumes from the staging step, or deskewed OME-Zarr volumes from `Top_shear`.

## Input Discovery

Package runs link selected TIFF files directly before deconvolution:

```text
input_tiff/<sample>.tif
```

Non-TIFF package runs normalize selected files before deconvolution:

```text
input_zarr/<sample>.ome.zarr/
```

Full light-sheet runs pass the deskewed output into `DECON`:

```text
Top_shear/<sample>.ome.zarr/
```

Decon-only runs use the staged selected input directly. Existing OME-Zarr inputs
are copied into the normalized input directory, and OZX inputs are unzipped
there. CZI, ND2, LIF, and HDF5 inputs are converted there before any GPU work
starts. Mixed TIFF and non-TIFF selections also use the normalization path.

Select inputs from one optical configuration at a time. The PSF is estimated
from the first sorted input volume and reused for every volume in the same
`DECON` call, so mixing wavelengths or acquisition modes can skew the result.

## PSF Handoff

Blind PSF estimation uses TIFF files as the process boundary for both backends.
OME-Zarr inputs are opened directly at level 0 for PSF tiling; only temporary
tile inputs, seed PSFs, tile PSF outputs, and merged `estimated_psf.tif` are
written as TIFFs. Native CuPy workers read and write those files in isolated
spawned processes. MATLAB mode passes the same files to `deconvblind`. The
wrapper normalizes the merged PSF. Low and medium presets pass it directly to
the generic CuPy implementation; high mode gives it to MATLAB `deconvlucy`.
No external Petakit installation is required.

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

## Per-Chunk Restoration Work

For low/medium presets, each overlapped chunk, `_decon_chunk`:

1. Transfers the chunk and normalized PSF to CuPy.
2. Runs generic accelerated CuPy Richardson-Lucy with `n_iters = iter`.
3. Copies the restored chunk back to host memory.
4. Synchronizes the device, clears the cuFFT plan cache, and releases all
   cached blocks from CuPy's default device-memory pool, including on errors.
5. Crops or pads the result to preserve the input chunk shape.
6. Clips values to the `uint16` range and returns the block to Dask.

Releasing cached allocations adds a small allocation and FFT-planning cost to
each chunk, but prevents completed chunks from reserving most of an 8 GiB GPU
and starving the next FFT allocation.

The deconvolution worker count is clamped to one because the Nextflow process
owns one GPU. Dask sequences chunks through that GPU without concurrent CuPy
contexts competing for device memory.

This is independent of `blind_workers`, which controls PSF tile tasks and
launches one spawned process per active CuPy tile. For a one-GPU Slurm
allocation, keep both values at `1` unless memory and throughput have been
measured on the target GPU.

For the high preset, the same chunk/halo handoff writes temporary TIFF inputs,
invokes MATLAB `deconvlucy`, and reads the uint16 result back into Dask. The
high preset already binds the host MATLAB installation into Singularity and has
no GPU allocation; it uses MATLAB for both blind PSF estimation and final
deconvolution.

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

The final array is rounded, clipped to `uint16`, and written as an internal
OME-Zarr directory. The `DECON` task then zips each `DB2_*.ome.zarr` directory
to `DB2_*.ozx` and removes the expanded directory from task scratch space.
TIFF inputs still materialize the deconvolved array. A TIFF request writes that
array directly; an OZX request writes it to OME-Zarr before the process archive
step. Inputs of every other supported type are normalized to OME-Zarr before
deconvolution and retain the streaming output path.

## Outputs

The merged PSF is published to:

```text
<output_dir>/estimated_psf.tif
```

For each non-TIFF input, or a TIFF input requesting OZX, the process publishes:

```text
<output_dir>/DB2_<sample>.ozx
```

Set `output_formats` to `tiff` to publish direct or converted TIFF stacks under
`<output_dir>/deconvolved_tiff/`.
