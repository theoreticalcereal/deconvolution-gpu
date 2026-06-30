# ctASLM2 Deconvolution

Nextflow DSL2 workflow for ctASLM/light-sheet deskewing, blind PSF estimation,
GPU Richardson-Lucy deconvolution, and Neuroglancer visualization on BioHPC.

The current workflow is OME-Zarr-first. Selected inputs are normalized to
chunked OME-Zarr before deskew or deconvolution. TIFF remains supported as an
input format and as a compatibility/export format, but it is no longer the main
internal image contract.

## Pipeline

1. `BUILD_DECON_CONTAINER` builds the per-run conda runtime used by Python,
   Zarr, Dask, pycudadecon, and Neuroglancer conversion.
2. `STAGE_DECON_INPUT` links selected files, preserves original filenames, and
   normalizes supported images into `input_zarr/*.ome.zarr`.
3. `DESKEW` reads normalized OME-Zarr volumes and writes deskewed OME-Zarr
   volumes under `Top_shear/`.
4. `DECON` reads TIFF or OME-Zarr volumes, estimates one blind PSF from the
   first selected volume, runs chunked GPU deconvolution, and writes
   `DB2_*.ome.zarr` for native Zarr inputs.
5. `CONVERT_TIFFS_TO_NEUROGLANCER` writes `neuroglancer/layers.json`. If
   deconvolution output is already OME-Zarr, it points directly to that data.

## Supported Inputs

Astrocyte `input` accepts:

- TIFF: `.tif`, `.tiff`
- OME-Zarr: `.ome.zarr`
- CZI: `.czi`
- ND2: `.nd2`
- LIF: `.lif`
- HDF5: `.h5`, `.hdf5`

CZI, ND2, LIF, and HDF5 support depends on the reader packages installed in the
runtime. If a reader is unavailable, normalization fails with a clear error.
Select one channel/optical condition per run because one blind PSF is estimated
from the first selected volume and applied to all selected volumes.

## Common Runs

Full light-sheet deskew plus deconvolution:

```bash
nextflow run workflow/main.nf -profile light_sheet \
  --input '/path/to/CH00_000000.tiff,/path/to/CH00_000001.tiff' \
  --dx 0.167 \
  --dz 0.2 \
  --dxy 0.167 \
  --wavelength 0.515 \
  --ni 1.56 \
  --ns 1.56 \
  --detection_na 0.7467 \
  --output_dir ./workflow/output
```

Deconvolution only for already deskewed/stacked data:

```bash
nextflow run workflow/main.nf -profile light_sheet_decon \
  --decon_only true \
  --decon_input_dir /path/to/deskewed_or_zarr_dir \
  --dxy 0.167 \
  --dz 0.2 \
  --wavelength 0.515 \
  --ni 1.56 \
  --ns 1.56 \
  --detection_na 0.7467 \
  --illumination_na 0.7 \
  --output_dir ./workflow/output
```

Use `-resume` to restart from successful Nextflow checkpoints.

## Important Parameters

| Parameter | Purpose |
|---|---|
| `input` | Astrocyte/file-picker inputs. Supported formats are normalized to OME-Zarr. |
| `output_dir` | Published output root. Default: `./workflow/output`. |
| `output_formats` | Requested leaf exports. Default: `ome_zarr`; `ome_zarr,tiff` is reserved for TIFF export support. |
| `decon_only` | Skip `DESKEW` and run `DECON` directly. |
| `decon_input_dir` | Manual decon-only input directory. |
| `dx`, `dz`, `angle`, `flip` | Deskew geometry. |
| `dxy`, `wavelength`, `detection_na`, `illumination_na`, `ni`, `ns` | Optical/acquisition parameters for PSF generation and deconvolution. |
| `blind_iters`, `chunk_xy`, `blind_z_slices`, `matlab_workers` | Blind PSF estimation controls. |
| `decon_chunk_xy`, `overlap_xy`, `decon_workers`, `vram_gb` | GPU deconvolution chunking controls. |
| `neuroglancer_data_mode` | TIFF dimensionality policy for legacy TIFF-to-Zarr visualization conversion. |

## Output Layout

```text
<output_dir>/
|-- Top_shear/
|   |-- <sample>.ome.zarr/
|   `-- note.txt
|-- estimated_psf.tif
|-- deconvolved/
|   `-- DB2_<sample>.ome.zarr/
`-- neuroglancer/
    `-- layers.json
```

`estimated_psf.tif` remains TIFF because the PSF estimation and OTF creation
path uses TIFF-compatible tooling. The image data products are OME-Zarr.

## Documentation

Start with [docs/index.md](docs/index.md). The most relevant pages are:

- [Workflow Overview](docs/workflow-overview.md)
- [Deskew Process](docs/deskew-process.md)
- [GPU Deconvolution Process](docs/gpu-deconvolution-process.md)
- [Profiles and Parameters](docs/profiles-and-parameters.md)
- [workflow/output](docs/workflow-output.md)
- [Outputs and Troubleshooting](docs/outputs-and-troubleshooting.md)

## Notes

- Load the cluster Nextflow module before manual runs.
- `DESKEW` runs on the `super` queue; `DECON` runs on the `GPU` queue.
- The Astrocyte VizApp follows the standalone Neuroglancer package pattern:
  Astrocyte pulls the dummy `docker://hello-world` VizApp container, then
  `vizapp/run_neuroglancer.sh` loads the BioHPC `neuroglancer/2.40.1` module
  and runs `vizapp/neuroloader.py`.
- The theoretical PSF is only a seed. The active deconvolution PSF is the
  merged blind estimate.
- Nextflow work directories can be large. Clean with `nextflow clean -f` after
  preserving needed results from `output_dir`.
