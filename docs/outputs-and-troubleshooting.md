# Outputs and Troubleshooting

## Input Shape

This workflow expects deskewed or otherwise ready-to-deconvolve volumes. Raw
ctASLM/light-sheet data should be processed with `deskew-gpu` first.

## Missing Optical Parameters

`decon_wrapper.py` validates required optical and sampling parameters before
starting PSF estimation. If the run fails early, check `wavelength`, `na` or
`detection_na`, `ni`, `ns`, `dxy`, and `dz`.

## Blind PSF Backend

Native CuPy blind PSF estimation is the default. It requires a Slurm GPU
allocation; CUDA initialization errors are expected if it is invoked directly
on a login node without a GPU. Use `blind_workers = 1` for a one-GPU job and
allow Slurm to set `CUDA_VISIBLE_DEVICES`.

MATLAB `deconvblind` remains available with `blind_backend = matlab`. The
BioHPC config loads MATLAB and passes its executable into the process for that
backend.

## Runtime Environment

The workflow runs in the prebuilt
`git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.2` Singularity
container. If Python dependencies are missing at runtime, rebuild and republish
that image rather than adding per-run conda build steps back to the workflow.

The image must provide CUDA 11.8, CuPy 13.6, cuCIM 23.06, SciPy, and cupyx FFT
convolution. Legacy `pycudadecon` and `cudadecon` packages are excluded from
the exported Conda lock. Check the published image with:

```bash
workflow/images/deconvolution-gpu/check-deployment-container.sh
```

CuPy JIT compilation uses a writable UID-specific cache under `SLURM_TMPDIR`,
`TMPDIR`, or `/tmp`. Set `CUPY_CACHE_DIR` explicitly to override that location.
This avoids attempts to write beneath a read-only home directory mounted into
the Singularity image.

The publishing script loads `mamba/2.3.0` through Lmod and exports the already
validated `../decon_env` environment; it does not run a dependency solver.
Set `SOURCE_ENV_PREFIX=/path/to/environment` to use another compatible source.
The script verifies the pinned scientific/GPU versions, then Mamba exports exact
package URLs and checksums to a temporary explicit lock. Podman installs that
lock without solving, while pip-only packages come from `requirements.txt`.
The temporary build context is removed when publishing exits.
Mamba 2.3 emits package URLs with human-readable and comment lines but without
the legacy marker expected by Conda. The publishing script writes `@EXPLICIT`,
retains only HTTP or file package URLs, and confirms that CuPy and cuCIM are in
the resulting lock before Podman starts.

The publishing script prints a timestamped heartbeat while Mamba resolution,
Podman building, or Singularity verification is active. The default interval is
30 seconds and can be changed with `HEARTBEAT_SECONDS`. Each wrapped command
reports an explicit completed or failed status while preserving its exit code.

Pip-only image readers are installed with `constraints.txt` so they cannot
upgrade NumPy beyond Numba's supported range or downgrade the validated TIFF
and Zarr stack. The Docker build runs `pip check` and imports the pinned
NumPy/Numba/SciPy/scikit-image/TIFF/Zarr versions before the image can be pushed.
The source environment currently contains TIFFFile 2022.10.10; the pip stage
explicitly upgrades the container to TIFFFile 2025.5.10 under the same NumPy
constraints. Source validation reports package names and actual versions rather
than an anonymous assertion failure.

After exporting registry credentials, rebuild and republish the existing tag:

```bash
NO_CACHE=1 workflow/images/deconvolution-gpu/build-deconvolution-image.sh
```

If a CuPy tile process fails, check the Slurm allocation, container `--nv`
binding, CUDA compatibility, and VRAM before increasing worker counts. Cache
entries are separated by backend and normalization settings.
