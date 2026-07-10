# Outputs and Troubleshooting

## Input Shape

This workflow expects deskewed or otherwise ready-to-deconvolve volumes. Raw
ctASLM/light-sheet data should be processed with `deskew-gpu` first.

## Missing Optical Parameters

`decon_wrapper.py` validates required optical and sampling parameters before
starting PSF estimation. If the run fails early, check `wavelength`, `na` or
`detection_na`, `ni`, `ns`, `dxy`, and `dz`.

## MATLAB

Blind PSF estimation uses MATLAB `deconvblind`. The BioHPC config loads the
MATLAB module and passes the resolved executable into the deconvolution process.

## Runtime Environment

The workflow runs in the prebuilt
`git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.0` Singularity
container. If Python dependencies are missing at runtime, rebuild and republish
that image rather than adding per-run conda build steps back to the workflow.
