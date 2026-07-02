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

The workflow builds its conda runtime from `workflow/envs/decon-conda.txt` and
`workflow/envs/decon-pip-requirements.txt`. Visualization dependencies are not
installed by this package.
