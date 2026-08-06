# Deconvolution Comparison Issues Handoff

Last updated: 2026-08-06

## Purpose

This handoff records the MATLAB reference comparison work for the current GPU
deconvolution blur issue. The immediate question is whether the workflow blur
comes from blind PSF generation or from the final cuCIM Richardson-Lucy
deconvolution step.

## Reproduction Assets

Input TIFF pulled from upstream:

```text
1_CH00_000000.tif
```

Comparison runner:

```text
workflow/scripts/run_matlab_reference_comparison.sh
```

Volume comparison helper:

```text
workflow/scripts/compare_volumes.py
```

Volume comparison tests:

```text
tests/test_compare_volumes.py
```

The runner stages the repo-root `1_CH00_000000.tif` into each comparison run
directory and runs:

1. Current Nextflow workflow on the staged TIFF.
2. Standalone MATLAB reference using the staged `1_CH00_000000.tif`.
3. PSF comparison with `workflow/scripts/compare_psfs.py`.
4. Full-volume comparison with `workflow/scripts/compare_volumes.py`.

The runner defaults the Nextflow `DECON` process queue to `GPUp40`.

## Acquisition Parameters Used

Current runner defaults:

| Parameter | Value |
| --- | --- |
| XY voxel size | `0.104 um` |
| Z voxel size | `0.3 um` |
| Detection NA | `1.1` |
| Illumination NA | `0.19` |
| Emission wavelength | `0.610 um` |
| MATLAB reference iterations | `10` |
| MATLAB padding | `[20 20 20]` |
| MATLAB threads | `8` |

MATLAB reference PSF default:

```text
/archive/bioinformatics/Danuser_lab/Fiolka/MicroscopeDevelopment/SyntheticPSF/omniOPM/oil/NA0.2_ill_561_det_610_NA1_40degree_0.118umxyz_BottomtoTop.tif
```

Important caveat: the smaller test volume is smaller than the full synthetic
PSF in Z after padding. The generated MATLAB script center-crops the PSF before
`deconvblind` when needed. The observed warning was:

```text
PSF shape [256  256  256] exceeds padded input shape [552  552  191];
center-cropping PSF to fit deconvblind OUTSIZE.
```

This makes the standalone reference runnable, but it is not identical to using
the full uncropped PSF on a larger volume.

## Recovered Completed Run

Completed comparison run recovered after the bash post-MATLAB syntax failure:

```text
comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809
```

Recovered files:

| Artifact | Path |
| --- | --- |
| MATLAB PSF | `matlab_reference/PSFr_Top_Cell19/Top_Cell19psfr1.tif` |
| MATLAB volume | `matlab_reference/DB2_Top_Cell19/1_CH00_000000.tif` |
| Workflow PSF | `workflow/estimated_psf.tif` |
| Workflow volume | `workflow/deconvolved_tiff/DB2_1_CH00_000000.tif` |
| PSF metrics | `metrics/psf_comparison.csv`, `metrics/psf_comparison.json` |
| Volume metrics | `metrics/volume_comparison.csv`, `metrics/volume_comparison.json` |

The runner syntax error has since been fixed by simplifying the MATLAB `-batch`
launch command.

## Key Findings

The workflow output is measurably blurrier than the MATLAB reference volume.

Full-volume metrics from the recovered run:

| Metric | Value |
| --- | ---: |
| NCC | `0.9639101352` |
| SSIM | `0.9829456832` |
| MAE | `7.2529304201` |
| RMSE | `21.8896151931` |
| Gradient energy ratio, workflow/MATLAB | `0.6182452213` |
| High-frequency fraction ratio, workflow/MATLAB | `0.0751393834` |

The high-frequency ratio is the clearest blur signal: the workflow output
retains only about 7.5 percent of the MATLAB reference high-frequency fraction
under the current comparison metric.

The PSF mismatch points first at blind PSF estimation:

| PSF metric | MATLAB reference | Workflow |
| --- | ---: | ---: |
| Shape | `191x256x256` | `101x61x61` |
| XY FWHM X, voxels | `3.0822722722` | `5.1021442018` |
| XY FWHM Y, voxels | `3.0322164620` | `5.7875704988` |
| Z FWHM, voxels | `9.1857024746` | `9.2951820111` |
| Gaussian R2 | `0.9382261299` | `0.5682643097` |
| NCC | `0.8068171468` | compared pair |
| SSIM | `0.9750090351` | compared pair |

Interpretation: workflow and MATLAB are similar in Z width, but the workflow
estimated PSF is much broader in XY. That broader XY PSF is consistent with
the full-volume blur.

## Issues To Address

### 1. CuPy scout mode estimates an overly broad XY PSF

Current evidence:

- MATLAB reference XY FWHM is about `3.0` voxels.
- Workflow CuPy scout XY FWHM is about `5.1` to `5.8` voxels.
- Final workflow volume has strongly reduced high-frequency content.

Likely contributing differences:

- Scout mode performs a short adaptive pass, selects/merges tiles, then refines.
- The current small test volume may not benefit from scout selection.
- Tile merge and SNR weighting can broaden the merged PSF.
- Workflow PSF seed shape and MATLAB recovered PSF shape differ substantially.

Recommended first experiment:

```bash
WF_CUPY_FFT_ENGINE=cupyx \
NEXTFLOW_EXTRA_ARGS="--chunk_xy 512 --blind_max_tiles 1 --blind_z_slices 0 --pad_xy 20 --pad_z 20 --blind_latent_update_period 1" \
workflow/scripts/run_matlab_reference_comparison.sh
```

Goal: make CuPy blind estimation use one whole-frame tile, full Z, MATLAB-like
padding, and every-iteration latent updates.

Recommended scout-mode experiment:

```bash
WF_CUPY_FFT_ENGINE=scout \
NEXTFLOW_EXTRA_ARGS="--adaptive_scout_iters 5 --adaptive_keep_tiles 16 --blind_max_tiles 16 --blind_z_slices 0 --pad_xy 20 --pad_z 20 --blind_latent_update_period 1" \
workflow/scripts/run_matlab_reference_comparison.sh
```

Goal: keep scout mode but reduce aggressive pruning and make refinement closer
to full alternating updates.

Success criterion:

- Workflow PSF XY FWHM moves toward `3.0` voxels.
- Full-volume high-frequency fraction ratio increases substantially above
  `0.075`.

### 2. Need an explicit final-deconvolution isolation test

The current comparison suggests PSF generation is the primary problem, but it
does not completely rule out cuCIM behavior.

Needed workflow/debug feature:

- Add a mode to run final cuCIM deconvolution with an externally supplied PSF.
- Use the recovered MATLAB `Top_Cell19psfr1.tif` as the fixed PSF.
- Compare cuCIM output with standalone MATLAB `deconvlucy` using the same PSF.

Success criterion:

- If cuCIM with MATLAB PSF matches MATLAB volume closely, the main defect is
  PSF estimation.
- If cuCIM with MATLAB PSF is still blurred, investigate final RL settings,
  chunk overlap, boundary handling, intensity scaling, and cuCIM API behavior.

### 3. MATLAB reference is slow and opaque during long steps

Observed behavior:

- Full MATLAB reference can run for around 18 minutes on the small test volume.
- The expensive step is full-volume `deconvblind`.

Mitigations already added:

- `MATLAB_THREADS` defaults to `8`.
- MATLAB script prints before/after `deconvblind` and `deconvlucy`.

Remaining issue:

- Full-volume MATLAB reference is still slow. Do not chunk it unless the goal
  changes, because chunking would no longer match the standalone reference.

### 4. PSF shape mismatch complicates reference fairness

The full synthetic PSF is `256x256x256`, while the smaller test volume produces
a padded input of `552x552x191`. MATLAB cannot run `deconvblind` when the PSF is
larger than the output size in any dimension, so the script center-crops the
PSF in Z.

Issue:

- MATLAB reference uses cropped PSF on the smaller test volume.
- Workflow seed size is `101x61x61` by default.
- Shape differences make direct PSF comparison less clean.

Potential fixes:

- Generate a workflow seed with a Z size matching the cropped MATLAB reference
  where feasible.
- Add a comparison experiment with `WF_PSF_SIZE_Z=191` and `WF_PSF_SIZE_XY=256`
  if GPU memory permits.
- Alternatively, make both reference and workflow use a common smaller PSF size
  and document that as the controlled benchmark.

### 5. Comparison artifacts should be kept out of normal commits

Generated comparison outputs are large and should not be committed by default:

```text
comparison_runs/
workflow/*.tif
out.json
```

The upstream test TIFF `1_CH00_000000.tif` is intentionally tracked and should
remain available for this benchmark.

## Useful Commands

Run the default comparison:

```bash
workflow/scripts/run_matlab_reference_comparison.sh
```

Run only comparisons after both MATLAB and workflow outputs already exist:

```bash
RUN_WORKFLOW=0 RUN_MATLAB=0 workflow/scripts/run_matlab_reference_comparison.sh
```

Use more MATLAB CPU threads:

```bash
MATLAB_THREADS=16 workflow/scripts/run_matlab_reference_comparison.sh
```

Use an explicit input TIFF:

```bash
INPUT_TIFF=/path/to/1_CH00_000000.tif workflow/scripts/run_matlab_reference_comparison.sh
```

Manually compare recovered PSFs:

```bash
../decon_env/bin/python workflow/scripts/compare_psfs.py \
  comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809/matlab_reference/PSFr_Top_Cell19/Top_Cell19psfr1.tif \
  comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809/workflow/estimated_psf.tif \
  --spacing 0.3 0.104 0.104 \
  --csv comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809/metrics/psf_comparison.csv \
  --json comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809/metrics/psf_comparison.json
```

Manually compare recovered volumes:

```bash
../decon_env/bin/python workflow/scripts/compare_volumes.py \
  comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809/matlab_reference/DB2_Top_Cell19/1_CH00_000000.tif \
  comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809/workflow/deconvolved_tiff/DB2_1_CH00_000000.tif \
  --spacing 0.3 0.104 0.104 \
  --csv comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809/metrics/volume_comparison.csv \
  --json comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_123809/metrics/volume_comparison.json
```

## Next Recommended Work Order

1. Run the `cupyx` whole-frame experiment and compare PSF XY FWHM plus volume
   high-frequency ratio.
2. If `cupyx` improves the result, adjust defaults or add an Astrocyte-visible
   preset for small-field compatibility mode.
3. If `cupyx` does not improve the result, implement the fixed external PSF
   cuCIM isolation test.
4. Based on the fixed-PSF result, address either blind PSF estimation or final
   cuCIM restoration.
5. Add regression documentation and keep the small upstream TIFF as a benchmark
   input.

## 2026-08-06 Resolution

The low PSF metric was caused primarily by seed mismatch. The workflow's
generated light-sheet seed remained very close to its final CuPy estimate
(`NCC=0.9963`), but only reached `NCC=0.8081` against the MATLAB recovered
PSF. The workflow now accepts a calibrated TIFF through `psf_seed_path` and a
recovered fixed TIFF through `fixed_psf_path`. Fixed mode bypasses blind
estimation and publishes the fitted PSF through the existing
`estimated_psf.tif` output contract.

Scout refinement also now continues from the merged scout consensus. It
previously discarded the scout update, restarted from the original theoretical
seed, and ran only the remaining iteration count.

MATLAB `deconvlucy` is accelerated; the measured match for its 10 iterations is
20 cuCIM iterations. The comparison runner exposes this separately as
`WF_DECON_ITERS`, defaulting to twice `ITER`, and can reuse completed MATLAB
outputs through `REFERENCE_RUN_DIR`.

Validated cached-reference command:

```bash
module load nextflow/24.10.0
REFERENCE_RUN_DIR="$PWD/comparison_runs/matlab_reference_Top_Cell19_CH00_000000_20260806_125550"
WF_FIXED_PSF_PATH="$REFERENCE_RUN_DIR/matlab_reference/PSFr_Top_Cell19/Top_Cell19psfr1.tif"
RUN_MATLAB=0 RUN_ID=fixed_matlab_psf_iter20 \
  REFERENCE_RUN_DIR="$REFERENCE_RUN_DIR" \
  WF_FIXED_PSF_PATH="$WF_FIXED_PSF_PATH" \
  workflow/scripts/run_matlab_reference_comparison.sh
```

Measured results:

| Output | NCC | SSIM |
| --- | ---: | ---: |
| PSF | `1.0000000000` | `1.0000000000` |
| Full volume | `0.9805697769` | `0.9806358292` |
