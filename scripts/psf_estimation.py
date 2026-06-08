# psf_estimation.py
# Blind PSF estimation via chunked MATLAB deconvblind + median merge.
#
# Workflow:
#   Load the first deskewed TIFF.
#   Dask splits it into (nz, chunk_xy, chunk_xy) tiles with full Z, 256x256 XY.
#   Each tile is written to a temp TIFF and sent to MATLAB deconvblind.
#   MATLAB writes back an estimated PSF TIFF per tile.
#   Python collects all per-tile PSFs and returns np.median across them.
#
# The returned PSF is float32, normalised to sum=1, and saved as estimated_psf.tif
# next to the input image so pycudadecon can pick it up via TemporaryOTF.

import argparse
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np
import psfmodels as pm
from tifffile import imread, imwrite


# ---------------------------------------------------------------------------
# Theoretical PSF (fallback when --no_blind is passed)
# ---------------------------------------------------------------------------

def generate_theoretical_psf(
    na: float = 1.0,
    wavelength: float = 0.525,      # µm
    ni: float = 1.33,
    dxy: float = 0.1,               # µm, lateral pixel size
    dz: float = 0.3,                # µm, axial step
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    background: float = 0.0,
) -> np.ndarray:
    """
    Generate a 3-D Gibson-Lanni PSF using psfmodels.

    All parameters are optional; defaults are reasonable for a single-objective
    light-sheet with water immersion at 525 nm.

    Returns float32 array of shape (psf_size_z, psf_size_xy, psf_size_xy),
    background-subtracted and normalised to sum = 1.
    """
    psf = pm.make_psf(
        nz=psf_size_z,
        nx=psf_size_xy,
        dz=dz,
        dx=dxy,
        NA=na,
        wvl=wavelength,
        ni=ni,
        model="vectorial",
    ).astype(np.float32)

    psf = np.abs(psf - background)
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


# ---------------------------------------------------------------------------
# Per-chunk blind estimation via MATLAB deconvblind
# ---------------------------------------------------------------------------

def _write_chunk(chunk: np.ndarray, path: Path) -> None:
    imwrite(str(path), chunk.astype(np.float32))


def _run_matlab_deconvblind(
    chunk_path: Path,
    psf_seed: np.ndarray,
    psf_seed_path: Path,
    output_psf_path: Path,
    n_iters: int,
    script_dir: Path,
) -> None:
    """
    Call MATLAB deconvblind on one chunk.  The script writes the recovered PSF
    to output_psf_path as a float32 TIFF.

    MATLAB is invoked with -batch so it exits cleanly on completion or error.
    """
    imwrite(str(psf_seed_path), psf_seed.astype(np.float32))

    matlab_cmd = (
        f"addpath('{script_dir}'); "
        f"chunk = single(readtiffstack('{chunk_path}')); "
        f"psf_seed = single(readtiffstack('{psf_seed_path}')); "
        f"[~, psf_est] = deconvblind(chunk, psf_seed, {n_iters}); "
        f"psf_est = single(psf_est); "
        f"psf_est = psf_est / sum(psf_est(:)); "
        f"writetiffstack(psf_est, '{output_psf_path}');"
    )

    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MATLAB deconvblind failed for chunk {chunk_path.name}.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# Main estimation entry point
# ---------------------------------------------------------------------------

def estimate_psf_from_chunks(
    image_path: str | Path,
    psf_seed: np.ndarray,
    n_iters: int = 10,
    chunk_xy: int = 256,
    pad_xy: int = 32,
    script_dir: str | Path | None = None,
) -> np.ndarray:
    """
    Estimate a PSF by running MATLAB deconvblind on spatial XY chunks of the
    first deskewed TIFF and merging the per-chunk estimates with a median.

    Parameters
    ----------
    image_path  : path to the deskewed input TIFF (full Z stack, 3-D).
    psf_seed    : initial PSF guess, float32 numpy array (nz_psf, ny_psf, nx_psf).
                  Typically the output of generate_theoretical_psf().
    n_iters     : number of deconvblind iterations per chunk.
    chunk_xy    : XY tile size in pixels (256 for ~4 GB volumes).
    pad_xy      : pixels of reflect padding added to each XY edge before
                  deconvblind to suppress edge ringing.  The padding is applied
                  to the image chunk only — deconvblind always returns a PSF
                  the same shape as the seed, so no crop is needed afterwards.
    script_dir  : directory containing readtiffstack.m / writetiffstack.m.
                  Defaults to the directory of this script.

    Returns
    -------
    float32 numpy array of shape matching psf_seed, normalised to sum = 1.
    """
    image_path = Path(image_path)
    script_dir = Path(script_dir) if script_dir else Path(__file__).parent

    print(f"Loading {image_path} for PSF estimation...", flush=True)
    volume = imread(str(image_path))  # (nz, ny, nx)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}")

    nz, ny, nx = volume.shape
    print(f"  Volume shape: {volume.shape}", flush=True)

    # Build list of (y_start, x_start) tile origins, skipping tiles that are
    # too small to produce a reliable PSF estimate.
    min_tile = chunk_xy // 2
    tile_origins = []
    for y0 in range(0, ny, chunk_xy):
        for x0 in range(0, nx, chunk_xy):
            y1 = min(y0 + chunk_xy, ny)
            x1 = min(x0 + chunk_xy, nx)
            if (y1 - y0) >= min_tile and (x1 - x0) >= min_tile:
                tile_origins.append((y0, x0, y1, x1))

    print(f"  Processing {len(tile_origins)} chunk(s) of size "
          f"(nz={nz}, xy≤{chunk_xy}, pad={pad_xy})...", flush=True)

    psf_estimates = []

    with tempfile.TemporaryDirectory(prefix="psf_est_") as tmpdir:
        tmpdir = Path(tmpdir)

        for idx, (y0, x0, y1, x1) in enumerate(tile_origins):
            chunk = volume[:, y0:y1, x0:x1]

            # Reflect-pad in XY to suppress edge ringing during deconvblind.
            # Z is full-depth so no padding needed there.
            if pad_xy > 0:
                chunk = np.pad(
                    chunk,
                    pad_width=((0, 0), (pad_xy, pad_xy), (pad_xy, pad_xy)),
                    mode="reflect",
                )

            chunk_path      = tmpdir / f"chunk_{idx:04d}.tif"
            seed_path       = tmpdir / f"seed_{idx:04d}.tif"
            psf_out_path    = tmpdir / f"psf_out_{idx:04d}.tif"

            _write_chunk(chunk, chunk_path)

            try:
                _run_matlab_deconvblind(
                    chunk_path, psf_seed, seed_path, psf_out_path,
                    n_iters, script_dir,
                )
            except RuntimeError as exc:
                print(f"  WARNING: chunk {idx} failed, skipping. {exc}", flush=True)
                continue

            if psf_out_path.exists():
                psf_chunk = imread(str(psf_out_path)).astype(np.float32)
                # Resize to match seed shape if MATLAB padded differently
                if psf_chunk.shape != psf_seed.shape:
                    print(
                        f"  WARNING: chunk {idx} PSF shape {psf_chunk.shape} "
                        f"!= seed shape {psf_seed.shape}, skipping.", flush=True
                    )
                    continue
                psf_estimates.append(psf_chunk)
                print(f"  Chunk {idx + 1}/{len(tile_origins)} done.", flush=True)
            else:
                print(f"  WARNING: no PSF output for chunk {idx}, skipping.", flush=True)

    if not psf_estimates:
        raise RuntimeError(
            "All chunks failed during PSF estimation. "
            "Check MATLAB logs above and ensure deconvblind is available."
        )

    print(f"Merging {len(psf_estimates)} PSF estimate(s) via median...", flush=True)
    stack = np.stack(psf_estimates, axis=0)        # (n_chunks, nz, ny, nx)
    merged = np.median(stack, axis=0).astype(np.float32)

    total = merged.sum()
    if total > 0:
        merged /= total

    return merged


# ---------------------------------------------------------------------------
# CLI (for standalone testing)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate PSF from a deskewed TIFF using chunked deconvblind."
    )
    parser.add_argument("--image_path",  required=True)
    parser.add_argument("--output_path", required=True,
                        help="Where to save the merged PSF TIFF.")
    parser.add_argument("--n_iters",    type=int,   default=10)
    parser.add_argument("--chunk_xy",   type=int,   default=256)
    parser.add_argument("--pad_xy",     type=int,   default=32,
                        help="XY reflect-padding per edge before deconvblind (pixels).")
    parser.add_argument("--script_dir", default=str(Path(__file__).parent))

    # Optional optical parameters for the PSF seed
    parser.add_argument("--na",         type=float, default=1.0)
    parser.add_argument("--wavelength", type=float, default=0.525)
    parser.add_argument("--ni",         type=float, default=1.33)
    parser.add_argument("--dxy",        type=float, default=0.1)
    parser.add_argument("--dz",         type=float, default=0.3)
    parser.add_argument("--psf_size_z", type=int,   default=61)
    parser.add_argument("--psf_size_xy",type=int,   default=128)
    parser.add_argument("--background", type=float, default=0.0)
    args = parser.parse_args()

    psf_seed = generate_theoretical_psf(
        na=args.na,
        wavelength=args.wavelength,
        ni=args.ni,
        dxy=args.dxy,
        dz=args.dz,
        psf_size_z=args.psf_size_z,
        psf_size_xy=args.psf_size_xy,
        background=args.background,
    )

    merged_psf = estimate_psf_from_chunks(
        image_path=args.image_path,
        psf_seed=psf_seed,
        n_iters=args.n_iters,
        chunk_xy=args.chunk_xy,
        pad_xy=args.pad_xy,
        script_dir=args.script_dir,
    )

    imwrite(args.output_path, merged_psf)
    print(f"Merged PSF saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
