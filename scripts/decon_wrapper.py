# decon_wrapper.py
# Dask-orchestrated GPU deconvolution with optional blind PSF estimation.
#
# PSF resolution order:
#   1. Blind (default): estimate PSF from first TIFF via chunked MATLAB deconvblind,
#      merge per-chunk estimates with median, save as estimated_psf.tif.
#   2. No-blind (--no_blind): generate a theoretical Gibson-Lanni PSF from
#      optical parameters.  All optical params are optional with defaults.
#
# Deconvolution:
#   pycudadecon (TemporaryOTF + RLContext) processes each TIFF as full-Z,
#   256x256 XY Dask chunks using map_overlap.  Chunks are computed
#   single-threaded to keep one clean GPU context per job.

import argparse
from glob import glob
from pathlib import Path

import dask.array as da
import numpy as np
from pycudadecon import TemporaryOTF, RLContext, rl_decon
from tifffile import imread, imwrite

from psf_estimation import estimate_psf_from_chunks, generate_theoretical_psf


# ---------------------------------------------------------------------------
# Dask worker
# ---------------------------------------------------------------------------

def _decon_chunk(chunk: np.ndarray, otf_path: str, dz: float, n_iters: int) -> np.ndarray:
    """
    Process one spatial chunk with pycudadecon.
    Each call opens and closes its own RLContext so chunks can be dispatched
    sequentially without GPU context leakage.
    """
    with RLContext(chunk.shape, otf_path, dzdata=dz) as ctx:
        result = rl_decon(chunk, output_shape=ctx.out_shape, n_iters=n_iters)
    return np.clip(result, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# Per-TIFF deconvolution
# ---------------------------------------------------------------------------

def deconvolve_tiff(
    image_path: Path,
    psf: np.ndarray,
    n_iters: int,
    dz: float,
    chunk_xy: int = 256,
) -> np.ndarray:
    """
    Deconvolve a single TIFF using the supplied PSF.

    Chunks are (nz, chunk_xy, chunk_xy) with 32-px XY overlap so tile
    boundaries are invisible in the merged output.  Z is never split —
    pycudadecon needs full Z depth to build its FFT plan.
    """
    volume = imread(str(image_path))
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")

    nz = volume.shape[0]
    lazy = da.from_array(volume, chunks=(nz, chunk_xy, chunk_xy))

    print(f"  Deconvolving {image_path.name}  shape={volume.shape}", flush=True)

    with TemporaryOTF(psf) as otf:
        processed = lazy.map_overlap(
            _decon_chunk,
            depth={0: 0, 1: 32, 2: 32},
            boundary="reflect",
            dtype=np.uint16,
            otf_path=otf.path,
            dz=dz,
            n_iters=n_iters,
        )
        output = processed.compute(scheduler="single-threaded")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dask-orchestrated GPU deconvolution with blind PSF estimation."
    )

    # Required options
    parser.add_argument("--image_path", required=True,
                        help="Directory containing deskewed CH*_registered_consistent.tif files.")

    # PSF generation mode
    parser.add_argument("--no_blind", action="store_true",
                        help="Skip blind PSF estimation; use a theoretical PSF instead.")

    # Blind estimation options (only used when --no_blind is NOT set)
    parser.add_argument("--blind_iters", type=int, default=10,
                        help="deconvblind iterations per chunk during PSF estimation.")
    parser.add_argument("--chunk_xy",    type=int, default=256,
                        help="XY tile size (pixels) for blind PSF estimation.")
    parser.add_argument("--pad_xy",      type=int, default=32,
                        help="XY reflect-padding per edge added to each chunk before deconvblind (pixels).")

    # Deconvolution options
    parser.add_argument("--iter",       type=int,   default=10,
                        help="RL deconvolution iterations.")
    parser.add_argument("--background", type=float, default=0.0,
                        help="Background value to subtract before decon.")

    # Optional optical parameters (used for PSF seed / theoretical PSF)
    parser.add_argument("--na",          type=float, default=1.0,
                        help="Numerical aperture.")
    parser.add_argument("--wavelength",  type=float, default=0.525,
                        help="Emission wavelength in µm.")
    parser.add_argument("--ni",          type=float, default=1.33,
                        help="Refractive index of immersion medium.")
    parser.add_argument("--dxy",         type=float, default=0.1,
                        help="Lateral pixel size in µm.")
    parser.add_argument("--dz",          type=float, default=0.3,
                        help="Axial step size in µm.")
    parser.add_argument("--psf_size_z",  type=int,   default=61,
                        help="Z size of PSF volume.")
    parser.add_argument("--psf_size_xy", type=int,   default=128,
                        help="XY size of PSF volume.")

    # Misc, usually unneeded
    parser.add_argument("--script_dir",  default=str(Path(__file__).parent),
                        help="Directory containing readtiffstack.m / writetiffstack.m.")

    args = parser.parse_args()

    image_dir = Path(args.image_path)

    # Collect all deskewed TIFFs, sorted so index 0 is deterministic
    tiff_list = sorted(
        glob(str(image_dir / "CH*_registered_consistent.tif")) +
        glob(str(image_dir / "CH*_registered_consistent.tiff"))
    )
    if not tiff_list:
        print(f"Error: no CH*_registered_consistent.tif files found in {image_dir}")
        raise SystemExit(1)

    print(f"Found {len(tiff_list)} TIFF(s) to process.", flush=True)

    # ------------------------------------------------------------------
    # PSF resolution
    # ------------------------------------------------------------------

    # Build PSF seed from optical params regardless of mode — used as either
    # the blind-estimation seed or the final theoretical PSF.
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

    if args.no_blind:
        print("Using theoretical PSF (--no_blind).", flush=True)
        psf = psf_seed
    else:
        print("Running blind PSF estimation on first TIFF...", flush=True)
        psf = estimate_psf_from_chunks(
            image_path=tiff_list[0],
            psf_seed=psf_seed,
            n_iters=args.blind_iters,
            chunk_xy=args.chunk_xy,
            pad_xy=args.pad_xy,
            script_dir=args.script_dir,
        )
        psf_save_path = image_dir / "estimated_psf.tif"
        imwrite(str(psf_save_path), psf)
        print(f"Merged PSF saved to {psf_save_path}", flush=True)

    # ------------------------------------------------------------------
    # Deconvolve all TIFFs with the resolved PSF
    # ------------------------------------------------------------------

    for tiff_path in tiff_list:
        tiff_path = Path(tiff_path)
        output = deconvolve_tiff(
            image_path=tiff_path,
            psf=psf,
            n_iters=args.iter,
            dz=args.dz,
            chunk_xy=args.chunk_xy,
        )

        stem = tiff_path.name.replace(".tiff", "").replace(".tif", "")
        out_name = f"DB2_{stem}.tif" if "CH" in stem else "DB2_deconvolved_output.tif"
        imwrite(out_name, output)
        print(f"  Saved {out_name}", flush=True)

    print("All TIFFs deconvolved.", flush=True)


if __name__ == "__main__":
    main()
