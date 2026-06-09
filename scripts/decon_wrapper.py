# decon_wrapper.py
# Dask-orchestrated GPU deconvolution with optional blind PSF estimation.
#
# PSF resolution order:
#   1. Blind (default): estimate PSF from first TIFF via chunked MATLAB deconvblind,
#      merge per-chunk estimates with SNR weighting, save as estimated_psf.tif.
#   2. No-blind (--no_blind): generate a theoretical Gibson-Lanni PSF from
#      optical parameters.  All optical params are optional with defaults.
#
# Deconvolution:
#   pycudadecon (TemporaryOTF + RLContext) processes each TIFF as full-Z
#   XY chunks using map_overlap.  The requested chunk_xy is treated as the
#   core tile size; <=0 auto-sizes from available VRAM.

import argparse
import time
from glob import glob
from pathlib import Path

import dask.array as da
import numpy as np
from pycudadecon import TemporaryOTF, RLContext, rl_decon
from tifffile import imwrite

from psf_estimation import (
    estimate_psf_from_chunks,
    generate_theoretical_psf,
    open_tiff_memmap,
    resolve_chunk_xy,
)


# ---------------------------------------------------------------------------
# Dask worker
# ---------------------------------------------------------------------------

def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _chunk_progress(block_info: dict | None, total_chunks: int) -> tuple[int, str]:
    if not isinstance(block_info, dict) or None not in block_info:
        return 0, "unknown"

    info = block_info[None]
    location = info.get("chunk-location")
    num_chunks = info.get("num-chunks")
    if not location or not num_chunks:
        return 0, "unknown"

    chunk_index = 1
    stride = 1
    for loc, count in zip(reversed(location), reversed(num_chunks)):
        chunk_index += loc * stride
        stride *= count

    return chunk_index, f"{chunk_index}/{total_chunks}"


def _decon_chunk(
    chunk: np.ndarray,
    otf_path: str,
    dz: float,
    n_iters: int,
    total_chunks: int,
    block_info: dict | None = None,
) -> np.ndarray:
    """
    Process one spatial chunk with pycudadecon.
    Each call opens and closes its own RLContext so chunks can be dispatched
    sequentially without GPU context leakage.
    """
    _, chunk_label = _chunk_progress(block_info, total_chunks)
    if chunk.size == 0:
        return chunk

    print(
        f"  Chunk {chunk_label} started: shape={chunk.shape}, iterations={n_iters}",
        flush=True,
    )

    start = time.perf_counter()
    with RLContext(chunk.shape, otf_path, dzdata=dz) as ctx:
        result = rl_decon(chunk, output_shape=ctx.out_shape, n_iters=n_iters)
    elapsed = time.perf_counter() - start
    avg_iter = elapsed / n_iters if n_iters > 0 else elapsed

    print(
        f"  Iteration {n_iters}/{n_iters} of chunk {chunk_label} completed: "
        f"chunk_time={_format_seconds(elapsed)}, "
        f"avg_iteration_time={_format_seconds(avg_iter)}",
        flush=True,
    )
    return np.clip(result, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# Per-TIFF deconvolution
# ---------------------------------------------------------------------------

def _psf_overlap_xy(psf: np.ndarray) -> int:
    """Use the PSF support as the minimum XY halo needed at chunk boundaries."""
    psf_xy = max(psf.shape[-2:])
    return max(16, int(np.ceil(psf_xy / 2)))


def deconvolve_tiff(
    image_path: Path,
    psf: np.ndarray,
    n_iters: int,
    dz: float,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
) -> np.ndarray:
    """
    Deconvolve a single TIFF using the supplied PSF.

    Chunks are full-Z XY tiles with PSF-dependent XY overlap so tile boundaries
    are invisible in the merged output.  Z is never split.  `chunk_xy` is the
    core tile size; <=0 chooses a VRAM-aware size.
    """
    volume = open_tiff_memmap(image_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")

    original_shape = volume.shape
    overlap_xy = _psf_overlap_xy(psf)
    overlap_xy = min(overlap_xy, max(1, (min(volume.shape[1:]) - 1) // 2))
    decon_workers = max(1, decon_workers)
    core_chunk_xy = resolve_chunk_xy(
        chunk_xy,
        volume.shape,
        volume.dtype,
        overlap_xy=overlap_xy,
        vram_gb=vram_gb,
        workers=decon_workers,
        min_xy=max(128, overlap_xy * 2),
    )
    if core_chunk_xy <= 0:
        raise ValueError(f"Resolved decon chunk size must be positive, got {core_chunk_xy}")

    nz, ny, nx = volume.shape
    lazy = da.from_array(
        volume,
        chunks=(nz, core_chunk_xy, core_chunk_xy),
        asarray=False,
        lock=False,
    )
    total_chunks = int(np.prod(lazy.numblocks))

    print(f"  Deconvolving {image_path.name}  shape={original_shape}", flush=True)
    print(
        f"  Deconvolution chunks: total={total_chunks}, "
        f"core_chunk_shape=(z={nz}, y={core_chunk_xy}, x={core_chunk_xy}), "
        f"psf_overlap_xy={overlap_xy}, image_xy=({ny}, {nx}), "
        f"iterations_per_chunk={n_iters}, workers={decon_workers}",
        flush=True,
    )

    with TemporaryOTF(psf) as otf:
        processed = lazy.map_overlap(
            _decon_chunk,
            depth={0: 0, 1: overlap_xy, 2: overlap_xy},
            boundary="reflect",
            dtype=np.uint16,
            otf_path=otf.path,
            dz=dz,
            n_iters=n_iters,
            total_chunks=total_chunks,
        )
        scheduler = "threads" if decon_workers > 1 else "single-threaded"
        output = processed.compute(scheduler=scheduler, num_workers=decon_workers)

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
    parser.add_argument("--chunk_xy",    type=int, default=0,
                        help="XY tile size for blind PSF estimation. <=0 auto-sizes from VRAM.")
    parser.add_argument("--decon_chunk_xy", type=int, default=0,
                        help="Core XY tile size for CUDA deconvolution. <=0 auto-sizes from VRAM.")
    parser.add_argument("--pad_xy",      type=int, default=32,
                        help="XY halo per edge added to each blind PSF chunk (pixels).")
    parser.add_argument("--blind_workers", type=int, default=0,
                        help="Concurrent MATLAB deconvblind chunks. <=0 uses a bounded auto value.")
    parser.add_argument("--prefetch_chunks", type=int, default=0,
                        help="Number of PSF tiles to keep submitted/read ahead. <=0 uses 2x workers.")
    parser.add_argument("--decon_workers", type=int, default=1,
                        help="Dask workers for CUDA deconvolution chunks.")
    parser.add_argument("--vram_gb", type=float, default=None,
                        help="Override detected free VRAM in GiB for auto chunk sizing.")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory for cached blind PSF estimates.")
    parser.add_argument("--no_psf_cache", action="store_true",
                        help="Disable reuse of cached blind PSF estimates.")

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
    parser.add_argument("--psf_size_xy", type=int,   default=129,
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
            max_workers=args.blind_workers,
            prefetch_chunks=args.prefetch_chunks,
            vram_gb=args.vram_gb,
            cache_dir=args.cache_dir,
            use_cache=not args.no_psf_cache,
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
            chunk_xy=args.decon_chunk_xy,
            vram_gb=args.vram_gb,
            decon_workers=args.decon_workers,
        )

        stem = tiff_path.name.replace(".tiff", "").replace(".tif", "")
        out_name = f"DB2_{stem}.tif" if "CH" in stem else "DB2_deconvolved_output.tif"
        imwrite(out_name, output)
        print(f"  Saved {out_name}", flush=True)

    print("All TIFFs deconvolved.", flush=True)


if __name__ == "__main__":
    main()
