# decon_wrapper.py
# Dask-orchestrated GPU deconvolution with blind PSF estimation.
#
# PSF resolution:
#   Generate a theoretical Gibson-Lanni PSF from the optical parameters, use it
#   only as the starting guess for chunked MATLAB deconvblind, then merge the
#   recovered per-chunk blind PSFs with SNR weighting and save estimated_psf.tif.
#
# Deconvolution:
#   cuCIM Richardson-Lucy processes each volume as full-Z
#   XY chunks using map_overlap.  The requested chunk_xy is treated as the
#   core tile size; <=0 auto-sizes from available VRAM.

from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import dask.array as da
import numpy as np
from tifffile import imwrite

from blind_rl import deconvolve_with_cucim

from psf_estimation import (
    DEFAULT_BLIND_CHUNK_XY,
    DEFAULT_BLIND_LATENT_UPDATE_PERIOD,
    DEFAULT_BLIND_MAX_TILES,
    DEFAULT_BLIND_Z_SLICES,
    DEFAULT_SNR_WEIGHT_CAP,
    estimate_psf_from_chunks,
    detect_vram_bytes,
    normalize_blind_backend,
    open_tiff_memmap,
    resolve_dxy,
    resolve_chunk_xy,
)
from psf_modes import generate_psf_seed

try:
    from ome_zarr_io import (
        discover_image_volumes,
        image_stem,
        is_ome_zarr_path,
        is_ozx_path,
        log_progress,
        create_ome_zarr_array,
        open_ome_zarr_array,
        unzip_ozx_to_ome_zarr,
        write_downsampled_pyramid,
        write_ome_zarr_array,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ome_zarr_io import (
        discover_image_volumes,
        image_stem,
        is_ome_zarr_path,
        is_ozx_path,
        log_progress,
        create_ome_zarr_array,
        open_ome_zarr_array,
        unzip_ozx_to_ome_zarr,
        write_downsampled_pyramid,
        write_ome_zarr_array,
    )


# ---------------------------------------------------------------------------
# Dask worker
# ---------------------------------------------------------------------------

CHANNEL_TIMEPOINT_RE = re.compile(r"^CH(?P<channel>\d+)_(?P<timepoint>\d+)(?:_registered_consistent)?$")


def _tiff_stem(filename: str) -> str:
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return path.name[: -len(path.suffix)]
    return path.name


def _load_original_name_map(image_dir: Path) -> dict[str, str]:
    map_path = image_dir / "original_filenames.tsv"
    if not map_path.exists():
        return {}

    original_name_map: dict[str, str] = {}
    with map_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(
                    f"  WARNING: ignoring malformed original filename map line "
                    f"{line_number} in {map_path}",
                    flush=True,
                )
                continue
            current_name, original_name = parts
            original_name_map[current_name] = original_name
            current_stem = _tiff_stem(current_name)
            original_stem = _tiff_stem(original_name)
            original_name_map[f"{current_stem}.tif"] = f"{original_stem}.tif"
            original_name_map[f"{current_stem}.tiff"] = f"{original_stem}.tiff"
    return original_name_map


def _decon_output_name(tiff_path: Path, original_name_map: dict[str, str]) -> str:
    original_name = original_name_map.get(tiff_path.name, tiff_path.name)
    return f"DB2_{_tiff_stem(original_name)}.tif"


def _decon_ome_zarr_output_name(image_path: Path, original_name_map: dict[str, str]) -> str:
    original_name = original_name_map.get(image_path.name, image_path.name)
    return f"DB2_{image_stem(original_name)}.ome.zarr"


def _write_materialized_decon_output(
    restored: np.ndarray,
    image_path: Path,
    original_name_map: dict[str, str],
    output_format: str,
    max_downsample: int,
) -> Path:
    """Write a TIFF-input result in the requested final representation."""
    if output_format == "tiff":
        output_path = Path(_decon_output_name(image_path, original_name_map))
        log_progress(f"Writing deconvolved TIFF output: {output_path}")
        imwrite(output_path, restored)
        return output_path
    if output_format == "ozx":
        output_path = Path(
            _decon_ome_zarr_output_name(image_path, original_name_map)
        )
        log_progress(
            f"Writing deconvolved OME-Zarr for OZX output: {output_path}"
        )
        write_ome_zarr_array(
            output_path,
            restored,
            layer_name=image_stem(output_path),
            max_downsample=max_downsample,
        )
        return output_path
    raise ValueError(f"Unsupported output format: {output_format}")


def _write_tiff_near_input_or_cwd(path: Path, data: np.ndarray) -> Path:
    try:
        imwrite(str(path), data)
        return path
    except OSError as exc:
        fallback = Path.cwd() / path.name
        print(
            f"  WARNING: cannot write {path}: {exc}; writing {fallback} instead",
            flush=True,
        )
        imwrite(str(fallback), data)
        return fallback


def _detected_channels(tiff_list: list[str]) -> list[int]:
    detected = set()
    for tiff_path in tiff_list:
        match = CHANNEL_TIMEPOINT_RE.match(Path(tiff_path).stem)
        if match:
            detected.add(int(match.group("channel")))
    return sorted(detected)

def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _require_positive(name: str, value: float | None) -> float:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be provided and > 0")
    return value


def _resolve_required_dxy(
    dxy: float | None,
    camera_pixel_size: float | None,
    magnification: float | None,
) -> float:
    if dxy is not None:
        return _require_positive("dxy", dxy)
    if camera_pixel_size is not None or magnification is not None:
        _require_positive("camera_pixel_size", camera_pixel_size)
        _require_positive("magnification", magnification)
        return resolve_dxy(0, camera_pixel_size, magnification)
    raise ValueError(
        "dxy must be provided and > 0, or camera_pixel_size and magnification must both be provided"
    )


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


def _center_crop_or_pad_to_shape(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return an array with exactly `shape`, preserving the centered content."""
    target_shape = tuple(int(axis) for axis in shape)
    current_shape = tuple(int(axis) for axis in array.shape)
    if len(current_shape) != len(target_shape):
        raise ValueError(f"Cannot reshape {current_shape} to {target_shape}")
    if current_shape == target_shape:
        return array

    slices = []
    pad_width = []
    for current_axis, target_axis in zip(current_shape, target_shape):
        if current_axis > target_axis:
            start = (current_axis - target_axis) // 2
            slices.append(slice(start, start + target_axis))
            pad_width.append((0, 0))
        else:
            slices.append(slice(None))
            missing = target_axis - current_axis
            before = missing // 2
            pad_width.append((before, missing - before))

    adjusted = array[tuple(slices)]
    if any(before or after for before, after in pad_width):
        adjusted = np.pad(adjusted, pad_width, mode="edge")
    return adjusted


def _decon_chunk(
    chunk: np.ndarray,
    psf: np.ndarray,
    n_iters: int,
    total_chunks: int,
    block_info: dict | None = None,
) -> np.ndarray:
    """
    Process one spatial chunk with cuCIM Richardson-Lucy.
    """
    _, chunk_label = _chunk_progress(block_info, total_chunks)
    if chunk.size == 0:
        return chunk

    print(
        f"  Chunk {chunk_label} started: shape={chunk.shape}, iterations={n_iters}",
        flush=True,
    )

    start = time.perf_counter()
    result = deconvolve_with_cucim(chunk, psf, n_iters)
    elapsed = time.perf_counter() - start
    avg_iter = elapsed / n_iters if n_iters > 0 else elapsed

    print(
        f"  Iteration {n_iters}/{n_iters} of chunk {chunk_label} completed: "
        f"chunk_time={_format_seconds(elapsed)}, "
        f"avg_iteration_time={_format_seconds(avg_iter)}",
        flush=True,
    )
    output = np.clip(result, 0, 65535).astype(np.uint16)
    return _center_crop_or_pad_to_shape(output, chunk.shape)


def _match_input_intensity_range(output: np.ndarray, input_volume: np.ndarray) -> np.ndarray:
    """Map deconvolved output to the original TIFF intensity range."""
    in_min = float(np.min(input_volume))
    in_max = float(np.max(input_volume))
    out_min = float(np.min(output))
    out_max = float(np.max(output))
    dtype_max = float(np.iinfo(np.uint16).max)

    if not np.isfinite([in_min, in_max, out_min, out_max]).all():
        raise ValueError("Cannot rescale deconvolution output with non-finite intensity bounds")

    if out_max > out_min and in_max > in_min:
        scaled = output.astype(np.float32, copy=False)
        scaled = (scaled - out_min) / (out_max - out_min)
        scaled = scaled * (in_max - in_min) + in_min
    else:
        scaled = output

    return np.clip(np.rint(scaled), 0, dtype_max).astype(np.uint16)


def _match_block_intensity_range(
    block: np.ndarray,
    input_min: float,
    input_max: float,
    output_min: float,
    output_max: float,
) -> np.ndarray:
    dtype_max = float(np.iinfo(np.uint16).max)
    if output_max > output_min and input_max > input_min:
        scaled = block.astype(np.float32, copy=False)
        scaled = (scaled - output_min) / (output_max - output_min)
        scaled = scaled * (input_max - input_min) + input_min
    else:
        scaled = block
    return np.clip(np.rint(scaled), 0, dtype_max).astype(np.uint16)


# ---------------------------------------------------------------------------
# Per-TIFF deconvolution
# ---------------------------------------------------------------------------

def _psf_overlap_xy(psf: np.ndarray) -> int:
    """Use a moderate PSF-support halo at chunk boundaries."""
    psf_xy = max(psf.shape[-2:])
    return min(48, max(16, int(np.ceil(psf_xy / 4))))


def _auto_decon_max_xy(
    volume_shape: tuple[int, int, int],
    dtype: np.dtype,
    workers: int,
    vram_gb: float | None,
    fallback_xy: int = 512,
) -> int:
    nz, ny, nx = volume_shape
    image_max_xy = min(ny, nx)
    if nz > 4:
        return min(1024, image_max_xy)

    vram_bytes = int(vram_gb * (1024 ** 3)) if vram_gb and vram_gb > 0 else detect_vram_bytes()
    if not vram_bytes:
        return min(fallback_xy, image_max_xy)

    workers = max(1, workers)
    bytes_per_voxel = np.dtype(dtype).itemsize
    target_bytes = vram_bytes * 0.05 / workers
    memory_multiplier = 2048.0
    denom = max(1, nz) * bytes_per_voxel * memory_multiplier
    max_xy = int(math.sqrt(max(1.0, target_bytes / denom)))
    max_xy = max(128, (max_xy // 32) * 32)
    return min(max_xy, 1024, image_max_xy)


def _build_deconvolution_graph(
    volume,
    image_name: str,
    psf: np.ndarray,
    n_iters: int,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
) -> tuple[da.Array, int]:
    """
    Build a lazy deconvolution graph for a single 3-D volume.

    Chunks are full-Z XY tiles with PSF-dependent XY overlap so tile boundaries
    are invisible in the merged output.  Z is never split.  `chunk_xy` is the
    core tile size; <=0 chooses a VRAM-aware size.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")

    original_shape = volume.shape
    overlap_xy = overlap_xy if overlap_xy > 0 else _psf_overlap_xy(psf)
    overlap_xy = min(overlap_xy, max(1, (min(volume.shape[1:]) - 1) // 2))
    if decon_workers != 1:
        log_progress(
            f"  cuCIM uses one GPU worker per allocation; "
            f"clamping decon_workers={decon_workers} to 1"
        )
    decon_workers = 1
    core_chunk_xy = resolve_chunk_xy(
        chunk_xy,
        volume.shape,
        volume.dtype,
        overlap_xy=overlap_xy,
        vram_gb=vram_gb,
        workers=decon_workers,
        min_xy=max(128, overlap_xy * 2),
        max_xy=_auto_decon_max_xy(volume.shape, volume.dtype, decon_workers, vram_gb),
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

    log_progress(f"Deconvolving {image_name}: shape={original_shape}, dtype={volume.dtype}")
    log_progress(
        f"  Deconvolution chunks: total={total_chunks}, "
        f"core_chunk_shape=(z={nz}, y={core_chunk_xy}, x={core_chunk_xy}), "
        f"psf_overlap_xy={overlap_xy}, image_xy=({ny}, {nx}), "
        f"iterations_per_chunk={n_iters}, workers={decon_workers}"
    )

    return lazy.map_overlap(
        _decon_chunk,
        depth={0: 0, 1: overlap_xy, 2: overlap_xy},
        boundary="reflect",
        dtype=np.uint16,
        psf=psf,
        n_iters=n_iters,
        total_chunks=total_chunks,
    ), decon_workers


def _deconvolution_scheduler(decon_workers: int) -> str:
    return "threads" if decon_workers > 1 else "single-threaded"


def deconvolve_volume(
    volume,
    image_name: str,
    psf: np.ndarray,
    n_iters: int,
    dz: float,
    dxy: float,
    wavelength: float,
    na: float,
    ni: float,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
) -> np.ndarray:
    del dz, dxy, wavelength, na, ni
    processed, decon_workers = _build_deconvolution_graph(
        volume,
        image_name,
        psf,
        n_iters,
        chunk_xy=chunk_xy,
        vram_gb=vram_gb,
        decon_workers=decon_workers,
        overlap_xy=overlap_xy,
    )
    scheduler = _deconvolution_scheduler(decon_workers)
    log_progress(
        f"Computing cuCIM deconvolution graph for {image_name}: "
        f"scheduler={scheduler}, workers={decon_workers}"
    )
    output = processed.compute(scheduler=scheduler, num_workers=decon_workers)

    output = _match_input_intensity_range(output, volume)
    log_progress(
        f"  Matched deconvolution intensity range to input: "
        f"min={int(output.min())}, max={int(output.max())}"
    )

    return output


def deconvolve_tiff(
    image_path: Path,
    psf: np.ndarray,
    n_iters: int,
    dz: float,
    dxy: float,
    wavelength: float,
    na: float,
    ni: float,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
) -> np.ndarray:
    log_progress(f"Opening TIFF for deconvolution: {image_path}")
    volume = open_tiff_memmap(image_path)
    return deconvolve_volume(
        volume,
        image_path.name,
        psf,
        n_iters,
        dz,
        dxy,
        wavelength,
        na,
        ni,
        chunk_xy=chunk_xy,
        vram_gb=vram_gb,
        decon_workers=decon_workers,
        overlap_xy=overlap_xy,
    )


def deconvolve_ome_zarr(
    image_path: Path,
    psf: np.ndarray,
    n_iters: int,
    dz: float,
    dxy: float,
    wavelength: float,
    na: float,
    ni: float,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
) -> np.ndarray:
    log_progress(f"Opening OME-Zarr for deconvolution: {image_path}")
    volume = open_ome_zarr_array(image_path, mode="r")
    return deconvolve_volume(
        volume,
        image_path.name,
        psf,
        n_iters,
        dz,
        dxy,
        wavelength,
        na,
        ni,
        chunk_xy=chunk_xy,
        vram_gb=vram_gb,
        decon_workers=decon_workers,
        overlap_xy=overlap_xy,
    )


def _zarr_chunks_from_dask(array: da.Array) -> tuple[int, int, int]:
    return tuple(int(axis_chunks[0]) for axis_chunks in array.chunks)


def _default_output_chunks(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (min(16, shape[0]), min(256, shape[1]), min(256, shape[2]))


def _expand_ozx_inputs_for_processing(image_inputs: list[Path], temp_dir: Path) -> list[Path]:
    expanded = []
    for image_input in image_inputs:
        if is_ozx_path(image_input):
            target = temp_dir / f"{image_stem(image_input)}.ome.zarr"
            expanded.append(unzip_ozx_to_ome_zarr(image_input, target))
        else:
            expanded.append(image_input)
    return expanded


def deconvolve_ome_zarr_to_zarr(
    image_path: Path,
    output_path: Path | str,
    psf: np.ndarray,
    n_iters: int,
    dz: float,
    dxy: float,
    wavelength: float,
    na: float,
    ni: float,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
    max_downsample: int = 16,
) -> Path:
    log_progress(f"Opening OME-Zarr for streaming deconvolution: {image_path}")
    volume = open_ome_zarr_array(image_path, mode="r")
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D OME-Zarr volume, got shape {volume.shape}")

    output_path = Path(output_path)
    del dz, dxy, wavelength, na, ni
    processed, decon_workers = _build_deconvolution_graph(
        volume,
        image_path.name,
        psf,
        n_iters,
        chunk_xy=chunk_xy,
        vram_gb=vram_gb,
        decon_workers=decon_workers,
        overlap_xy=overlap_xy,
    )
    scheduler = _deconvolution_scheduler(decon_workers)
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "Missing required dependency 'zarr' for streaming OME-Zarr deconvolution"
        ) from exc

    with tempfile.TemporaryDirectory(prefix=".decon_raw_", dir=Path.cwd()) as temp_dir:
        raw_path = Path(temp_dir) / "raw.zarr"
        raw_chunks = _zarr_chunks_from_dask(processed)
        log_progress(
            "Streaming raw cuCIM deconvolution chunks to temporary Zarr: "
            f"path={raw_path}, shape={processed.shape}, chunks={raw_chunks}"
        )
        raw_array = zarr.open(
            str(raw_path),
            mode="w",
            shape=tuple(int(axis) for axis in processed.shape),
            chunks=raw_chunks,
            dtype=np.uint16,
            compressor=None,
        )
        da.store(processed, raw_array, lock=False, compute=False).compute(
            scheduler=scheduler,
            num_workers=decon_workers,
        )

        raw_lazy = da.from_array(
            raw_array, chunks=processed.chunks, asarray=False, lock=False
        )
        input_lazy = da.from_array(
            volume, chunks=processed.chunks, asarray=False, lock=False
        )
        input_min, input_max, output_min, output_max = da.compute(
            input_lazy.min(),
            input_lazy.max(),
            raw_lazy.min(),
            raw_lazy.max(),
            scheduler="threads",
            num_workers=max(1, decon_workers),
        )
        input_min = float(input_min)
        input_max = float(input_max)
        output_min = float(output_min)
        output_max = float(output_max)
        log_progress(
            "  Streaming intensity match: "
            f"input=({input_min:.6g}, {input_max:.6g}), "
            f"raw_output=({output_min:.6g}, {output_max:.6g})"
        )

        final_array = create_ome_zarr_array(
            output_path,
            shape=tuple(int(axis) for axis in volume.shape),
            dtype=np.uint16,
            chunks=_default_output_chunks(tuple(int(axis) for axis in volume.shape)),
            layer_name=image_stem(output_path),
            max_downsample=int(max_downsample),
        )
        scaled = raw_lazy.map_blocks(
            _match_block_intensity_range,
            dtype=np.uint16,
            input_min=input_min,
            input_max=input_max,
            output_min=output_min,
            output_max=output_max,
        )
        log_progress(f"Streaming scaled deconvolution output to OME-Zarr: {output_path}")
        da.store(scaled, final_array, lock=False, compute=False).compute(
            scheduler="threads",
            num_workers=max(1, decon_workers),
        )

    write_downsampled_pyramid(output_path, max_downsample=int(max_downsample))
    log_progress(f"Finished streaming OME-Zarr deconvolution output: {output_path.resolve()}")
    return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Dask-orchestrated GPU deconvolution with blind PSF estimation."
    )

    # Required options
    parser.add_argument("--image_path", required=True,
                        help="Directory containing normalized OME-Zarr or TIFF image volumes.")
    parser.add_argument("--output_format", choices=("ozx", "tiff"), default="ozx",
                        help="Final output representation requested by the workflow.")

    # Blind estimation options
    parser.add_argument("--blind_iters", type=int, default=10,
                        help="deconvblind iterations per chunk during PSF estimation.")
    parser.add_argument("--blind_backend", default="cupy", choices=("matlab", "cupy", "scout", "cupyx"),
                        help="Backend for blind PSF estimation: 'cupy' or 'matlab'. Legacy scout/cupyx values select cupy with that CuPy mode.")
    parser.add_argument("--chunk_xy",    type=int, default=DEFAULT_BLIND_CHUNK_XY,
                        help="XY tile size for blind PSF estimation. <=0 auto-sizes from VRAM.")
    parser.add_argument("--blind_max_tiles", type=int, default=DEFAULT_BLIND_MAX_TILES,
                        help="Maximum representative PSF tiles; 0 processes the full grid.")
    parser.add_argument("--cupy_fft_engine", choices=("cupyx", "scout"), default="scout",
                        help="CuPy PSF estimation mode: scout filters tiles before final CuPy refinement; cupyx runs all selected tiles directly.")
    parser.add_argument("--adaptive_scout_iters", type=int, default=2,
                        help="Short blind-RL iterations used by the scout pass.")
    parser.add_argument("--adaptive_keep_tiles", type=int, default=4,
                        help="Number of scout-approved tiles to finish with the full CuPy pass.")
    parser.add_argument("--tile_selection_strategy", choices=("spatial_snr_v1", "coarse_to_fine_snr"), default="spatial_snr_v1",
                        help="Strategy for selecting representative blind PSF tiles.")
    parser.add_argument("--coarse_region_rows", type=int, default=4,
                        help="Coarse region row count for coarse_to_fine_snr tile selection.")
    parser.add_argument("--coarse_region_columns", type=int, default=4,
                        help="Coarse region column count for coarse_to_fine_snr tile selection.")
    parser.add_argument("--coarse_region_limit", type=int, default=8,
                        help="Maximum coarse regions considered by coarse_to_fine_snr tile selection.")
    parser.add_argument("--decon_chunk_xy", type=int, default=0,
                        help="Core XY tile size for CUDA deconvolution. <=0 auto-sizes from VRAM.")
    parser.add_argument("--pad_xy",      type=int, default=32,
                        help="XY halo per edge added to each blind PSF chunk (pixels).")
    parser.add_argument("--pad_z",       type=int, default=20,
                        help="Z halo per edge added before MATLAB deconvblind (pixels).")
    parser.add_argument("--blind_peak_normalization", default="none",
                        choices=("none", "gamma", "unit"),
                        help="Optional peak normalization behavior for CuPy backend.")
    parser.add_argument("--blind_peak_gamma_max", type=float, default=2.5,
                        help="Maximum gamma scaling when blind_peak_normalization='gamma'.")
    parser.add_argument("--blind_latent_update_period", type=int, default=DEFAULT_BLIND_LATENT_UPDATE_PERIOD,
                        help="Update latent image every N blind iterations; 1 preserves full alternating updates.")
    parser.add_argument("--blind_workers", type=int, default=1,
                        help="Concurrent MATLAB deconvblind chunks. <=0 uses CPU affinity, falling back to 32.")
    parser.add_argument("--matlab_threads", type=int, default=1,
                        help="Threads per MATLAB deconvblind process; clamped to 1 or 2.")
    parser.add_argument("--matlab_workers", type=int, default=1,
                        help="Concurrent MATLAB deconvblind processes; default 1 avoids MATLAB orchestration hangs.")
    parser.add_argument("--matlab_bin", default="matlab",
                        help="MATLAB executable used for deconvblind.")
    parser.add_argument("--matlab_timeout", type=int, default=1800,
                        help="Seconds before killing one MATLAB deconvblind chunk. <=0 disables.")
    parser.add_argument("--blind_z_slices", type=int, default=DEFAULT_BLIND_Z_SLICES,
                        help="Z planes used per blind PSF tile. <=0 uses full Z.")
    parser.add_argument("--snr_weight_cap", type=float, default=DEFAULT_SNR_WEIGHT_CAP,
                        help="Maximum per-chunk SNR weight before weighted PSF merge; <=0 disables cap.")
    parser.add_argument("--prefetch_chunks", type=int, default=0,
                        help="Number of PSF tiles to keep submitted/read ahead. <=0 uses one worker batch.")
    parser.add_argument("--decon_workers", type=int, default=1,
                        help="Dask workers for CUDA deconvolution chunks.")
    parser.add_argument("--overlap_xy", type=int, default=0,
                        help="Override CUDA decon XY overlap. <=0 uses a capped PSF/4 estimate.")
    parser.add_argument("--vram_gb", type=float, default=None,
                        help="Override detected free VRAM in GiB for auto chunk sizing.")
    parser.add_argument("--pyramid_max_downsample", type=int, choices=(1, 2, 4, 8, 16), default=16,
                        help="Maximum XY downsampling factor for OME-Zarr pyramid output.")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory for cached blind PSF estimates.")
    parser.add_argument("--no_psf_cache", action="store_true",
                        help="Disable reuse of cached blind PSF estimates.")

    # Deconvolution options
    parser.add_argument("--iter",       type=int,   default=10,
                        help="RL deconvolution iterations.")
    parser.add_argument("--background", type=float, default=0.0,
                        help="Background value to subtract before decon.")

    # Optical/acquisition parameters used to generate the blind-estimation PSF seed.
    parser.add_argument("--na",          type=float, default=None,
                        help="Backward-compatible detection numerical aperture.")
    parser.add_argument("--detection_na", type=float, default=None,
                        help="Detection objective numerical aperture. Overrides --na when provided.")
    parser.add_argument("--illumination_na", type=float, default=None,
                        help="Illumination numerical aperture metadata; not used by psfmodels.")
    parser.add_argument("--wavelength",  type=float, default=None,
                        help="Emission wavelength in µm.")
    parser.add_argument("--ni",          type=float, default=None,
                        help="Refractive index of immersion medium.")
    parser.add_argument("--ns",          type=float, default=None,
                        help="Sample refractive index.")
    parser.add_argument("--ni0",         type=float, default=None,
                        help="Design immersion refractive index.")
    parser.add_argument("--tg",          type=float, default=None,
                        help="Experimental coverslip thickness in µm.")
    parser.add_argument("--tg0",         type=float, default=None,
                        help="Design coverslip thickness in µm.")
    parser.add_argument("--ng",          type=float, default=None,
                        help="Experimental coverslip refractive index.")
    parser.add_argument("--ng0",         type=float, default=None,
                        help="Design coverslip refractive index.")
    parser.add_argument("--ti0",         type=float, default=None,
                        help="Objective working distance in µm.")
    parser.add_argument("--oversample_factor", type=int, default=3,
                        help="PSF model oversampling factor.")
    parser.add_argument("--psf_model", choices=("vectorial", "scalar", "gaussian"), default="vectorial",
                        help="psfmodels PSF model.")
    parser.add_argument("--psf_mode", choices=("single", "light_sheet"), default="single",
                        help="Seed PSF mode: single detection PSF or light-sheet detection x rotated illumination PSF.")
    parser.add_argument("--light_sheet_angle", type=float, default=90.0,
                        help="Degrees to rotate illumination PSF in Z/X for light_sheet PSF mode.")
    parser.add_argument("--camera_pixel_size", type=float, default=None,
                        help="Camera pixel size in µm; used to derive dxy when --dxy <= 0.")
    parser.add_argument("--magnification", type=float, default=None,
                        help="Total magnification; used to derive dxy when --dxy <= 0.")
    parser.add_argument("--dxy",         type=float, default=None,
                        help="Lateral pixel size in µm.")
    parser.add_argument("--dz",          type=float, default=None,
                        help="Axial step size in µm.")
    parser.add_argument("--psf_size_z",  type=int,   default=61,
                        help="Z size of PSF volume.")
    parser.add_argument("--psf_size_xy", type=int,   default=129,
                        help="XY size of PSF volume.")

    # Misc, usually unneeded
    parser.add_argument("--script_dir",  default=str(Path(__file__).parent),
                        help="Directory containing readtiffstack.m / writetiffstack.m.")

    args = parser.parse_args()
    args.blind_backend, args.cupy_fft_engine = normalize_blind_backend(
        args.blind_backend,
        args.cupy_fft_engine,
    )

    image_dir = Path(args.image_path)
    log_progress(f"DECON starting: image_path={image_dir}")

    # Collect all image volumes, sorted so index 0 is deterministic. File
    # selection is handled by the workflow before this wrapper runs.
    image_inputs = discover_image_volumes(image_dir)
    if not image_inputs:
        print(f"Error: no TIFF, OME-Zarr, or OZX image volumes found in {image_dir}")
        raise SystemExit(1)

    original_name_map = _load_original_name_map(image_dir)
    ozx_temp_context = None
    if any(is_ozx_path(path) for path in image_inputs):
        ozx_temp_context = tempfile.TemporaryDirectory(prefix=".ozx_input_", dir=Path.cwd())
        image_inputs = _expand_ozx_inputs_for_processing(image_inputs, Path(ozx_temp_context.name))

    log_progress(f"Found {len(image_inputs)} selected image volume(s) to process")
    for index, image_input in enumerate(image_inputs, start=1):
        log_progress(f"  Input {index}/{len(image_inputs)}: {image_input.name}")
    selected_channels = _detected_channels([str(path) for path in image_inputs if not is_ome_zarr_path(path)])
    if len(selected_channels) > 1:
        print(
            "WARNING: Multiple channels were detected in the selected TIFF inputs: "
            f"{selected_channels}. This workflow estimates one PSF from the first "
            "selected image and applies it to all selected images. Process one "
            "channel at a time unless applying one PSF across wavelengths is intentional.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # PSF resolution
    # ------------------------------------------------------------------

    dxy = _resolve_required_dxy(args.dxy, args.camera_pixel_size, args.magnification)
    dz = _require_positive("dz", args.dz)
    wavelength = _require_positive("wavelength", args.wavelength)
    ni = _require_positive("ni", args.ni)
    ns = _require_positive("ns", args.ns)
    detection_na = args.detection_na if args.detection_na is not None else args.na
    detection_na = _require_positive("detection_na", detection_na)
    if args.psf_mode == "light_sheet":
        _require_positive("illumination_na", args.illumination_na)
    log_progress(
        "Resolved optical parameters: "
        f"dxy={dxy}, dz={dz}, wavelength={wavelength}, "
        f"detection_na={detection_na}, ni={ni}, ns={ns}, psf_mode={args.psf_mode}"
    )

    # Build the optical-model PSF seed. This is intentionally not accepted as
    # the final deconvolution PSF because the measured blind estimates are much
    # closer to the observed data.
    psf_seed = generate_psf_seed(
        psf_mode=args.psf_mode,
        na=args.na,
        detection_na=args.detection_na,
        illumination_na=args.illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=args.ni0,
        tg=args.tg,
        tg0=args.tg0,
        ng=args.ng,
        ng0=args.ng0,
        ti0=args.ti0,
        oversample_factor=args.oversample_factor,
        psf_model=args.psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=args.psf_size_z,
        psf_size_xy=args.psf_size_xy,
        background=args.background,
        light_sheet_angle=args.light_sheet_angle,
    )
    log_progress(
        f"Using PSF seed mode={args.psf_mode}, shape={psf_seed.shape}, "
        f"sum={float(psf_seed.sum()):.6g}"
    )

    psf_input_path = image_inputs[0]

    log_progress(f"Running blind PSF estimation on first image volume: {psf_input_path}")
    psf_start = time.perf_counter()
    psf = estimate_psf_from_chunks(
        image_path=str(psf_input_path),
        psf_seed=psf_seed,
        n_iters=args.blind_iters,
        blind_backend=args.blind_backend,
        chunk_xy=args.chunk_xy,
        pad_xy=args.pad_xy,
        pad_z=args.pad_z,
        script_dir=args.script_dir,
        max_workers=args.blind_workers,
        prefetch_chunks=args.prefetch_chunks,
        vram_gb=args.vram_gb,
        cache_dir=args.cache_dir,
        use_cache=not args.no_psf_cache,
        matlab_threads=args.matlab_threads,
        matlab_workers=args.matlab_workers,
        matlab_bin=args.matlab_bin,
        matlab_timeout=args.matlab_timeout,
        snr_weight_cap=args.snr_weight_cap,
        blind_peak_normalization=args.blind_peak_normalization,
        blind_peak_gamma_max=args.blind_peak_gamma_max,
        blind_latent_update_period=args.blind_latent_update_period,
        blind_z_slices=args.blind_z_slices,
        blind_max_tiles=args.blind_max_tiles,
        cupy_fft_engine=args.cupy_fft_engine,
        adaptive_scout_iters=args.adaptive_scout_iters,
        adaptive_keep_tiles=args.adaptive_keep_tiles,
        tile_selection_strategy=args.tile_selection_strategy,
        coarse_region_rows=args.coarse_region_rows,
        coarse_region_columns=args.coarse_region_columns,
        coarse_region_limit=args.coarse_region_limit,
    )
    psf_save_path = image_dir / "estimated_psf.tif"
    psf_save_path = _write_tiff_near_input_or_cwd(psf_save_path, psf)
    published_psf_path = Path.cwd() / "estimated_psf.tif"
    if psf_save_path.resolve() != published_psf_path.resolve():
        imwrite(str(published_psf_path), psf)
    log_progress(
        f"Merged PSF saved to {psf_save_path}; "
        f"shape={psf.shape}, elapsed={time.perf_counter() - psf_start:.2f}s"
    )

    # ------------------------------------------------------------------
    # Deconvolve all image volumes with the resolved PSF
    # ------------------------------------------------------------------

    for index, image_path in enumerate(image_inputs, start=1):
        image_path = Path(image_path)
        volume_start = time.perf_counter()
        log_progress(f"Starting deconvolution input {index}/{len(image_inputs)}: {image_path.name}")
        if is_ome_zarr_path(image_path):
            out_name = _decon_ome_zarr_output_name(image_path, original_name_map)
            log_progress(f"Writing deconvolved OME-Zarr output: {out_name}")
            deconvolve_ome_zarr_to_zarr(
                image_path=image_path,
                output_path=out_name,
                psf=psf,
                n_iters=args.iter,
                dz=dz,
                dxy=dxy,
                wavelength=wavelength,
                na=detection_na,
                ni=ni,
                chunk_xy=args.decon_chunk_xy,
                vram_gb=args.vram_gb,
                decon_workers=args.decon_workers,
                overlap_xy=args.overlap_xy,
                max_downsample=args.pyramid_max_downsample,
            )
        else:
            output = deconvolve_tiff(
                image_path=image_path,
                psf=psf,
                n_iters=args.iter,
                dz=dz,
                dxy=dxy,
                wavelength=wavelength,
                na=detection_na,
                ni=ni,
                chunk_xy=args.decon_chunk_xy,
                vram_gb=args.vram_gb,
                decon_workers=args.decon_workers,
                overlap_xy=args.overlap_xy,
            )
            out_name = _write_materialized_decon_output(
                output,
                image_path,
                original_name_map,
                output_format=args.output_format,
                max_downsample=args.pyramid_max_downsample,
            )
        log_progress(
            f"Saved {out_name}; elapsed={time.perf_counter() - volume_start:.2f}s"
        )

    log_progress(f"All image volumes deconvolved in {time.perf_counter() - run_start:.2f}s")
    if ozx_temp_context is not None:
        ozx_temp_context.cleanup()


if __name__ == "__main__":
    main()
