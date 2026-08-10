# psf_estimation.py
# Blind PSF estimation via chunked deconvblind + weighted merge.
#
# Workflow:
#   Open the first deskewed TIFF or OME-Zarr volume as an array-like object.
#   Split it into full-Z XY tiles sized from available VRAM unless overridden.
#   MATLAB tiles are written to temporary TIFFs for deconvblind compatibility.
#   CuPy tiles are estimated directly from in-memory arrays to avoid per-tile
#   process startup and TIFF roundtrips.
#   Python collects all per-tile PSFs and returns an SNR-weighted PSF merge.
#
# The returned PSF is float32, normalised to sum=1, and saved as estimated_psf.tif
# next to the input image for direct use by the restoration stage.

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import inspect
import json
import math
import os
import subprocess
import tempfile
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psfmodels as pm
from tifffile import TiffFile, imread, imwrite, memmap as tiff_memmap

try:
    from ome_zarr_io import is_ome_zarr_path, open_ome_zarr_array
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ome_zarr_io import is_ome_zarr_path, open_ome_zarr_array

DEFAULT_CPU_THREADS = 32
DEFAULT_SNR_WEIGHT_CAP = 100.0
DEFAULT_BLIND_ITERS = 8
DEFAULT_BLIND_MEMORY_MULTIPLIER = 28.0
DEFAULT_BLIND_MEMORY_OVERHEAD_GB = 1.0
DEFAULT_BLIND_Z_SLICES = 128
DEFAULT_BLIND_CHUNK_XY = 256
DEFAULT_BLIND_MAX_TILES = 16
DEFAULT_BLIND_LATENT_UPDATE_PERIOD = 2
DEFAULT_BLIND_BACKEND = "cupy"
DEFAULT_BLIND_PEAK_NORMALIZATION = "none"
DEFAULT_BLIND_PEAK_GAMMA_MAX = 2.5
DEFAULT_CUPY_VRAM_FRACTION = 0.72
DEFAULT_CUPY_FFT_BYTES_PER_VOXEL = 208
DEFAULT_ADAPTIVE_SCOUT_ITERS = 2
DEFAULT_ADAPTIVE_KEEP_TILES = 4
BLIND_CHUNK_ALIGNMENT = 32
BLIND_TILE_SELECTION_STRATEGY = "spatial_snr_v1"
COARSE_TO_FINE_TILE_SELECTION_STRATEGY = "coarse_to_fine_snr"
TILE_SELECTION_STRATEGIES = (
    BLIND_TILE_SELECTION_STRATEGY,
    COARSE_TO_FINE_TILE_SELECTION_STRATEGY,
)
DEFAULT_COARSE_REGION_ROWS = 4
DEFAULT_COARSE_REGION_COLUMNS = 4
DEFAULT_COARSE_REGION_LIMIT = 8
BLIND_BACKENDS = ("matlab", "cupy")
BLIND_BACKEND_CLI_CHOICES = ("matlab", "cupy", "scout", "cupyx")


def normalize_blind_backend(
    blind_backend: str | None,
    cupy_fft_engine: str | None = "scout",
) -> tuple[str, str]:
    """Return canonical backend and CuPy mode values.

    Accept scout/cupyx as legacy backend aliases from older params files.
    """
    backend = str(blind_backend or DEFAULT_BLIND_BACKEND).lower()
    engine = str(cupy_fft_engine or "scout").lower()

    if backend == "scout":
        return "cupy", "scout"
    if backend == "cupyx":
        return "cupy", "cupyx"
    return backend, engine


def _ensure_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path,
        prefix=".write_test_",
        delete=True,
    ):
        pass


def _resolve_psf_cache_root(image_path: Path, cache_dir: str | Path | None) -> Path:
    if cache_dir:
        cache_root = Path(cache_dir)
        _ensure_writable_dir(cache_root)
        return cache_root

    preferred = image_path.parent / ".psf_cache"
    fallback = Path.cwd() / ".psf_cache"
    for cache_root in (preferred, fallback):
        try:
            _ensure_writable_dir(cache_root)
        except OSError as exc:
            if cache_root == preferred:
                print(
                    f"  WARNING: cannot use PSF cache {cache_root}: {exc}; "
                    f"falling back to {fallback}",
                    flush=True,
                )
            continue
        return cache_root

    raise PermissionError(
        f"Unable to create PSF cache in {preferred} or fallback {fallback}"
    )


# ---------------------------------------------------------------------------
# Theoretical PSF seed for blind deconvolution
# ---------------------------------------------------------------------------

def _available_cpu_threads(default: int = DEFAULT_CPU_THREADS) -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    count = os.cpu_count()
    return count if count and count > 0 else default


def resolve_worker_count(requested_workers: int, default: int = DEFAULT_CPU_THREADS) -> int:
    if requested_workers > 0:
        return requested_workers
    return min(_available_cpu_threads(default=default), default)


def _parse_memory_bytes(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip().upper()
    if not text:
        return None
    multiplier = 1
    if text.endswith(("K", "KB")):
        multiplier = 1024
        text = text.rstrip("B").rstrip("K")
    elif text.endswith(("M", "MB")):
        multiplier = 1024 ** 2
        text = text.rstrip("B").rstrip("M")
    elif text.endswith(("G", "GB")):
        multiplier = 1024 ** 3
        text = text.rstrip("B").rstrip("G")
    elif text.endswith(("T", "TB")):
        multiplier = 1024 ** 4
        text = text.rstrip("B").rstrip("T")
    try:
        number = float(text)
    except ValueError:
        return None
    # Slurm memory variables without suffix are MB.
    if multiplier == 1 and number < 10_000_000:
        multiplier = 1024 ** 2
    return int(number * multiplier)


def _cgroup_memory_limit_bytes() -> int | None:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text or text == "max":
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if 0 < value < 1 << 60:
            return value
    return None


def _allocated_memory_bytes() -> int | None:
    slurm_node = _parse_memory_bytes(os.environ.get("SLURM_MEM_PER_NODE"))
    if slurm_node:
        return slurm_node

    slurm_per_cpu = _parse_memory_bytes(os.environ.get("SLURM_MEM_PER_CPU"))
    if slurm_per_cpu:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or _available_cpu_threads())
        return slurm_per_cpu * max(1, cpus)

    cgroup_limit = _cgroup_memory_limit_bytes()
    if cgroup_limit:
        return cgroup_limit

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _blind_chunk_input_bytes(
    volume_shape: tuple[int, int, int],
    dtype: np.dtype,
    chunk_xy: int,
    halo_xy: int,
) -> int:
    nz, ny, nx = volume_shape
    tile_y = min(ny, chunk_xy + 2 * halo_xy)
    tile_x = min(nx, chunk_xy + 2 * halo_xy)
    return int(nz * tile_y * tile_x * np.dtype(dtype).itemsize)


def select_blind_z_window(
    volume: np.ndarray,
    max_z_slices: int = DEFAULT_BLIND_Z_SLICES,
    sample_planes: int = 64,
) -> tuple[slice, str]:
    nz = volume.shape[0]
    if max_z_slices <= 0 or nz <= max_z_slices:
        return slice(None), f"full_z=0:{nz}"

    sample_count = max(1, min(sample_planes, nz))
    sample_indices = np.unique(np.linspace(0, nz - 1, sample_count, dtype=int))
    scores = []
    for z in sample_indices:
        plane = np.asarray(volume[z], dtype=np.float32)
        scores.append(float(np.percentile(plane, 99.9)))

    if max(scores) <= min(scores):
        center_z = nz // 2
        score_detail = "flat_sample_scores"
    else:
        center_z = int(sample_indices[int(np.argmax(scores))])
        score_detail = "brightest_sample"
    start = max(0, center_z - (max_z_slices // 2))
    stop = min(nz, start + max_z_slices)
    start = max(0, stop - max_z_slices)
    return (
        slice(start, stop),
        f"bright_z_window={start}:{stop}, center={center_z}, "
        f"sampled_planes={len(sample_indices)}, selector={score_detail}",
    )


def resolve_blind_worker_count(
    requested_workers: int,
    cpu_workers: int,
    volume_shape: tuple[int, int, int],
    dtype: np.dtype,
    chunk_xy: int,
    halo_xy: int,
    memory_multiplier: float = DEFAULT_BLIND_MEMORY_MULTIPLIER,
    overhead_gb: float = DEFAULT_BLIND_MEMORY_OVERHEAD_GB,
) -> tuple[int, str]:
    if requested_workers > 0:
        return requested_workers, "explicit"

    memory_bytes = _allocated_memory_bytes()
    if not memory_bytes:
        return cpu_workers, "cpu"

    chunk_bytes = _blind_chunk_input_bytes(volume_shape, dtype, chunk_xy, halo_xy)
    per_worker = chunk_bytes * memory_multiplier + overhead_gb * (1024 ** 3)
    usable = memory_bytes * 0.70
    memory_workers = max(1, int(usable // max(1, per_worker)))
    resolved = max(1, min(cpu_workers, memory_workers))
    detail = (
        f"cpu={cpu_workers}, memory_cap={memory_workers}, "
        f"allocated={memory_bytes / (1024 ** 3):.1f}GiB, "
        f"estimated_per_worker={per_worker / (1024 ** 3):.1f}GiB"
    )
    return resolved, detail


def resolve_backend_executor_workers(
    blind_backend: str,
    blind_workers: int,
    matlab_workers: int,
) -> int:
    """Resolve the thread-pool size needed to feed the selected backend."""
    blind_backend, _ = normalize_blind_backend(blind_backend)
    blind_workers = max(1, int(blind_workers))
    matlab_workers = max(1, int(matlab_workers))
    if blind_backend == "cupy":
        return 1
    if blind_backend == "matlab":
        return max(blind_workers, matlab_workers)
    raise ValueError(f"Unsupported blind backend '{blind_backend}'")


def generate_theoretical_psf(
    na: float | None = None,
    detection_na: float | None = None,
    illumination_na: float | None = None,
    wavelength: float | None = None,      # µm
    ni: float | None = None,
    ns: float | None = None,
    ni0: float | None = None,
    tg: float | None = None,
    tg0: float | None = None,
    ng: float | None = None,
    ng0: float | None = None,
    ti0: float | None = None,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
    dxy: float | None = None,       # µm, lateral pixel size
    dz: float | None = None,        # µm, axial step
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    background: float = 0.0,
) -> np.ndarray:
    """
    Generate a 3-D Gibson-Lanni PSF using psfmodels.

    psfmodels generates the detection PSF.  `illumination_na` is accepted for
    pipeline metadata, but this scalar/vectorial PSF model does not use it.

    Returns float32 array of shape (psf_size_z, psf_size_xy, psf_size_xy),
    background-subtracted and normalised to sum = 1.
    """
    detection_na = detection_na if detection_na is not None else na
    required_values = {
        "detection_na": detection_na,
        "wavelength": wavelength,
        "ni": ni,
        "ns": ns,
        "dxy": dxy,
        "dz": dz,
    }
    missing = [name for name, value in required_values.items() if value is None or value <= 0]
    if missing:
        raise ValueError(
            "Missing required optical/acquisition parameter(s): " + ", ".join(missing)
        )
    requested_kwargs = {
        "z": psf_size_z,
        "nx": psf_size_xy,
        "dz": dz,
        "dxy": dxy,
        "NA": detection_na,
        "wvl": wavelength,
        "ni": ni,
        "oversample_factor": oversample_factor,
        "model": psf_model,
    }
    optional_kwargs = {
        "ns": ns,
        "ni0": ni0,
        "tg": tg,
        "tg0": tg0,
        "ng": ng,
        "ng0": ng0,
        "ti0": ti0,
    }
    requested_kwargs.update({name: value for name, value in optional_kwargs.items() if value is not None})
    signature = inspect.signature(pm.make_psf)
    missing = [name for name in requested_kwargs if name not in signature.parameters]
    if missing:
        raise RuntimeError(
            "psfmodels.make_psf API mismatch; missing expected parameter(s): "
            + ", ".join(missing)
        )
    psf = pm.make_psf(**requested_kwargs).astype(np.float32)

    psf = np.maximum(psf - background, 0)
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


def resolve_dxy(
    dxy: float | None,
    camera_pixel_size: float | None = None,
    magnification: float | None = None,
) -> float:
    if dxy is not None and dxy > 0:
        return dxy
    if camera_pixel_size and magnification and camera_pixel_size > 0 and magnification > 0:
        return camera_pixel_size / magnification
    raise ValueError("dxy must be > 0, or camera_pixel_size and magnification must be provided")


# ---------------------------------------------------------------------------
# Per-chunk blind estimation via MATLAB deconvblind
# ---------------------------------------------------------------------------

def _normalise_psf(psf: np.ndarray) -> np.ndarray:
    psf = np.nan_to_num(psf.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.clip(psf, 0, None)
    total = float(psf.sum())
    if total > 0:
        psf = psf / total
    return psf.astype(np.float32, copy=False)


def ensure_3d_volume(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 2:
        return volume[np.newaxis, :, :]
    return volume


def adapt_psf_seed_to_volume(psf_seed: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    volume_z = int(volume_shape[0])
    if volume_z <= 0:
        raise ValueError(f"Volume Z size must be positive, got {volume_shape}")
    if psf_seed.shape[0] <= volume_z:
        return psf_seed

    z_start = (psf_seed.shape[0] - volume_z) // 2
    z_stop = z_start + volume_z
    adapted = psf_seed[z_start:z_stop, :, :]
    return _normalise_psf(adapted)


def open_tiff_memmap(path: str | Path) -> np.ndarray:
    """
    Return a read-only array-like TIFF volume without forcing a full RAM load.

    `tifffile.memmap` maps compatible contiguous TIFF data directly.  Some TIFFs
    cannot be directly memory-mapped; for those, tifffile can materialise a
    temporary memmap via `asarray(out="memmap")`, which still keeps downstream
    chunking bounded instead of holding the whole image as an ndarray.
    """
    path = Path(path)
    try:
        return ensure_3d_volume(tiff_memmap(str(path), mode="r"))
    except Exception:
        with TiffFile(str(path)) as tif:
            return ensure_3d_volume(tif.asarray(out="memmap"))


def open_psf_source(path: str | Path):
    """
    Return a read-only array-like 3-D volume for blind PSF estimation.

    TIFF inputs are memory-mapped. OME-Zarr inputs are opened at level 0 so
    downstream tile slicing can read only the chunks needed for MATLAB.
    """
    path = Path(path)
    if is_ome_zarr_path(path):
        print(f"Opening OME-Zarr {path} for PSF estimation...", flush=True)
        return open_ome_zarr_array(path, mode="r")

    print(f"Memory-mapping {path} for PSF estimation...", flush=True)
    return open_tiff_memmap(path)


def detect_vram_bytes() -> int | None:
    """Best-effort free VRAM query using nvidia-smi."""
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    command = ["nvidia-smi"]
    if visible_device and visible_device not in ("NoDevFiles", "-1"):
        command.extend(["--id", visible_device])
    command.extend(
        [
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    free_mb = []
    for line in result.stdout.splitlines():
        try:
            free_mb.append(int(line.strip().split()[0]))
        except (IndexError, ValueError):
            continue
    if not free_mb:
        return None
    return min(free_mb) * 1024 * 1024


def _cupy_blind_fft_bytes(
    core_xy: int,
    volume_z: int,
    halo_xy: int,
    pad_z: int,
    psf_shape: tuple[int, int, int],
) -> int:
    """Estimate peak alternating-RL storage from the padded FFT domain."""
    from scipy.fft import next_fast_len

    image_shape = (
        int(volume_z) + 2 * int(pad_z),
        int(core_xy) + 2 * int(halo_xy),
        int(core_xy) + 2 * int(halo_xy),
    )
    fft_shape = tuple(
        next_fast_len(image_size + int(kernel_size) - 1)
        for image_size, kernel_size in zip(image_shape, psf_shape)
    )
    return int(math.prod(fft_shape) * DEFAULT_CUPY_FFT_BYTES_PER_VOXEL)


def resolve_cupy_blind_chunk_xy(
    requested_xy: int,
    volume_shape: tuple[int, int, int],
    psf_shape: tuple[int, int, int],
    halo_xy: int,
    pad_z: int,
    vram_gb: float | None = None,
) -> tuple[int, str]:
    """Clamp a requested CuPy core size to the available FFT workspace."""
    _, ny, nx = volume_shape
    maximum = min(ny, nx, requested_xy if requested_xy > 0 else min(ny, nx))
    minimum = min(
        maximum,
        max(
            64,
            int(
                math.ceil(max(int(psf_shape[-2]), int(psf_shape[-1])) /
                          BLIND_CHUNK_ALIGNMENT)
            ) * BLIND_CHUNK_ALIGNMENT,
        ),
    )
    candidate = max(
        minimum,
        (maximum // BLIND_CHUNK_ALIGNMENT) * BLIND_CHUNK_ALIGNMENT,
    )
    vram_bytes = (
        int(vram_gb * (1024 ** 3))
        if vram_gb and vram_gb > 0
        else detect_vram_bytes()
    )
    if not vram_bytes:
        return candidate, "VRAM unavailable; OOM retry enabled"

    budget = int(vram_bytes * DEFAULT_CUPY_VRAM_FRACTION)
    while candidate > minimum:
        estimated = _cupy_blind_fft_bytes(
            candidate,
            volume_shape[0],
            halo_xy,
            pad_z,
            psf_shape,
        )
        if estimated <= budget:
            break
        candidate = max(minimum, candidate - BLIND_CHUNK_ALIGNMENT)

    estimated = _cupy_blind_fft_bytes(
        candidate,
        volume_shape[0],
        halo_xy,
        pad_z,
        psf_shape,
    )
    detail = (
        f"free_vram={vram_bytes / (1024 ** 3):.1f}GiB, "
        f"budget={budget / (1024 ** 3):.1f}GiB, "
        f"estimated_peak={estimated / (1024 ** 3):.1f}GiB"
    )
    return candidate, detail


def _is_cupy_out_of_memory(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        name = f"{type(current).__module__}.{type(current).__name__}"
        message = f"{name}: {current}"
        if "cupy.cuda.memory.OutOfMemoryError" in message or (
            type(current).__name__ == "OutOfMemoryError" and "cupy" in name
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _next_smaller_blind_chunk_xy(current: int, minimum: int) -> int:
    if current <= minimum:
        return current
    return max(
        minimum,
        ((current - 1) // BLIND_CHUNK_ALIGNMENT) * BLIND_CHUNK_ALIGNMENT,
    )


def resolve_chunk_xy(
    requested_xy: int,
    volume_shape: tuple[int, int, int],
    dtype: np.dtype,
    overlap_xy: int = 0,
    vram_gb: float | None = None,
    workers: int = 1,
    safety_fraction: float = 0.55,
    memory_multiplier: float = 18.0,
    min_xy: int = 128,
    max_xy: int | None = None,
) -> int:
    """
    Resolve an XY chunk size.  Positive `requested_xy` is treated as an explicit
    override; zero or negative values trigger a VRAM-aware estimate.
    """
    if requested_xy > 0:
        return requested_xy

    nz, ny, nx = volume_shape
    max_xy = max_xy or min(ny, nx)
    vram_bytes = int(vram_gb * (1024 ** 3)) if vram_gb and vram_gb > 0 else detect_vram_bytes()
    if not vram_bytes:
        return min(512, max_xy)

    workers = max(1, workers)
    bytes_per_voxel = np.dtype(dtype).itemsize
    usable = vram_bytes * safety_fraction / workers
    denom = max(1, nz) * bytes_per_voxel * memory_multiplier
    overlapped_xy = int(math.sqrt(max(1, usable / denom)))
    core_xy = max(min_xy, overlapped_xy - (2 * overlap_xy))
    core_xy = min(core_xy, max_xy)
    aligned = max(min_xy, (core_xy // 32) * 32)
    return max(32, min(aligned, max_xy))


def _psf_cache_key(
    image_path: Path,
    psf_seed: np.ndarray,
    n_iters: int,
    chunk_xy: int,
    pad_xy: int,
    pad_z: int,
    script_dir: Path,
    merge_mode: str,
    snr_weight_cap: float,
    z_window: tuple[int | None, int | None],
    blind_backend: str,
    blind_peak_normalization: str,
    blind_peak_gamma_max: float,
    blind_latent_update_period: int,
    blind_max_tiles: int,
    tile_selection_strategy: str,
    cupy_fft_engine: str = "cupyx",
    adaptive_scout_iters: int = DEFAULT_ADAPTIVE_SCOUT_ITERS,
    adaptive_keep_tiles: int = DEFAULT_ADAPTIVE_KEEP_TILES,
    coarse_region_rows: int = DEFAULT_COARSE_REGION_ROWS,
    coarse_region_columns: int = DEFAULT_COARSE_REGION_COLUMNS,
    coarse_region_limit: int = DEFAULT_COARSE_REGION_LIMIT,
) -> str:
    stat = image_path.stat()
    payload = {
        "image": str(image_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "seed_shape": psf_seed.shape,
        "seed_sha256": hashlib.sha256(np.ascontiguousarray(psf_seed).view(np.uint8)).hexdigest(),
        "n_iters": n_iters,
        "chunk_xy": chunk_xy,
        "pad_xy": pad_xy,
        "pad_z": pad_z,
        "script_dir": str(script_dir.resolve()),
        "merge_mode": merge_mode,
        "snr_weight_cap": snr_weight_cap,
        "z_window": z_window,
        "blind_backend": blind_backend,
        "blind_peak_normalization": blind_peak_normalization,
        "blind_peak_gamma_max": blind_peak_gamma_max,
        "blind_latent_update_period": blind_latent_update_period,
        "blind_max_tiles": blind_max_tiles,
        "tile_selection_strategy": tile_selection_strategy,
        "cupy_fft_engine": cupy_fft_engine,
        "adaptive_scout_iters": adaptive_scout_iters,
        "adaptive_keep_tiles": adaptive_keep_tiles,
        "coarse_region_rows": coarse_region_rows,
        "coarse_region_columns": coarse_region_columns,
        "coarse_region_limit": coarse_region_limit,
        "version": 9,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]

def _write_matlab_stack(array: np.ndarray, path: Path, scale_float: bool = False) -> None:
    """Write a stack in a TIFF format that MATLAB's Tiff reader handles reliably."""
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        finite = np.clip(finite, 0, None)
        if scale_float:
            max_value = finite.max()
            if max_value > 0:
                finite = finite / max_value * np.iinfo(np.uint16).max
        array = np.clip(np.rint(finite), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    imwrite(str(path), array)


def _write_chunk(chunk: np.ndarray, path: Path) -> None:
    _write_matlab_stack(chunk, path)


def _run_matlab_deconvblind(
    chunk_path: Path,
    psf_seed: np.ndarray,
    psf_seed_path: Path,
    output_psf_path: Path,
    n_iters: int,
    pad_z: int,
    script_dir: Path,
    matlab_bin: str,
    matlab_threads: int,
    matlab_timeout: int,
) -> None:
    """
    Call MATLAB deconvblind on one chunk.  The script writes the recovered PSF
    to output_psf_path as a float32 TIFF.

    MATLAB is invoked with -batch so it exits cleanly on completion or error.
    """
    _write_matlab_stack(psf_seed, psf_seed_path, scale_float=True)

    matlab_threads = min(2, max(1, matlab_threads))
    pad_z = max(0, pad_z)
    matlab_thread_cmd = f"maxNumCompThreads({matlab_threads}); "
    z_pad_cmd = (
        f"chunk = padarray(chunk, [0 0 {pad_z}], 'symmetric'); "
        if pad_z > 0 else ""
    )
    matlab_cmd = (
        f"addpath('{script_dir}'); "
        f"{matlab_thread_cmd}"
        f"chunk = single(readtiffstack('{chunk_path}')); "
        f"psf_seed = single(readtiffstack('{psf_seed_path}')); "
        f"psf_seed = psf_seed / sum(psf_seed(:)); "
        f"{z_pad_cmd}"
        f"[~, psf_est] = deconvblind(chunk, psf_seed, {n_iters}); "
        f"psf_est = single(psf_est); "
        f"psf_est = psf_est / sum(psf_est(:)); "
        f"writetiffstack(psf_est, '{output_psf_path}');"
    )
    matlab_args = [matlab_bin, "-nojvm", "-nodisplay", "-nosplash"]
    if matlab_threads == 1:
        matlab_args.append("-singleCompThread")
    matlab_args.extend(["-batch", matlab_cmd])
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = str(matlab_threads)

    try:
        result = subprocess.run(
            matlab_args,
            capture_output=True,
            text=True,
            env=env,
            timeout=matlab_timeout if matlab_timeout > 0 else None,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to start MATLAB executable '{matlab_bin}' for chunk "
            f"{chunk_path.name}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"MATLAB deconvblind timed out for chunk {chunk_path.name} "
            f"after {matlab_timeout}s.\n"
            f"STDOUT: {exc.stdout or ''}\nSTDERR: {exc.stderr or ''}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"MATLAB deconvblind failed for chunk {chunk_path.name} "
            f"(returncode={result.returncode}).\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


def _run_cupy_deconvblind(
    chunk_path: Path,
    psf_seed: np.ndarray,
    psf_seed_path: Path,
    output_psf_path: Path,
    n_iters: int,
    pad_z: int,
    script_dir: Path,
    matlab_bin: str,
    matlab_threads: int,
    matlab_timeout: int,
    blind_peak_normalization: str,
    blind_peak_gamma_max: float,
    blind_latent_update_period: int,
) -> None:
    """
    Run native blind Richardson-Lucy estimation in an isolated GPU process.

    MATLAB-only arguments remain in the signature so both backends use the
    same tile task payload.
    """
    if blind_peak_normalization not in ("none", "gamma", "unit"):
        raise ValueError(
            f"Unknown blind_peak_normalization='{blind_peak_normalization}'"
        )
    if blind_peak_normalization != "none" and blind_peak_gamma_max <= 0:
        raise ValueError(
            "blind_peak_gamma_max must be > 0 when blind_peak_normalization is enabled"
        )

    imwrite(
        str(psf_seed_path),
        np.asarray(psf_seed, dtype=np.float32),
        photometric="minisblack",
    )
    del psf_seed, script_dir, matlab_bin, matlab_threads, matlab_timeout
    try:
        from blind_rl import estimate_psf_file_in_process
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import the native CuPy blind-RL backend"
        ) from exc

    estimate_psf_file_in_process(
        chunk_path,
        psf_seed_path,
        output_psf_path,
        n_iters,
        pad_z,
        peak_normalization=blind_peak_normalization,
        peak_gamma_max=blind_peak_gamma_max,
        latent_update_period=blind_latent_update_period,
    )


def _run_cupy_deconvblind_array(
    chunk: np.ndarray,
    psf_seed: np.ndarray,
    n_iters: int,
    pad_z: int,
    blind_peak_normalization: str,
    blind_peak_gamma_max: float,
    blind_latent_update_period: int,
    cupy_pool_trim_bytes: int | None,
) -> np.ndarray:
    """Run native CuPy blind-RL directly on in-memory tile arrays."""
    try:
        from blind_rl import estimate_psf_array_cupy, trim_cupy_memory_pool
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import the native CuPy blind-RL backend"
        ) from exc

    psf = estimate_psf_array_cupy(
        chunk,
        psf_seed,
        n_iters,
        pad_z,
        peak_normalization=blind_peak_normalization,
        peak_gamma_max=blind_peak_gamma_max,
        clear_plan_cache=False,
        free_memory_pool=False,
        latent_update_period=blind_latent_update_period,
    )
    trim_cupy_memory_pool(cupy_pool_trim_bytes)
    return psf


def _clear_cupy_blind_memory() -> None:
    try:
        from blind_rl import clear_cupy_memory
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import the native CuPy blind-RL backend"
        ) from exc

    clear_cupy_memory(clear_plan_cache=True, free_memory_pool=True)


# ---------------------------------------------------------------------------
# Main estimation entry point
# ---------------------------------------------------------------------------

def _tile_origins(ny: int, nx: int, chunk_xy: int) -> list[tuple[int, int, int, int]]:
    min_tile = max(1, chunk_xy // 2)
    origins = []
    for y0 in range(0, ny, chunk_xy):
        for x0 in range(0, nx, chunk_xy):
            y1 = min(y0 + chunk_xy, ny)
            x1 = min(x0 + chunk_xy, nx)
            if (y1 - y0) >= min_tile and (x1 - x0) >= min_tile:
                origins.append((y0, x0, y1, x1))
    return origins


def _extract_tile_with_halo(
    volume: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    halo_xy: int,
) -> np.ndarray:
    _, ny, nx = volume.shape
    read_y0 = max(0, y0 - halo_xy)
    read_y1 = min(ny, y1 + halo_xy)
    read_x0 = max(0, x0 - halo_xy)
    read_x1 = min(nx, x1 + halo_xy)

    chunk = np.asarray(volume[:, read_y0:read_y1, read_x0:read_x1])

    before_y = read_y0 - (y0 - halo_xy)
    after_y = (y1 + halo_xy) - read_y1
    before_x = read_x0 - (x0 - halo_xy)
    after_x = (x1 + halo_xy) - read_x1
    if before_y or after_y or before_x or after_x:
        chunk = np.pad(
            chunk,
            pad_width=((0, 0), (before_y, after_y), (before_x, after_x)),
            mode="reflect",
        )
    return chunk


def _snr_weight(core: np.ndarray, weight_cap: float = DEFAULT_SNR_WEIGHT_CAP) -> float:
    sample = np.asarray(core, dtype=np.float32)
    if sample.size == 0:
        return 0.0
    p50, p90, p99 = np.percentile(sample, [50, 90, 99])
    noise_region = sample[sample <= p90]
    if noise_region.size == 0:
        noise_region = sample
    mad = np.median(np.abs(noise_region - np.median(noise_region)))
    noise = max(1.4826 * float(mad), float(np.std(noise_region)), 1.0)
    snr = max(0.0, float(p99 - p50)) / noise
    weight = max(1e-3, snr * snr)
    if weight_cap > 0:
        weight = min(weight, weight_cap)
    return weight


def _log_selected_blind_tiles(
    selected: list[tuple[float, tuple[int, int, int, int]]],
    candidate_count: int,
    strategy: str,
) -> None:
    reduction = 100.0 * (1.0 - len(selected) / candidate_count) if candidate_count else 0.0
    print(
        f"  Blind tile selection: strategy={strategy}, "
        f"candidates={candidate_count}, selected={len(selected)}, "
        f"reduction={reduction:.1f}%",
        flush=True,
    )
    for index, (score, tile) in enumerate(selected, start=1):
        y0, x0, y1, x1 = tile
        print(
            f"    Selected blind tile {index}/{len(selected)}: "
            f"tile=({y0}:{y1}, {x0}:{x1}), snr_weight={score:.3g}",
            flush=True,
        )


def _select_coarse_to_fine_snr_tiles(
    volume,
    tile_origins: list[tuple[int, int, int, int]],
    *,
    max_tiles: int,
    snr_weight_cap: float,
    coarse_region_rows: int = DEFAULT_COARSE_REGION_ROWS,
    coarse_region_columns: int = DEFAULT_COARSE_REGION_COLUMNS,
    coarse_region_limit: int = DEFAULT_COARSE_REGION_LIMIT,
) -> list[tuple[int, int, int, int]]:
    if max_tiles < 0:
        raise ValueError(f"blind_max_tiles cannot be negative, got {max_tiles}")
    if max_tiles == 0 or len(tile_origins) <= max_tiles:
        print(
            f"  Blind tile selection: strategy={COARSE_TO_FINE_TILE_SELECTION_STRATEGY}, "
            f"candidates={len(tile_origins)}, selected={len(tile_origins)}, "
            "reduction=0.0%",
            flush=True,
        )
        return list(tile_origins)

    coarse_region_rows = max(1, int(coarse_region_rows))
    coarse_region_columns = max(1, int(coarse_region_columns))
    coarse_region_limit = max(1, int(coarse_region_limit))
    y_min = min(tile[0] for tile in tile_origins)
    y_max = max(tile[2] for tile in tile_origins)
    x_min = min(tile[1] for tile in tile_origins)
    x_max = max(tile[3] for tile in tile_origins)
    y_span = max(1, y_max - y_min)
    x_span = max(1, x_max - x_min)

    scored_tiles: list[tuple[float, tuple[int, int, int, int]]] = []
    regions: dict[
        tuple[int, int], list[tuple[float, tuple[int, int, int, int]]]
    ] = {}
    for tile in tile_origins:
        y0, x0, y1, x1 = tile
        core = np.asarray(volume[:, y0:y1, x0:x1])
        score = _snr_weight(core, weight_cap=snr_weight_cap)
        scored = (score, tile)
        scored_tiles.append(scored)
        y_center = ((y0 + y1) / 2.0) - y_min
        x_center = ((x0 + x1) / 2.0) - x_min
        region = (
            min(coarse_region_rows - 1, int(y_center * coarse_region_rows / y_span)),
            min(coarse_region_columns - 1, int(x_center * coarse_region_columns / x_span)),
        )
        regions.setdefault(region, []).append(scored)

    region_scores = [
        (max(score for score, _ in candidates), region)
        for region, candidates in regions.items()
    ]
    selected_regions = {
        region
        for _, region in sorted(region_scores, key=lambda item: (-item[0], item[1]))[
            : min(coarse_region_limit, len(region_scores))
        ]
    }

    selected: list[tuple[float, tuple[int, int, int, int]]] = []
    selected_tiles: set[tuple[int, int, int, int]] = set()
    for region in sorted(selected_regions):
        best = min(regions[region], key=lambda item: (-item[0], item[1]))
        selected.append(best)
        selected_tiles.add(best[1])
        if len(selected) >= max_tiles:
            break

    ranked_by_region = {
        region: sorted(candidates, key=lambda item: (-item[0], item[1]))
        for region, candidates in regions.items()
        if region in selected_regions
    }
    while len(selected) < max_tiles:
        added = False
        for region in sorted(selected_regions):
            for scored in ranked_by_region.get(region, []):
                if scored[1] in selected_tiles:
                    continue
                selected.append(scored)
                selected_tiles.add(scored[1])
                added = True
                break
            if len(selected) >= max_tiles:
                break
        if not added:
            break

    selected.sort(key=lambda item: item[1])
    _log_selected_blind_tiles(
        selected,
        len(tile_origins),
        COARSE_TO_FINE_TILE_SELECTION_STRATEGY,
    )
    return [tile for _, tile in selected]


def _select_representative_tiles(
    volume,
    tile_origins: list[tuple[int, int, int, int]],
    max_tiles: int,
    snr_weight_cap: float,
    *,
    strategy: str = BLIND_TILE_SELECTION_STRATEGY,
    coarse_region_rows: int = DEFAULT_COARSE_REGION_ROWS,
    coarse_region_columns: int = DEFAULT_COARSE_REGION_COLUMNS,
    coarse_region_limit: int = DEFAULT_COARSE_REGION_LIMIT,
) -> list[tuple[int, int, int, int]]:
    """Select high-SNR tiles across balanced spatial regions."""
    if max_tiles < 0:
        raise ValueError(f"blind_max_tiles cannot be negative, got {max_tiles}")
    if max_tiles == 0 or len(tile_origins) <= max_tiles:
        print(
            f"  Blind tile selection: strategy={strategy}, "
            f"candidates={len(tile_origins)}, selected={len(tile_origins)}, "
            "reduction=0.0%",
            flush=True,
        )
        return list(tile_origins)
    strategy = str(strategy)
    if strategy == COARSE_TO_FINE_TILE_SELECTION_STRATEGY:
        return _select_coarse_to_fine_snr_tiles(
            volume,
            tile_origins,
            max_tiles=max_tiles,
            snr_weight_cap=snr_weight_cap,
            coarse_region_rows=coarse_region_rows,
            coarse_region_columns=coarse_region_columns,
            coarse_region_limit=coarse_region_limit,
        )
    if strategy != BLIND_TILE_SELECTION_STRATEGY:
        raise ValueError(
            f"Unknown tile selection strategy {strategy!r}; "
            f"expected one of {TILE_SELECTION_STRATEGIES!r}"
        )

    y_positions = sorted({tile[0] for tile in tile_origins})
    x_positions = sorted({tile[1] for tile in tile_origins})
    y_indices = {position: index for index, position in enumerate(y_positions)}
    x_indices = {position: index for index, position in enumerate(x_positions)}

    region_rows = max(
        1,
        min(
            len(y_positions),
            max_tiles,
            int(round(math.sqrt(max_tiles * len(y_positions) / len(x_positions)))),
        ),
    )
    region_columns = max(
        1,
        min(len(x_positions), max_tiles // region_rows),
    )

    scored_tiles: list[tuple[float, tuple[int, int, int, int]]] = []
    regions: dict[
        tuple[int, int], list[tuple[float, tuple[int, int, int, int]]]
    ] = {}
    for tile in tile_origins:
        y0, x0, y1, x1 = tile
        core = np.asarray(volume[:, y0:y1, x0:x1])
        score = _snr_weight(core, weight_cap=snr_weight_cap)
        scored = (score, tile)
        scored_tiles.append(scored)
        region = (
            min(
                region_rows - 1,
                y_indices[y0] * region_rows // len(y_positions),
            ),
            min(
                region_columns - 1,
                x_indices[x0] * region_columns // len(x_positions),
            ),
        )
        regions.setdefault(region, []).append(scored)

    selected: list[tuple[float, tuple[int, int, int, int]]] = []
    selected_tiles: set[tuple[int, int, int, int]] = set()
    for region in sorted(regions):
        best = min(regions[region], key=lambda item: (-item[0], item[1]))
        selected.append(best)
        selected_tiles.add(best[1])

    for scored in sorted(scored_tiles, key=lambda item: (-item[0], item[1])):
        if len(selected) >= max_tiles:
            break
        if scored[1] not in selected_tiles:
            selected.append(scored)
            selected_tiles.add(scored[1])

    selected.sort(key=lambda item: item[1])
    _log_selected_blind_tiles(selected, len(tile_origins), strategy)
    return [tile for _, tile in selected]


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _estimate_one_tile(
    idx: int,
    total_tiles: int,
    volume: np.ndarray,
    tile: tuple[int, int, int, int],
    psf_seed: np.ndarray,
    pad_xy: int,
    pad_z: int,
    n_iters: int,
    script_dir: Path,
    tmpdir: Path,
    backend: str,
    backend_lock: threading.Semaphore | None,
    matlab_threads: int,
    matlab_bin: str,
    matlab_timeout: int,
    blind_peak_normalization: str,
    blind_peak_gamma_max: float,
    blind_latent_update_period: int,
    snr_weight_cap: float,
    cupy_pool_trim_bytes: int | None = None,
) -> tuple[int, np.ndarray | None, float, str | None]:
    chunk_start = time.perf_counter()
    y0, x0, y1, x1 = tile
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"  Blind chunk {idx + 1}/{total_tiles} started at {started_at}: "
        f"tile=({y0}:{y1}, {x0}:{x1})",
        flush=True,
    )
    core = np.asarray(volume[:, y0:y1, x0:x1])
    weight = _snr_weight(core, weight_cap=snr_weight_cap)
    chunk = _extract_tile_with_halo(volume, y0, x0, y1, x1, pad_xy)
    chunk_shape = chunk.shape
    read_elapsed = time.perf_counter() - chunk_start

    chunk_path = tmpdir / f"chunk_{idx:04d}.tif"
    seed_path = tmpdir / f"seed_{idx:04d}.tif"
    psf_out_path = tmpdir / f"psf_out_{idx:04d}.tif"
    psf_chunk = None
    output_read_elapsed = 0.0
    if backend == "cupy":
        write_elapsed = 0.0
    else:
        write_start = time.perf_counter()
        _write_chunk(chunk, chunk_path)
        del chunk
        write_elapsed = time.perf_counter() - write_start

    backend_wait_elapsed = 0.0
    backend_elapsed = 0.0
    try:
        if backend not in ("matlab", "cupy"):
            raise ValueError(f"Unsupported blind backend: {backend}")
        backend_wait_start = time.perf_counter()
        backend_lock_ctx = backend_lock if backend == "matlab" else None
        if backend_lock_ctx is None:
            backend_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"  Blind chunk {idx + 1}/{total_tiles} {backend} started at "
                f"{backend_started_at}: wait={_format_seconds(backend_wait_elapsed)}, "
                f"input_shape={chunk_shape}",
                flush=True,
            )
            backend_start = time.perf_counter()
            if backend == "cupy":
                psf_chunk = _run_cupy_deconvblind_array(
                    chunk,
                    psf_seed,
                    n_iters,
                    pad_z,
                    blind_peak_normalization,
                    blind_peak_gamma_max,
                    blind_latent_update_period,
                    cupy_pool_trim_bytes,
                )
                del chunk
            else:
                _run_matlab_deconvblind(
                    chunk_path,
                    psf_seed,
                    seed_path,
                    psf_out_path,
                    n_iters,
                    pad_z,
                    script_dir,
                    matlab_bin,
                    matlab_threads,
                    matlab_timeout,
                )
            backend_elapsed = time.perf_counter() - backend_start
        else:
            with backend_lock_ctx:
                backend_wait_elapsed = time.perf_counter() - backend_wait_start
                backend_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"  Blind chunk {idx + 1}/{total_tiles} {backend} started at "
                    f"{backend_started_at}: wait={_format_seconds(backend_wait_elapsed)}, "
                    f"input_shape={chunk_shape}",
                    flush=True,
                )
                backend_start = time.perf_counter()
                _run_matlab_deconvblind(
                    chunk_path,
                    psf_seed,
                    seed_path,
                    psf_out_path,
                    n_iters,
                    pad_z,
                    script_dir,
                    matlab_bin,
                    matlab_threads,
                    matlab_timeout,
                )
                backend_elapsed = time.perf_counter() - backend_start
    except BaseException as exc:
        if backend == "cupy" and _is_cupy_out_of_memory(exc):
            raise
        if not isinstance(exc, RuntimeError):
            raise
        return idx, None, weight, str(exc)

    if psf_chunk is None:
        if not psf_out_path.exists():
            return idx, None, weight, f"{backend.upper()} produced no PSF output"

        output_read_start = time.perf_counter()
        psf_chunk = ensure_3d_volume(imread(str(psf_out_path))).astype(np.float32)
        output_read_elapsed = time.perf_counter() - output_read_start
    if psf_chunk.shape != psf_seed.shape:
        return (
            idx,
            None,
            weight,
            f"PSF shape {psf_chunk.shape} != seed shape {psf_seed.shape}",
        )

    total_elapsed = time.perf_counter() - chunk_start
    completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"  Blind chunk {idx + 1}/{total_tiles} completed at {completed_at}: "
        f"tile=({y0}:{y1}, {x0}:{x1}), "
        f"snr_weight={weight:.3g}, "
        f"read={_format_seconds(read_elapsed)}, "
        f"write={_format_seconds(write_elapsed)}, "
        f"backend_wait={_format_seconds(backend_wait_elapsed)}, "
        f"backend={_format_seconds(backend_elapsed)}, "
        f"output_read={_format_seconds(output_read_elapsed)}, "
        f"total={_format_seconds(total_elapsed)}",
        flush=True,
    )
    return idx, _normalise_psf(psf_chunk), weight, None


def _merge_weighted_psfs(
    psf_estimates: list[np.ndarray],
    psf_weights: list[float],
    snr_weight_cap: float,
) -> np.ndarray:
    if not psf_estimates:
        raise RuntimeError("Cannot merge an empty PSF estimate list")
    stack = np.stack(psf_estimates, axis=0)
    weights = np.asarray(psf_weights, dtype=np.float32)
    max_weight = snr_weight_cap if snr_weight_cap > 0 else None
    weights = np.clip(weights, 1e-3, max_weight)
    weights = weights / weights.sum()
    merged = np.tensordot(weights, stack, axes=(0, 0)).astype(np.float32)
    return _normalise_psf(merged)


def _psf_shape_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=np.float32).ravel()
    second_values = np.asarray(second, dtype=np.float32).ravel()
    if first_values.shape != second_values.shape:
        raise ValueError(
            f"PSF shapes do not match for NCC: {np.asarray(first).shape} "
            f"vs {np.asarray(second).shape}"
        )
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    if denominator <= 0.0:
        return 1.0 if np.allclose(first_values, second_values) else 0.0
    return float(np.dot(first_values, second_values) / denominator)


def _select_adaptive_scout_tiles(
    origins: list[tuple[int, int, int, int]],
    scout_estimates: list[np.ndarray],
    scout_weights: list[float],
    *,
    keep_tiles: int,
    snr_weight_cap: float,
) -> tuple[list[tuple[int, int, int, int]], list[float], np.ndarray]:
    if not scout_estimates:
        raise RuntimeError("Adaptive blind PSF scout produced no estimates")
    if len(scout_estimates) != len(scout_weights):
        raise ValueError("Adaptive scout estimates and weights must have the same length")
    if len(origins) < len(scout_estimates):
        raise ValueError("Adaptive scout received more estimates than tile origins")

    keep_count = min(max(1, int(keep_tiles)), len(scout_estimates))
    max_weight = snr_weight_cap if snr_weight_cap > 0 else None
    scored: list[tuple[float, float, tuple[int, int, int, int], int]] = []
    for index, (tile, psf, weight) in enumerate(
        zip(origins, scout_estimates, scout_weights)
    ):
        similarities = [
            _psf_shape_similarity(psf, other)
            for other_index, other in enumerate(scout_estimates)
            if other_index != index
        ]
        consensus_score = float(np.mean(similarities)) if similarities else 1.0
        clipped_weight = float(np.clip(float(weight), 1e-3, max_weight))
        scored.append((consensus_score, clipped_weight, tile, index))

    selected = sorted(scored, key=lambda item: (-item[0], -item[1], item[2]))[
        :keep_count
    ]
    selected.sort(key=lambda item: item[3])
    selected_indices = [index for _, _, _, index in selected]
    kept_origins = [tile for _, _, tile, _ in selected]
    kept_weights = [float(scout_weights[index]) for index in selected_indices]
    scout_seed = _merge_weighted_psfs(
        [scout_estimates[index] for index in selected_indices],
        [1.0] * len(selected_indices),
        snr_weight_cap,
    )
    return kept_origins, kept_weights, scout_seed


def _run_blind_tile_adaptive_cupyx_pass(
    volume: np.ndarray,
    psf_seed: np.ndarray,
    tile_origins: list[tuple[int, int, int, int]],
    *,
    pad_xy: int,
    pad_z: int,
    n_iters: int,
    script_dir: Path,
    max_workers: int,
    prefetch_chunks: int,
    matlab_workers: int,
    matlab_threads: int,
    matlab_bin: str,
    matlab_timeout: int,
    blind_peak_normalization: str,
    blind_peak_gamma_max: float,
    blind_latent_update_period: int,
    snr_weight_cap: float,
    cupy_pool_trim_bytes: int | None,
    adaptive_scout_iters: int,
    adaptive_keep_tiles: int,
) -> tuple[list[np.ndarray], list[float]]:
    scout_iters = min(max(1, int(adaptive_scout_iters)), max(1, int(n_iters)))
    refine_iters = max(0, int(n_iters) - scout_iters)
    scout_estimates, scout_weights = _run_blind_tile_pass(
        volume,
        psf_seed,
        tile_origins,
        pad_xy=pad_xy,
        pad_z=pad_z,
        n_iters=scout_iters,
        script_dir=script_dir,
        blind_backend="cupy",
        max_workers=max_workers,
        prefetch_chunks=prefetch_chunks,
        matlab_workers=matlab_workers,
        matlab_threads=matlab_threads,
        matlab_bin=matlab_bin,
        matlab_timeout=matlab_timeout,
        blind_peak_normalization=blind_peak_normalization,
        blind_peak_gamma_max=blind_peak_gamma_max,
        blind_latent_update_period=blind_latent_update_period,
        snr_weight_cap=snr_weight_cap,
        cupy_pool_trim_bytes=cupy_pool_trim_bytes,
        cupy_fft_engine="cupyx",
        adaptive_scout_iters=adaptive_scout_iters,
        adaptive_keep_tiles=adaptive_keep_tiles,
    )
    kept_origins, kept_weights, scout_seed = _select_adaptive_scout_tiles(
        tile_origins,
        scout_estimates,
        scout_weights,
        keep_tiles=adaptive_keep_tiles,
        snr_weight_cap=snr_weight_cap,
    )
    total_weight = float(np.sum(kept_weights, dtype=np.float64))
    print(
        f"  Adaptive CuPy blind PSF scout kept {len(kept_origins)}/"
        f"{len(tile_origins)} tile(s); scout_iters={scout_iters}; "
        f"final_iters={refine_iters}; final_engine=cupyx",
        flush=True,
    )
    if refine_iters == 0:
        return [scout_seed], [float(np.sum(scout_weights, dtype=np.float64))]

    final_estimates, final_weights = _run_blind_tile_pass(
        volume,
        scout_seed,
        kept_origins,
        pad_xy=pad_xy,
        pad_z=pad_z,
        n_iters=refine_iters,
        script_dir=script_dir,
        blind_backend="cupy",
        max_workers=max_workers,
        prefetch_chunks=prefetch_chunks,
        matlab_workers=matlab_workers,
        matlab_threads=matlab_threads,
        matlab_bin=matlab_bin,
        matlab_timeout=matlab_timeout,
        blind_peak_normalization=blind_peak_normalization,
        blind_peak_gamma_max=blind_peak_gamma_max,
        blind_latent_update_period=blind_latent_update_period,
        snr_weight_cap=snr_weight_cap,
        cupy_pool_trim_bytes=cupy_pool_trim_bytes,
        cupy_fft_engine="cupyx",
        adaptive_scout_iters=adaptive_scout_iters,
        adaptive_keep_tiles=adaptive_keep_tiles,
    )
    merged = _merge_weighted_psfs(final_estimates, final_weights, snr_weight_cap)
    return [merged], [total_weight]


def _run_blind_tile_pass(
    volume: np.ndarray,
    psf_seed: np.ndarray,
    tile_origins: list[tuple[int, int, int, int]],
    *,
    pad_xy: int,
    pad_z: int,
    n_iters: int,
    script_dir: Path,
    blind_backend: str,
    max_workers: int,
    prefetch_chunks: int,
    matlab_workers: int,
    matlab_threads: int,
    matlab_bin: str,
    matlab_timeout: int,
    blind_peak_normalization: str,
    blind_peak_gamma_max: float,
    blind_latent_update_period: int,
    snr_weight_cap: float,
    cupy_pool_trim_bytes: int | None = None,
    cupy_fft_engine: str = "cupyx",
    adaptive_scout_iters: int = DEFAULT_ADAPTIVE_SCOUT_ITERS,
    adaptive_keep_tiles: int = DEFAULT_ADAPTIVE_KEEP_TILES,
) -> tuple[list[np.ndarray], list[float]]:
    if blind_backend == "cupy" and cupy_fft_engine == "scout":
        return _run_blind_tile_adaptive_cupyx_pass(
            volume,
            psf_seed,
            tile_origins,
            pad_xy=pad_xy,
            pad_z=pad_z,
            n_iters=n_iters,
            script_dir=script_dir,
            max_workers=max_workers,
            prefetch_chunks=prefetch_chunks,
            matlab_workers=matlab_workers,
            matlab_threads=matlab_threads,
            matlab_bin=matlab_bin,
            matlab_timeout=matlab_timeout,
            blind_peak_normalization=blind_peak_normalization,
            blind_peak_gamma_max=blind_peak_gamma_max,
            blind_latent_update_period=blind_latent_update_period,
            snr_weight_cap=snr_weight_cap,
            cupy_pool_trim_bytes=cupy_pool_trim_bytes,
            adaptive_scout_iters=adaptive_scout_iters,
            adaptive_keep_tiles=adaptive_keep_tiles,
        )
    if blind_backend == "cupy" and cupy_fft_engine != "cupyx":
        raise ValueError(
            f"cupy_fft_engine must be 'cupyx' or 'scout', got {cupy_fft_engine!r}"
        )

    psf_estimates: list[np.ndarray] = []
    psf_weights: list[float] = []
    failure_details: list[str] = []
    failed_chunks = 0
    completed_chunks = 0
    prefetch_limit = prefetch_chunks if prefetch_chunks > 0 else max_workers
    heartbeat_seconds = 60.0
    last_heartbeat = time.perf_counter()

    try:
        with tempfile.TemporaryDirectory(prefix="psf_est_") as tmpdir_value:
            tmpdir = Path(tmpdir_value)
            backend_lock = (
                threading.Semaphore(matlab_workers)
                if blind_backend == "matlab"
                else None
            )
            next_idx = 0
            pending: set[futures.Future] = set()

            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                while next_idx < len(tile_origins) or pending:
                    submitted_before = next_idx
                    while (
                        next_idx < len(tile_origins)
                        and len(pending) < prefetch_limit
                    ):
                        pending.add(
                            executor.submit(
                                _estimate_one_tile,
                                next_idx,
                                len(tile_origins),
                                volume,
                                tile_origins[next_idx],
                                psf_seed,
                                pad_xy,
                                pad_z,
                                n_iters,
                                script_dir,
                                tmpdir,
                                blind_backend,
                                backend_lock,
                                matlab_threads,
                                matlab_bin,
                                matlab_timeout,
                                blind_peak_normalization,
                                blind_peak_gamma_max,
                                blind_latent_update_period,
                                snr_weight_cap,
                                cupy_pool_trim_bytes,
                            )
                        )
                        next_idx += 1
                    if next_idx > submitted_before:
                        print(
                            f"  Submitted blind chunks {submitted_before + 1}-{next_idx}/"
                            f"{len(tile_origins)}; pending={len(pending)}, "
                            f"completed={completed_chunks}, failed={failed_chunks}",
                            flush=True,
                        )

                    done, pending = futures.wait(
                        pending,
                        timeout=heartbeat_seconds,
                        return_when=futures.FIRST_COMPLETED,
                    )
                    if not done:
                        now = time.perf_counter()
                        if now - last_heartbeat >= heartbeat_seconds:
                            print(
                                f"  Blind PSF heartbeat: submitted={next_idx}/"
                                f"{len(tile_origins)}, completed={completed_chunks}, "
                                f"failed={failed_chunks}, pending={len(pending)}",
                                flush=True,
                            )
                            last_heartbeat = now
                        continue

                    for future in done:
                        try:
                            idx, psf_chunk, weight, error = future.result()
                        except BaseException:
                            for pending_future in pending:
                                pending_future.cancel()
                            raise
                        completed_chunks += 1
                        if error:
                            failed_chunks += 1
                            failure_details.append(f"chunk {idx}: {error}")
                            if "initial PSF must have at least one non-zero element" in error:
                                raise RuntimeError(
                                    "MATLAB read the PSF seed as all zeros. "
                                    "The seed TIFF writer is incompatible with MATLAB."
                                )
                            if failed_chunks >= 3 and not psf_estimates:
                                for pending_future in pending:
                                    pending_future.cancel()
                                detail = "\n\n".join(failure_details[:3])
                                raise RuntimeError(
                                    "First three chunks failed during PSF estimation; "
                                    "aborting instead of submitting every tile to the backend.\n\n"
                                    f"{detail}"
                                )
                            print(
                                f"  WARNING: chunk {idx} failed, skipping. {error}",
                                flush=True,
                            )
                            continue
                        if psf_chunk is not None:
                            psf_estimates.append(psf_chunk)
                            psf_weights.append(weight)
    finally:
        if blind_backend == "cupy":
            _clear_cupy_blind_memory()

    return psf_estimates, psf_weights

def estimate_psf_from_chunks(
    image_path: str | Path,
    psf_seed: np.ndarray,
    n_iters: int = 10,
    blind_backend: str = DEFAULT_BLIND_BACKEND,
    chunk_xy: int = DEFAULT_BLIND_CHUNK_XY,
    pad_xy: int = 32,
    pad_z: int = 20,
    script_dir: str | Path | None = None,
    max_workers: int = 0,
    prefetch_chunks: int = 0,
    vram_gb: float | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    matlab_threads: int = 1,
    matlab_workers: int = 1,
    matlab_bin: str = "matlab",
    matlab_timeout: int = 1800,
    snr_weight_cap: float = DEFAULT_SNR_WEIGHT_CAP,
    blind_peak_normalization: str = DEFAULT_BLIND_PEAK_NORMALIZATION,
    blind_peak_gamma_max: float = DEFAULT_BLIND_PEAK_GAMMA_MAX,
    blind_latent_update_period: int = DEFAULT_BLIND_LATENT_UPDATE_PERIOD,
    blind_z_slices: int = DEFAULT_BLIND_Z_SLICES,
    blind_max_tiles: int = DEFAULT_BLIND_MAX_TILES,
    cupy_fft_engine: str = "scout",
    adaptive_scout_iters: int = DEFAULT_ADAPTIVE_SCOUT_ITERS,
    adaptive_keep_tiles: int = DEFAULT_ADAPTIVE_KEEP_TILES,
    tile_selection_strategy: str = BLIND_TILE_SELECTION_STRATEGY,
    coarse_region_rows: int = DEFAULT_COARSE_REGION_ROWS,
    coarse_region_columns: int = DEFAULT_COARSE_REGION_COLUMNS,
    coarse_region_limit: int = DEFAULT_COARSE_REGION_LIMIT,
) -> np.ndarray:
    """
    Estimate a PSF by running deconvblind-like estimation on spatial XY chunks of
    the first deskewed volume and merging per-chunk estimates by SNR-weighted
    mean.

    Parameters
    ----------
    image_path  : path to the deskewed input TIFF or OME-Zarr (full Z stack, 3-D).
    psf_seed    : initial PSF guess, float32 numpy array (nz_psf, ny_psf, nx_psf).
                  Typically the output of generate_theoretical_psf().
    blind_backend : 'matlab' or 'cupy'. 'matlab' runs MATLAB deconvblind.
    n_iters       : number of blind iterations per chunk.
    chunk_xy    : XY tile size.  <= 0 chooses a VRAM-aware size.
    pad_xy      : XY halo per edge before deconvblind. Interior tiles include
                  real neighboring pixels; only image borders are reflect-padded.
    pad_z       : Z halo per edge before blind estimation.
    script_dir  : directory containing readtiffstack.m / writetiffstack.m.
                  Defaults to the directory of this script.

    Returns
    -------
    float32 numpy array of shape matching psf_seed, normalised to sum = 1.
    """
    image_path = Path(image_path)
    script_dir = Path(script_dir) if script_dir else Path(__file__).parent

    volume = open_psf_source(image_path)  # (nz, ny, nx)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}")

    original_shape = volume.shape
    z_window, z_window_detail = select_blind_z_window(volume, blind_z_slices)
    z_start = z_window.start
    z_stop = z_window.stop
    volume = volume[z_window]
    nz, ny, nx = volume.shape
    original_psf_seed_shape = psf_seed.shape
    psf_seed = adapt_psf_seed_to_volume(psf_seed, volume.shape)
    if psf_seed.shape != original_psf_seed_shape:
        print(
            f"  Adapted PSF seed shape from {original_psf_seed_shape} to "
            f"{psf_seed.shape} to fit blind volume Z={nz}.",
            flush=True,
        )
    requested_workers = max_workers
    cpu_workers = resolve_worker_count(requested_workers)
    blind_backend, cupy_fft_engine = normalize_blind_backend(
        blind_backend,
        cupy_fft_engine,
    )
    matlab_threads = min(2, max(1, matlab_threads))
    matlab_workers = max(1, matlab_workers)
    matlab_timeout = max(0, matlab_timeout)
    if blind_backend not in BLIND_BACKENDS:
        raise ValueError(
            f"Unsupported blind backend '{blind_backend}'. Expected 'matlab' or 'cupy'."
        )
    pad_z = max(0, pad_z)
    if nz == 1 and pad_z > 0:
        print(
            f"  Single-slice blind volume detected; disabling Z padding "
            f"(requested pad_z={pad_z}).",
            flush=True,
        )
        pad_z = 0
    snr_weight_cap = max(0.0, snr_weight_cap)
    blind_latent_update_period = max(1, int(blind_latent_update_period))
    if cupy_fft_engine not in ("cupyx", "scout"):
        raise ValueError(
            f"cupy_fft_engine must be 'cupyx' or 'scout', got {cupy_fft_engine!r}"
        )
    adaptive_scout_iters = max(1, int(adaptive_scout_iters))
    adaptive_keep_tiles = max(1, int(adaptive_keep_tiles))
    tile_selection_strategy = str(tile_selection_strategy)
    if tile_selection_strategy not in TILE_SELECTION_STRATEGIES:
        raise ValueError(
            f"tile_selection_strategy must be one of {TILE_SELECTION_STRATEGIES!r}, "
            f"got {tile_selection_strategy!r}"
        )
    coarse_region_rows = max(1, int(coarse_region_rows))
    coarse_region_columns = max(1, int(coarse_region_columns))
    coarse_region_limit = max(1, int(coarse_region_limit))
    if blind_max_tiles < 0:
        raise ValueError(
            f"blind_max_tiles cannot be negative, got {blind_max_tiles}"
        )
    chunk_xy = resolve_chunk_xy(
        chunk_xy,
        volume.shape,
        volume.dtype,
        overlap_xy=pad_xy,
        vram_gb=vram_gb,
        workers=cpu_workers,
        min_xy=max(128, psf_seed.shape[-1]),
        max_xy=min(DEFAULT_BLIND_CHUNK_XY, ny, nx),
    )
    cupy_sizing_detail = None
    min_cupy_chunk_xy = min(
        ny,
        nx,
        max(
            64,
            int(
                math.ceil(max(psf_seed.shape[-2:]) / BLIND_CHUNK_ALIGNMENT)
            ) * BLIND_CHUNK_ALIGNMENT,
        ),
    )
    if blind_backend == "cupy":
        chunk_xy, cupy_sizing_detail = resolve_cupy_blind_chunk_xy(
            chunk_xy,
            volume.shape,
            psf_seed.shape,
            pad_xy,
            pad_z,
            vram_gb=vram_gb,
        )
    cupy_pool_trim_bytes = None
    if blind_backend == "cupy":
        cupy_vram_bytes = (
            int(vram_gb * (1024 ** 3))
            if vram_gb and vram_gb > 0
            else detect_vram_bytes()
        )
        if cupy_vram_bytes:
            cupy_pool_trim_bytes = int(
                cupy_vram_bytes * DEFAULT_CUPY_VRAM_FRACTION
            )
    max_workers, worker_detail = resolve_blind_worker_count(
        requested_workers,
        cpu_workers,
        volume.shape,
        volume.dtype,
        chunk_xy,
        pad_xy,
    )

    print(
        f"  Volume shape: {original_shape}; blind_volume_shape={volume.shape}; "
        f"{z_window_detail}; resolved_chunk_xy={chunk_xy}",
        flush=True,
    )
    backend_executor_workers = resolve_backend_executor_workers(
        blind_backend,
        blind_workers=max_workers,
        matlab_workers=matlab_workers,
    )
    if blind_backend == "cupy":
        if backend_executor_workers != max_workers:
            print(
                f"  CuPy backend clamps blind_workers from {max_workers} to 1 "
                "for one allocated GPU.",
                flush=True,
            )
            worker_detail = f"{worker_detail}; cupy_gpu_cap=1"
        max_workers = backend_executor_workers
        print(
            "  Blind backend configured as 'cupy'; running each chunk in an "
            "in-process direct CuPy array path.",
            flush=True,
        )
        print(
            f"  CuPy FFT workspace sizing: chunk_xy={chunk_xy}; "
            f"{cupy_sizing_detail}",
            flush=True,
        )
        if cupy_pool_trim_bytes:
            print(
                "  CuPy memory-pool trim threshold: "
                f"{cupy_pool_trim_bytes / (1024 ** 3):.1f}GiB retained "
                "(dynamic per visible GPU budget).",
                flush=True,
            )
        matlab_workers = min(matlab_workers, max_workers)
    else:
        blind_backend = "matlab"
        if backend_executor_workers > max_workers:
            print(
                f"  MATLAB backend expands the blind executor from "
                f"{max_workers} to {backend_executor_workers} workers to honor "
                f"matlab_workers={matlab_workers}.",
                flush=True,
            )
            worker_detail = (
                f"{worker_detail}; matlab_backend_floor={matlab_workers}"
            )
        max_workers = backend_executor_workers
    print(
        f"  Blind worker selection: executor_workers={max_workers} "
        f"({worker_detail}); "
        f"backend='{blind_backend}', backend_workers={matlab_workers}, "
        f"matlab_threads={matlab_threads}, "
        f"matlab_timeout={matlab_timeout}s, snr_weight_cap={snr_weight_cap:g}, "
        f"blind_latent_update_period={blind_latent_update_period}",
        flush=True,
    )
    cache_path = None
    cache_root = None
    if use_cache:
        cache_root = _resolve_psf_cache_root(image_path, cache_dir)
    def _cache_path_for_chunk(resolved_chunk_xy: int) -> Path | None:
        if cache_root is None:
            return None
        cache_key = _psf_cache_key(
            image_path,
            psf_seed,
            n_iters,
            resolved_chunk_xy,
            pad_xy,
            pad_z,
            script_dir,
            "snr_weighted_mean",
            snr_weight_cap,
            (z_start, z_stop),
            blind_backend,
            blind_peak_normalization,
            blind_peak_gamma_max,
            blind_latent_update_period,
            blind_max_tiles,
            tile_selection_strategy,
            cupy_fft_engine,
            adaptive_scout_iters,
            adaptive_keep_tiles,
            coarse_region_rows,
            coarse_region_columns,
            coarse_region_limit,
        )
        return cache_root / f"estimated_psf_{cache_key}.tif"

    cache_path = _cache_path_for_chunk(chunk_xy)
    if cache_path is not None:
        if cache_path.exists():
            print(f"Using cached PSF estimate: {cache_path}", flush=True)
            return _normalise_psf(imread(str(cache_path)))

    while True:
        candidate_tile_origins = _tile_origins(ny, nx, chunk_xy)
        tile_origins = _select_representative_tiles(
            volume,
            candidate_tile_origins,
            max_tiles=blind_max_tiles,
            snr_weight_cap=snr_weight_cap,
            strategy=tile_selection_strategy,
            coarse_region_rows=coarse_region_rows,
            coarse_region_columns=coarse_region_columns,
            coarse_region_limit=coarse_region_limit,
        )
        print(
            f"  Processing {len(tile_origins)} chunk(s) of size "
            f"(nz={nz}, xy<={chunk_xy}, halo_xy={pad_xy}, pad_z={pad_z}, "
            f"executor_workers={max_workers}, backend='{blind_backend}', "
            f"backend_workers={matlab_workers}, matlab_threads={matlab_threads}, "
            f"matlab_timeout={matlab_timeout}s, snr_weight_cap={snr_weight_cap:g}, "
            f"blind_latent_update_period={blind_latent_update_period}, "
            f"cupy_fft_engine={cupy_fft_engine}, "
            f"adaptive_scout_iters={adaptive_scout_iters}, "
            f"adaptive_keep_tiles={adaptive_keep_tiles})...",
            flush=True,
        )
        try:
            psf_estimates, psf_weights = _run_blind_tile_pass(
                volume,
                psf_seed,
                tile_origins,
                pad_xy=pad_xy,
                pad_z=pad_z,
                n_iters=n_iters,
                script_dir=script_dir,
                blind_backend=blind_backend,
                max_workers=max_workers,
                prefetch_chunks=prefetch_chunks,
                matlab_workers=matlab_workers,
                matlab_threads=matlab_threads,
                matlab_bin=matlab_bin,
                matlab_timeout=matlab_timeout,
                blind_peak_normalization=blind_peak_normalization,
                blind_peak_gamma_max=blind_peak_gamma_max,
                blind_latent_update_period=blind_latent_update_period,
                cupy_pool_trim_bytes=cupy_pool_trim_bytes,
                snr_weight_cap=snr_weight_cap,
                cupy_fft_engine=cupy_fft_engine,
                adaptive_scout_iters=adaptive_scout_iters,
                adaptive_keep_tiles=adaptive_keep_tiles,
            )
            break
        except BaseException as exc:
            reduced_chunk_xy = _next_smaller_blind_chunk_xy(
                chunk_xy, min_cupy_chunk_xy
            )
            if (
                blind_backend != "cupy"
                or not _is_cupy_out_of_memory(exc)
                or reduced_chunk_xy >= chunk_xy
            ):
                raise
            print(
                f"  WARNING: CuPy exhausted VRAM at chunk_xy={chunk_xy}; "
                f"discarding the partial pass and retrying all tiles with "
                f"chunk_xy={reduced_chunk_xy}.",
                flush=True,
            )
            chunk_xy = reduced_chunk_xy
            cache_path = _cache_path_for_chunk(chunk_xy)
            if cache_path is not None and cache_path.exists():
                print(f"Using cached PSF estimate: {cache_path}", flush=True)
                return _normalise_psf(imread(str(cache_path)))

    if not psf_estimates:
        raise RuntimeError(
            "All chunks failed during PSF estimation. "
            "Check backend logs above and ensure blind estimation is available."
        )

    print(f"Merging {len(psf_estimates)} PSF estimate(s) via SNR-weighted mean...", flush=True)
    merged = _merge_weighted_psfs(psf_estimates, psf_weights, snr_weight_cap)

    if cache_path is not None:
        imwrite(str(cache_path), merged)
        print(f"Cached PSF estimate: {cache_path}", flush=True)

    return merged


# ---------------------------------------------------------------------------
# CLI (for standalone testing)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate PSF from a deskewed TIFF or OME-Zarr using chunked deconvblind."
    )
    parser.add_argument("--image_path",  required=True)
    parser.add_argument("--output_path", required=True,
                        help="Where to save the merged PSF TIFF.")
    parser.add_argument("--blind_backend", default=DEFAULT_BLIND_BACKEND, choices=("matlab", "cupy", "scout", "cupyx"),
                        help="Backend for blind estimation: cupy (default) or matlab. Legacy scout/cupyx values select cupy with that CuPy mode.")
    parser.add_argument("--n_iters",    type=int,   default=DEFAULT_BLIND_ITERS)
    parser.add_argument("--chunk_xy",   type=int,   default=DEFAULT_BLIND_CHUNK_XY,
                        help="XY tile size. <=0 auto-sizes from available VRAM.")
    parser.add_argument("--blind_max_tiles", type=int, default=DEFAULT_BLIND_MAX_TILES,
                        help="Maximum representative PSF tiles; 0 processes the full grid.")
    parser.add_argument("--cupy_fft_engine", choices=("cupyx", "scout"), default="scout",
                        help="CuPy PSF estimation mode: scout filters tiles before final CuPy refinement; cupyx runs all selected tiles directly.")
    parser.add_argument("--adaptive_scout_iters", type=int, default=DEFAULT_ADAPTIVE_SCOUT_ITERS,
                        help="Short blind-RL iterations used by the scout pass.")
    parser.add_argument("--adaptive_keep_tiles", type=int, default=DEFAULT_ADAPTIVE_KEEP_TILES,
                        help="Number of scout-approved tiles to finish with the full CuPy pass.")
    parser.add_argument("--tile_selection_strategy", choices=TILE_SELECTION_STRATEGIES, default=BLIND_TILE_SELECTION_STRATEGY,
                        help="Strategy for selecting representative blind PSF tiles.")
    parser.add_argument("--coarse_region_rows", type=int, default=DEFAULT_COARSE_REGION_ROWS,
                        help="Coarse region row count for coarse_to_fine_snr tile selection.")
    parser.add_argument("--coarse_region_columns", type=int, default=DEFAULT_COARSE_REGION_COLUMNS,
                        help="Coarse region column count for coarse_to_fine_snr tile selection.")
    parser.add_argument("--coarse_region_limit", type=int, default=DEFAULT_COARSE_REGION_LIMIT,
                        help="Maximum coarse regions considered by coarse_to_fine_snr tile selection.")
    parser.add_argument("--pad_xy",     type=int,   default=32,
                        help="XY halo per edge before deconvblind (pixels).")
    parser.add_argument("--pad_z",      type=int,   default=20,
                        help="Z halo per edge before deconvblind (pixels).")
    parser.add_argument("--blind_workers", type=int, default=1,
                        help="Concurrent MATLAB deconvblind chunks. <=0 uses CPU affinity, falling back to 32.")
    parser.add_argument("--matlab_threads", type=int, default=1,
                        help="Threads per MATLAB deconvblind process; clamped to 1 or 2.")
    parser.add_argument("--matlab_workers", type=int, default=1,
                        help="Concurrent MATLAB deconvblind processes; default 1 avoids MATLAB orchestration hangs.")
    parser.add_argument("--blind_peak_normalization", default=DEFAULT_BLIND_PEAK_NORMALIZATION,
                        choices=("none", "gamma", "unit"),
                        help="Optional peak normalization behavior used by the CuPy backend.")
    parser.add_argument("--blind_peak_gamma_max", type=float, default=DEFAULT_BLIND_PEAK_GAMMA_MAX,
                        help="Maximum gamma scaling value used when blind_peak_normalization='gamma'.")
    parser.add_argument("--blind_latent_update_period", type=int, default=DEFAULT_BLIND_LATENT_UPDATE_PERIOD,
                        help="Update latent image every N blind iterations; 1 preserves full alternating updates.")
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
    parser.add_argument("--vram_gb", type=float, default=None,
                        help="Override detected free VRAM in GiB for auto chunk sizing.")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory for cached PSF estimates.")
    parser.add_argument("--no_psf_cache", action="store_true",
                        help="Disable reuse of cached blind PSF estimates.")
    parser.add_argument("--script_dir", default=str(Path(__file__).parent))

    # Optical/acquisition parameters for the PSF seed
    parser.add_argument("--na",         type=float, default=None)
    parser.add_argument("--detection_na", type=float, default=None)
    parser.add_argument("--illumination_na", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ni",         type=float, default=None)
    parser.add_argument("--ns",         type=float, default=None)
    parser.add_argument("--ni0",        type=float, default=None)
    parser.add_argument("--tg",         type=float, default=None)
    parser.add_argument("--tg0",        type=float, default=None)
    parser.add_argument("--ng",         type=float, default=None)
    parser.add_argument("--ng0",        type=float, default=None)
    parser.add_argument("--ti0",        type=float, default=None)
    parser.add_argument("--oversample_factor", type=int, default=3)
    parser.add_argument("--psf_model", choices=("vectorial", "scalar", "gaussian"), default="vectorial")
    parser.add_argument("--camera_pixel_size", type=float, default=None)
    parser.add_argument("--magnification", type=float, default=None)
    parser.add_argument("--dxy",        type=float, default=None)
    parser.add_argument("--dz",         type=float, default=None)
    parser.add_argument("--psf_size_z", type=int,   default=61)
    parser.add_argument("--psf_size_xy",type=int,   default=128)
    parser.add_argument("--background", type=float, default=0.0)
    args = parser.parse_args()
    args.blind_backend, args.cupy_fft_engine = normalize_blind_backend(
        args.blind_backend,
        args.cupy_fft_engine,
    )

    dxy = resolve_dxy(args.dxy, args.camera_pixel_size, args.magnification)
    psf_seed = generate_theoretical_psf(
        na=args.na,
        detection_na=args.detection_na,
        illumination_na=args.illumination_na,
        wavelength=args.wavelength,
        ni=args.ni,
        ns=args.ns,
        ni0=args.ni0,
        tg=args.tg,
        tg0=args.tg0,
        ng=args.ng,
        ng0=args.ng0,
        ti0=args.ti0,
        oversample_factor=args.oversample_factor,
        psf_model=args.psf_model,
        dxy=dxy,
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
        pad_z=args.pad_z,
        script_dir=args.script_dir,
        max_workers=args.blind_workers,
        blind_backend=args.blind_backend,
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
        blind_max_tiles=args.blind_max_tiles,
        blind_z_slices=args.blind_z_slices,
        cupy_fft_engine=args.cupy_fft_engine,
        adaptive_scout_iters=args.adaptive_scout_iters,
        adaptive_keep_tiles=args.adaptive_keep_tiles,
        tile_selection_strategy=args.tile_selection_strategy,
        coarse_region_rows=args.coarse_region_rows,
        coarse_region_columns=args.coarse_region_columns,
        coarse_region_limit=args.coarse_region_limit,
    )

    imwrite(args.output_path, merged_psf)
    print(f"Merged PSF saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
