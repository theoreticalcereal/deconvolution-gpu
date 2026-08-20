# decon_wrapper.py
# Dask-orchestrated GPU deconvolution with blind PSF estimation.
#
# PSF resolution:
#   Generate a theoretical Gibson-Lanni PSF from the optical parameters, use it
#   only as the starting guess for chunked MATLAB deconvblind, then merge the
#   recovered per-chunk blind PSFs with SNR weighting and save estimated_psf.tif.
#
# Deconvolution:
#   Petakit-compatible accelerated Richardson-Lucy runs as one whole-volume
#   CuPy FFT when VRAM permits, with PSF-sized 3-D overlap as a fallback.

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
import yaml
from tifffile import imwrite

from petakit_rl import restore_uint16_cupy, restore_uint16_petakit_cpu

from psf_estimation import (
    DEFAULT_BLIND_CHUNK_XY,
    DEFAULT_BLIND_ITERS,
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
from psf_modes import generate_psf_seed, load_fixed_psf, load_psf_seed

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

IMAGE_AGGRESSIVENESS_PRESETS = {
    "low": {
        "blind_backend": "cupy",
        "cupy_fft_engine": "scout",
        "decon_backend": "cupy",
    },
    "medium": {
        "blind_backend": "cupy",
        "cupy_fft_engine": "cupyx",
        "decon_backend": "cupy",
        "blind_max_tiles": 0,
    },
    "high": {
        "blind_backend": "matlab",
        "cupy_fft_engine": "scout",
        "decon_backend": "petakit",
    },
}

MICROSCOPE_PROFILES = {
    "upright_aslm_36x_ri_1_33": {
        "label": "Upright ASLM — 36x — RI 1.33",
        "detection_na": 0.643,
        "illumination_na": 0.643,
        "magnification": 36.0,
        "dxy": 0.181,
        "ri": 1.33,
        "light_sheet_angle": 45.0,
    },
    "benchtop_mesospim_4x_ri_1_56": {
        "label": "BenchTop MesoSPIM — 4x — RI 1.56",
        "detection_na": 0.25,
        "illumination_na": 0.1,
        "magnification": 4.0,
        "dxy": 1.609,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "benchtop_mesospim_10x_ri_1_56": {
        "label": "BenchTop MesoSPIM — 10x — RI 1.56",
        "detection_na": 0.431,
        "illumination_na": 0.1,
        "magnification": 10.0,
        "dxy": 0.65,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "benchtop_mesospim_10x_ri_1_52": {
        "label": "BenchTop MesoSPIM — 10x — RI 1.52",
        "detection_na": 0.42,
        "illumination_na": 0.1,
        "magnification": 10.0,
        "dxy": 0.65,
        "ri": 1.52,
        "light_sheet_angle": 0.0,
    },
    "ctaslm_v3_50x_ri_1_56": {
        "label": "ctASLM v3 — 50x — RI 1.56",
        "detection_na": 1.2,
        "illumination_na": 0.7,
        "magnification": 50.0,
        "dxy": 0.128,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "ctaslm_v3_50x_ri_1_52": {
        "label": "ctASLM v3 — 50x — RI 1.52",
        "detection_na": 1.2,
        "illumination_na": 0.7,
        "magnification": 50.0,
        "dxy": 0.128,
        "ri": 1.52,
        "light_sheet_angle": 0.0,
    },
    "multiscale_low_res_0_63x_ri_1_56": {
        "label": "Multiscale - Low Res — 0.63x — RI 1.56",
        "detection_na": 0.25,
        "illumination_na": 0.1,
        "magnification": 0.63,
        "dxy": 9.7,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "multiscale_low_res_1x_ri_1_56": {
        "label": "Multiscale - Low Res — 1x — RI 1.56",
        "detection_na": 0.25,
        "illumination_na": 0.1,
        "magnification": 1.0,
        "dxy": 6.38,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "multiscale_low_res_2x_ri_1_56": {
        "label": "Multiscale - Low Res — 2x — RI 1.56",
        "detection_na": 0.25,
        "illumination_na": 0.1,
        "magnification": 2.0,
        "dxy": 3.14,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "multiscale_low_res_3x_ri_1_56": {
        "label": "Multiscale - Low Res — 3x — RI 1.56",
        "detection_na": 0.25,
        "illumination_na": 0.1,
        "magnification": 3.0,
        "dxy": 2.12,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "multiscale_low_res_4x_ri_1_56": {
        "label": "Multiscale - Low Res — 4x — RI 1.56",
        "detection_na": 0.25,
        "illumination_na": 0.1,
        "magnification": 4.0,
        "dxy": 1.609,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "multiscale_low_res_5x_ri_1_56": {
        "label": "Multiscale - Low Res — 5x — RI 1.56",
        "detection_na": 0.25,
        "illumination_na": 0.1,
        "magnification": 5.0,
        "dxy": 1.255,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "multiscale_low_res_6x_ri_1_56": {
        "label": "Multiscale - Low Res — 6x — RI 1.56",
        "detection_na": 0.25,
        "illumination_na": 0.1,
        "magnification": 6.0,
        "dxy": 1.044,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "multiscale_high_res_38x_ri_1_56": {
        "label": "Multiscale - High Res — 38x — RI 1.56",
        "detection_na": 0.753,
        "illumination_na": 0.753,
        "magnification": 38.0,
        "dxy": 0.171,
        "ri": 1.56,
        "light_sheet_angle": 0.0,
    },
    "multiscale_high_res_37x_ri_1_52": {
        "label": "Multiscale - High Res — 37x — RI 1.52",
        "detection_na": 0.734,
        "illumination_na": 0.734,
        "magnification": 37.0,
        "dxy": 0.171,
        "ri": 1.52,
        "light_sheet_angle": 0.0,
    },
}

ACQUISITION_MICROSCOPE_NAMES = {
    "Nanoscale": "Multiscale - High Res",
    "Macroscale": "Multiscale - Low Res",
}

SOLVENT_REFRACTIVE_INDICES = {"BABB": 1.56}


def resolve_image_aggressiveness(mode: str) -> dict[str, object]:
    """Return the non-overridable processing policy for an Astrocyte mode."""
    try:
        return dict(IMAGE_AGGRESSIVENESS_PRESETS[str(mode).lower()])
    except KeyError as exc:
        choices = ", ".join(IMAGE_AGGRESSIVENESS_PRESETS)
        raise ValueError(f"image_aggressiveness must be one of {choices}, got {mode!r}") from exc


def _read_acquisition_metadata(config_file: str | Path) -> dict[str, object]:
    """Read the optional Navigate acquisition YAML used for inference only."""
    path = Path(config_file)
    suffix = path.suffix.lower()
    if suffix not in {".yml", ".yaml"}:
        raise ValueError("Acquisition metadata file must have a .yml or .yaml extension")

    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Acquisition metadata file must contain a top-level mapping")
    return values


def _required_mapping(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Acquisition metadata is missing mapping {key!r}")
    return value


def _infer_profile_from_acquisition(metadata: dict[str, object]) -> str:
    saving = _required_mapping(metadata, "Saving")
    state = _required_mapping(metadata, "MicroscopeState")
    microscope_name = state.get("microscope_name")
    microscope = ACQUISITION_MICROSCOPE_NAMES.get(str(microscope_name))
    if microscope is None:
        supported = ", ".join(ACQUISITION_MICROSCOPE_NAMES)
        raise ValueError(
            f"Cannot infer a microscope profile from microscope_name={microscope_name!r}; "
            f"supported values are {supported}"
        )

    prefix = str(saving.get("prefix", ""))
    match = re.search(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*x", prefix, re.IGNORECASE)
    if match is None:
        raise ValueError("Cannot infer magnification: Saving.prefix must contain a value such as '38x_'")
    magnification = float(match.group(1))
    candidates = [
        profile_id
        for profile_id, profile in MICROSCOPE_PROFILES.items()
        if profile["label"].startswith(microscope)
        and math.isclose(profile["magnification"], magnification)
    ]

    solvent = str(saving.get("solvent", "")).strip().upper()
    ri = SOLVENT_REFRACTIVE_INDICES.get(solvent)
    if ri is not None:
        candidates = [
            profile_id
            for profile_id in candidates
            if math.isclose(MICROSCOPE_PROFILES[profile_id]["ri"], ri)
        ]
    if len(candidates) != 1:
        raise ValueError(
            "Cannot infer a unique microscope profile from acquisition metadata; "
            "select a microscope profile explicitly."
        )
    return candidates[0]


def _infer_wavelength_from_acquisition(metadata: dict[str, object]) -> float:
    state = _required_mapping(metadata, "MicroscopeState")
    channels = _required_mapping(state, "channels")
    selected_lasers = [
        channel.get("laser")
        for channel in channels.values()
        if isinstance(channel, dict) and channel.get("is_selected") is True
    ]
    if len(selected_lasers) != 1:
        raise ValueError(
            "Cannot infer wavelength: acquisition metadata must select exactly one channel"
        )
    match = re.search(r"(\d+(?:\.\d+)?)\s*nm", str(selected_lasers[0]), re.IGNORECASE)
    if match is None:
        raise ValueError("Cannot infer wavelength from the selected channel laser")
    return float(match.group(1)) / 1000.0


def _infer_dz_from_acquisition(metadata: dict[str, object]) -> float:
    state = _required_mapping(metadata, "MicroscopeState")
    try:
        dz = abs(float(state["step_size"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cannot infer dz from MicroscopeState.step_size") from exc
    if dz <= 0:
        raise ValueError("MicroscopeState.step_size must be non-zero to infer dz")
    return dz


def apply_acquisition_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve a profile and optional Navigate-derived wavelength/Z spacing."""
    metadata = _read_acquisition_metadata(args.config_file) if args.config_file else None
    profile_id = args.microscope_profile
    profile = None
    if profile_id == "auto":
        if metadata is not None:
            profile_id = _infer_profile_from_acquisition(metadata)
    if profile_id != "auto":
        try:
            profile = MICROSCOPE_PROFILES[profile_id]
        except KeyError as exc:
            raise ValueError(f"Unsupported microscope profile: {profile_id!r}") from exc

        args.microscope_profile = profile_id
        args.na = profile["detection_na"]
        args.detection_na = profile["detection_na"]
        args.illumination_na = profile["illumination_na"]
        args.magnification = profile["magnification"]
        args.dxy = profile["dxy"]
        args.ni = profile["ri"]
        args.ns = profile["ri"]
        args.light_sheet_angle = profile["light_sheet_angle"]
    if args.wavelength is None and metadata is not None:
        args.wavelength = _infer_wavelength_from_acquisition(metadata)
    if args.dz is None and metadata is not None:
        args.dz = _infer_dz_from_acquisition(metadata)
    for key, value in resolve_image_aggressiveness(args.image_aggressiveness).items():
        setattr(args, key, value)
    if args.decon_backend == "petakit":
        args.vram_gb = None
    return args


def parse_workflow_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse Astrocyte's public inputs for parameter-intake contract tests."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--config_file", default="")
    parser.add_argument("--microscope_profile", choices=("auto", *MICROSCOPE_PROFILES), required=True)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--dz", type=float, default=None)
    parser.add_argument(
        "--image_aggressiveness",
        choices=tuple(IMAGE_AGGRESSIVENESS_PRESETS),
        required=True,
    )
    args = parser.parse_args(argv)
    return apply_acquisition_settings(args)


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


def _resolve_sample_refractive_index(ns: float | None, ni: float) -> float:
    if ns is None or ns == -1:
        return ni
    return _require_positive("ns", ns)


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
    background: float,
    total_chunks: int,
    decon_backend: str = "cupy",
    block_info: dict | None = None,
) -> np.ndarray:
    """Process one spatial chunk with the selected Petakit-compatible backend."""
    _, chunk_label = _chunk_progress(block_info, total_chunks)
    if chunk.size == 0:
        return chunk

    print(
        f"  Chunk {chunk_label} started: shape={chunk.shape}, iterations={n_iters}",
        flush=True,
    )

    start = time.perf_counter()
    if decon_backend == "cupy":
        result = restore_uint16_cupy(
            chunk,
            psf,
            n_iters,
            background=background,
        )
    elif decon_backend == "petakit":
        result = restore_uint16_petakit_cpu(
            chunk,
            psf,
            n_iters,
            background=background,
        )
    else:
        raise ValueError(f"Unsupported deconvolution backend {decon_backend!r}")
    elapsed = time.perf_counter() - start
    avg_iter = elapsed / n_iters if n_iters > 0 else elapsed

    print(
        f"  Iteration {n_iters}/{n_iters} of chunk {chunk_label} completed: "
        f"chunk_time={_format_seconds(elapsed)}, "
        f"avg_iteration_time={_format_seconds(avg_iter)}",
        flush=True,
    )
    return _center_crop_or_pad_to_shape(result, chunk.shape)


# ---------------------------------------------------------------------------
# Per-TIFF deconvolution
# ---------------------------------------------------------------------------

DECON_WORKSPACE_BYTES_PER_VOXEL = 80.0


def _psf_halo(psf_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return Petakit's large-file border size for each PSF axis."""
    if len(psf_shape) != 3 or any(int(size) <= 0 for size in psf_shape):
        raise ValueError(f"PSF shape must contain three positive axes, got {psf_shape}")
    return tuple((int(size) + 11) // 2 for size in psf_shape)


def _available_vram_bytes(vram_gb: float | None) -> int | None:
    if vram_gb and vram_gb > 0:
        return int(vram_gb * (1024 ** 3))
    return detect_vram_bytes()


def _whole_volume_fits(
    volume_shape: tuple[int, int, int],
    vram_gb: float | None,
    safety_fraction: float = 0.65,
) -> bool:
    available = _available_vram_bytes(vram_gb)
    if not available:
        return False
    estimated = float(np.prod(volume_shape)) * DECON_WORKSPACE_BYTES_PER_VOXEL
    return estimated <= available * safety_fraction


def _expanded_chunk_shape(
    core_shape: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
    halo: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(
        min(int(volume), int(core) + 2 * int(depth))
        for core, volume, depth in zip(core_shape, volume_shape, halo)
    )


def _fit_core_chunks_to_vram(
    core_shape: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
    halo: tuple[int, int, int],
    available_bytes: int,
    safety_fraction: float = 0.55,
) -> tuple[int, int, int]:
    """Shrink core chunks until their halo-expanded FFT domain fits VRAM."""
    max_voxels = int(
        int(available_bytes) * safety_fraction / DECON_WORKSPACE_BYTES_PER_VOXEL
    )
    volume_shape = tuple(int(size) for size in volume_shape)
    halo = tuple(int(size) for size in halo)
    minimum = [
        min(volume, max(1, depth))
        for volume, depth in zip(volume_shape, halo)
    ]
    core = [
        max(minimum[axis], min(int(size), volume_shape[axis]))
        for axis, size in enumerate(core_shape)
    ]

    while np.prod(_expanded_chunk_shape(tuple(core), volume_shape, halo)) > max_voxels:
        reducible = [
            axis for axis, size in enumerate(core) if size > minimum[axis]
        ]
        if not reducible:
            raise MemoryError(
                "PSF halo-compatible minimum chunk exceeds the configured VRAM budget"
            )
        axis = max(
            reducible,
            key=lambda item: _expanded_chunk_shape(
                tuple(core), volume_shape, halo
            )[item],
        )
        reduced = max(minimum[axis], int(core[axis] * 0.8))
        if axis in (1, 2) and reduced >= 32:
            reduced = max(32, (reduced // 32) * 32)
        if reduced >= core[axis]:
            reduced = max(minimum[axis], core[axis] - 1)
        core[axis] = reduced

    return tuple(core)


def _balanced_axis_chunks(
    axis_size: int, target_size: int, minimum_size: int
) -> tuple[int, ...]:
    """Partition an axis without a remainder smaller than the overlap depth."""
    axis_size = int(axis_size)
    target_size = max(1, int(target_size))
    minimum_size = max(1, min(axis_size, int(minimum_size)))
    chunk_count = min(
        math.ceil(axis_size / target_size),
        max(1, axis_size // minimum_size),
    )
    base, remainder = divmod(axis_size, chunk_count)
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(chunk_count)
    )


def _auto_decon_max_xy(
    volume_shape: tuple[int, int, int],
    dtype: np.dtype,
    workers: int,
    vram_gb: float | None,
    fallback_xy: int = 512,
) -> int:
    nz, ny, nx = volume_shape
    image_max_xy = min(ny, nx)
    vram_bytes = _available_vram_bytes(vram_gb)
    if not vram_bytes:
        return min(fallback_xy, image_max_xy)

    workers = max(1, workers)
    bytes_per_voxel = np.dtype(dtype).itemsize
    target_bytes = vram_bytes * 0.55 / workers
    memory_multiplier = DECON_WORKSPACE_BYTES_PER_VOXEL / bytes_per_voxel
    denom = max(1, nz) * bytes_per_voxel * memory_multiplier
    max_xy = int(math.sqrt(max(1.0, target_bytes / denom)))
    max_xy = max(128, (max_xy // 32) * 32)
    return min(max_xy, 1024, image_max_xy)


def _build_deconvolution_graph(
    volume,
    image_name: str,
    psf: np.ndarray,
    n_iters: int,
    background: float = 0.0,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
    decon_backend: str = "cupy",
) -> tuple[da.Array, int]:
    """
    Build a lazy deconvolution graph for a single 3-D volume.

    Whole volumes use one unpadded block. Larger volumes use PSF-derived 3-D
    overlap. `chunk_xy` is the core XY tile size; <=0 is VRAM-aware.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")

    original_shape = volume.shape
    halo_z, halo_y, halo_x = _psf_halo(tuple(int(size) for size in psf.shape))
    if overlap_xy > 0:
        halo_y = max(halo_y, int(overlap_xy))
        halo_x = max(halo_x, int(overlap_xy))
    if decon_backend == "cupy" and decon_workers != 1:
        log_progress(
            f"  CuPy restoration uses one GPU worker per allocation; "
            f"clamping decon_workers={decon_workers} to 1"
        )
    if decon_backend == "cupy":
        decon_workers = 1
    nz, ny, nx = (int(size) for size in volume.shape)
    halo_z = min(halo_z, max(0, nz - 1))
    halo_y = min(halo_y, max(0, ny - 1))
    halo_x = min(halo_x, max(0, nx - 1))
    if chunk_xy <= 0 and _whole_volume_fits((nz, ny, nx), vram_gb):
        core_chunk_xy = max(ny, nx)
    else:
        core_chunk_xy = resolve_chunk_xy(
            chunk_xy,
            volume.shape,
            volume.dtype,
            overlap_xy=max(halo_y, halo_x),
            vram_gb=vram_gb,
            workers=decon_workers,
            safety_fraction=0.55,
            memory_multiplier=(
                DECON_WORKSPACE_BYTES_PER_VOXEL / np.dtype(volume.dtype).itemsize
            ),
            min_xy=max(128, halo_y * 2, halo_x * 2),
            max_xy=_auto_decon_max_xy(
                volume.shape, volume.dtype, decon_workers, vram_gb
            ),
        )
    if core_chunk_xy <= 0:
        raise ValueError(f"Resolved decon chunk size must be positive, got {core_chunk_xy}")

    core_chunk_xy = min(core_chunk_xy, max(ny, nx))
    chunk_y = min(core_chunk_xy, ny)
    chunk_x = min(core_chunk_xy, nx)
    chunk_z = nz
    available = _available_vram_bytes(vram_gb)
    if available:
        chunk_z, chunk_y, chunk_x = _fit_core_chunks_to_vram(
            (chunk_z, chunk_y, chunk_x),
            (nz, ny, nx),
            (halo_z, halo_y, halo_x),
            available,
        )

    chunk_divisions = tuple(
        _balanced_axis_chunks(axis, core, max(1, depth))
        for axis, core, depth in zip(
            (nz, ny, nx),
            (chunk_z, chunk_y, chunk_x),
            (halo_z, halo_y, halo_x),
        )
    )
    largest_core = tuple(max(chunks) for chunks in chunk_divisions)
    if available:
        expanded = _expanded_chunk_shape(
            largest_core, (nz, ny, nx), (halo_z, halo_y, halo_x)
        )
        estimated = np.prod(expanded) * DECON_WORKSPACE_BYTES_PER_VOXEL
        if estimated > available * 0.55:
            raise MemoryError(
                "Halo-compatible Dask chunks exceed the configured VRAM budget"
            )

    lazy = da.from_array(
        volume,
        chunks=chunk_divisions,
        asarray=False,
        lock=False,
    )
    total_chunks = int(np.prod(lazy.numblocks))

    log_progress(f"Deconvolving {image_name}: shape={original_shape}, dtype={volume.dtype}")
    log_progress(
        f"  Deconvolution chunks: total={total_chunks}, "
        f"max_core_chunk_shape={largest_core}, "
        f"psf_halo=(z={halo_z}, y={halo_y}, x={halo_x}), "
        f"image_shape=({nz}, {ny}, {nx}), "
        f"iterations_per_chunk={n_iters}, workers={decon_workers}"
    )

    common_kwargs = {
        "dtype": np.uint16,
        "psf": psf,
        "n_iters": n_iters,
        "background": background,
        "total_chunks": total_chunks,
        "decon_backend": decon_backend,
    }
    if total_chunks == 1:
        return da.map_blocks(
            _decon_chunk,
            lazy,
            **common_kwargs,
        ), decon_workers

    return da.map_overlap(
        _decon_chunk,
        lazy,
        depth={0: halo_z, 1: halo_y, 2: halo_x},
        boundary="none",
        allow_rechunk=False,
        **common_kwargs,
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
    background: float = 0.0,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
    decon_backend: str = "cupy",
) -> np.ndarray:
    del dz, dxy, wavelength, na, ni
    processed, decon_workers = _build_deconvolution_graph(
        volume,
        image_name,
        psf,
        n_iters,
        background=background,
        chunk_xy=chunk_xy,
        vram_gb=vram_gb,
        decon_workers=decon_workers,
        overlap_xy=overlap_xy,
        decon_backend=decon_backend,
    )
    scheduler = _deconvolution_scheduler(decon_workers)
    log_progress(
        f"Computing {decon_backend} Petakit-compatible deconvolution graph for {image_name}: "
        f"scheduler={scheduler}, workers={decon_workers}"
    )
    return processed.compute(scheduler=scheduler, num_workers=decon_workers)


def deconvolve_tiff(
    image_path: Path,
    psf: np.ndarray,
    n_iters: int,
    dz: float,
    dxy: float,
    wavelength: float,
    na: float,
    ni: float,
    background: float = 0.0,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
    decon_backend: str = "cupy",
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
        background=background,
        chunk_xy=chunk_xy,
        vram_gb=vram_gb,
        decon_workers=decon_workers,
        overlap_xy=overlap_xy,
        decon_backend=decon_backend,
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
    background: float = 0.0,
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
        background=background,
        chunk_xy=chunk_xy,
        vram_gb=vram_gb,
        decon_workers=decon_workers,
        overlap_xy=overlap_xy,
    )


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
    background: float = 0.0,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
    max_downsample: int = 16,
    decon_backend: str = "cupy",
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
        background=background,
        chunk_xy=chunk_xy,
        vram_gb=vram_gb,
        decon_workers=decon_workers,
        overlap_xy=overlap_xy,
        decon_backend=decon_backend,
    )
    scheduler = _deconvolution_scheduler(decon_workers)
    final_array = create_ome_zarr_array(
        output_path,
        shape=tuple(int(axis) for axis in volume.shape),
        dtype=np.uint16,
        chunks=_default_output_chunks(tuple(int(axis) for axis in volume.shape)),
        layer_name=image_stem(output_path),
        max_downsample=int(max_downsample),
    )
    log_progress(f"Streaming deconvolution output to OME-Zarr: {output_path}")
    da.store(processed, final_array, lock=False, compute=False).compute(
        scheduler=scheduler,
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
    parser.add_argument("--config_file", default="",
                        help="Optional Navigate acquisition YAML used to infer profile, wavelength, and dz.")
    parser.add_argument("--microscope_profile", default="auto",
                        choices=("auto", *MICROSCOPE_PROFILES),
                        help="Complete microscope/magnification/RI profile, or auto to infer it from acquisition YAML.")
    parser.add_argument("--image_aggressiveness", default="medium",
                        choices=tuple(IMAGE_AGGRESSIVENESS_PRESETS),
                        help="Astrocyte speed/accuracy preset that fixes processing engines and queue policy.")
    parser.add_argument("--output_format", choices=("ozx", "tiff"), default="ozx",
                        help="Final output representation requested by the workflow.")

    # Blind estimation options
    parser.add_argument("--blind_iters", type=int, default=DEFAULT_BLIND_ITERS,
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
    parser.add_argument("--decon_backend", choices=("cupy", "petakit"), default="cupy",
                        help="Deconvolution backend; controlled by image_aggressiveness in Astrocyte runs.")
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
    parser.add_argument("--psf_seed_path", default="",
                        help="Optional calibrated TIFF PSF seed; center-fitted to the configured PSF support.")
    parser.add_argument("--fixed_psf_path", default="",
                        help="Optional calibrated TIFF PSF used directly for final deconvolution.")
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
    try:
        args = apply_acquisition_settings(args)
    except ValueError as exc:
        parser.error(str(exc))
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
    ns = _resolve_sample_refractive_index(args.ns, ni)
    detection_na = args.detection_na if args.detection_na is not None else args.na
    detection_na = _require_positive("detection_na", detection_na)
    if args.psf_mode == "light_sheet":
        _require_positive("illumination_na", args.illumination_na)
    log_progress(
        "Resolved optical parameters: "
        f"dxy={dxy}, dz={dz}, wavelength={wavelength}, "
        f"detection_na={detection_na}, ni={ni}, ns={ns}, psf_mode={args.psf_mode}"
    )

    psf_start = time.perf_counter()
    psf_shape = (args.psf_size_z, args.psf_size_xy, args.psf_size_xy)
    if args.fixed_psf_path:
        psf = load_fixed_psf(args.fixed_psf_path)
        log_progress(
            f"Loaded fixed PSF from {args.fixed_psf_path}; "
            f"native shape={psf.shape}. Skipping blind PSF estimation."
        )
    else:
        if args.psf_seed_path:
            psf_seed = load_psf_seed(args.psf_seed_path, psf_shape)
            log_progress(
                f"Loaded calibrated PSF seed from {args.psf_seed_path}; "
                f"center-fitted shape={psf_seed.shape}"
            )
        else:
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
        log_progress(
            f"Running blind PSF estimation on first image volume: {psf_input_path}"
        )
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
                background=args.background,
                chunk_xy=args.decon_chunk_xy,
                vram_gb=args.vram_gb,
                decon_workers=args.decon_workers,
                overlap_xy=args.overlap_xy,
                max_downsample=args.pyramid_max_downsample,
                decon_backend=args.decon_backend,
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
                background=args.background,
                chunk_xy=args.decon_chunk_xy,
                vram_gb=args.vram_gb,
                decon_workers=args.decon_workers,
                overlap_xy=args.overlap_xy,
                decon_backend=args.decon_backend,
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
