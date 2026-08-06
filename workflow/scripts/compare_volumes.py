#!/usr/bin/env python3
"""Compare two deconvolved image volumes with correlation, error, and sharpness metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
from tifffile import imread

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ome_zarr_io import (
    is_ome_zarr_path,
    is_ozx_path,
    open_ome_zarr_array,
    unzip_ozx_to_ome_zarr,
)


TIFF_SUFFIXES = {".tif", ".tiff"}


def _as_zyx(volume, *, label: str) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} is empty")
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"{label} must be a 2-D or 3-D volume, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return array


def _load_mat_volume(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("MAT input requires scipy.io.loadmat") from exc

    data = loadmat(path)
    candidates = [
        value
        for key, value in data.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.squeeze(value).ndim in {2, 3}
    ]
    if not candidates:
        raise ValueError(f"No 2-D or 3-D numeric volume found in {path}")
    return max(candidates, key=lambda value: np.squeeze(value).size)


def load_volume(path: Path | str) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TIFF_SUFFIXES:
        return _as_zyx(imread(str(path)), label=str(path))
    if suffix == ".npy":
        return _as_zyx(np.load(path), label=str(path))
    if suffix == ".mat":
        return _as_zyx(_load_mat_volume(path), label=str(path))
    if is_ome_zarr_path(path):
        return _as_zyx(open_ome_zarr_array(path, mode="r")[:], label=str(path))
    if is_ozx_path(path):
        with tempfile.TemporaryDirectory(prefix="volume_ozx_") as tmpdir:
            zarr_path = Path(tmpdir) / f"{path.stem}.ome.zarr"
            unzip_ozx_to_ome_zarr(path, zarr_path)
            return _as_zyx(open_ome_zarr_array(zarr_path, mode="r")[:], label=str(path))

    raise ValueError(
        f"Unsupported volume file type for {path}. "
        "Use TIFF, OME-Zarr, OZX, NPY, or MAT."
    )


def _unit_range(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float64)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum == minimum:
        return np.zeros_like(array, dtype=np.float64)
    return (array - minimum) / (maximum - minimum)


def _center_crop(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    slices = []
    for axis_size, target_size in zip(volume.shape, shape):
        if axis_size < target_size:
            raise ValueError(f"Cannot crop shape {volume.shape} to larger shape {shape}")
        start = (axis_size - target_size) // 2
        slices.append(slice(start, start + target_size))
    return volume[tuple(slices)]


def _align_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    align: str = "center-crop",
) -> tuple[np.ndarray, np.ndarray]:
    reference = _as_zyx(reference, label="reference")
    candidate = _as_zyx(candidate, label="candidate")
    if reference.shape == candidate.shape:
        return reference, candidate
    if align == "strict":
        raise ValueError(
            f"Volume shapes differ: reference={reference.shape}, candidate={candidate.shape}"
        )
    if align != "center-crop":
        raise ValueError(f"Unsupported alignment mode: {align}")
    common_shape = tuple(min(a, b) for a, b in zip(reference.shape, candidate.shape))
    return _center_crop(reference, common_shape), _center_crop(candidate, common_shape)


def ncc_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    ref = ref - float(np.mean(ref))
    cand = cand - float(np.mean(cand))
    denominator = math.sqrt(float(np.sum(ref * ref)) * float(np.sum(cand * cand)))
    if denominator <= 0.0:
        return 1.0 if np.allclose(reference, candidate) else float("nan")
    return float(np.sum(ref * cand) / denominator)


def ssim_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = _unit_range(reference)
    cand = _unit_range(candidate)
    mu_ref = float(np.mean(ref))
    mu_cand = float(np.mean(cand))
    var_ref = float(np.mean((ref - mu_ref) ** 2))
    var_cand = float(np.mean((cand - mu_cand) ** 2))
    covariance = float(np.mean((ref - mu_ref) * (cand - mu_cand)))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_ref * mu_cand + c1) * (2.0 * covariance + c2)
    denominator = (mu_ref ** 2 + mu_cand ** 2 + c1) * (var_ref + var_cand + c2)
    if denominator <= 0.0:
        return 1.0 if np.allclose(ref, cand) else float("nan")
    return float(numerator / denominator)


def gradient_energy(
    volume: np.ndarray,
    *,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
) -> float:
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != 3:
        raise ValueError(f"spacing must have 3 values for z/y/x, got {spacing}")
    normalized = _unit_range(volume)
    gradients = np.gradient(normalized, *spacing, edge_order=1)
    return float(np.mean(sum(axis_gradient ** 2 for axis_gradient in gradients)))


def high_frequency_fraction(volume: np.ndarray, *, cutoff: float = 0.25) -> float:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")
    normalized = _unit_range(volume)
    centered = normalized - float(np.mean(normalized))
    spectrum = np.fft.fftn(centered)
    energy = np.abs(spectrum) ** 2
    total = float(np.sum(energy))
    if total <= 0.0:
        return float("nan")
    frequency_grids = np.meshgrid(
        *(np.fft.fftfreq(axis_size) for axis_size in normalized.shape),
        indexing="ij",
    )
    radius = np.sqrt(sum(axis_frequency ** 2 for axis_frequency in frequency_grids))
    return float(np.sum(energy[radius >= cutoff]) / total)


def _summary(prefix: str, volume: np.ndarray) -> dict[str, float | str]:
    return {
        f"{prefix}_shape": "x".join(str(axis) for axis in volume.shape),
        f"{prefix}_min": float(np.min(volume)),
        f"{prefix}_max": float(np.max(volume)),
        f"{prefix}_mean": float(np.mean(volume)),
        f"{prefix}_std": float(np.std(volume)),
    }


def compare_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
    align: str = "center-crop",
) -> dict[str, float | str]:
    ref, cand = _align_pair(reference, candidate, align=align)
    diff = cand - ref
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(float(np.mean(diff ** 2))))
    ref_range = float(np.max(ref) - np.min(ref))
    nrmse = float(rmse / ref_range) if ref_range > 0.0 else float("nan")
    reference_gradient_energy = gradient_energy(ref, spacing=spacing)
    candidate_gradient_energy = gradient_energy(cand, spacing=spacing)
    gradient_energy_ratio = (
        float(candidate_gradient_energy / reference_gradient_energy)
        if reference_gradient_energy > 0.0
        else float("nan")
    )
    reference_high_frequency_fraction = high_frequency_fraction(ref)
    candidate_high_frequency_fraction = high_frequency_fraction(cand)
    high_frequency_fraction_ratio = (
        float(candidate_high_frequency_fraction / reference_high_frequency_fraction)
        if reference_high_frequency_fraction > 0.0
        else float("nan")
    )
    row: dict[str, float | str] = {
        "crop_shape": "x".join(str(axis) for axis in ref.shape),
        "ncc": ncc_score(ref, cand),
        "ssim": ssim_score(ref, cand),
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "mean_error": float(np.mean(diff)),
        "max_abs_error": float(np.max(np.abs(diff))),
        "reference_gradient_energy": reference_gradient_energy,
        "candidate_gradient_energy": candidate_gradient_energy,
        "gradient_energy_ratio": gradient_energy_ratio,
        "reference_high_frequency_fraction": reference_high_frequency_fraction,
        "candidate_high_frequency_fraction": candidate_high_frequency_fraction,
        "high_frequency_fraction_ratio": high_frequency_fraction_ratio,
    }
    row.update(_summary("reference", ref))
    row.update(_summary("candidate", cand))
    return row


def compare_two_volumes(
    reference_name: str,
    reference_volume: np.ndarray,
    candidate_name: str,
    candidate_volume: np.ndarray,
    *,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
    align: str = "center-crop",
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "reference": str(reference_name),
        "candidate": str(candidate_name),
    }
    row.update(
        compare_pair(
            reference_volume,
            candidate_volume,
            spacing=spacing,
            align=align,
        )
    )
    return row


def compare_volume_paths(
    reference_path: Path | str,
    candidate_path: Path | str,
    *,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
    align: str = "center-crop",
) -> list[dict[str, float | str]]:
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)
    return [
        compare_two_volumes(
            str(reference_path),
            load_volume(reference_path),
            str(candidate_path),
            load_volume(candidate_path),
            spacing=spacing,
            align=align,
        )
    ]


def _json_ready(rows: list[dict[str, float | str]]) -> list[dict[str, float | str | None]]:
    ready = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                converted[key] = None
            else:
                converted[key] = value
        ready.append(converted)
    return ready


def write_csv(rows: list[dict[str, float | str]], handle) -> None:
    fieldnames = [
        "reference",
        "candidate",
        "reference_shape",
        "candidate_shape",
        "crop_shape",
        "ncc",
        "ssim",
        "mae",
        "rmse",
        "nrmse",
        "mean_error",
        "max_abs_error",
        "reference_gradient_energy",
        "candidate_gradient_energy",
        "gradient_energy_ratio",
        "reference_high_frequency_fraction",
        "candidate_high_frequency_fraction",
        "high_frequency_fraction_ratio",
        "reference_min",
        "reference_max",
        "reference_mean",
        "reference_std",
        "candidate_min",
        "candidate_max",
        "candidate_mean",
        "candidate_std",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare exactly two deconvolved image volumes."
    )
    parser.add_argument("reference", help="Reference volume path")
    parser.add_argument("candidate", help="Candidate volume path")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        metavar=("Z", "Y", "X"),
        default=(1.0, 1.0, 1.0),
        help="Voxel spacing used by gradient sharpness metrics. Defaults to 1 1 1.",
    )
    parser.add_argument(
        "--align",
        choices=("center-crop", "strict"),
        default="center-crop",
        help="How to align different volume shapes for pair metrics. Default: center-crop.",
    )
    parser.add_argument("--csv", help="Optional CSV output path. CSV is always printed to stdout.")
    parser.add_argument("--json", help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = compare_volume_paths(
        args.reference,
        args.candidate,
        spacing=args.spacing,
        align=args.align,
    )
    write_csv(rows, sys.stdout)

    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            write_csv(rows, handle)
    if args.json:
        Path(args.json).write_text(
            json.dumps(_json_ready(rows), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
