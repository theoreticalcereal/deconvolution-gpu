#!/usr/bin/env python3
"""Compare estimated PSFs with Gaussian fit, FWHM, NCC, and SSIM metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from dataclasses import dataclass
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
FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
AXES = ("z", "y", "x")


@dataclass(frozen=True)
class GaussianProfileFit:
    center: float
    sigma: float
    r2: float

    @property
    def fwhm(self) -> float:
        return FWHM_FACTOR * self.sigma


def _as_zyx(volume, *, label: str) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} is empty")
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"{label} must be a 2-D or 3-D PSF, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return array


def load_psf(path: Path | str) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TIFF_SUFFIXES:
        return _as_zyx(imread(str(path)), label=str(path))
    if suffix == ".npy":
        return _as_zyx(np.load(path), label=str(path))
    if suffix == ".mat":
        return _as_zyx(_load_mat_psf(path), label=str(path))
    if is_ome_zarr_path(path):
        return _as_zyx(open_ome_zarr_array(path, mode="r")[:], label=str(path))
    if is_ozx_path(path):
        with tempfile.TemporaryDirectory(prefix="psf_ozx_") as tmpdir:
            zarr_path = Path(tmpdir) / f"{path.stem}.ome.zarr"
            unzip_ozx_to_ome_zarr(path, zarr_path)
            return _as_zyx(open_ome_zarr_array(zarr_path, mode="r")[:], label=str(path))

    raise ValueError(
        f"Unsupported PSF file type for {path}. "
        "Use TIFF, OME-Zarr, OZX, NPY, or MAT."
    )


def _load_mat_psf(path: Path) -> np.ndarray:
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
        raise ValueError(f"No 2-D or 3-D numeric PSF array found in {path}")
    return max(candidates, key=lambda value: np.squeeze(value).size)


def _positive_sum_normalized(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float64)
    array = array - float(np.min(array))
    array = np.clip(array, 0.0, None)
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError("PSF has no positive signal after background subtraction")
    return array / total


def _unit_range(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float64)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum == minimum:
        return np.zeros_like(array, dtype=np.float64)
    return (array - minimum) / (maximum - minimum)


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denominator = float(np.sum((observed - float(np.mean(observed))) ** 2))
    residual = float(np.sum((observed - predicted) ** 2))
    if denominator <= 0.0:
        return 1.0 if residual <= 1e-15 else float("nan")
    return 1.0 - (residual / denominator)


def _gaussian_1d(x, offset, amplitude, center, sigma):
    sigma = max(float(sigma), 1e-9)
    return offset + amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def _profile_moment_initial(profile: np.ndarray) -> tuple[float, float, float, float]:
    x = np.arange(profile.size, dtype=np.float64)
    offset = float(np.min(profile))
    signal = np.clip(profile - offset, 0.0, None)
    total = float(np.sum(signal))
    if total <= 0.0:
        center = (profile.size - 1.0) / 2.0
        sigma = max(profile.size / 6.0, 1e-3)
        amplitude = max(float(np.max(profile)) - offset, 1e-12)
        return offset, amplitude, center, sigma

    center = float(np.sum(x * signal) / total)
    variance = float(np.sum(signal * (x - center) ** 2) / total)
    sigma = max(math.sqrt(max(variance, 0.0)), 1e-3)
    amplitude = max(float(np.max(profile)) - offset, 1e-12)
    return offset, amplitude, center, sigma


def _fit_profile(profile: np.ndarray) -> GaussianProfileFit:
    profile = np.asarray(profile, dtype=np.float64)
    x = np.arange(profile.size, dtype=np.float64)
    initial = _profile_moment_initial(profile)

    try:
        from scipy.optimize import curve_fit

        upper_offset = max(float(np.max(profile)), initial[0] + initial[1], 1e-12)
        bounds = (
            (0.0, 0.0, 0.0, 1e-6),
            (upper_offset, np.inf, max(profile.size - 1.0, 0.0), max(profile.size * 2.0, 1.0)),
        )
        params, _ = curve_fit(
            _gaussian_1d,
            x,
            profile,
            p0=initial,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        params = initial

    predicted = _gaussian_1d(x, *params)
    return GaussianProfileFit(
        center=float(params[2]),
        sigma=float(abs(params[3])),
        r2=float(_r2_score(profile, predicted)),
    )


def gaussian_profile_fits(volume: np.ndarray) -> dict[str, GaussianProfileFit]:
    psf = _positive_sum_normalized(volume)
    return {
        "z": _fit_profile(np.sum(psf, axis=(1, 2))),
        "y": _fit_profile(np.sum(psf, axis=(0, 2))),
        "x": _fit_profile(np.sum(psf, axis=(0, 1))),
    }


def _gaussian_model(shape: tuple[int, int, int], fits: dict[str, GaussianProfileFit]) -> np.ndarray:
    z, y, x = np.indices(shape, dtype=np.float64)
    exponent = (
        ((z - fits["z"].center) ** 2) / (2.0 * fits["z"].sigma ** 2)
        + ((y - fits["y"].center) ** 2) / (2.0 * fits["y"].sigma ** 2)
        + ((x - fits["x"].center) ** 2) / (2.0 * fits["x"].sigma ** 2)
    )
    model = np.exp(-exponent)
    total = float(np.sum(model))
    if total <= 0.0:
        raise ValueError("Gaussian model has no positive signal")
    return model / total


def summarize_psf(
    name: str,
    volume: np.ndarray,
    *,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
) -> dict[str, float | str]:
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != 3:
        raise ValueError(f"spacing must have 3 values for z/y/x, got {spacing}")

    psf = _positive_sum_normalized(_as_zyx(volume, label=name))
    fits = gaussian_profile_fits(psf)
    model = _gaussian_model(psf.shape, fits)
    row: dict[str, float | str] = {
        "path": str(name),
        "shape": "x".join(str(axis) for axis in psf.shape),
        "gaussian_r2": float(_r2_score(psf, model)),
    }
    for index, axis in enumerate(AXES):
        fit = fits[axis]
        row[f"center_{axis}_voxels"] = fit.center
        row[f"sigma_{axis}_voxels"] = fit.sigma
        row[f"fwhm_{axis}_voxels"] = fit.fwhm
        row[f"fwhm_{axis}"] = fit.fwhm * spacing[index]
        row[f"profile_r2_{axis}"] = fit.r2
    return row


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
            f"PSF shapes differ: reference={reference.shape}, candidate={candidate.shape}"
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


def compare_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    align: str = "center-crop",
) -> dict[str, float | str]:
    ref, cand = _align_pair(reference, candidate, align=align)
    return {
        "ncc": ncc_score(ref, cand),
        "ssim": ssim_score(ref, cand),
        "crop_shape": "x".join(str(axis) for axis in ref.shape),
    }


def _prefixed_summary(prefix: str, summary: dict[str, float | str]) -> dict[str, float | str]:
    return {
        f"{prefix}_{key}": value
        for key, value in summary.items()
        if key not in {"path"}
    }


def compare_two_psfs(
    reference_name: str,
    reference_volume: np.ndarray,
    candidate_name: str,
    candidate_volume: np.ndarray,
    *,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
    align: str = "center-crop",
) -> dict[str, float | str]:
    reference_summary = summarize_psf(reference_name, reference_volume, spacing=spacing)
    candidate_summary = summarize_psf(candidate_name, candidate_volume, spacing=spacing)
    row: dict[str, float | str] = {
        "reference": str(reference_name),
        "candidate": str(candidate_name),
    }
    row.update(_prefixed_summary("reference", reference_summary))
    row.update(_prefixed_summary("candidate", candidate_summary))
    row.update(compare_pair(reference_volume, candidate_volume, align=align))
    return row


def compare_psf_paths(
    reference_path: Path | str,
    candidate_path: Path | str,
    *,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
    align: str = "center-crop",
) -> list[dict[str, float | str]]:
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)
    return [
        compare_two_psfs(
            str(reference_path),
            load_psf(reference_path),
            str(candidate_path),
            load_psf(candidate_path),
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
        "reference_gaussian_r2",
        "candidate_gaussian_r2",
        "ncc",
        "ssim",
        "reference_fwhm_z",
        "reference_fwhm_y",
        "reference_fwhm_x",
        "candidate_fwhm_z",
        "candidate_fwhm_y",
        "candidate_fwhm_x",
        "reference_fwhm_z_voxels",
        "reference_fwhm_y_voxels",
        "reference_fwhm_x_voxels",
        "candidate_fwhm_z_voxels",
        "candidate_fwhm_y_voxels",
        "candidate_fwhm_x_voxels",
        "reference_center_z_voxels",
        "reference_center_y_voxels",
        "reference_center_x_voxels",
        "candidate_center_z_voxels",
        "candidate_center_y_voxels",
        "candidate_center_x_voxels",
        "reference_sigma_z_voxels",
        "reference_sigma_y_voxels",
        "reference_sigma_x_voxels",
        "candidate_sigma_z_voxels",
        "candidate_sigma_y_voxels",
        "candidate_sigma_x_voxels",
        "reference_profile_r2_z",
        "reference_profile_r2_y",
        "reference_profile_r2_x",
        "candidate_profile_r2_z",
        "candidate_profile_r2_y",
        "candidate_profile_r2_x",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exactly two PSF volumes: a reference PSF and a candidate PSF."
        )
    )
    parser.add_argument("reference", help="Reference PSF path")
    parser.add_argument("candidate", help="Candidate PSF path")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        metavar=("Z", "Y", "X"),
        default=(1.0, 1.0, 1.0),
        help="Voxel spacing used to scale fwhm_z/fwhm_y/fwhm_x. Defaults to 1 1 1.",
    )
    parser.add_argument(
        "--align",
        choices=("center-crop", "strict"),
        default="center-crop",
        help="How to align different PSF shapes for pair metrics. Default: center-crop.",
    )
    parser.add_argument("--csv", help="Optional CSV output path. CSV is always printed to stdout.")
    parser.add_argument("--json", help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = compare_psf_paths(
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
