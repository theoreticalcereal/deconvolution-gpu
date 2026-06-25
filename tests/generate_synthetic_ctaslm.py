#!/usr/bin/env python3
"""
Generate synthetic raw ctASLM TIFF stacks for pipeline checks.

The script builds an internal ground-truth volume containing beads and
filaments, applies a skew transform, and writes only the raw skewed input TIFFs
using pipeline-compatible names such as CH00_000000.tiff.
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import scipy.ndimage as ndimage
    import tifffile
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency {exc.name!r}. Install numpy, scipy, and tifffile "
        "to generate synthetic ctASLM TIFFs."
    ) from exc


def _center_bounds(size: int) -> tuple[int, int]:
    margin = max(2, min(20, size // 5))
    lower = margin
    upper = max(lower + 1, size - margin)
    return lower, upper


def _add_beads(volume: np.ndarray, rng: np.random.Generator, count: int, sigma: float) -> None:
    nz, ny, nx = volume.shape
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    z_low, z_high = _center_bounds(nz)
    y_low, y_high = _center_bounds(ny)
    x_low, x_high = _center_bounds(nx)

    for _ in range(max(0, count)):
        zc = rng.integers(z_low, z_high)
        yc = rng.integers(y_low, y_high)
        xc = rng.integers(x_low, x_high)
        dist_sq = (z - zc) ** 2 + (y - yc) ** 2 + (x - xc) ** 2
        volume += np.exp(-dist_sq / (2 * sigma**2)) * 8000.0


def _add_filaments(
    volume: np.ndarray,
    rng: np.random.Generator,
    count: int,
    samples: int,
    radius: int,
) -> None:
    nz, ny, nx = volume.shape
    z_low, z_high = _center_bounds(nz)
    y_low, y_high = _center_bounds(ny)
    x_low, x_high = _center_bounds(nx)
    radius = max(0, radius)

    for _ in range(max(0, count)):
        t = np.linspace(0.0, 1.0, max(2, samples))
        z_path = np.interp(t, [0.0, 0.5, 1.0], rng.integers(z_low, z_high, 3))
        y_path = np.interp(t, [0.0, 0.5, 1.0], rng.integers(y_low, y_high, 3))
        x_path = np.interp(t, [0.0, 0.5, 1.0], rng.integers(x_low, x_high, 3))

        for zi, yi, xi in zip(z_path.astype(int), y_path.astype(int), x_path.astype(int)):
            volume[
                max(0, zi - radius):min(nz, zi + radius + 1),
                max(0, yi - radius):min(ny, yi + radius + 1),
                max(0, xi - radius):min(nx, xi + radius + 1),
            ] = 12000.0


def _skew_volume(
    volume: np.ndarray,
    skew_angle_deg: float,
    skew_axis: str,
    background: float,
) -> np.ndarray:
    nz, _, _ = volume.shape
    theta = np.radians(skew_angle_deg)
    skew_factor = np.tan(theta)
    matrix = np.eye(3, dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64)

    if skew_axis == "x":
        matrix[2, 0] = skew_factor
        offset[2] = -skew_factor * (nz - 1) / 2.0
    elif skew_axis == "y":
        matrix[1, 0] = skew_factor
        offset[1] = -skew_factor * (nz - 1) / 2.0
    else:
        raise ValueError("skew_axis must be 'x' or 'y'")

    return ndimage.affine_transform(
        volume,
        matrix,
        offset=offset,
        order=1,
        mode="constant",
        cval=float(background),
    )


def generate_synthetic_ctaslm(
    shape=(64, 64, 64),
    skew_angle_deg=31.5,
    num_beads=30,
    num_filaments=5,
    seed=42,
    skew_axis="x",
    background=200,
    bead_sigma=2.0,
    filament_samples=100,
    filament_radius=1,
):
    """
    Return one synthetic raw skewed ctASLM volume as uint16.

    Ground truth is generated internally to create the skewed raw volume, but it
    is not returned or written by the CLI.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("shape must contain three positive integers: Z Y X")

    gt = np.zeros(shape, dtype=np.float32)
    _add_beads(gt, rng, num_beads, bead_sigma)
    _add_filaments(gt, rng, num_filaments, filament_samples, filament_radius)

    if background > 0:
        gt += rng.poisson(lam=background, size=shape).astype(np.float32)

    raw_skewed = _skew_volume(gt, skew_angle_deg, skew_axis, background)
    raw_skewed = np.clip(raw_skewed, 0, np.iinfo(np.uint16).max)
    return raw_skewed.astype(np.uint16)


def synthetic_filename(channel: int, timepoint: int, suffix: str = ".tiff") -> str:
    return f"CH{int(channel):02d}_{int(timepoint):06d}{suffix}"


def write_synthetic_ctaslm_series(
    output_dir,
    shape=(64, 64, 64),
    skew_angle_deg=31.5,
    num_beads=30,
    num_filaments=5,
    seed=42,
    channel=0,
    start_timepoint=0,
    count=1,
    skew_axis="x",
    background=200,
    bead_sigma=2.0,
    filament_samples=100,
    filament_radius=1,
    suffix=".tiff",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for index in range(max(1, int(count))):
        timepoint = int(start_timepoint) + index
        volume = generate_synthetic_ctaslm(
            shape=shape,
            skew_angle_deg=skew_angle_deg,
            num_beads=num_beads,
            num_filaments=num_filaments,
            seed=int(seed) + index,
            skew_axis=skew_axis,
            background=background,
            bead_sigma=bead_sigma,
            filament_samples=filament_samples,
            filament_radius=filament_radius,
        )
        path = output_dir / synthetic_filename(channel, timepoint, suffix)
        tifffile.imwrite(str(path), volume, imagej=True)
        paths.append(path)

    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic raw skewed ctASLM TIFF inputs."
    )
    parser.add_argument("--output-dir", default="tests/synthetic_ctaslm")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("Z", "Y", "X"), default=(64, 64, 64))
    parser.add_argument("--skew-angle-deg", type=float, default=31.5)
    parser.add_argument("--skew-axis", choices=("x", "y"), default="x")
    parser.add_argument("--num-beads", type=int, default=30)
    parser.add_argument("--num-filaments", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--start-timepoint", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--background", type=float, default=200.0)
    parser.add_argument("--bead-sigma", type=float, default=2.0)
    parser.add_argument("--filament-samples", type=int, default=100)
    parser.add_argument("--filament-radius", type=int, default=1)
    parser.add_argument("--suffix", choices=(".tif", ".tiff"), default=".tiff")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_synthetic_ctaslm_series(
        output_dir=args.output_dir,
        shape=tuple(args.shape),
        skew_angle_deg=args.skew_angle_deg,
        num_beads=args.num_beads,
        num_filaments=args.num_filaments,
        seed=args.seed,
        channel=args.channel,
        start_timepoint=args.start_timepoint,
        count=args.count,
        skew_axis=args.skew_axis,
        background=args.background,
        bead_sigma=args.bead_sigma,
        filament_samples=args.filament_samples,
        filament_radius=args.filament_radius,
        suffix=args.suffix,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
