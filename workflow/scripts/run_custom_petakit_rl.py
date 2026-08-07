#!/usr/bin/env python3
"""Run the Petakit-compatible CuPy RL kernel on one TIFF volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from tifffile import imread, imwrite

from petakit_rl import fit_psf_to_shape, restore_uint16_cupy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Petakit-compatible accelerated RL with CuPy."
    )
    parser.add_argument("--input", required=True, help="Input 3-D TIFF")
    parser.add_argument("--psf", required=True, help="Input 3-D PSF TIFF")
    parser.add_argument("--output", required=True, help="Output uint16 TIFF")
    parser.add_argument("--iter", type=int, default=10, help="RL iterations")
    parser.add_argument(
        "--background", type=float, default=0.0, help="Background to subtract"
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device")
    parser.add_argument(
        "--timing-json",
        default=None,
        help="Timing JSON path; defaults next to the output TIFF",
    )
    return parser


def _as_3d(array: np.ndarray, label: str) -> np.ndarray:
    result = np.squeeze(np.asarray(array))
    if result.ndim == 2:
        result = result[np.newaxis, :, :]
    if result.ndim != 3:
        raise ValueError(f"{label} must be 3-D, got shape {result.shape}")
    return result


def main() -> None:
    args = build_parser().parse_args()
    total_start = time.perf_counter()

    load_start = time.perf_counter()
    observed = _as_3d(imread(args.input), "input")
    psf = _as_3d(imread(args.psf), "PSF")
    psf = fit_psf_to_shape(psf, observed.shape)
    load_seconds = time.perf_counter() - load_start

    restoration_start = time.perf_counter()
    restored = restore_uint16_cupy(
        observed,
        psf,
        args.iter,
        background=args.background,
        device_id=args.device_id,
    )
    restoration_seconds = time.perf_counter() - restoration_start

    write_start = time.perf_counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(output_path, restored, photometric="minisblack")
    write_seconds = time.perf_counter() - write_start

    timing_path = Path(args.timing_json) if args.timing_json else output_path.with_suffix(".timing.json")
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing = {
        "input": str(Path(args.input).resolve()),
        "psf": str(Path(args.psf).resolve()),
        "output": str(output_path.resolve()),
        "input_shape": list(observed.shape),
        "psf_shape": list(psf.shape),
        "iterations": int(args.iter),
        "background": float(args.background),
        "device_id": int(args.device_id),
        "load_seconds": load_seconds,
        "restoration_seconds": restoration_seconds,
        "write_seconds": write_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(timing, sort_keys=True))


if __name__ == "__main__":
    main()
