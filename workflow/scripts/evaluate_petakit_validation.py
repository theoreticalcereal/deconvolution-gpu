#!/usr/bin/env python3
"""Evaluate two-stage Petakit compatibility and runtime gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STAGE1_LIMITS = {
    "psf_ncc_min": 0.90,
    "center_displacement_max_voxels": 1.0,
    "fwhm_relative_delta_max": 0.20,
    "volume_ncc_min": 0.95,
    "volume_ssim_min": 0.95,
    "gradient_ratio_min": 0.80,
    "gradient_ratio_max": 1.25,
    "frequency_ratio_min": 0.80,
    "frequency_ratio_max": 1.25,
}

STAGE2_LIMITS = {
    "volume_ncc_min": 0.99,
    "volume_ssim_min": 0.99,
    "mean_ratio_min": 0.99,
    "mean_ratio_max": 1.01,
    "gradient_ratio_min": 0.95,
    "gradient_ratio_max": 1.05,
    "frequency_ratio_min": 0.90,
    "frequency_ratio_max": 1.10,
}


def _check(value: float | None, *, minimum: float | None = None,
           maximum: float | None = None) -> dict[str, Any]:
    numeric = None
    if value is not None:
        try:
            candidate = float(value)
            if math.isfinite(candidate):
                numeric = candidate
        except (TypeError, ValueError):
            pass
    passed = numeric is not None and (
        minimum is None or numeric >= minimum
    ) and (maximum is None or numeric <= maximum)
    result: dict[str, Any] = {"value": numeric, "passed": passed}
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    return result


def _shape(value: str | list[int]) -> tuple[int, int, int]:
    parts = value.split("x") if isinstance(value, str) else value
    result = tuple(int(part) for part in parts)
    if len(result) != 3:
        raise ValueError(f"Expected a 3-D shape, got {value!r}")
    return result


def _psf_checks(row: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "ncc": _check(row["ncc"], minimum=STAGE1_LIMITS["psf_ncc_min"]),
    }
    reference_shape = _shape(row["reference_shape"])
    candidate_shape = _shape(row["candidate_shape"])
    for index, axis in enumerate(("z", "y", "x")):
        reference_offset = row[f"reference_center_{axis}_voxels"] - (
            reference_shape[index] - 1
        ) / 2
        candidate_offset = row[f"candidate_center_{axis}_voxels"] - (
            candidate_shape[index] - 1
        ) / 2
        checks[f"center_displacement_{axis}_voxels"] = _check(
            abs(candidate_offset - reference_offset),
            maximum=STAGE1_LIMITS["center_displacement_max_voxels"],
        )

        reference_fwhm = float(row[f"reference_fwhm_{axis}_voxels"])
        candidate_fwhm = float(row[f"candidate_fwhm_{axis}_voxels"])
        relative_delta = (
            abs(candidate_fwhm - reference_fwhm) / reference_fwhm
            if reference_fwhm > 0
            else float("inf")
        )
        checks[f"fwhm_relative_delta_{axis}"] = _check(
            relative_delta, maximum=STAGE1_LIMITS["fwhm_relative_delta_max"]
        )
    return checks


def _volume_checks(row: dict[str, Any], limits: dict[str, float],
                   *, include_mean: bool) -> dict[str, Any]:
    checks = {
        "ncc": _check(row["ncc"], minimum=limits["volume_ncc_min"]),
        "ssim": _check(row["ssim"], minimum=limits["volume_ssim_min"]),
        "gradient_energy_ratio": _check(
            row["gradient_energy_ratio"],
            minimum=limits["gradient_ratio_min"],
            maximum=limits["gradient_ratio_max"],
        ),
        "high_frequency_fraction_ratio": _check(
            row["high_frequency_fraction_ratio"],
            minimum=limits["frequency_ratio_min"],
            maximum=limits["frequency_ratio_max"],
        ),
    }
    if include_mean:
        reference_mean = float(row["reference_mean"])
        mean_ratio = (
            float(row["candidate_mean"]) / reference_mean
            if reference_mean != 0
            else float("inf")
        )
        checks["mean_ratio"] = _check(
            mean_ratio,
            minimum=limits["mean_ratio_min"],
            maximum=limits["mean_ratio_max"],
        )
    return checks


def _stage(checks: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def evaluate(psf_row: dict[str, Any], stage1_volume_row: dict[str, Any],
             stage2_volume_row: dict[str, Any], workflow_seconds: float | None,
             reference_seconds: float | None) -> dict[str, Any]:
    stage1_checks = _psf_checks(psf_row)
    stage1_checks.update(
        {
            f"petakit_{name}": check
            for name, check in _volume_checks(
                stage1_volume_row, STAGE1_LIMITS, include_mean=False
            ).items()
        }
    )
    stage1 = _stage(stage1_checks)
    stage2 = _stage(
        _volume_checks(stage2_volume_row, STAGE2_LIMITS, include_mean=True)
    )
    speed_evaluated = workflow_seconds is not None and reference_seconds is not None
    speed_passed = bool(
        speed_evaluated and workflow_seconds < reference_seconds
    )
    speed = {
        "evaluated": speed_evaluated,
        "passed": speed_passed,
        "workflow_seconds": workflow_seconds,
        "reference_seconds": reference_seconds,
        "speedup": (
            float(reference_seconds / workflow_seconds)
            if speed_evaluated and workflow_seconds > 0
            else None
        ),
    }
    return {
        "passed": stage1["passed"] and stage2["passed"] and speed["passed"],
        "stage1_psf_effect": stage1,
        "stage2_application": stage2,
        "end_to_end_speed": speed,
    }


def _first_row(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, list):
        if not value:
            raise ValueError(f"No metric rows in {path}")
        return value[0]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-psf", required=True)
    parser.add_argument("--stage1-volume", required=True)
    parser.add_argument("--stage2-volume", required=True)
    parser.add_argument("--timing-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enforce", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    timing = json.loads(Path(args.timing_summary).read_text(encoding="utf-8"))
    summary = evaluate(
        _first_row(args.stage1_psf),
        _first_row(args.stage1_volume),
        _first_row(args.stage2_volume),
        timing["workflow_total_seconds"],
        timing["reference_total_seconds"],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.enforce and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
