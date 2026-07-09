#!/usr/bin/env python3
"""Export final deconvolution OME-Zarr outputs to TIFF stacks."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

import tifffile

from ome_zarr_io import (
    OME_ZARR_SUFFIX,
    image_stem,
    is_ome_zarr_path,
    is_ozx_path,
    log_progress,
    open_ome_zarr_array,
    unzip_ozx_to_ome_zarr,
)


TIFF_SUFFIXES = {".tif", ".tiff"}
SUPPORTED_OUTPUT_FORMATS = {"tiff"}


def discover_decon_outputs(input_dir: Path | str) -> list[Path]:
    root = Path(input_dir)
    outputs = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name.startswith("DB2_")
        and (path.suffix.lower() in TIFF_SUFFIXES or is_ozx_path(path))
    ]
    outputs.extend(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("DB2_") and is_ome_zarr_path(path)
    )
    return sorted(outputs, key=lambda path: (0 if path.is_file() else 1, path.name))


def tiff_output_name(path: Path) -> str:
    if is_ome_zarr_path(path):
        return f"{path.name[:-len(OME_ZARR_SUFFIX)]}.tif"
    if is_ozx_path(path):
        return f"{image_stem(path)}.tif"
    return f"{image_stem(path)}.tif"


def export_directory(input_dir: Path | str, output_dir: Path | str, output_format: str = "tiff") -> list[Path]:
    normalized_format = str(output_format).lower()
    if normalized_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}. Supported export formats: tiff")

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    outputs = []
    for source in discover_decon_outputs(input_root):
        destination = output_root / tiff_output_name(source)
        if source.is_file() and source.suffix.lower() in TIFF_SUFFIXES:
            log_progress(f"Copying deconvolved TIFF output: {source.name} -> {destination}")
            shutil.copy2(source, destination)
        elif source.is_file() and is_ozx_path(source):
            log_progress(f"Exporting OZX output to TIFF: {source.name} -> {destination}")
            with tempfile.TemporaryDirectory(prefix=".ozx_export_", dir=Path.cwd()) as temp_dir:
                extracted = unzip_ozx_to_ome_zarr(
                    source,
                    Path(temp_dir) / f"{image_stem(source)}{OME_ZARR_SUFFIX}",
                )
                volume = open_ome_zarr_array(extracted, mode="r")
                tifffile.imwrite(str(destination), volume, bigtiff=True)
        else:
            log_progress(f"Exporting OME-Zarr output to TIFF: {source.name} -> {destination}")
            volume = open_ome_zarr_array(source, mode="r")
            tifffile.imwrite(str(destination), volume, bigtiff=True)
        outputs.append(destination)

    if not outputs:
        raise FileNotFoundError(f"No DB2 deconvolution outputs found in {input_root}")
    log_progress(f"Exported {len(outputs)} TIFF output(s) to {output_root}")
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export deconvolved OME-Zarr outputs to TIFF stacks.")
    parser.add_argument("--input", required=True, help="Directory containing DB2 OME-Zarr/TIFF outputs.")
    parser.add_argument("--output", required=True, help="Directory where exported TIFF outputs should be written.")
    parser.add_argument("--output-format", default="tiff", help="Requested output format. Currently supports tiff.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    export_directory(args.input, args.output, output_format=args.output_format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
