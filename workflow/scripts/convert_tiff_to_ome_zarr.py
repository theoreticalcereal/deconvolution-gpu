#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
import shutil
import sys
import os
from datetime import datetime
from dataclasses import dataclass


TIFF_SUFFIXES = {".tif", ".tiff"}
TIFF_STEM_PREFIX = "db2_"
SUPPORTED_DTYPES = {"uint16"}
VOLUME_MODES = {"auto", "2d", "3d"}
DEFAULT_MAX_LEVELS = 5


class ConversionError(Exception):
    pass


def log_progress(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass(frozen=True)
class LayerManifest:
    name: str
    path: pathlib.Path

    @property
    def source(self):
        return f"zarr://{self.path.resolve().as_uri()}"


def resolve_input_directory(selected_path):
    path = pathlib.Path(selected_path).expanduser().resolve()
    if not path.exists():
        raise ConversionError(f"Selected path does not exist: {path}")
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    raise ConversionError(f"Selected path is not a file or directory: {path}")


def discover_tiffs(input_dir):
    root = pathlib.Path(input_dir)
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TIFF_SUFFIXES
        and path.stem.lower().startswith(TIFF_STEM_PREFIX)
    ]
    matches = sorted(set(matches), key=lambda path: path.relative_to(root).as_posix())
    return matches


def discover_ome_zarrs(input_dir):
    root = pathlib.Path(input_dir)
    matches = [
        path
        for path in root.rglob("*.ome.zarr")
        if path.is_dir()
        and path.name.lower().startswith(TIFF_STEM_PREFIX)
        and (path / ".zattrs").is_file()
    ]
    return sorted(set(matches), key=lambda path: path.relative_to(root).as_posix())


def sanitize_layer_name(path):
    stem = pathlib.Path(path).stem
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    return sanitized or "layer"


def sanitize_layer_names(paths):
    counts = {}
    names = []
    for path in paths:
        base_name = sanitize_layer_name(path)
        count = counts.get(base_name, 0) + 1
        counts[base_name] = count
        names.append(base_name if count == 1 else f"{base_name}_{count}")
    return names


def ome_zarr_layer_name(path):
    name = pathlib.Path(path).name
    suffix = ".ome.zarr"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return pathlib.Path(name).stem


def write_manifest(output_dir, layers):
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "layers.json"
    payload = {
        "layers": [
            {
                "name": layer.name,
                "source": layer.source,
            }
            for layer in layers
        ]
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log_progress(f"Wrote Neuroglancer OME-Zarr manifest: {manifest_path}")
    return manifest_path


def normalize_volume_mode(volume_mode):
    normalized = str(volume_mode).lower()
    if normalized not in VOLUME_MODES:
        raise ConversionError(
            f"Unsupported Neuroglancer volume mode '{volume_mode}'; expected one of: "
            f"{', '.join(sorted(VOLUME_MODES))}"
        )
    return normalized


def load_tiff_volume(tiff_path, volume_mode="auto"):
    volume_mode = normalize_volume_mode(volume_mode)
    try:
        import tifffile
    except ImportError as exc:
        raise ConversionError(
            "Missing required Python dependency 'tifffile'. Rebuild decon_runtime "
            "before running Neuroglancer conversion."
        ) from exc

    log_progress(f"Loading TIFF for OME-Zarr conversion: {tiff_path}")
    try:
        data = tifffile.memmap(tiff_path)
    except ValueError:
        log_progress(f"TIFF memmap unavailable; reading full image: {tiff_path}")
        data = tifffile.imread(tiff_path)

    dtype = str(data.dtype)
    if dtype not in SUPPORTED_DTYPES:
        raise ConversionError(
            f"Unsupported TIFF datatype '{dtype}' for {tiff_path}; supported: "
            f"{', '.join(sorted(SUPPORTED_DTYPES))}"
        )

    if data.ndim == 2:
        if volume_mode == "3d":
            raise ConversionError(f"Expected a 3D TIFF volume, found 2D: {tiff_path}")
        volume = data.reshape((1,) + data.shape)
        log_progress(f"Loaded 2-D TIFF as one-Z-slice volume: shape={volume.shape}, dtype={volume.dtype}")
        return volume

    if data.ndim == 3:
        if volume_mode == "2d" and int(data.shape[0]) != 1:
            raise ConversionError(f"Expected a 2D TIFF image, found 3D: {tiff_path}")
        log_progress(f"Loaded 3-D TIFF volume: shape={data.shape}, dtype={data.dtype}")
        return data

    expected = "2D or 3D" if volume_mode == "auto" else f"{volume_mode.upper()} TIFF"
    raise ConversionError(f"Expected a {expected}, found {data.ndim}D: {tiff_path}")


def zarr_dtype(dtype):
    text = str(dtype)
    if text == "uint16":
        return "<u2"
    raise ConversionError(f"Unsupported OME-Zarr dtype '{dtype}'")


def bounded_chunks(shape):
    z_size, y_size, x_size = shape
    return [min(16, z_size), min(256, y_size), min(256, x_size)]


def pyramid_shapes(shape, max_levels=DEFAULT_MAX_LEVELS):
    shapes = [tuple(int(axis) for axis in shape)]
    while len(shapes) < max_levels:
        z_size, y_size, x_size = shapes[-1]
        if y_size <= 1 and x_size <= 1:
            break
        shapes.append((z_size, max(1, (y_size + 1) // 2), max(1, (x_size + 1) // 2)))
    return shapes


def downsample_xy(volume):
    try:
        import numpy as np
    except ImportError as exc:
        raise ConversionError(
            "Missing required Python dependency 'numpy'. Rebuild decon_runtime before "
            "running OME-Zarr conversion."
        ) from exc

    array = np.asarray(volume)
    z_size, y_size, x_size = array.shape
    padded_y = y_size + (y_size % 2)
    padded_x = x_size + (x_size % 2)
    if padded_y != y_size or padded_x != x_size:
        padded = np.zeros((z_size, padded_y, padded_x), dtype=array.dtype)
        padded[:, :y_size, :x_size] = array
        array = padded

    return array.reshape(z_size, padded_y // 2, 2, padded_x // 2, 2).max(axis=(2, 4))


def chunk_bytes(chunk):
    if hasattr(chunk, "tobytes"):
        return chunk.tobytes(order="C")

    try:
        import numpy as np
    except ImportError as exc:
        raise ConversionError(
            "Missing required Python dependency 'numpy'. Rebuild decon_runtime before "
            "running OME-Zarr conversion."
        ) from exc
    return np.asarray(chunk).tobytes(order="C")


def write_zarr_array(array, array_dir):
    array_path = pathlib.Path(array_dir)
    array_path.mkdir(parents=True, exist_ok=True)
    shape = [int(axis) for axis in array.shape]
    chunks = bounded_chunks(shape)
    (array_path / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": shape,
                "chunks": chunks,
                "dtype": zarr_dtype(array.dtype),
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    z_chunk, y_chunk, x_chunk = chunks
    z_size, y_size, x_size = shape
    total_chunks = (
        ((z_size + z_chunk - 1) // z_chunk)
        * ((y_size + y_chunk - 1) // y_chunk)
        * ((x_size + x_chunk - 1) // x_chunk)
    )
    written = 0
    log_progress(
        f"Writing Zarr scale array: path={array_path}, shape={shape}, "
        f"chunks={chunks}, total_chunks={total_chunks}"
    )
    for z_start in range(0, z_size, z_chunk):
        z_end = min(z_start + z_chunk, z_size)
        for y_start in range(0, y_size, y_chunk):
            y_end = min(y_start + y_chunk, y_size)
            for x_start in range(0, x_size, x_chunk):
                x_end = min(x_start + x_chunk, x_size)
                chunk = array[z_start:z_end, y_start:y_end, x_start:x_end]
                chunk_name = f"{z_start // z_chunk}.{y_start // y_chunk}.{x_start // x_chunk}"
                (array_path / chunk_name).write_bytes(chunk_bytes(chunk))
                written += 1
                if written % 100 == 0 or written == total_chunks:
                    log_progress(f"  Wrote Zarr chunks {written}/{total_chunks} for {array_path}")


def multiscales_metadata(layer_name, shapes):
    datasets = []
    for index, _shape in enumerate(shapes):
        scale = 2 ** index
        datasets.append(
            {
                "path": str(index),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1, scale, scale],
                    }
                ],
            }
        )
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": layer_name,
                "axes": [
                    {"name": "z", "type": "space", "unit": "pixel"},
                    {"name": "y", "type": "space", "unit": "pixel"},
                    {"name": "x", "type": "space", "unit": "pixel"},
                ],
                "datasets": datasets,
            }
        ]
    }


def write_ome_zarr(tiff_path, output_layer_dir, volume_mode="auto", max_levels=DEFAULT_MAX_LEVELS):
    start = datetime.now()
    volume = load_tiff_volume(tiff_path, volume_mode=volume_mode)
    layer_path = pathlib.Path(output_layer_dir).resolve()
    if layer_path.exists():
        log_progress(f"Removing existing OME-Zarr layer: {layer_path}")
        shutil.rmtree(layer_path)
    layer_path.mkdir(parents=True)
    (layer_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}) + "\n")

    shapes = pyramid_shapes(volume.shape, max_levels=max_levels)
    current = volume
    written_shapes = []
    for index, expected_shape in enumerate(shapes):
        if tuple(current.shape) != tuple(expected_shape):
            break
        log_progress(f"Writing OME-Zarr scale {index}: shape={tuple(current.shape)}")
        write_zarr_array(current, layer_path / str(index))
        written_shapes.append(tuple(current.shape))
        if index < len(shapes) - 1:
            current = downsample_xy(current)

    (layer_path / ".zattrs").write_text(
        json.dumps(multiscales_metadata(ome_zarr_layer_name(layer_path), written_shapes), indent=2, sort_keys=True)
        + "\n"
    )
    elapsed = (datetime.now() - start).total_seconds()
    log_progress(f"Finished OME-Zarr layer: {layer_path} in {elapsed:.2f}s")
    return layer_path


def hardlink_or_copy_file(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_existing_ome_zarr(source_path, output_layer_dir):
    source = pathlib.Path(source_path).resolve()
    target = pathlib.Path(output_layer_dir).resolve()
    if source == target:
        log_progress(f"Existing OME-Zarr layer already in output directory: {target}")
        return target
    if target.exists():
        log_progress(f"Removing existing staged OME-Zarr layer: {target}")
        shutil.rmtree(target)
    log_progress(f"Staging existing OME-Zarr layer: {source} -> {target}")
    shutil.copytree(source, target, copy_function=hardlink_or_copy_file)
    log_progress(f"Finished staging OME-Zarr layer: {target}")
    return target


def convert_directory(
    selected_path,
    output_dir,
    manifest_dir=None,
    volume_mode="auto",
    max_levels=DEFAULT_MAX_LEVELS,
):
    input_dir = resolve_input_directory(selected_path)
    log_progress(f"Neuroglancer conversion scanning: {input_dir}")
    tiff_paths = discover_tiffs(input_dir)
    zarr_paths = discover_ome_zarrs(input_dir)
    log_progress(
        f"Found {len(tiff_paths)} DB2 TIFF output(s) and "
        f"{len(zarr_paths)} DB2 OME-Zarr output(s)"
    )
    output = pathlib.Path(output_dir).resolve()
    manifest_output = pathlib.Path(manifest_dir).resolve() if manifest_dir else output
    tiff_layer_names = sanitize_layer_names(tiff_paths)
    zarr_layer_names = sanitize_layer_names(
        [path.with_name(ome_zarr_layer_name(path)) for path in zarr_paths]
    )

    layers = []
    for index, (tiff_path, layer_name) in enumerate(zip(tiff_paths, tiff_layer_names), start=1):
        layer_dir = output / f"{layer_name}.ome.zarr"
        log_progress(f"Converting TIFF layer {index}/{len(tiff_paths)}: {tiff_path.name}")
        write_ome_zarr(tiff_path, layer_dir, volume_mode=volume_mode, max_levels=max_levels)
        layers.append(LayerManifest(layer_name, layer_dir))
    for index, (zarr_path, layer_name) in enumerate(zip(zarr_paths, zarr_layer_names), start=1):
        layer_dir = output / f"{layer_name}.ome.zarr"
        log_progress(f"Adding existing OME-Zarr layer {index}/{len(zarr_paths)}: {zarr_path.name}")
        staged_layer = stage_existing_ome_zarr(zarr_path, layer_dir)
        layers.append(LayerManifest(layer_name, staged_layer))

    if not layers:
        raise ConversionError(f"No DB2 TIFF or OME-Zarr volumes found under {input_dir}")

    return write_manifest(manifest_output, layers)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Convert DB2 TIFF outputs to chunked multiscale OME-Zarr for Neuroglancer."
    )
    parser.add_argument("--input", required=True, help="Selected file or directory to search")
    parser.add_argument("--output", required=True, help="Output directory for OME-Zarr datasets")
    parser.add_argument(
        "--manifest-output",
        help="Directory where layers.json should be written. Defaults to --output.",
    )
    parser.add_argument(
        "--volume-mode",
        choices=sorted(VOLUME_MODES),
        default="auto",
        help="TIFF dimensionality to accept: auto accepts 2D or 3D, 2d requires 2D, 3d requires 3D.",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=DEFAULT_MAX_LEVELS,
        help="Maximum OME-Zarr multiscale levels to write.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    try:
        manifest_path = convert_directory(
            args.input,
            args.output,
            manifest_dir=args.manifest_output,
            volume_mode=args.volume_mode,
            max_levels=args.max_levels,
        )
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    log_progress(f"OME-Zarr visualization step complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
