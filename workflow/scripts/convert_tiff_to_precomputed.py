#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass


TIFF_SUFFIXES = {".tif", ".tiff"}
TIFF_STEM_PREFIX = "db2_"
SUPPORTED_DTYPES = {"uint16"}
VOLUME_MODES = {"auto", "2d", "3d"}


class ConversionError(Exception):
    pass


@dataclass(frozen=True)
class LayerManifest:
    name: str
    path: pathlib.Path

    @property
    def source(self):
        return f"precomputed://file://{self.path.resolve()}"


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
    if not matches:
        raise ConversionError(f"No DB2 TIFF volumes found under {root}")
    return matches


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

    try:
        data = tifffile.memmap(tiff_path)
    except ValueError:
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
        return data.reshape((1,) + data.shape)

    if data.ndim == 3:
        if volume_mode == "2d":
            raise ConversionError(f"Expected a 2D TIFF image, found 3D: {tiff_path}")
        return data

    expected = "2D or 3D" if volume_mode == "auto" else f"{volume_mode.upper()} TIFF"
    raise ConversionError(f"Expected a {expected}, found {data.ndim}D: {tiff_path}")


def write_precomputed(tiff_path, output_layer_dir, volume_mode="auto"):
    volume = load_tiff_volume(tiff_path, volume_mode=volume_mode)
    try:
        from cloudvolume import CloudVolume
    except ImportError as exc:
        raise ConversionError(
            "Missing required Python dependency 'cloud-volume'. Rebuild decon_runtime "
            "so the workflow environment includes cloud-volume."
        ) from exc

    z_size, y_size, x_size = volume.shape
    layer_path = pathlib.Path(output_layer_dir).resolve()
    layer_path.mkdir(parents=True, exist_ok=True)

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type=str(volume.dtype),
        encoding="raw",
        resolution=[1, 1, 1],
        voxel_offset=[0, 0, 0],
        chunk_size=[
            min(64, x_size),
            min(64, y_size),
            min(64, z_size),
        ],
        volume_size=[x_size, y_size, z_size],
    )
    cloud_path = f"file://{layer_path}"
    neuro_volume = CloudVolume(cloud_path, info=info, compress=False)
    neuro_volume.commit_info()
    neuro_volume[:, :, :] = volume.transpose(2, 1, 0)[:, :, :, None]
    return layer_path


def convert_directory(selected_path, output_dir, volume_mode="auto"):
    input_dir = resolve_input_directory(selected_path)
    tiff_paths = discover_tiffs(input_dir)
    output = pathlib.Path(output_dir).resolve()
    layer_names = sanitize_layer_names(tiff_paths)

    layers = []
    for tiff_path, layer_name in zip(tiff_paths, layer_names):
        layer_dir = output / layer_name
        write_precomputed(tiff_path, layer_dir, volume_mode=volume_mode)
        layers.append(LayerManifest(layer_name, layer_dir))

    return write_manifest(output, layers)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Convert DB2 TIFF volumes to Neuroglancer precomputed datasets."
    )
    parser.add_argument("--input", required=True, help="Selected file or directory to search")
    parser.add_argument("--output", required=True, help="Output neuroglancer directory")
    parser.add_argument(
        "--volume-mode",
        choices=sorted(VOLUME_MODES),
        default="auto",
        help="TIFF dimensionality to accept: auto accepts 2D or 3D, 2d requires 2D, 3d requires 3D.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    try:
        manifest_path = convert_directory(args.input, args.output, volume_mode=args.volume_mode)
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Neuroglancer manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
