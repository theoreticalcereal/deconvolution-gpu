#!/usr/bin/env python3
"""Small OME-Zarr v0.4 helpers for native workflow image volumes."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile
from datetime import datetime
from typing import Iterable


TIFF_SUFFIXES = {".tif", ".tiff"}
OME_ZARR_SUFFIX = ".ome.zarr"
OZX_SUFFIX = ".ozx"
PYRAMID_DOWNSAMPLE_FACTORS = (1, 2, 4, 8, 16)
SPATIAL_AXES_ZYX = ("z", "y", "x")


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def is_ome_zarr_path(path: Path | str) -> bool:
    return Path(path).name.lower().endswith(OME_ZARR_SUFFIX)


def is_ozx_path(path: Path | str) -> bool:
    return Path(path).name.lower().endswith(OZX_SUFFIX)


def image_stem(path: Path | str) -> str:
    path = Path(path)
    if is_ome_zarr_path(path):
        return path.name[: -len(OME_ZARR_SUFFIX)]
    if is_ozx_path(path):
        return path.name[: -len(OZX_SUFFIX)]
    if path.suffix.lower() in TIFF_SUFFIXES:
        return path.name[: -len(path.suffix)]
    return path.stem


def discover_image_volumes(input_dir: Path | str) -> list[Path]:
    root = Path(input_dir)
    if root.is_dir() and is_ome_zarr_path(root):
        return [root]
    if root.is_file() and is_ozx_path(root):
        return [root]
    paths = [
        path
        for path in root.iterdir()
        if path.is_dir() and is_ome_zarr_path(path)
    ]
    paths.extend(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )
    paths.extend(
        path
        for path in root.iterdir()
        if path.is_file() and is_ozx_path(path)
    )
    return sorted(paths, key=lambda path: path.name)


def _validate_ozx_member(name: str) -> tuple[str, ...]:
    path = Path(name)
    parts = path.parts
    if path.is_absolute() or ".." in parts:
        raise ValueError(f"Unsafe OZX archive member: {name}")
    return tuple(part for part in parts if part not in {"", "."})


def zip_ome_zarr_to_ozx(zarr_path: Path | str, archive_path: Path | str) -> Path:
    source = Path(zarr_path)
    archive = Path(archive_path)
    if not source.is_dir() or not is_ome_zarr_path(source):
        raise ValueError(f"OZX export requires an OME-Zarr directory, got {source}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    log_progress(f"Zipping OME-Zarr output: {source} -> {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
        for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
            handle.write(file_path, file_path.relative_to(source).as_posix())
    return archive.resolve()


def unzip_ozx_to_ome_zarr(archive_path: Path | str, zarr_path: Path | str) -> Path:
    archive = Path(archive_path)
    target = Path(zarr_path)
    if not is_ozx_path(archive):
        raise ValueError(f"OZX input must end with {OZX_SUFFIX}: {archive}")

    with zipfile.ZipFile(archive, "r") as handle:
        members = [info for info in handle.infolist() if not info.is_dir()]
        member_parts = [_validate_ozx_member(info.filename) for info in members]
        top_levels = {parts[0] for parts in member_parts if parts}
        has_root_zgroup = any(parts == (".zgroup",) for parts in member_parts)
        strip_prefix = None
        if not has_root_zgroup and len(top_levels) == 1:
            candidate = next(iter(top_levels))
            if candidate.lower().endswith(OME_ZARR_SUFFIX):
                strip_prefix = candidate

        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
        log_progress(f"Unzipping OZX input: {archive} -> {target}")

        for info, parts in zip(members, member_parts):
            if strip_prefix is not None:
                if not parts or parts[0] != strip_prefix:
                    raise ValueError(f"Unsafe OZX archive member: {info.filename}")
                parts = parts[1:]
            if not parts:
                continue
            destination = target.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info, "r") as source_handle, destination.open("wb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle)

    if not (target / ".zgroup").exists():
        raise ValueError(f"OZX archive does not contain an OME-Zarr root .zgroup: {archive}")
    return target.resolve()


def _selected_downsample_factors(
    max_downsample: int,
    downsample_factors: Iterable[int] = PYRAMID_DOWNSAMPLE_FACTORS,
) -> tuple[int, ...]:
    max_downsample = int(max_downsample)
    if max_downsample < 1:
        raise ValueError(f"max_downsample must be >= 1, got {max_downsample}")
    return tuple(
        int(factor)
        for factor in downsample_factors
        if int(factor) <= max_downsample
    )


def multiscales_metadata(
    layer_name: str,
    downsample_factors: Iterable[int] = PYRAMID_DOWNSAMPLE_FACTORS,
    max_downsample: int = 16,
) -> dict:
    selected_factors = _selected_downsample_factors(max_downsample, downsample_factors)
    datasets = [
        {
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": [1, int(factor), int(factor)]}
            ],
        }
        for level, factor in enumerate(selected_factors)
    ]
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": layer_name,
                "axes": [
                    {"name": axis, "type": "space"}
                    for axis in SPATIAL_AXES_ZYX
                ],
                "datasets": datasets,
            }
        ]
    }


def set_zyx_array_dimensions(array) -> None:
    array.attrs["_ARRAY_DIMENSIONS"] = list(SPATIAL_AXES_ZYX)


def downsample_xy(array, factor: int):
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"downsample factor must be >= 1, got {factor}")
    if factor == 1:
        return array
    return array[:, ::factor, ::factor]


def _normalise_chunks(chunks, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if chunks is None:
        return (min(16, shape[0]), min(256, shape[1]), min(256, shape[2]))
    normalised = []
    for axis_chunks, axis_size in zip(chunks, shape, strict=True):
        if isinstance(axis_chunks, (tuple, list)):
            chunk_size = int(axis_chunks[0])
        else:
            chunk_size = int(axis_chunks)
        normalised.append(max(1, min(chunk_size, int(axis_size))))
    return tuple(normalised)


def _downsampled_chunks(base_chunks: tuple[int, int, int], shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    return (
        max(1, min(int(base_chunks[0]), int(shape[0]))),
        max(1, min(max(1, int(base_chunks[1]) // int(factor)), int(shape[1]))),
        max(1, min(max(1, int(base_chunks[2]) // int(factor)), int(shape[2]))),
    )


def _downsampled_shape_xy(shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    return (
        int(shape[0]),
        max(1, ((int(shape[1]) - 1) // int(factor)) + 1),
        max(1, ((int(shape[2]) - 1) // int(factor)) + 1),
    )


def open_ome_zarr_array(path: Path | str, mode: str = "r"):
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'zarr' for OME-Zarr volume access") from exc

    return zarr.open(str(Path(path) / "0"), mode=mode)


def create_ome_zarr_array(
    path: Path | str,
    *,
    shape: Iterable[int],
    dtype,
    chunks: Iterable[int],
    layer_name: str | None = None,
    max_downsample: int = 16,
    overwrite: bool = True,
):
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'zarr' for OME-Zarr volume access") from exc

    zarr_path = Path(path)
    if overwrite and zarr_path.exists():
        log_progress(f"Removing existing OME-Zarr output: {zarr_path}")
        shutil.rmtree(zarr_path)
    zarr_path.mkdir(parents=True, exist_ok=True)
    (zarr_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}) + "\n")
    (zarr_path / ".zattrs").write_text(
        json.dumps(
            multiscales_metadata(layer_name or image_stem(zarr_path), max_downsample=max_downsample),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    log_progress(
        "Creating OME-Zarr array: "
        f"path={zarr_path}, shape={tuple(int(axis) for axis in shape)}, "
        f"chunks={tuple(int(axis) for axis in chunks)}, dtype={dtype}"
    )
    array = zarr.open(
        str(zarr_path / "0"),
        mode="w",
        shape=tuple(int(axis) for axis in shape),
        chunks=tuple(int(axis) for axis in chunks),
        dtype=dtype,
        compressor=None,
    )
    set_zyx_array_dimensions(array)
    return array


def write_downsampled_pyramid(
    path: Path | str,
    *,
    max_downsample: int = 16,
    downsample_factors: Iterable[int] = PYRAMID_DOWNSAMPLE_FACTORS,
) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'zarr' for OME-Zarr pyramid generation") from exc

    zarr_path = Path(path)
    source = zarr.open(str(zarr_path / "0"), mode="r")
    source_shape = tuple(int(axis) for axis in source.shape)
    if len(source_shape) != 3:
        raise ValueError(f"OME-Zarr pyramid source must be 3-D, got shape {source_shape}")
    source_chunks = _normalise_chunks(getattr(source, "chunks", None), source_shape)
    dtype = getattr(source, "dtype", None)

    for level, factor in enumerate(_selected_downsample_factors(max_downsample, downsample_factors)):
        if level == 0 or factor == 1:
            continue
        shape = _downsampled_shape_xy(source_shape, factor)
        chunks = _downsampled_chunks(source_chunks, shape, factor)
        log_progress(
            "Writing OME-Zarr pyramid level: "
            f"path={zarr_path / str(level)}, downsample={factor}x, "
            f"shape={shape}, chunks={chunks}"
        )
        target = zarr.open(
            str(zarr_path / str(level)),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=None,
        )
        set_zyx_array_dimensions(target)
        for z_start in range(0, shape[0], chunks[0]):
            z_stop = min(z_start + chunks[0], shape[0])
            target[z_start:z_stop, :, :] = source[z_start:z_stop, ::factor, ::factor]


def write_ome_zarr_array(
    path: Path | str,
    array,
    *,
    chunks=None,
    layer_name: str | None = None,
    max_downsample: int = 16,
) -> Path:
    shape = tuple(int(axis) for axis in array.shape)
    if len(shape) != 3:
        raise ValueError(f"OME-Zarr workflow volumes must be 3-D, got shape {shape}")
    if chunks is None:
        chunks = (min(16, shape[0]), min(256, shape[1]), min(256, shape[2]))
    output = Path(path)
    log_progress(
        "Writing OME-Zarr volume: "
        f"path={output}, shape={shape}, chunks={tuple(int(axis) for axis in chunks)}, "
        f"dtype={array.dtype}"
    )
    zarr_array = create_ome_zarr_array(
        output,
        shape=shape,
        dtype=array.dtype,
        chunks=chunks,
        layer_name=layer_name,
        max_downsample=max_downsample,
    )
    zarr_array[:] = array
    write_downsampled_pyramid(output, max_downsample=max_downsample)
    log_progress(f"Finished OME-Zarr volume: {output.resolve()}")
    return output.resolve()
