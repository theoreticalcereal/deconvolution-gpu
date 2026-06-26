#!/usr/bin/env python3
"""Chunked deskew/top-view TIFF writer.

This follows the existing MATLAB geometry but avoids materialising the
resized/rotated top-view volume.  The final TIFF is written one output page at
a time.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import tifffile


def _selected_input_dir(image_path: str, cell_name: str | None) -> Path:
    base = Path(image_path)
    return base / cell_name if cell_name else base


def _discover_tiffs(input_dir: Path) -> list[Path]:
    paths = sorted([*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")])
    if not paths:
        raise FileNotFoundError(f"No TIFF files found in {input_dir}")
    return paths


def _open_volume(path: Path) -> np.ndarray:
    try:
        volume = tifffile.memmap(str(path), mode="r")
    except Exception:
        volume = tifffile.imread(str(path))
    array = np.asarray(volume)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D image or 3-D stack at {path}, got {array.shape}")
    return array


def _resize_source_z(output_z: np.ndarray, source_z_size: int, scaled_z_size: int) -> np.ndarray:
    # Match image-resize center mapping closely enough for the MATLAB top-view
    # path: output pixel centers map into input pixel centers.
    source = ((output_z.astype(np.float64) + 0.5) * source_z_size / scaled_z_size) - 0.5
    return np.clip(source, 0.0, float(source_z_size - 1))


def _build_rotation_lookup(
    *,
    shear_y: int,
    x_size: int,
    x_out: int,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_out = np.arange(shear_y, dtype=np.float64)
    center_y = (shear_y - 1) / 2.0
    center_x = (x_size - 1) / 2.0
    theta = math.radians(-angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    yy = y_out - center_y
    xx = float(x_out) - center_x
    src_y = (cos_t * yy) - (sin_t * xx) + center_y
    src_x = (sin_t * yy) + (cos_t * xx) + center_x
    src_y_i = np.rint(src_y).astype(np.int64)
    src_x_i = np.rint(src_x).astype(np.int64)
    valid = (
        (src_y_i >= 0)
        & (src_y_i < shear_y)
        & (src_x_i >= 0)
        & (src_x_i < x_size)
    )
    return src_y_i, src_x_i, valid


def _shear_column(
    volume_zyx: np.ndarray,
    *,
    source_y: np.ndarray,
    source_x: np.ndarray,
    valid_yx: np.ndarray,
    z_index: int,
    offsets: np.ndarray,
    max_yoffset: int,
) -> np.ndarray:
    z_index = int(z_index)
    y_size = int(volume_zyx.shape[1])
    out = np.zeros(source_y.shape, dtype=np.float32)

    def sample_one(z: int) -> np.ndarray:
        raw_y = source_y - int(offsets[z]) - int(max_yoffset)
        valid = valid_yx & (raw_y >= 0) & (raw_y < y_size)
        values = np.zeros(source_y.shape, dtype=np.float32)
        if np.any(valid):
            values[valid] = volume_zyx[z, raw_y[valid], source_x[valid]].astype(np.float32)
        return values

    if z_index >= int(volume_zyx.shape[0]) - 1:
        return sample_one(z_index)
    out = 0.5 * (sample_one(z_index) + sample_one(z_index + 1))
    return out


def _write_top_shear(
    volume_zyx: np.ndarray,
    output_path: Path,
    *,
    dx: float,
    dz: float,
    angle: float,
    flip: int,
    z_chunk: int,
) -> tuple[int, int, int]:
    z_size, y_size, x_size = (int(v) for v in volume_zyx.shape)
    new_dz = float(dz) * math.cos(math.radians(float(angle)))
    cz = math.floor(z_size / 2) + 1
    z_one_based = np.arange(1, z_size + 1, dtype=np.float64)
    offsets = np.rint(float(flip) * (z_one_based - cz) * (new_dz / float(dx))).astype(np.int64)
    max_yoffset = int(np.max(np.abs(offsets)))
    shear_y = int(y_size + (2 * max_yoffset))

    scale_z = float(dz) * math.sin(math.radians(float(angle))) / float(dx)
    scaled_z = max(1, int(round(z_size * scale_z)))
    output_shape = (shear_y, scaled_z, x_size)
    print(
        "Chunked top-view geometry: "
        f"input_zyx={volume_zyx.shape}, shear_y={shear_y}, "
        f"scaled_z={scaled_z}, output_yzx={output_shape}, "
        f"scale_z={scale_z:.6g}",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    z_positions = np.arange(scaled_z, dtype=np.float64)
    source_z = _resize_source_z(z_positions, z_size, scaled_z)
    z0_all = np.floor(source_z).astype(np.int64)
    z1_all = np.clip(z0_all + 1, 0, z_size - 1)
    wz_all = (source_z - z0_all).astype(np.float32)

    with tifffile.TiffWriter(str(output_path), bigtiff=True) as writer:
        for x_out in range(x_size):
            src_y, src_x, valid_yx = _build_rotation_lookup(
                shear_y=shear_y,
                x_size=x_size,
                x_out=x_out,
                angle_deg=float(flip) * float(angle),
            )
            page = np.zeros((shear_y, scaled_z), dtype=np.uint16)
            for z_start in range(0, scaled_z, int(z_chunk)):
                z_stop = min(z_start + int(z_chunk), scaled_z)
                z0 = z0_all[z_start:z_stop]
                z1 = z1_all[z_start:z_stop]
                wz = wz_all[z_start:z_stop]
                unique_z = sorted(set(z0.tolist()) | set(z1.tolist()))
                columns = {
                    z: _shear_column(
                        volume_zyx,
                        source_y=src_y,
                        source_x=src_x,
                        valid_yx=valid_yx,
                        z_index=z,
                        offsets=offsets,
                        max_yoffset=max_yoffset,
                    )
                    for z in unique_z
                }
                tile = np.empty((shear_y, z_stop - z_start), dtype=np.float32)
                for local_i, (a, b, w) in enumerate(zip(z0, z1, wz, strict=False)):
                    tile[:, local_i] = ((1.0 - float(w)) * columns[int(a)]) + (
                        float(w) * columns[int(b)]
                    )
                page[:, z_start:z_stop] = np.clip(np.rint(tile), 0, 65535).astype(np.uint16)
            writer.write(page, photometric="minisblack", compression=None, contiguous=True)
            if (x_out + 1) % 50 == 0 or (x_out + 1) == x_size:
                print(f"  Wrote top-view page {x_out + 1}/{x_size}", flush=True)
    return output_shape


def run_chunked_deskew(
    *,
    image_path: str,
    cell_name: str,
    dx: float,
    dz: float,
    angle: float,
    flip: int,
    output_dir: str,
    z_chunk: int,
) -> None:
    input_dir = _selected_input_dir(image_path, cell_name)
    output_root = Path(output_dir)
    top_shear_dir = output_root / "Top_shear"
    top_shear_dir.mkdir(parents=True, exist_ok=True)

    for path in _discover_tiffs(input_dir):
        print(f"Processing TIFF with chunked deskew: {path.name}", flush=True)
        volume = _open_volume(path)
        output_shape = _write_top_shear(
            volume,
            top_shear_dir / f"{path.stem}.tif",
            dx=float(dx),
            dz=float(dz),
            angle=float(angle),
            flip=int(flip),
            z_chunk=int(z_chunk),
        )
        (top_shear_dir / "note.txt").write_text(
            "Chunked top-view deskew output. "
            f"output_yzx={output_shape}; z pixel = x(y) pixel.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked deskew/top-view TIFF writer.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--cell_name", default="")
    parser.add_argument("--cell_index", default="")
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dz", type=float, required=True)
    parser.add_argument("--angle", type=float, required=True)
    parser.add_argument("--flip", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--z_chunk", type=int, default=256)
    args = parser.parse_args()
    if args.cell_index:
        print("cell_index is accepted for compatibility but ignored by chunked deskew.", flush=True)
    run_chunked_deskew(
        image_path=args.image_path,
        cell_name=args.cell_name,
        dx=args.dx,
        dz=args.dz,
        angle=args.angle,
        flip=args.flip,
        output_dir=args.output_dir,
        z_chunk=max(1, args.z_chunk),
    )


if __name__ == "__main__":
    main()
