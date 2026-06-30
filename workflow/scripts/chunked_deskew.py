#!/usr/bin/env python3
"""Chunked deskew/top-view TIFF writer.

This follows the existing MATLAB geometry but avoids materialising the
resized/rotated top-view volume.  The final TIFF is written one output page at
a time.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent import futures
import math
from pathlib import Path
import shutil
import time

import numpy as np
import tifffile

from ome_zarr_io import (
    create_ome_zarr_array,
    discover_image_volumes,
    image_stem,
    is_ome_zarr_path,
    log_progress,
    open_ome_zarr_array,
)


def _selected_input_dir(image_path: str, cell_name: str | None) -> Path:
    base = Path(image_path)
    return base / cell_name if cell_name else base


def _discover_inputs(input_dir: Path) -> list[Path]:
    paths = discover_image_volumes(input_dir)
    if not paths:
        raise FileNotFoundError(f"No TIFF or OME-Zarr volumes found in {input_dir}")
    return paths


def _open_volume(path: Path) -> np.ndarray:
    if is_ome_zarr_path(path):
        log_progress(f"Opening OME-Zarr input volume: {path}")
        return open_ome_zarr_array(path, mode="r")

    try:
        log_progress(f"Opening TIFF input volume with memmap: {path}")
        volume = tifffile.memmap(str(path), mode="r")
    except Exception:
        log_progress(f"TIFF memmap failed; reading full TIFF volume: {path}")
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
    deskew_workers: int,
    deskew_prefetch: int,
) -> tuple[int, int, int]:
    start_time = time.perf_counter()
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
    write_ome_zarr = is_ome_zarr_path(output_path)
    if write_ome_zarr:
        log_progress(f"Writing deskew output as OME-Zarr: {output_path}")
        zarr_output = create_ome_zarr_array(
            output_path,
            shape=(x_size, shear_y, scaled_z),
            chunks=(1, min(256, shear_y), min(256, scaled_z)),
            dtype=np.dtype("uint16"),
            layer_name=image_stem(output_path),
        )
        writer_context = nullcontext()
        writer = None
    else:
        log_progress(f"Writing deskew output as TIFF: {output_path}")
        zarr_output = None
        writer_context = tifffile.TiffWriter(str(output_path), bigtiff=True)

    z_positions = np.arange(scaled_z, dtype=np.float64)
    source_z = _resize_source_z(z_positions, z_size, scaled_z)
    z0_all = np.floor(source_z).astype(np.int64)
    z1_all = np.clip(z0_all + 1, 0, z_size - 1)
    wz_all = (source_z - z0_all).astype(np.float32)

    def compute_page(x_out: int) -> tuple[int, np.ndarray]:
        page_start = time.perf_counter()
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
        elapsed = time.perf_counter() - page_start
        print(f"  Computed top-view page {x_out + 1}/{x_size}: compute={elapsed:.2f}s", flush=True)
        return x_out, page

    deskew_workers = max(1, int(deskew_workers))
    deskew_prefetch = max(1, int(deskew_prefetch))
    deskew_prefetch = max(deskew_workers, deskew_prefetch)
    print(
        f"  Chunked deskew scheduler: workers={deskew_workers}, "
        f"prefetch={deskew_prefetch}, pages={x_size}",
        flush=True,
    )

    with writer_context as writer:
        pending_pages: set[futures.Future[tuple[int, np.ndarray]]] = set()
        write_buffer: dict[int, np.ndarray] = {}
        next_submit = 0
        next_write = 0
        completed = 0
        last_heartbeat = time.perf_counter()
        heartbeat_seconds = 60.0

        with futures.ThreadPoolExecutor(max_workers=deskew_workers) as executor:
            while next_write < x_size:
                submitted_before = next_submit
                while next_submit < x_size and len(pending_pages) < deskew_prefetch:
                    pending_pages.add(executor.submit(compute_page, next_submit))
                    next_submit += 1
                if next_submit > submitted_before:
                    print(
                        f"  Submitted deskew pages {submitted_before + 1}-{next_submit}/"
                        f"{x_size}; pending={len(pending_pages)}, completed={completed}",
                        flush=True,
                    )

                while next_write in write_buffer:
                    page = write_buffer.pop(next_write)
                    if zarr_output is not None:
                        zarr_output[next_write, :, :] = page
                    else:
                        writer.write(
                            page,
                            photometric="minisblack",
                            compression=None,
                            contiguous=True,
                        )
                    next_write += 1
                    if next_write % 50 == 0 or next_write == x_size:
                        log_progress(f"Wrote top-view page {next_write}/{x_size}")

                if next_write >= x_size:
                    break

                done, pending_pages = futures.wait(
                    pending_pages,
                    timeout=heartbeat_seconds,
                    return_when=futures.FIRST_COMPLETED,
                )
                if not done:
                    now = time.perf_counter()
                    if now - last_heartbeat >= heartbeat_seconds:
                        print(
                            f"  Chunked deskew heartbeat: submitted={next_submit}/"
                            f"{x_size}, completed={completed}, "
                            f"written={next_write}, pending={len(pending_pages)}, "
                            f"buffered={len(write_buffer)}",
                            flush=True,
                        )
                        last_heartbeat = now
                    continue

                for future in done:
                    x_out, page = future.result()
                    write_buffer[int(x_out)] = page
                    completed += 1
    log_progress(
        f"Finished top-view deskew output: {output_path} "
        f"in {time.perf_counter() - start_time:.2f}s"
    )
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
    deskew_workers: int,
    deskew_prefetch: int,
) -> None:
    run_start = time.perf_counter()
    input_dir = _selected_input_dir(image_path, cell_name)
    output_root = Path(output_dir)
    top_shear_dir = output_root / "Top_shear"
    top_shear_dir.mkdir(parents=True, exist_ok=True)
    original_name_map = input_dir / "original_filenames.tsv"
    if original_name_map.exists():
        shutil.copy2(original_name_map, top_shear_dir / "original_filenames.tsv")
        log_progress(f"Copied original filename map to {top_shear_dir}")

    inputs = _discover_inputs(input_dir)
    log_progress(f"Chunked deskew discovered {len(inputs)} input volume(s) in {input_dir}")
    for index, path in enumerate(inputs, start=1):
        volume_start = time.perf_counter()
        log_progress(f"Processing deskew input {index}/{len(inputs)}: {path.name}")
        volume = _open_volume(path)
        log_progress(f"Opened {path.name}: shape={volume.shape}, dtype={volume.dtype}")
        output_name = f"{image_stem(path)}.ome.zarr" if is_ome_zarr_path(path) else f"{image_stem(path)}.tif"
        output_shape = _write_top_shear(
            volume,
            top_shear_dir / output_name,
            dx=float(dx),
            dz=float(dz),
            angle=float(angle),
            flip=int(flip),
            z_chunk=int(z_chunk),
            deskew_workers=int(deskew_workers),
            deskew_prefetch=int(deskew_prefetch),
        )
        (top_shear_dir / "note.txt").write_text(
            "Chunked top-view deskew output. "
            f"output_yzx={output_shape}; z pixel = x(y) pixel.\n"
        )
        log_progress(
            f"Finished deskew input {path.name}: output={output_name}, "
            f"elapsed={time.perf_counter() - volume_start:.2f}s"
        )
    log_progress(f"Chunked deskew complete in {time.perf_counter() - run_start:.2f}s")


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
    parser.add_argument("--deskew_workers", type=int, default=32)
    parser.add_argument("--deskew_prefetch", type=int, default=64)
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
        deskew_workers=max(1, args.deskew_workers),
        deskew_prefetch=max(1, args.deskew_prefetch),
    )


if __name__ == "__main__":
    main()
