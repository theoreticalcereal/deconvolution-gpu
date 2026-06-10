import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

from psf_estimation import (
    _normalise_psf,
    _run_matlab_deconvblind,
    _write_chunk,
    generate_theoretical_psf,
    estimate_psf_from_chunks,
    open_tiff_memmap,
    select_blind_z_window,
)


def _find_input_tiff(image_dir: Path, index: int) -> Path:
    tiffs = sorted(
        list(image_dir.glob("CH*_registered_consistent.tif"))
        + list(image_dir.glob("CH*_registered_consistent.tiff"))
    )
    if not tiffs:
        raise FileNotFoundError(f"No CH*_registered_consistent TIFFs found in {image_dir}")
    if index < 0 or index >= len(tiffs):
        raise IndexError(f"TIFF index {index} out of range for {len(tiffs)} file(s)")
    return tiffs[index]


def _center_crop_xy(volume: np.ndarray, crop_xy: int) -> np.ndarray:
    if crop_xy <= 0:
        return np.asarray(volume)
    _, ny, nx = volume.shape
    crop_y = min(crop_xy, ny)
    crop_x = min(crop_xy, nx)
    y0 = max(0, (ny - crop_y) // 2)
    x0 = max(0, (nx - crop_x) // 2)
    return np.asarray(volume[:, y0:y0 + crop_y, x0:x0 + crop_x])


def _run_full_blind(
    volume: np.ndarray,
    psf_seed: np.ndarray,
    n_iters: int,
    pad_z: int,
    script_dir: Path,
    matlab_threads: int,
    matlab_timeout: int,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="full_blind_psf_") as tmpdir:
        tmpdir = Path(tmpdir)
        chunk_path = tmpdir / "full_window.tif"
        seed_path = tmpdir / "seed.tif"
        psf_out_path = tmpdir / "full_blind_psf.tif"
        _write_chunk(volume, chunk_path)
        _run_matlab_deconvblind(
            chunk_path=chunk_path,
            psf_seed=psf_seed,
            psf_seed_path=seed_path,
            output_psf_path=psf_out_path,
            n_iters=n_iters,
            pad_z=pad_z,
            script_dir=script_dir,
            matlab_threads=matlab_threads,
            matlab_timeout=matlab_timeout,
        )
        if not psf_out_path.exists():
            raise RuntimeError("MATLAB produced no full-blind PSF output")
        psf = imread(str(psf_out_path)).astype(np.float32)
    if psf.shape != psf_seed.shape:
        raise ValueError(f"Full-blind PSF shape {psf.shape} != seed shape {psf_seed.shape}")
    return _normalise_psf(psf)


def _fwhm_pixels(line: np.ndarray) -> float:
    line = np.asarray(line, dtype=np.float32)
    if line.size == 0 or float(line.max()) <= 0:
        return 0.0
    half = float(line.max()) * 0.5
    above = np.flatnonzero(line >= half)
    if above.size == 0:
        return 0.0
    left = float(above[0])
    right = float(above[-1])
    if above[0] > 0:
        i = above[0]
        denom = float(line[i] - line[i - 1])
        if denom != 0:
            left = (i - 1) + (half - float(line[i - 1])) / denom
    if above[-1] < line.size - 1:
        i = above[-1]
        denom = float(line[i + 1] - line[i])
        if denom != 0:
            right = i + (half - float(line[i])) / denom
    return max(0.0, right - left)


def _psf_stats(name: str, psf: np.ndarray, dxy: float, dz: float) -> dict:
    zc, yc, xc = np.unravel_index(int(np.argmax(psf)), psf.shape)
    fwhm_x = _fwhm_pixels(psf[zc, yc, :])
    fwhm_y = _fwhm_pixels(psf[zc, :, xc])
    fwhm_z = _fwhm_pixels(psf[:, yc, xc])
    return {
        "name": name,
        "shape": list(psf.shape),
        "peak_index_zyx": [int(zc), int(yc), int(xc)],
        "sum": float(psf.sum()),
        "max": float(psf.max()),
        "fwhm_x_px": fwhm_x,
        "fwhm_y_px": fwhm_y,
        "fwhm_z_px": fwhm_z,
        "fwhm_x_um": fwhm_x * dxy,
        "fwhm_y_um": fwhm_y * dxy,
        "fwhm_z_um": fwhm_z * dz,
    }


def _pair_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = _normalise_psf(a)
    b = _normalise_psf(b)
    av = a.ravel().astype(np.float64)
    bv = b.ravel().astype(np.float64)
    da = av - av.mean()
    db = bv - bv.mean()
    denom = np.linalg.norm(da) * np.linalg.norm(db)
    ncc = float(np.dot(da, db) / denom) if denom > 0 else 0.0

    data_range = max(float(av.max()), float(bv.max())) - min(float(av.min()), float(bv.min()))
    if data_range <= 0:
        ssim = 1.0
    else:
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        mux = float(av.mean())
        muy = float(bv.mean())
        varx = float(av.var())
        vary = float(bv.var())
        cov = float(((av - mux) * (bv - muy)).mean())
        ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / (
            (mux * mux + muy * muy + c1) * (varx + vary + c2)
        )

    diff = av - bv
    return {
        "ncc": ncc,
        "ssim_global": float(ssim),
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def _normalise_plane(plane: np.ndarray) -> np.ndarray:
    plane = np.asarray(plane, dtype=np.float32)
    plane = plane - float(plane.min())
    max_value = float(plane.max())
    if max_value > 0:
        plane = plane / max_value
    return np.clip(np.rint(plane * 65535), 0, 65535).astype(np.uint16)


def _resize_nearest(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out_y, out_x = shape
    in_y, in_x = plane.shape
    y_idx = np.linspace(0, in_y - 1, out_y).astype(int)
    x_idx = np.linspace(0, in_x - 1, out_x).astype(int)
    return plane[np.ix_(y_idx, x_idx)]


def _make_cross_section_montage(psfs: dict[str, np.ndarray]) -> np.ndarray:
    panels = []
    target_shape = (160, 160)
    for name in ("theoretical", "chunked_blind", "full_blind"):
        psf = psfs[name]
        zc, yc, xc = np.unravel_index(int(np.argmax(psf)), psf.shape)
        planes = [
            psf[zc, :, :],
            psf[:, yc, :],
            psf[:, :, xc],
        ]
        row = [_resize_nearest(_normalise_plane(p), target_shape) for p in planes]
        panels.append(np.concatenate(row, axis=1))
    for left, right in (("chunked_blind", "full_blind"), ("chunked_blind", "theoretical")):
        diff = np.abs(psfs[left] - psfs[right])
        zc, yc, xc = np.unravel_index(int(np.argmax(psfs[left])), psfs[left].shape)
        planes = [
            diff[zc, :, :],
            diff[:, yc, :],
            diff[:, :, xc],
        ]
        row = [_resize_nearest(_normalise_plane(p), target_shape) for p in planes]
        panels.append(np.concatenate(row, axis=1))
    return np.stack(panels, axis=0)


def _write_metrics(metrics: dict, output_dir: Path) -> None:
    (output_dir / "psf_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = ["comparison\tncc\tssim_global\tmse\tmae\tmax_abs_diff"]
    for name, values in metrics["comparisons"].items():
        rows.append(
            "\t".join(
                [
                    name,
                    f"{values['ncc']:.8g}",
                    f"{values['ssim_global']:.8g}",
                    f"{values['mse']:.8g}",
                    f"{values['mae']:.8g}",
                    f"{values['max_abs_diff']:.8g}",
                ]
            )
        )
    (output_dir / "psf_metrics.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare theoretical, chunked blind, and full blind PSFs.")
    parser.add_argument("--image_path", required=True, help="Deskewed Top_shear directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--script_dir", default=str(Path(__file__).parent))
    parser.add_argument("--tiff_index", type=int, default=0)
    parser.add_argument("--sanity_xy", type=int, default=512,
                        help="Center XY crop for sanity check. <=0 uses full XY.")
    parser.add_argument("--blind_iters", type=int, default=3)
    parser.add_argument("--chunk_xy", type=int, default=256)
    parser.add_argument("--pad_xy", type=int, default=32)
    parser.add_argument("--pad_z", type=int, default=20)
    parser.add_argument("--blind_workers", type=int, default=1)
    parser.add_argument("--matlab_threads", type=int, default=1)
    parser.add_argument("--matlab_workers", type=int, default=1)
    parser.add_argument("--matlab_timeout", type=int, default=1800)
    parser.add_argument("--blind_z_slices", type=int, default=64)
    parser.add_argument("--snr_weight_cap", type=float, default=100.0)
    parser.add_argument("--prefetch_chunks", type=int, default=0)
    parser.add_argument("--vram_gb", type=float, default=None)
    parser.add_argument("--na", type=float, default=1.0)
    parser.add_argument("--wavelength", type=float, default=0.520)
    parser.add_argument("--ni", type=float, default=1.515)
    parser.add_argument("--dxy", type=float, default=0.118)
    parser.add_argument("--dz", type=float, default=0.118)
    parser.add_argument("--psf_size_z", type=int, default=101)
    parser.add_argument("--psf_size_xy", type=int, default=61)
    parser.add_argument("--background", type=float, default=0.0)
    args = parser.parse_args()

    image_dir = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(args.script_dir)

    input_tiff = _find_input_tiff(image_dir, args.tiff_index)
    print(f"Using TIFF for PSF sanity check: {input_tiff}", flush=True)

    psf_seed = generate_theoretical_psf(
        na=args.na,
        wavelength=args.wavelength,
        ni=args.ni,
        dxy=args.dxy,
        dz=args.dz,
        psf_size_z=args.psf_size_z,
        psf_size_xy=args.psf_size_xy,
        background=args.background,
    )

    volume = open_tiff_memmap(input_tiff)
    z_window, z_detail = select_blind_z_window(volume, args.blind_z_slices)
    cropped = _center_crop_xy(volume[z_window], args.sanity_xy)
    crop_tiff = output_dir / "sanity_input_crop.tif"
    imwrite(str(crop_tiff), cropped)
    print(
        f"Saved sanity input crop: {crop_tiff} shape={cropped.shape}; {z_detail}",
        flush=True,
    )

    print("Estimating chunked blind PSF on sanity crop...", flush=True)
    chunked_blind = estimate_psf_from_chunks(
        image_path=crop_tiff,
        psf_seed=psf_seed,
        n_iters=args.blind_iters,
        chunk_xy=args.chunk_xy,
        pad_xy=args.pad_xy,
        pad_z=args.pad_z,
        script_dir=script_dir,
        max_workers=args.blind_workers,
        prefetch_chunks=args.prefetch_chunks,
        vram_gb=args.vram_gb,
        cache_dir=output_dir / ".psf_cache",
        use_cache=False,
        matlab_threads=args.matlab_threads,
        matlab_workers=args.matlab_workers,
        matlab_timeout=args.matlab_timeout,
        snr_weight_cap=args.snr_weight_cap,
        blind_z_slices=args.blind_z_slices,
    )

    print("Estimating full-window blind PSF on same sanity crop/window...", flush=True)
    full_volume = open_tiff_memmap(crop_tiff)
    full_window = np.asarray(full_volume)
    print(f"Full blind window shape={full_window.shape}", flush=True)
    full_blind = _run_full_blind(
        volume=full_window,
        psf_seed=psf_seed,
        n_iters=args.blind_iters,
        pad_z=args.pad_z,
        script_dir=script_dir,
        matlab_threads=args.matlab_threads,
        matlab_timeout=args.matlab_timeout,
    )

    psfs = {
        "theoretical": _normalise_psf(psf_seed),
        "chunked_blind": _normalise_psf(chunked_blind),
        "full_blind": _normalise_psf(full_blind),
    }
    for name, psf in psfs.items():
        imwrite(str(output_dir / f"{name}_psf.tif"), psf.astype(np.float32, copy=False))

    metrics = {
        "input_tiff": str(input_tiff),
        "sanity_crop_shape_zyx": list(cropped.shape),
        "full_blind_window_shape_zyx": list(full_window.shape),
        "blind_z_window": [z_window.start, z_window.stop],
        "blind_z_window_detail": z_detail,
        "parameters": vars(args),
        "psfs": {
            name: _psf_stats(name, psf, dxy=args.dxy, dz=args.dz)
            for name, psf in psfs.items()
        },
        "comparisons": {
            "chunked_blind_vs_full_blind": _pair_metrics(psfs["chunked_blind"], psfs["full_blind"]),
            "chunked_blind_vs_theoretical": _pair_metrics(psfs["chunked_blind"], psfs["theoretical"]),
            "full_blind_vs_theoretical": _pair_metrics(psfs["full_blind"], psfs["theoretical"]),
        },
    }
    _write_metrics(metrics, output_dir)
    imwrite(str(output_dir / "psf_cross_sections.tif"), _make_cross_section_montage(psfs))
    print(f"PSF sanity outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
