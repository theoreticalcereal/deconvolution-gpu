import argparse
import inspect
import json
import re
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

from decon_wrapper import deconvolve_tiff
from psf_estimation import (
    _normalise_psf,
    _write_chunk,
    _write_matlab_stack,
    generate_theoretical_psf,
    estimate_psf_from_chunks,
    open_tiff_memmap,
    resolve_dxy,
    select_blind_z_window,
)

CHANNEL_TIMEPOINT_RE = re.compile(r"^CH(?P<channel>\d+)_(?P<timepoint>\d+)(?:_registered_consistent)?$")


def _parse_int_filter(value: str | None) -> set[int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    tokens = [token for token in re.split(r"[\s,]+", text) if token]
    return {int(token) for token in tokens}


def _find_input_tiff(
    image_dir: Path,
    index: int,
    channels: set[int] | None,
    timepoints: set[int] | None,
) -> Path:
    tiffs = sorted(
        list(image_dir.glob("CH*.tif"))
        + list(image_dir.glob("CH*.tiff"))
    )
    filtered = []
    for tiff in tiffs:
        match = CHANNEL_TIMEPOINT_RE.match(tiff.stem)
        if not match:
            continue
        channel = int(match.group("channel"))
        timepoint = int(match.group("timepoint"))
        if channels is not None and channel not in channels:
            continue
        if timepoints is not None and timepoint not in timepoints:
            continue
        filtered.append(tiff)
    tiffs = filtered
    if not tiffs:
        raise FileNotFoundError(f"No matching CH*.tif TIFFs found in {image_dir}")
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
    pad_xy: int,
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
        _write_matlab_stack(psf_seed, seed_path, scale_float=True)

        matlab_threads = min(2, max(1, matlab_threads))
        pad_xy = max(0, pad_xy)
        pad_z = max(0, pad_z)
        matlab_thread_cmd = f"maxNumCompThreads({matlab_threads}); "
        pad_cmd = (
            f"chunk = padarray(chunk, [{pad_xy} {pad_xy} {pad_z}], 'symmetric'); "
            if pad_xy > 0 or pad_z > 0 else ""
        )
        matlab_cmd = (
            f"addpath('{script_dir}'); "
            f"{matlab_thread_cmd}"
            f"chunk = single(readtiffstack('{chunk_path}')); "
            f"psf_seed = single(readtiffstack('{seed_path}')); "
            f"psf_seed = psf_seed / sum(psf_seed(:)); "
            f"{pad_cmd}"
            f"[~, psf_est] = deconvblind(chunk, psf_seed, {n_iters}); "
            f"psf_est = single(psf_est); "
            f"psf_est = psf_est / sum(psf_est(:)); "
            f"writetiffstack(psf_est, '{psf_out_path}');"
        )
        matlab_args = ["matlab"]
        if matlab_threads == 1:
            matlab_args.append("-singleCompThread")
        matlab_args.extend(["-batch", matlab_cmd])
        env = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = str(matlab_threads)

        try:
            result = subprocess.run(
                matlab_args,
                capture_output=True,
                text=True,
                env=env,
                timeout=matlab_timeout if matlab_timeout > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MATLAB full-window deconvblind timed out after {matlab_timeout}s.\n"
                f"STDOUT: {exc.stdout or ''}\nSTDERR: {exc.stderr or ''}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB full-window deconvblind failed (returncode={result.returncode}).\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        if not psf_out_path.exists():
            raise RuntimeError("MATLAB produced no full-blind PSF output")
        psf = imread(str(psf_out_path)).astype(np.float32)
    if psf.shape != psf_seed.shape:
        raise ValueError(f"Full-blind PSF shape {psf.shape} != seed shape {psf_seed.shape}")
    return _normalise_psf(psf)


def _run_matlab_deconvlucy(
    volume: np.ndarray,
    psf: np.ndarray,
    n_iters: int,
    pad_xy: int,
    pad_z: int,
    script_dir: Path,
    matlab_threads: int,
    matlab_timeout: int,
) -> np.ndarray:
    """
    Mirror the reference script's second pass:
    pad image -> deconvlucy(image, psfr, iter) -> crop -> rescale to input range.
    """
    with tempfile.TemporaryDirectory(prefix="full_lucy_decon_") as tmpdir:
        tmpdir = Path(tmpdir)
        image_path = tmpdir / "full_window.tif"
        psf_path = tmpdir / "psfr.tif"
        output_path = tmpdir / "Dec2.tif"
        _write_chunk(volume, image_path)
        _write_matlab_stack(psf, psf_path, scale_float=True)

        matlab_threads = min(2, max(1, matlab_threads))
        pad_xy = max(0, pad_xy)
        pad_z = max(0, pad_z)
        matlab_thread_cmd = f"maxNumCompThreads({matlab_threads}); "
        pad_cmd = (
            f"E1 = padarray(E1, [{pad_xy} {pad_xy} {pad_z}], 'symmetric'); "
            if pad_xy > 0 or pad_z > 0 else ""
        )
        crop_cmd = (
            f"Dec2 = Dec2({pad_xy + 1}:{pad_xy}+mImage, "
            f"{pad_xy + 1}:{pad_xy}+nImage, "
            f"{pad_z + 1}:{pad_z}+NumberImages); "
        )
        matlab_cmd = (
            f"addpath('{script_dir}'); "
            f"{matlab_thread_cmd}"
            f"FinalImage = readtiffstack('{image_path}'); "
            f"mImage = size(FinalImage, 1); "
            f"nImage = size(FinalImage, 2); "
            f"NumberImages = size(FinalImage, 3); "
            f"E1 = single(FinalImage); "
            f"maxE1 = max(E1(:)); "
            f"minE1 = min(E1(:)); "
            f"psfr = single(readtiffstack('{psf_path}')); "
            f"psfr = psfr / sum(psfr(:)); "
            f"{pad_cmd}"
            f"Dec2 = deconvlucy(E1, psfr, {n_iters}); "
            f"{crop_cmd}"
            f"decMin = min(Dec2(:)); "
            f"decMax = max(Dec2(:)); "
            f"if decMax > decMin; "
            f"Dec2 = (Dec2 - decMin) / (decMax - decMin); "
            f"Dec2 = times(Dec2, maxE1 - minE1) + minE1; "
            f"end; "
            f"Dec2 = uint16(max(0, min(65535, Dec2))); "
            f"writetiffstack(Dec2, '{output_path}');"
        )

        matlab_args = ["matlab"]
        if matlab_threads == 1:
            matlab_args.append("-singleCompThread")
        matlab_args.extend(["-batch", matlab_cmd])
        env = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = str(matlab_threads)

        try:
            result = subprocess.run(
                matlab_args,
                capture_output=True,
                text=True,
                env=env,
                timeout=matlab_timeout if matlab_timeout > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MATLAB deconvlucy timed out after {matlab_timeout}s.\n"
                f"STDOUT: {exc.stdout or ''}\nSTDERR: {exc.stderr or ''}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB deconvlucy failed (returncode={result.returncode}).\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError("MATLAB produced no Lucy-Richardson Dec2 output")
        return imread(str(output_path)).astype(np.uint16, copy=False)


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


def _volume_stats(name: str, volume: np.ndarray) -> dict:
    values = np.asarray(volume).astype(np.float64, copy=False)
    return {
        "name": name,
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p1": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
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


def _make_decon_montage(volumes: dict[str, np.ndarray]) -> np.ndarray:
    panels = []
    target_shape = (256, 256)
    reference = volumes["reference_matlab_dec2"]
    zc = reference.shape[0] // 2
    yc = reference.shape[1] // 2
    xc = reference.shape[2] // 2
    for name in ("pipeline_cuda_db2", "reference_matlab_dec2"):
        volume = volumes[name]
        planes = [
            volume[zc, :, :],
            volume[:, yc, :],
            volume[:, :, xc],
        ]
        row = [_resize_nearest(_normalise_plane(p), target_shape) for p in planes]
        panels.append(np.concatenate(row, axis=1))

    diff = np.abs(
        volumes["pipeline_cuda_db2"].astype(np.float32)
        - volumes["reference_matlab_dec2"].astype(np.float32)
    )
    planes = [
        diff[zc, :, :],
        diff[:, yc, :],
        diff[:, :, xc],
    ]
    row = [_resize_nearest(_normalise_plane(p), target_shape) for p in planes]
    panels.append(np.concatenate(row, axis=1))
    return np.stack(panels, axis=0)


def _write_metrics(metrics: dict, output_dir: Path) -> None:
    (output_dir / "decon_metrics.json").write_text(
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
    (output_dir / "decon_metrics.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare this pipeline's chunked blind + CUDA RL deconvolution "
            "against the reference MATLAB deconvblind -> deconvlucy output."
        )
    )
    parser.add_argument("--image_path", required=True, help="Deskewed Top_shear directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--script_dir", default=str(Path(__file__).parent))
    parser.add_argument("--channels", default="")
    parser.add_argument("--timepoints", default="")
    parser.add_argument("--tiff_index", type=int, default=0)
    parser.add_argument("--sanity_xy", type=int, default=512,
                        help="Center XY crop for sanity check. <=0 uses full XY.")
    parser.add_argument("--blind_iters", type=int, default=3)
    parser.add_argument("--chunk_xy", type=int, default=256)
    parser.add_argument("--blind_passes", type=int, default=2,
                        help="Chunked blind PSF passes for the pipeline-style output.")
    parser.add_argument("--decon_chunk_xy", type=int, default=0,
                        help="Core XY tile size for CUDA deconvolution. <=0 auto-sizes from VRAM.")
    parser.add_argument("--pad_xy", type=int, default=32)
    parser.add_argument("--pad_z", type=int, default=20)
    parser.add_argument("--blind_workers", type=int, default=1)
    parser.add_argument("--matlab_threads", type=int, default=1)
    parser.add_argument("--matlab_workers", type=int, default=1)
    parser.add_argument("--matlab_timeout", type=int, default=1800)
    parser.add_argument("--lucy_iters", type=int, default=10,
                        help="MATLAB deconvlucy iterations for the second-pass Dec2 output.")
    parser.add_argument("--blind_z_slices", type=int, default=64)
    parser.add_argument("--snr_weight_cap", type=float, default=100.0)
    parser.add_argument("--prefetch_chunks", type=int, default=0)
    parser.add_argument("--vram_gb", type=float, default=None)
    parser.add_argument("--decon_workers", type=int, default=1)
    parser.add_argument("--overlap_xy", type=int, default=0,
                        help="Override CUDA decon XY overlap. <=0 uses the pipeline default.")
    parser.add_argument("--na", type=float, default=1.0)
    parser.add_argument("--detection_na", type=float, default=None)
    parser.add_argument("--illumination_na", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=0.520)
    parser.add_argument("--ni", type=float, default=1.515)
    parser.add_argument("--ns", type=float, default=None)
    parser.add_argument("--ni0", type=float, default=None)
    parser.add_argument("--tg", type=float, default=None)
    parser.add_argument("--tg0", type=float, default=None)
    parser.add_argument("--ng", type=float, default=None)
    parser.add_argument("--ng0", type=float, default=None)
    parser.add_argument("--ti0", type=float, default=None)
    parser.add_argument("--oversample_factor", type=int, default=3)
    parser.add_argument("--psf_model", choices=("vectorial", "scalar", "gaussian"), default="vectorial")
    parser.add_argument("--camera_pixel_size", type=float, default=None)
    parser.add_argument("--magnification", type=float, default=None)
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

    channels = _parse_int_filter(args.channels)
    timepoints = _parse_int_filter(args.timepoints)
    input_tiff = _find_input_tiff(image_dir, args.tiff_index, channels, timepoints)
    print(f"Using TIFF for deconvolution comparison: {input_tiff}", flush=True)

    dxy = resolve_dxy(args.dxy, args.camera_pixel_size, args.magnification)
    psf_seed = generate_theoretical_psf(
        na=args.na,
        detection_na=args.detection_na,
        illumination_na=args.illumination_na,
        wavelength=args.wavelength,
        ni=args.ni,
        ns=args.ns,
        ni0=args.ni0,
        tg=args.tg,
        tg0=args.tg0,
        ng=args.ng,
        ng0=args.ng0,
        ti0=args.ti0,
        oversample_factor=args.oversample_factor,
        psf_model=args.psf_model,
        dxy=dxy,
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

    print("Estimating pipeline-style chunked blind PSF on comparison crop...", flush=True)
    psf_estimation_kwargs = {
        "image_path": crop_tiff,
        "psf_seed": psf_seed,
        "n_iters": args.blind_iters,
        "chunk_xy": args.chunk_xy,
        "pad_xy": args.pad_xy,
        "pad_z": args.pad_z,
        "script_dir": script_dir,
        "max_workers": args.blind_workers,
        "prefetch_chunks": args.prefetch_chunks,
        "vram_gb": args.vram_gb,
        "cache_dir": output_dir / ".psf_cache",
        "use_cache": False,
        "matlab_threads": args.matlab_threads,
        "matlab_workers": args.matlab_workers,
        "matlab_timeout": args.matlab_timeout,
        "snr_weight_cap": args.snr_weight_cap,
        "blind_z_slices": args.blind_z_slices,
    }
    psf_signature = inspect.signature(estimate_psf_from_chunks)
    if "blind_passes" in psf_signature.parameters:
        psf_estimation_kwargs["blind_passes"] = args.blind_passes
    elif args.blind_passes != 1:
        print(
            "  NOTE: this psf_estimation.py does not support blind_passes; "
            "using its single-pass chunked blind PSF estimation.",
            flush=True,
        )
    chunked_blind = estimate_psf_from_chunks(**psf_estimation_kwargs)

    print("Running pipeline-style CUDA Richardson-Lucy deconvolution...", flush=True)
    detection_na = args.detection_na if args.detection_na is not None else args.na
    pipeline_decon = deconvolve_tiff(
        image_path=crop_tiff,
        psf=chunked_blind,
        n_iters=args.lucy_iters,
        dz=args.dz,
        dxy=dxy,
        wavelength=args.wavelength,
        na=detection_na,
        ni=args.ni,
        chunk_xy=args.decon_chunk_xy,
        vram_gb=args.vram_gb,
        decon_workers=args.decon_workers,
        overlap_xy=args.overlap_xy,
    )
    pipeline_decon_path = output_dir / "pipeline_cuda_DB2.tif"
    imwrite(str(pipeline_decon_path), pipeline_decon)
    print(
        f"Saved pipeline-style deconvolution output: "
        f"{pipeline_decon_path} shape={pipeline_decon.shape}",
        flush=True,
    )

    print("Estimating reference full-window blind PSF on same comparison crop/window...", flush=True)
    full_volume = open_tiff_memmap(crop_tiff)
    full_window = np.asarray(full_volume)
    print(f"Full blind window shape={full_window.shape}", flush=True)
    full_blind = _run_full_blind(
        volume=full_window,
        psf_seed=psf_seed,
        n_iters=args.blind_iters,
        pad_xy=args.pad_xy,
        pad_z=args.pad_z,
        script_dir=script_dir,
        matlab_threads=args.matlab_threads,
        matlab_timeout=args.matlab_timeout,
    )

    print("Running reference MATLAB second-pass Lucy-Richardson deconvolution...", flush=True)
    reference_decon = _run_matlab_deconvlucy(
        volume=full_window,
        psf=full_blind,
        n_iters=args.lucy_iters,
        pad_xy=args.pad_xy,
        pad_z=args.pad_z,
        script_dir=script_dir,
        matlab_threads=args.matlab_threads,
        matlab_timeout=args.matlab_timeout,
    )
    reference_decon_path = output_dir / "reference_matlab_Dec2.tif"
    imwrite(str(reference_decon_path), reference_decon)
    print(
        f"Saved reference MATLAB deconvolution output: "
        f"{reference_decon_path} shape={reference_decon.shape}",
        flush=True,
    )

    psfs = {
        "theoretical": _normalise_psf(psf_seed),
        "chunked_blind": _normalise_psf(chunked_blind),
        "full_blind": _normalise_psf(full_blind),
    }
    for name, psf in psfs.items():
        imwrite(str(output_dir / f"{name}_psf.tif"), psf.astype(np.float32, copy=False))

    decon_outputs = {
        "pipeline_cuda_db2": pipeline_decon,
        "reference_matlab_dec2": reference_decon,
    }
    if pipeline_decon.shape != reference_decon.shape:
        raise ValueError(
            f"Deconvolution shapes differ: pipeline={pipeline_decon.shape}, "
            f"reference={reference_decon.shape}"
        )

    metrics = {
        "input_tiff": str(input_tiff),
        "comparison_crop_shape_zyx": list(cropped.shape),
        "reference_window_shape_zyx": list(full_window.shape),
        "blind_z_window": [z_window.start, z_window.stop],
        "blind_z_window_detail": z_detail,
        "outputs": {
            "pipeline_cuda_db2": str(pipeline_decon_path),
            "reference_matlab_dec2": str(reference_decon_path),
            "decon_montage": str(output_dir / "decon_cross_sections.tif"),
        },
        "parameters": vars(args),
        "resolved_dxy": dxy,
        "decon_outputs": {
            name: _volume_stats(name, volume)
            for name, volume in decon_outputs.items()
        },
        "psfs": {
            name: _psf_stats(name, psf, dxy=dxy, dz=args.dz)
            for name, psf in psfs.items()
        },
        "comparisons": {
            "pipeline_cuda_db2_vs_reference_matlab_dec2": _pair_metrics(
                pipeline_decon,
                reference_decon,
            ),
        },
    }
    _write_metrics(metrics, output_dir)
    imwrite(str(output_dir / "decon_cross_sections.tif"), _make_decon_montage(decon_outputs))
    print(f"Deconvolution comparison outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
