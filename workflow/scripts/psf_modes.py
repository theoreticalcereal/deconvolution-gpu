from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import rotate
from tifffile import imread

from psf_estimation import generate_theoretical_psf


RIGHT_ANGLE_TOLERANCE = 1e-6


def _normalise(psf: np.ndarray) -> np.ndarray:
    psf = np.nan_to_num(psf.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.clip(psf, 0, None)
    total = float(psf.sum())
    if total > 0:
        psf = psf / total
    return psf.astype(np.float32, copy=False)


def _center_crop_or_pad(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a rotated PSF back to the detection support without resampling."""
    output = np.zeros(shape, dtype=volume.dtype)
    source_slices = []
    dest_slices = []
    for current, target in zip(volume.shape, shape):
        if current >= target:
            source_start = (current - target) // 2
            dest_start = 0
            length = target
        else:
            source_start = 0
            dest_start = (target - current) // 2
            length = current
        source_slices.append(slice(source_start, source_start + length))
        dest_slices.append(slice(dest_start, dest_start + length))
    output[tuple(dest_slices)] = volume[tuple(source_slices)]
    return output


def load_psf_seed(path: str | Path, shape: tuple[int, int, int]) -> np.ndarray:
    """Load a calibrated TIFF PSF and fit it to the configured support."""
    source = np.asarray(imread(path), dtype=np.float32)
    if source.ndim == 2:
        source = source[np.newaxis, :, :]
    if source.ndim != 3:
        raise ValueError(
            f"External PSF seed must be 3-D, got shape {source.shape} from {path}"
        )
    target_shape = tuple(int(axis) for axis in shape)
    if len(target_shape) != 3 or any(axis <= 0 for axis in target_shape):
        raise ValueError(f"External PSF target shape must be positive 3-D: {shape}")
    fitted = _center_crop_or_pad(source, target_shape)
    if not np.any(np.isfinite(fitted) & (fitted > 0)):
        raise ValueError(f"External PSF seed has no positive finite energy: {path}")
    return _normalise(fitted)


def _rotate_illumination_psf(illumination: np.ndarray, angle: float) -> np.ndarray:
    """Rotate illumination coordinates in the Z/X plane and preserve output shape."""
    right_angle_units = angle / 90.0
    if abs(right_angle_units - round(right_angle_units)) <= RIGHT_ANGLE_TOLERANCE:
        rotated = np.rot90(illumination, k=int(round(right_angle_units)), axes=(0, 2))
        return _center_crop_or_pad(rotated, illumination.shape)
    return rotate(
        illumination,
        angle=angle,
        axes=(0, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def generate_psf_seed(
    *,
    psf_mode: str,
    na: float,
    detection_na: float | None,
    illumination_na: float | None,
    wavelength: float,
    ni: float,
    ns: float | None,
    ni0: float | None,
    tg: float | None,
    tg0: float | None,
    ng: float | None,
    ng0: float | None,
    ti0: float | None,
    oversample_factor: int,
    psf_model: str,
    dxy: float,
    dz: float,
    psf_size_z: int,
    psf_size_xy: int,
    background: float,
    light_sheet_angle: float = 90.0,
) -> np.ndarray:
    """Create the blind-estimation seed PSF for the selected microscope mode."""
    detection = generate_theoretical_psf(
        na=na,
        detection_na=detection_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        tg=tg,
        tg0=tg0,
        ng=ng,
        ng0=ng0,
        ti0=ti0,
        oversample_factor=oversample_factor,
        psf_model=psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=background,
    )

    if psf_mode == "single":
        return _normalise(detection)
    if psf_mode != "light_sheet":
        raise ValueError(f"Unsupported psf_mode={psf_mode!r}")

    illumination = generate_theoretical_psf(
        na=na,
        detection_na=illumination_na if illumination_na is not None else detection_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        tg=tg,
        tg0=tg0,
        ng=ng,
        ng0=ng0,
        ti0=ti0,
        oversample_factor=oversample_factor,
        psf_model=psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=background,
    )

    # Rotate the illumination PSF in the Z/X plane so its axial waist models
    # the sheet/beam axis rather than the detection axis. The default 90 degree
    # case is an exact coordinate rotation; arbitrary angles use interpolation.
    rotated_illumination = _rotate_illumination_psf(illumination, light_sheet_angle)
    return _normalise(detection * rotated_illumination)
