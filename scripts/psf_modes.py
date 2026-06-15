from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate

from psf_estimation import generate_theoretical_psf


def _normalise(psf: np.ndarray) -> np.ndarray:
    psf = np.nan_to_num(psf.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.clip(psf, 0, None)
    total = float(psf.sum())
    if total > 0:
        psf = psf / total
    return psf.astype(np.float32, copy=False)


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
    # the sheet/beam axis rather than the detection axis. The resulting
    # effective seed is still one 3-D PSF, which keeps MATLAB deconvblind and
    # pycudadecon compatible.
    rotated_illumination = rotate(
        illumination,
        angle=light_sheet_angle,
        axes=(0, 2),
        reshape=False,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return _normalise(detection * rotated_illumination)
