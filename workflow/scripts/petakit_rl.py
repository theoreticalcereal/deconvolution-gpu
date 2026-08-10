"""CuPy implementation of Petakit's accelerated simplified Richardson-Lucy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


Array = Any


def _shape(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def fit_psf_to_shape(psf: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """Center-crop a PSF only along axes that exceed the image shape."""
    array = np.asarray(psf)
    target_shape = _shape(image_shape)
    if array.ndim != len(target_shape):
        raise ValueError(
            "PSF and image dimensionality must match: "
            f"psf={array.ndim}, image={len(target_shape)}"
        )
    if any(size <= 0 for size in target_shape):
        raise ValueError(f"Image shape must be positive, got {target_shape}")

    slices = []
    for psf_size, image_size in zip(array.shape, target_shape):
        crop_size = min(int(psf_size), image_size)
        start = max(0, (int(psf_size) - image_size) // 2)
        slices.append(slice(start, start + crop_size))
    return array[tuple(slices)]


def psf_to_otf(
    psf: Array,
    output_shape: Sequence[int],
    *,
    xp: Any = np,
) -> Array:
    """Convert a PSF to an OTF using Petakit's post-padding convention."""
    kernel = xp.asarray(psf)
    target_shape = _shape(output_shape)
    if kernel.ndim != len(target_shape):
        raise ValueError(
            "PSF and output dimensionality must match: "
            f"psf={kernel.ndim}, output={len(target_shape)}"
        )
    if any(int(psf_size) > output_size for psf_size, output_size in zip(kernel.shape, target_shape)):
        raise ValueError(
            f"PSF shape {_shape(kernel.shape)} is larger than output shape {target_shape}"
        )

    padded = xp.zeros(target_shape, dtype=kernel.dtype)
    padded[tuple(slice(0, int(size)) for size in kernel.shape)] = kernel
    for axis, size in enumerate(kernel.shape):
        padded = xp.roll(padded, -(int(size) // 2), axis=axis)
    return xp.fft.fftn(padded)


def _scalar(value: Array) -> float:
    item = getattr(value, "item", None)
    return float(item() if item is not None else value)


def _release_cupy_workspace(cp: Any) -> None:
    """Release shape-specific FFT plans and pooled blocks between Dask chunks."""
    fft_config = getattr(getattr(cp, "fft", None), "config", None)
    get_plan_cache = getattr(fft_config, "get_plan_cache", None)
    if callable(get_plan_cache):
        get_plan_cache().clear()

    for pool_name in ("get_default_memory_pool", "get_default_pinned_memory_pool"):
        get_pool = getattr(cp, pool_name, None)
        if callable(get_pool):
            get_pool().free_all_blocks()


def petakit_simplified_rl(
    observed: Array,
    psf: Array,
    n_iters: int,
    *,
    background: float = 0.0,
    xp: Any = np,
) -> Array:
    """Apply Petakit's accelerated simplified Richardson-Lucy iterations."""
    n_iters = int(n_iters)
    background = float(background)
    if n_iters < 0:
        raise ValueError(f"n_iters cannot be negative, got {n_iters}")
    if not np.isfinite(background) or background < 0:
        raise ValueError(
            f"background must be a non-negative finite value, got {background}"
        )

    image = xp.asarray(observed, dtype=xp.float32)
    kernel = xp.asarray(psf, dtype=xp.float32)
    if image.ndim != 3 or kernel.ndim != 3:
        raise ValueError(
            f"Observed image and PSF must both be 3-D, got {image.ndim} and {kernel.ndim}"
        )
    if image.size == 0 or kernel.size == 0:
        raise ValueError("Observed image and PSF must be non-empty")
    if not bool(xp.isfinite(image).all()) or not bool(xp.isfinite(kernel).all()):
        raise ValueError("Observed image and PSF must contain only finite values")
    if any(int(psf_size) > int(image_size) for psf_size, image_size in zip(kernel.shape, image.shape)):
        raise ValueError(
            f"PSF shape {_shape(kernel.shape)} is larger than image shape {_shape(image.shape)}"
        )

    kernel_sum = _scalar(xp.sum(kernel))
    if not np.isfinite(kernel_sum) or kernel_sum <= 0:
        raise ValueError(
            f"PSF must have a positive finite sum, got {kernel_sum}"
        )

    image = xp.maximum(image - xp.float32(background), xp.float32(0))
    kernel = kernel / xp.float32(kernel_sum)
    transfer = psf_to_otf(kernel, image.shape, xp=xp)
    transfer_adjoint = xp.conj(transfer)

    current = image.copy()
    previous = xp.zeros_like(current)
    delta = xp.zeros_like(current)
    extrapolated = None
    acceleration = xp.float32(0)
    epsilon = xp.float64(np.finfo(np.float64).eps)

    for iteration in range(n_iters):
        if iteration > 1:
            numerator = xp.sum((current - extrapolated) * delta)
            denominator = xp.sum(delta * delta) + epsilon
            acceleration = xp.clip(numerator / denominator, 0, 1).astype(
                xp.float32, copy=False
            )
            delta = current - extrapolated
        elif iteration == 1:
            delta = current - extrapolated

        extrapolated = xp.maximum(
            current + acceleration * (current - previous), 0
        )
        reblurred = xp.maximum(
            xp.fft.ifftn(transfer * xp.fft.fftn(extrapolated)).real,
            epsilon,
        )
        ratio = image / reblurred
        previous = current
        correction = xp.fft.ifftn(
            transfer_adjoint * xp.fft.fftn(ratio)
        ).real
        current = xp.maximum(extrapolated * correction, 0).astype(
            xp.float32, copy=False
        )

    return current


def restore_uint16_cupy(
    observed: np.ndarray,
    psf: np.ndarray,
    n_iters: int,
    *,
    background: float = 0.0,
    device_id: int = 0,
) -> np.ndarray:
    """Restore a host volume on one CUDA device and return uint16 data."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("Restoration requires cupy") from exc

    observed_host = np.asarray(observed)
    psf_host = fit_psf_to_shape(np.asarray(psf), observed_host.shape)
    image_gpu = None
    psf_gpu = None
    restored_gpu = None
    output_gpu = None
    with cp.cuda.Device(int(device_id)):
        try:
            image_gpu = cp.asarray(observed_host, dtype=cp.float32)
            psf_gpu = cp.asarray(psf_host, dtype=cp.float32)
            restored_gpu = petakit_simplified_rl(
                image_gpu,
                psf_gpu,
                n_iters,
                background=background,
                xp=cp,
            )
            output_gpu = cp.floor(
                cp.clip(restored_gpu, 0, 65535) + cp.float32(0.5)
            ).astype(cp.uint16, copy=False)
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(output_gpu)
        finally:
            output_gpu = None
            restored_gpu = None
            psf_gpu = None
            image_gpu = None
            _release_cupy_workspace(cp)
