"""MATLAB ``deconvlucy`` bridge used by the CPU-only accuracy preset."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import numpy as np
from tifffile import imread, imwrite

from richardson_lucy import fit_psf_to_shape


def _matlab_literal(value: str | Path) -> str:
    """Return a MATLAB single-quoted literal for a filesystem path."""
    return "'" + str(value).replace("'", "''") + "'"


def restore_uint16_matlab(
    observed: np.ndarray,
    psf: np.ndarray,
    n_iters: int,
    *,
    background: float = 0.0,
    matlab_bin: str = "matlab",
    matlab_threads: int = 1,
    matlab_timeout: int = 1800,
    script_dir: str | Path | None = None,
) -> np.ndarray:
    """Restore one 3-D chunk through MATLAB ``deconvlucy`` and return uint16."""
    observed_host = np.asarray(observed)
    if observed_host.ndim != 3:
        raise ValueError(f"Observed image must be 3-D, got {observed_host.ndim}-D")
    if not np.isfinite(background) or background < 0:
        raise ValueError(f"background must be a non-negative finite value, got {background}")

    psf_host = fit_psf_to_shape(np.asarray(psf), observed_host.shape)
    if not np.isfinite(psf_host).all() or float(np.sum(psf_host)) <= 0:
        raise ValueError("PSF must have a positive finite sum")

    n_iters = int(n_iters)
    if n_iters < 0:
        raise ValueError(f"n_iters cannot be negative, got {n_iters}")
    matlab_threads = min(2, max(1, int(matlab_threads)))
    helpers = Path(script_dir) if script_dir is not None else Path(__file__).parent

    with tempfile.TemporaryDirectory(prefix=".matlab_decon_", dir=Path.cwd()) as tmp_dir:
        work_dir = Path(tmp_dir)
        image_path = work_dir / "input.tif"
        psf_path = work_dir / "psf.tif"
        output_path = work_dir / "restored.tif"
        imwrite(image_path, observed_host, photometric="minisblack")
        imwrite(
            psf_path,
            psf_host.astype(np.float32, copy=False),
            photometric="minisblack",
        )

        matlab_cmd = (
            f"addpath({_matlab_literal(helpers)}); "
            f"maxNumCompThreads({matlab_threads}); "
            f"image = single(readtiffstack({_matlab_literal(image_path)})); "
            f"image = max(image - single({float(background)}), single(0)); "
            f"psf = single(readtiffstack({_matlab_literal(psf_path)})); "
            "psf = max(psf, single(0)); "
            "psf = psf / sum(psf(:)); "
            f"restored = deconvlucy(image, psf, {n_iters}); "
            "restored = min(max(round(restored), 0), 65535); "
            f"writetiffstack(uint16(restored), {_matlab_literal(output_path)});"
        )
        matlab_args = [matlab_bin, "-nojvm", "-nodisplay", "-nosplash"]
        if matlab_threads == 1:
            matlab_args.append("-singleCompThread")
        matlab_args.extend(["-batch", matlab_cmd])

        try:
            result = subprocess.run(
                matlab_args,
                capture_output=True,
                text=True,
                timeout=matlab_timeout if matlab_timeout > 0 else None,
            )
        except OSError as exc:
            raise RuntimeError(f"Unable to start MATLAB executable {matlab_bin!r}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MATLAB deconvlucy timed out after {matlab_timeout}s. "
                f"STDOUT: {exc.stdout or ''}\nSTDERR: {exc.stderr or ''}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB deconvlucy failed (returncode={result.returncode}). "
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        if not output_path.is_file():
            raise RuntimeError("MATLAB deconvlucy completed without writing its output TIFF")

        restored = np.asarray(imread(output_path), dtype=np.uint16)
    if restored.shape != observed_host.shape:
        raise RuntimeError(
            "MATLAB deconvlucy output shape does not match the input chunk: "
            f"expected {observed_host.shape}, got {restored.shape}"
        )
    return restored
