import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import psf_modes


def test_light_sheet_seed_uses_exact_right_angle_illumination_rotation(monkeypatch):
    calls = []
    detection = np.ones((5, 5, 5), dtype=np.float32)
    illumination = np.zeros((5, 5, 5), dtype=np.float32)
    illumination[:, :, 2] = 1.0

    def fake_generate_theoretical_psf(**kwargs):
        calls.append(kwargs)
        return detection if len(calls) == 1 else illumination

    monkeypatch.setattr(psf_modes, "generate_theoretical_psf", fake_generate_theoretical_psf)

    seed = psf_modes.generate_psf_seed(
        psf_mode="light_sheet",
        na=1.0,
        detection_na=0.8,
        illumination_na=0.3,
        wavelength=0.52,
        ni=1.33,
        ns=None,
        ni0=None,
        tg=None,
        tg0=None,
        ng=None,
        ng0=None,
        ti0=None,
        oversample_factor=3,
        psf_model="scalar",
        dxy=0.1,
        dz=0.2,
        psf_size_z=5,
        psf_size_xy=5,
        background=0.0,
        light_sheet_angle=90.0,
    )

    expected = np.zeros((5, 5, 5), dtype=np.float32)
    expected[2, :, :] = 1.0
    expected /= expected.sum()

    assert len(calls) == 2
    assert calls[0]["detection_na"] == 0.8
    assert calls[1]["detection_na"] == 0.3
    assert seed.dtype == np.float32
    np.testing.assert_allclose(seed, expected, atol=1e-6)


def test_right_angle_illumination_rotation_preserves_non_cubic_shape():
    illumination = np.zeros((7, 5, 3), dtype=np.float32)
    illumination[:, :, 1] = 1.0

    rotated = psf_modes._rotate_illumination_psf(illumination, 90.0)

    assert rotated.shape == illumination.shape
    assert rotated.dtype == np.float32
    assert np.isfinite(rotated).all()
