import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.signal import fftconvolve


MODULE_PATH = Path(__file__).parents[1] / "workflow" / "scripts" / "blind_rl.py"
SPEC = importlib.util.spec_from_file_location("blind_rl_under_test", MODULE_PATH)
blind_rl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(blind_rl)


class DependencyRemovalTests(unittest.TestCase):
    def test_blind_module_does_not_contain_cucim_restoration(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("cucim", source)
        self.assertNotIn("deconvolve_with_cucim", source)


def test_convolution_adjoints_match_for_even_psf():
    rng = np.random.default_rng(7)
    image = rng.random((5, 6, 7), dtype=np.float32)
    psf = rng.random((2, 3, 4), dtype=np.float32)
    values = rng.random(image.shape, dtype=np.float32)

    forward = blind_rl.convolve_same(image, psf, fftconvolve)
    image_back = blind_rl.image_adjoint(values, psf, np, fftconvolve)
    psf_back = blind_rl.psf_adjoint(
        values, image, psf.shape, np, fftconvolve
    )

    expected = np.vdot(forward, values)
    assert np.allclose(expected, np.vdot(image, image_back), rtol=2e-5)
    assert np.allclose(expected, np.vdot(psf, psf_back), rtol=2e-5)


def test_blind_rl_preserves_psf_constraints():
    image = np.zeros((9, 17, 17), dtype=np.float32)
    image[4, 5, 6] = 3.0
    image[4, 11, 12] = 2.0
    true_psf = np.zeros((3, 5, 5), dtype=np.float32)
    true_psf[1, 2, 2] = 0.7
    true_psf[1, 2, 3] = 0.2
    true_psf[2, 2, 2] = 0.1
    observed = fftconvolve(image, true_psf, mode="same")
    seed = np.ones_like(true_psf) / true_psf.size

    estimated, history = blind_rl.estimate_blind_psf_scipy(
        observed, seed, 4, return_history=True
    )

    assert estimated.shape == seed.shape
    assert np.isfinite(estimated).all()
    assert np.min(estimated) >= 0.0
    assert np.isclose(np.sum(estimated, dtype=np.float64), 1.0, atol=1e-6)
    assert len(history) == 4


class BlindRlAccelerationTests(unittest.TestCase):
    def test_lazy_latent_updates_reduce_fft_work_and_preserve_constraints(self):
        image = np.zeros((9, 17, 17), dtype=np.float32)
        image[4, 5, 6] = 3.0
        image[4, 11, 12] = 2.0
        true_psf = np.zeros((3, 5, 5), dtype=np.float32)
        true_psf[1, 2, 2] = 0.7
        true_psf[1, 2, 3] = 0.2
        true_psf[2, 2, 2] = 0.1
        observed = fftconvolve(image, true_psf, mode="same")
        seed = np.ones_like(true_psf) / true_psf.size

        baseline_calls = 0
        accelerated_calls = 0

        def baseline_fft(*args, **kwargs):
            nonlocal baseline_calls
            baseline_calls += 1
            return fftconvolve(*args, **kwargs)

        def accelerated_fft(*args, **kwargs):
            nonlocal accelerated_calls
            accelerated_calls += 1
            return fftconvolve(*args, **kwargs)

        blind_rl.estimate_blind_psf(
            observed,
            seed,
            4,
            xp=np,
            fftconvolve=baseline_fft,
            latent_update_period=1,
        )
        estimated = blind_rl.estimate_blind_psf(
            observed,
            seed,
            4,
            xp=np,
            fftconvolve=accelerated_fft,
            latent_update_period=2,
        )

        self.assertLess(accelerated_calls, baseline_calls)
        self.assertEqual(estimated.shape, seed.shape)
        self.assertTrue(np.isfinite(estimated).all())
        self.assertGreaterEqual(float(np.min(estimated)), 0.0)
        self.assertTrue(np.isclose(np.sum(estimated, dtype=np.float64), 1.0, atol=1e-6))

    def test_latent_update_period_must_be_positive(self):
        with self.assertRaises(ValueError):
            blind_rl.estimate_blind_psf_scipy(
                np.ones((3, 5, 5), dtype=np.float32),
                np.ones((3, 3, 3), dtype=np.float32),
                2,
                latent_update_period=0,
            )


def test_damping_neutralizes_small_residuals():
    observed = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = np.array([1.05, 1.5, 2.95], dtype=np.float32)
    ratio = blind_rl._error_ratio(observed, model, np, 1e-7, 0.1)
    assert ratio[0] == 1.0
    assert ratio[2] == 1.0
    assert ratio[1] != 1.0
