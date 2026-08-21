from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "workflow" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def reference_psf_to_otf(psf, output_shape):
    psf = np.asarray(psf)
    padded = np.zeros(output_shape, dtype=psf.dtype)
    padded[tuple(slice(0, size) for size in psf.shape)] = psf
    for axis, size in enumerate(psf.shape):
        padded = np.roll(padded, -(size // 2), axis=axis)
    return np.fft.fftn(padded)


def reference_accelerated_rl(observed, psf, n_iters, background=0.0):
    image = np.maximum(np.asarray(observed, dtype=np.float32) - background, 0)
    kernel = np.asarray(psf, dtype=np.float32)
    kernel = kernel / np.sum(kernel, dtype=np.float32)
    transfer = reference_psf_to_otf(kernel, image.shape)
    transfer_adjoint = np.conj(transfer)

    current = image.copy()
    previous = np.zeros_like(current)
    delta = np.zeros_like(current)
    extrapolated = None
    acceleration = np.float32(0)
    epsilon = np.finfo(np.float64).eps

    for iteration in range(n_iters):
        if iteration > 1:
            numerator = np.sum((current - extrapolated) * delta)
            denominator = np.sum(delta * delta) + epsilon
            acceleration = np.float32(np.clip(numerator / denominator, 0, 1))
            delta = current - extrapolated
        elif iteration == 1:
            delta = current - extrapolated

        extrapolated = np.maximum(
            current + acceleration * (current - previous), 0
        )
        reblurred = np.maximum(
            np.fft.ifftn(transfer * np.fft.fftn(extrapolated)).real,
            epsilon,
        )
        ratio = image / reblurred
        previous = current
        correction = np.fft.ifftn(
            transfer_adjoint * np.fft.fftn(ratio)
        ).real
        current = np.maximum(extrapolated * correction, 0).astype(
            np.float32, copy=False
        )

    return current


def asymmetric_psf():
    psf = np.zeros((3, 3, 5), dtype=np.float32)
    psf[1, 1, 2] = 5.0
    psf[0, 1, 2] = 2.0
    psf[1, 2, 2] = 1.0
    psf[1, 1, 4] = 3.0
    return psf


class PsfToOtfTests(unittest.TestCase):
    def test_psf_to_otf_matches_post_padding_and_center_roll(self):
        from richardson_lucy import psf_to_otf

        psf = np.zeros((3, 5, 7), dtype=np.float32)
        psf[0, 1, 6] = 2.0
        psf[1, 2, 3] = 5.0
        psf[2, 4, 0] = 3.0
        output_shape = (5, 7, 9)

        actual = psf_to_otf(psf, output_shape)
        expected = reference_psf_to_otf(psf, output_shape)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_psf_to_otf_rejects_dimensionality_mismatch(self):
        from richardson_lucy import psf_to_otf

        with self.assertRaisesRegex(ValueError, "dimensionality"):
            psf_to_otf(np.ones((3, 3), dtype=np.float32), (3, 3, 3))

    def test_psf_to_otf_rejects_psf_larger_than_output(self):
        from richardson_lucy import psf_to_otf

        with self.assertRaisesRegex(ValueError, "larger"):
            psf_to_otf(np.ones((5, 3, 3), dtype=np.float32), (3, 3, 3))

    def test_fit_psf_to_shape_uses_center_crop(self):
        from richardson_lucy import fit_psf_to_shape

        psf = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
        actual = fit_psf_to_shape(psf, (4, 5, 6))

        np.testing.assert_array_equal(actual, psf[1:5, 1:6, 1:7])


class AcceleratedRichardsonLucyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.observed = rng.uniform(10, 500, size=(5, 7, 9)).astype(np.float32)
        self.psf = asymmetric_psf()

    def test_iterations_match_independent_accelerated_reference(self):
        from richardson_lucy import accelerated_richardson_lucy

        for n_iters in (1, 2, 5):
            with self.subTest(n_iters=n_iters):
                expected = reference_accelerated_rl(
                    self.observed, self.psf, n_iters, background=7.0
                )
                actual = accelerated_richardson_lucy(
                    self.observed, self.psf, n_iters, background=7.0
                )
                np.testing.assert_allclose(
                    actual, expected, rtol=2e-5, atol=2e-4
                )

    def test_background_is_subtracted_before_zero_iteration_output(self):
        from richardson_lucy import accelerated_richardson_lucy

        observed = np.array([[[2, 5, 9]]], dtype=np.float32)
        psf = np.ones((1, 1, 1), dtype=np.float32)

        actual = accelerated_richardson_lucy(
            observed, psf, 0, background=5.0
        )

        np.testing.assert_array_equal(
            actual, np.array([[[0, 0, 4]]], dtype=np.float32)
        )

    def test_restoration_is_nonnegative_finite_and_preserves_flux(self):
        from richardson_lucy import accelerated_richardson_lucy

        actual = accelerated_richardson_lucy(self.observed, self.psf, 5)

        self.assertTrue(np.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0.0)
        self.assertAlmostEqual(
            float(actual.sum()), float(self.observed.sum()), delta=0.05
        )

    def test_zero_sum_psf_is_rejected(self):
        from richardson_lucy import accelerated_richardson_lucy

        with self.assertRaisesRegex(ValueError, "positive finite sum"):
            accelerated_richardson_lucy(
                self.observed, np.zeros_like(self.psf), 1
            )

    def test_negative_background_is_rejected(self):
        from richardson_lucy import accelerated_richardson_lucy

        with self.assertRaisesRegex(ValueError, "background"):
            accelerated_richardson_lucy(
                self.observed, self.psf, 1, background=-1
            )


class CupyIntegrationTests(unittest.TestCase):
    @staticmethod
    def _cupy_device_available():
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    def test_cupy_uint16_wrapper_matches_numpy_core(self):
        if not self._cupy_device_available():
            self.skipTest("CUDA device unavailable")

        from richardson_lucy import accelerated_richardson_lucy, restore_uint16_cupy

        rng = np.random.default_rng(7)
        observed = rng.integers(0, 2000, size=(5, 7, 9), dtype=np.uint16)
        psf = asymmetric_psf()
        expected_float = accelerated_richardson_lucy(
            observed, psf, 4, background=12.0
        )
        expected = np.floor(np.clip(expected_float, 0, 65535) + 0.5).astype(
            np.uint16
        )

        actual = restore_uint16_cupy(
            observed, psf, 4, background=12.0
        )

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1)

    def test_cupy_wrapper_transfers_once_and_returns_uint16_host_array(self):
        from richardson_lucy import accelerated_richardson_lucy, restore_uint16_cupy

        synchronized = []

        class FakeDevice:
            def __init__(self, device_id):
                self.device_id = device_id

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        fake_cp = types.SimpleNamespace(
            asarray=np.asarray,
            asnumpy=lambda array: np.asarray(array),
            clip=np.clip,
            conj=np.conj,
            fft=np.fft,
            floor=np.floor,
            float32=np.float32,
            float64=np.float64,
            isfinite=np.isfinite,
            maximum=np.maximum,
            roll=np.roll,
            sum=np.sum,
            uint16=np.uint16,
            zeros=np.zeros,
            zeros_like=np.zeros_like,
            cuda=types.SimpleNamespace(
                Device=FakeDevice,
                Stream=types.SimpleNamespace(
                    null=types.SimpleNamespace(
                        synchronize=lambda: synchronized.append(True)
                    )
                ),
            ),
        )
        observed = np.arange(5 * 7 * 9, dtype=np.uint16).reshape(5, 7, 9)
        psf = asymmetric_psf()
        expected = np.floor(
            np.clip(
                accelerated_richardson_lucy(observed, psf, 2, background=3),
                0,
                65535,
            ) + 0.5
        ).astype(np.uint16)

        with mock.patch.dict(sys.modules, {"cupy": fake_cp}):
            actual = restore_uint16_cupy(
                observed, psf, 2, background=3, device_id=2
            )

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.uint16)
        self.assertEqual(synchronized, [True])

    def test_cupy_wrapper_matches_matlab_uint16_rounding(self):
        from richardson_lucy import restore_uint16_cupy

        class FakeDevice:
            def __init__(self, device_id):
                self.device_id = device_id

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        fake_cp = types.SimpleNamespace(
            asarray=np.asarray,
            asnumpy=np.asarray,
            clip=np.clip,
            conj=np.conj,
            fft=np.fft,
            floor=np.floor,
            float32=np.float32,
            float64=np.float64,
            isfinite=np.isfinite,
            maximum=np.maximum,
            roll=np.roll,
            sum=np.sum,
            uint16=np.uint16,
            zeros=np.zeros,
            zeros_like=np.zeros_like,
            cuda=types.SimpleNamespace(
                Device=FakeDevice,
                Stream=types.SimpleNamespace(
                    null=types.SimpleNamespace(synchronize=lambda: None)
                ),
            ),
        )

        with mock.patch.dict(sys.modules, {"cupy": fake_cp}):
            actual = restore_uint16_cupy(
                np.array([[[1.4, 1.5, 2.5]]], dtype=np.float32),
                np.ones((1, 1, 1), dtype=np.float32),
                0,
            )

        np.testing.assert_array_equal(actual, np.array([[[1, 2, 3]]], dtype=np.uint16))

    def test_cupy_wrapper_releases_cached_fft_workspace_between_chunks(self):
        from richardson_lucy import restore_uint16_cupy

        released = []

        class FakeDevice:
            def __init__(self, device_id):
                self.device_id = device_id

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        class FakeFft:
            fftn = staticmethod(np.fft.fftn)
            ifftn = staticmethod(np.fft.ifftn)
            config = types.SimpleNamespace(
                get_plan_cache=lambda: types.SimpleNamespace(
                    clear=lambda: released.append("plan_cache")
                )
            )

        fake_cp = types.SimpleNamespace(
            asarray=np.asarray,
            asnumpy=np.asarray,
            clip=np.clip,
            conj=np.conj,
            fft=FakeFft,
            floor=np.floor,
            float32=np.float32,
            float64=np.float64,
            get_default_memory_pool=lambda: types.SimpleNamespace(
                free_all_blocks=lambda: released.append("memory_pool")
            ),
            get_default_pinned_memory_pool=lambda: types.SimpleNamespace(
                free_all_blocks=lambda: released.append("pinned_pool")
            ),
            isfinite=np.isfinite,
            maximum=np.maximum,
            roll=np.roll,
            sum=np.sum,
            uint16=np.uint16,
            zeros=np.zeros,
            zeros_like=np.zeros_like,
            cuda=types.SimpleNamespace(
                Device=FakeDevice,
                Stream=types.SimpleNamespace(
                    null=types.SimpleNamespace(synchronize=lambda: None)
                ),
            ),
        )

        with mock.patch.dict(sys.modules, {"cupy": fake_cp}):
            restore_uint16_cupy(
                np.ones((3, 3, 3), dtype=np.uint16),
                np.ones((1, 1, 1), dtype=np.float32),
                1,
            )

        self.assertEqual(
            released, ["plan_cache", "memory_pool", "pinned_pool"]
        )


if __name__ == "__main__":
    unittest.main()
