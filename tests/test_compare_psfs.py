import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/compare_psfs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_psfs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gaussian_volume(shape=(17, 19, 21), center=(8.0, 9.0, 10.0), sigma=(2.0, 3.0, 4.0)):
    z, y, x = np.indices(shape, dtype=np.float64)
    exponent = (
        ((z - center[0]) ** 2) / (2.0 * sigma[0] ** 2)
        + ((y - center[1]) ** 2) / (2.0 * sigma[1] ** 2)
        + ((x - center[2]) ** 2) / (2.0 * sigma[2] ** 2)
    )
    return np.exp(-exponent)


class ComparePsfsTests(unittest.TestCase):
    def test_summarize_psf_reports_gaussian_r2_and_fwhm(self):
        module = load_module()
        sigma = (2.0, 3.0, 4.0)
        volume = gaussian_volume(sigma=sigma)

        summary = module.summarize_psf("synthetic", volume, spacing=(1.0, 1.0, 1.0))

        self.assertGreater(summary["gaussian_r2"], 0.99)
        expected_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        self.assertAlmostEqual(summary["fwhm_z_voxels"], expected_factor * sigma[0], delta=0.25)
        self.assertAlmostEqual(summary["fwhm_y_voxels"], expected_factor * sigma[1], delta=0.25)
        self.assertAlmostEqual(summary["fwhm_x_voxels"], expected_factor * sigma[2], delta=0.25)

    def test_pair_metrics_treat_scaled_psfs_as_matching(self):
        module = load_module()
        reference = gaussian_volume()
        candidate = reference * 7.5

        metrics = module.compare_pair(reference, candidate)

        self.assertAlmostEqual(metrics["ncc"], 1.0, places=7)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=7)
        self.assertEqual(metrics["crop_shape"], "17x19x21")

    def test_center_crop_allows_different_candidate_shapes(self):
        module = load_module()
        reference = gaussian_volume(shape=(17, 19, 21))
        candidate = np.pad(reference, ((2, 2), (1, 1), (3, 3)))

        metrics = module.compare_pair(reference, candidate)

        self.assertAlmostEqual(metrics["ncc"], 1.0, places=7)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=7)
        self.assertEqual(metrics["crop_shape"], "17x19x21")

    def test_compare_two_psfs_returns_one_pairwise_row(self):
        module = load_module()
        reference = gaussian_volume(sigma=(2.0, 3.0, 4.0))
        candidate = gaussian_volume(sigma=(2.1, 3.2, 4.3))

        row = module.compare_two_psfs(
            "reference.tif",
            reference,
            "candidate.tif",
            candidate,
            spacing=(1.0, 1.0, 1.0),
        )

        self.assertEqual(row["reference"], "reference.tif")
        self.assertEqual(row["candidate"], "candidate.tif")
        self.assertIn("reference_gaussian_r2", row)
        self.assertIn("candidate_gaussian_r2", row)
        self.assertIn("reference_fwhm_z_voxels", row)
        self.assertIn("candidate_fwhm_z_voxels", row)
        self.assertIn("ncc", row)
        self.assertIn("ssim", row)


if __name__ == "__main__":
    unittest.main()
