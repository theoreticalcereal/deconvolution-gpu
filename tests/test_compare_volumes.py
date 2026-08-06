import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    np = None
    MISSING_DEPENDENCY = exc.name
else:
    MISSING_DEPENDENCY = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/compare_volumes.py"


def load_module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("compare_volumes", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CompareVolumesTests(unittest.TestCase):
    def setUp(self):
        if MISSING_DEPENDENCY:
            self.skipTest(f"missing test dependency: {MISSING_DEPENDENCY}")

    def test_compare_pair_reports_error_and_correlation_metrics(self):
        module = load_module()
        reference = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
        candidate = reference + 1.0

        row = module.compare_two_volumes(
            "matlab.tif",
            reference,
            "workflow.tif",
            candidate,
        )

        self.assertEqual(row["reference"], "matlab.tif")
        self.assertEqual(row["candidate"], "workflow.tif")
        self.assertEqual(row["crop_shape"], "3x3x3")
        self.assertAlmostEqual(row["ncc"], 1.0, places=7)
        self.assertAlmostEqual(row["mae"], 1.0, places=7)
        self.assertAlmostEqual(row["rmse"], 1.0, places=7)
        self.assertAlmostEqual(row["max_abs_error"], 1.0, places=7)

    def test_center_crop_allows_different_volume_shapes(self):
        module = load_module()
        reference = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
        candidate = np.pad(reference, ((1, 1), (2, 2), (1, 1)), mode="constant")

        row = module.compare_two_volumes(
            "reference.tif",
            reference,
            "candidate.tif",
            candidate,
        )

        self.assertEqual(row["crop_shape"], "3x3x3")
        self.assertAlmostEqual(row["ncc"], 1.0, places=7)
        self.assertAlmostEqual(row["mae"], 0.0, places=7)

    def test_high_frequency_ratio_drops_for_blurred_candidate(self):
        module = load_module()
        reference = np.zeros((7, 7, 7), dtype=np.float64)
        reference[3, 3, 3] = 100.0
        candidate = np.zeros((7, 7, 7), dtype=np.float64)
        candidate[3, 3, 3] = 100.0
        candidate[2:5, 3, 3] = 50.0
        candidate[3, 2:5, 3] = 50.0
        candidate[3, 3, 2:5] = 50.0

        row = module.compare_two_volumes(
            "reference.tif",
            reference,
            "candidate.tif",
            candidate,
        )

        self.assertLess(
            row["candidate_high_frequency_fraction"],
            row["reference_high_frequency_fraction"],
        )
        self.assertLess(row["high_frequency_fraction_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
