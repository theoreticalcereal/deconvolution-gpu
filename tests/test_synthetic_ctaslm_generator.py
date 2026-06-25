import tempfile
import unittest
from importlib import util
from pathlib import Path

try:
    import numpy as np
    import tifffile
except ModuleNotFoundError as exc:
    np = None
    tifffile = None
    MISSING_DEPENDENCY = exc.name
else:
    MISSING_DEPENDENCY = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tests/generate_synthetic_ctaslm.py"


def load_generator():
    if MISSING_DEPENDENCY:
        raise unittest.SkipTest(f"missing test dependency: {MISSING_DEPENDENCY}")
    if not SCRIPT_PATH.exists():
        raise AssertionError(f"Missing generator script: {SCRIPT_PATH}")
    spec = util.spec_from_file_location("generate_synthetic_ctaslm", SCRIPT_PATH)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticCtaslmGeneratorTests(unittest.TestCase):
    def test_generator_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing generator script: {SCRIPT_PATH}")

    def test_generate_raw_ctaslm_is_parameterized_and_reproducible(self):
        generator = load_generator()

        first = generator.generate_synthetic_ctaslm(
            shape=(24, 26, 28),
            skew_angle_deg=20.0,
            num_beads=3,
            num_filaments=2,
            seed=7,
        )
        second = generator.generate_synthetic_ctaslm(
            shape=(24, 26, 28),
            skew_angle_deg=20.0,
            num_beads=3,
            num_filaments=2,
            seed=7,
        )

        self.assertEqual(first.shape, (24, 26, 28))
        self.assertEqual(first.dtype, np.uint16)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(int(first.max()), int(first.min()))

    def test_write_series_uses_pipeline_names_without_ground_truth_output(self):
        generator = load_generator()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.write_synthetic_ctaslm_series(
                output_dir=tmpdir,
                shape=(16, 18, 20),
                skew_angle_deg=15.0,
                num_beads=2,
                num_filaments=1,
                seed=3,
                channel=3,
                start_timepoint=5,
                count=2,
            )

            names = [path.name for path in paths]
            self.assertEqual(names, ["CH03_000005.tiff", "CH03_000006.tiff"])
            self.assertFalse(list(Path(tmpdir).glob("*ground_truth*")))

            volume = tifffile.imread(paths[0])
            self.assertEqual(volume.shape, (16, 18, 20))
            self.assertEqual(volume.dtype, np.uint16)


if __name__ == "__main__":
    unittest.main()
