from pathlib import Path
import importlib.util
import re
import sys
import unittest
from unittest import mock

import numpy as np
from tifffile import imwrite


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "workflow" / "scripts"
MODULE_PATH = SCRIPT_DIR / "matlab_deconvolution.py"


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("matlab_deconvolution_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MatlabDeconvolutionTests(unittest.TestCase):
    def test_restoration_runs_matlab_deconvlucy_and_returns_its_uint16_stack(self):
        """Replacing MATLAB with a no-op must not silently return the input chunk."""
        module = load_module()
        expected = np.array([[[8, 5, 3], [2, 1, 0]]], dtype=np.uint16)
        invocations = []

        def fake_run(args, **kwargs):
            invocations.append((args, kwargs))
            output_match = re.search(r"writetiffstack\(uint16\(restored\), '([^']+)'\)", args[-1])
            self.assertIsNotNone(output_match)
            imwrite(output_match.group(1), expected, photometric="minisblack")
            return mock.Mock(returncode=0, stdout="MATLAB complete", stderr="")

        with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
            actual = module.restore_uint16_matlab(
                np.array([[[10, 7, 4], [2, 1, 0]]], dtype=np.uint16),
                np.array([[[1, 2, 1]]], dtype=np.float32),
                n_iters=3,
                background=2,
                matlab_bin="matlab-test",
                matlab_threads=4,
            )

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.uint16)
        args, kwargs = invocations[0]
        self.assertEqual(args[0], "matlab-test")
        self.assertIn("deconvlucy(image, psf, 3)", args[-1])
        self.assertIn("max(image - single(2.0), single(0))", args[-1])
        self.assertIn("maxNumCompThreads(2)", args[-1])
        self.assertEqual(kwargs["capture_output"], True)
