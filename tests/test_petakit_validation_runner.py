import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "workflow/scripts/run_custom_petakit_rl.py"
EVALUATOR_PATH = ROOT / "workflow/scripts/evaluate_petakit_validation.py"
RUNNER_PATH = ROOT / "workflow/scripts/run_petakit_reference_psf_comparison.sh"


def load_cli():
    script_dir = str(CLI_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "run_custom_petakit_rl_test", CLI_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "evaluate_petakit_validation_test", EVALUATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CustomPetakitRlCliTests(unittest.TestCase):
    def test_cli_exposes_restoration_and_timing_arguments(self):
        module = load_cli()

        args = module.build_parser().parse_args(
            [
                "--input", "input.tif",
                "--psf", "psf.tif",
                "--output", "output.tif",
                "--iter", "10",
                "--background", "7",
                "--device-id", "2",
                "--timing-json", "timing.json",
            ]
        )

        self.assertEqual(args.input, "input.tif")
        self.assertEqual(args.psf, "psf.tif")
        self.assertEqual(args.output, "output.tif")
        self.assertEqual(args.iter, 10)
        self.assertEqual(args.background, 7.0)
        self.assertEqual(args.device_id, 2)
        self.assertEqual(args.timing_json, "timing.json")


class TwoStageRunnerContractTests(unittest.TestCase):
    def test_runner_isolates_psf_and_application_comparisons(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")

        for expected in (
            "petakit_matlab_psf",
            "petakit_workflow_psf",
            "custom_matlab_psf",
            "stage1_psf_effect",
            "stage2_application",
            "decon_wrapper.py",
            "--fixed_psf_path",
            "CUSTOM_EXECUTOR",
            "NEXTFLOW_MODULE",
            "sbatch",
            'cd "${custom_matlab_psf}"',
            '"${PETAKIT_DECON_ITERS}"',
        ):
            self.assertIn(expected, source)

        self.assertGreaterEqual(source.count("XR_RLdeconFrame3D("), 2)
        self.assertIn('psfFullpath = $(matlab_literal "${matlab_psf}")', source)
        self.assertIn('psfFullpath = $(matlab_literal "${workflow_psf}")', source)

    def test_runner_defaults_to_fresh_enforced_end_to_end_validation(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")

        self.assertIn("RUN_WORKFLOW=${RUN_WORKFLOW:-1}", source)
        self.assertIn("ENFORCE_GATES=${ENFORCE_GATES:-1}", source)

    def test_runner_writes_machine_readable_stage_metrics(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")

        for expected in (
            "stage1_psf_comparison.json",
            "stage1_volume_comparison.json",
            "stage2_volume_comparison.json",
            "timing_summary.json",
            "validation_summary.json",
        ):
            self.assertIn(expected, source)


class ValidationEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _psf_row():
        return {
            "ncc": 0.95,
            "reference_shape": "101x61x61",
            "candidate_shape": "101x61x61",
            "reference_center_z_voxels": 50.0,
            "reference_center_y_voxels": 30.0,
            "reference_center_x_voxels": 30.0,
            "candidate_center_z_voxels": 50.5,
            "candidate_center_y_voxels": 30.0,
            "candidate_center_x_voxels": 29.5,
            "reference_fwhm_z_voxels": 10.0,
            "reference_fwhm_y_voxels": 4.0,
            "reference_fwhm_x_voxels": 4.0,
            "candidate_fwhm_z_voxels": 11.0,
            "candidate_fwhm_y_voxels": 4.5,
            "candidate_fwhm_x_voxels": 3.5,
        }

    @staticmethod
    def _volume_row(ncc, ssim, gradient_ratio, frequency_ratio):
        return {
            "ncc": ncc,
            "ssim": ssim,
            "gradient_energy_ratio": gradient_ratio,
            "high_frequency_fraction_ratio": frequency_ratio,
            "reference_mean": 100.0,
            "candidate_mean": 100.5,
        }

    def test_evaluator_reports_both_stage_and_runtime_passes(self):
        module = load_evaluator()

        summary = module.evaluate(
            self._psf_row(),
            self._volume_row(0.96, 0.97, 1.1, 0.9),
            self._volume_row(0.995, 0.995, 1.01, 0.98),
            workflow_seconds=60.0,
            reference_seconds=177.0,
        )

        self.assertTrue(summary["stage1_psf_effect"]["passed"])
        self.assertTrue(summary["stage2_application"]["passed"])
        self.assertTrue(summary["end_to_end_speed"]["passed"])
        self.assertTrue(summary["passed"])

    def test_evaluator_identifies_application_frequency_failure(self):
        module = load_evaluator()

        summary = module.evaluate(
            self._psf_row(),
            self._volume_row(0.96, 0.97, 1.1, 0.9),
            self._volume_row(0.995, 0.995, 1.01, 0.5),
            workflow_seconds=60.0,
            reference_seconds=177.0,
        )

        self.assertFalse(summary["stage2_application"]["passed"])
        self.assertFalse(summary["passed"])

    def test_evaluator_requires_exact_psf_shape_match(self):
        module = load_evaluator()
        mismatched_psf = self._psf_row()
        mismatched_psf["candidate_shape"] = "99x61x61"

        summary = module.evaluate(
            mismatched_psf,
            self._volume_row(0.96, 0.97, 1.1, 0.9),
            self._volume_row(0.995, 0.995, 1.01, 0.98),
            workflow_seconds=60.0,
            reference_seconds=177.0,
        )

        checks = summary["stage1_psf_effect"]["checks"]
        self.assertIn("shape_match", checks)
        shape_check = checks["shape_match"]
        self.assertEqual(shape_check["reference"], [101, 61, 61])
        self.assertEqual(shape_check["candidate"], [99, 61, 61])
        self.assertFalse(shape_check["passed"])
        self.assertFalse(summary["stage1_psf_effect"]["passed"])
        self.assertFalse(summary["passed"])

    def test_three_channel_psf_gate_requires_each_channel_to_pass(self):
        module = load_evaluator()
        rows = {
            "CH00": self._psf_row(),
            "CH01": self._psf_row(),
            "CH02": self._psf_row(),
        }
        rows["CH01"]["candidate_fwhm_z_voxels"] = 13.0

        self.assertTrue(hasattr(module, "evaluate_three_channel_psfs"))
        summary = module.evaluate_three_channel_psfs(rows)

        self.assertTrue(summary["channels"]["CH00"]["passed"])
        self.assertFalse(summary["channels"]["CH01"]["passed"])
        self.assertTrue(summary["channels"]["CH02"]["passed"])
        self.assertFalse(summary["passed"])

    def test_three_channel_psf_gate_rejects_a_missing_channel(self):
        module = load_evaluator()
        rows = {
            "CH00": self._psf_row(),
            "CH01": self._psf_row(),
        }

        try:
            module.evaluate_three_channel_psfs(rows)
        except ValueError as exc:
            self.assertRegex(str(exc), "missing=\\['CH02'\\]")
        except Exception as exc:
            self.fail(f"expected a descriptive ValueError, got {type(exc).__name__}: {exc}")
        else:
            self.fail("missing CH02 was accepted")

    def test_evaluator_marks_missing_runtime_as_not_evaluated(self):
        module = load_evaluator()

        summary = module.evaluate(
            self._psf_row(),
            self._volume_row(0.96, 0.97, 1.1, 0.9),
            self._volume_row(0.995, 0.995, 1.01, 0.98),
            workflow_seconds=None,
            reference_seconds=None,
        )

        self.assertFalse(summary["end_to_end_speed"]["evaluated"])
        self.assertFalse(summary["end_to_end_speed"]["passed"])
        self.assertFalse(summary["passed"])

    def test_evaluator_fails_nonfinite_or_missing_metrics_without_crashing(self):
        module = load_evaluator()
        stage2 = self._volume_row(None, 0.995, 1.01, 0.98)

        summary = module.evaluate(
            self._psf_row(),
            self._volume_row(0.96, 0.97, 1.1, 0.9),
            stage2,
            workflow_seconds=60.0,
            reference_seconds=177.0,
        )

        self.assertFalse(summary["stage2_application"]["checks"]["ncc"]["passed"])
        self.assertIsNone(summary["stage2_application"]["checks"]["ncc"]["value"])


if __name__ == "__main__":
    unittest.main()
