from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DeconvolutionWiringTest(unittest.TestCase):
    def test_main_wires_deconvolution_without_deskew_or_visualization(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")

        self.assertIn("include { BUILD_DECON_CONTAINER } from './modules'", main_text)
        self.assertIn("include { STAGE_DECON_INPUT } from './modules'", main_text)
        self.assertIn("include { DECON } from './modules'", main_text)
        self.assertIn("include { EXPORT_OUTPUT_FORMAT } from './modules'", main_text)
        self.assertNotIn("DESKEW", main_text)
        self.assertNotIn("CONVERT_TIFFS_TO_NEUROGLANCER", main_text)

    def test_modules_keep_decon_and_export_processes(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("process BUILD_DECON_CONTAINER", modules_text)
        self.assertIn("process STAGE_DECON_INPUT", modules_text)
        self.assertIn("process DECON", modules_text)
        self.assertIn("process EXPORT_OUTPUT_FORMAT", modules_text)
        self.assertNotIn("process DESKEW", modules_text)
        self.assertNotIn("process CONVERT_TIFFS_TO_NEUROGLANCER", modules_text)

    def test_matlab_stack_helpers_are_packaged_for_blind_psf_estimation(self):
        script_dir = ROOT / "workflow/scripts"

        self.assertTrue((script_dir / "readtiffstack.m").is_file())
        self.assertTrue((script_dir / "writetiffstack.m").is_file())

    def test_light_sheet_profile_preserves_psf_mode(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("params.psf_mode = 'light_sheet'", config_text)
        self.assertIn("params.light_sheet_angle = 90", config_text)
        self.assertNotIn("decon_only", config_text)
        self.assertNotIn("deskew_backend", config_text)

    def test_external_runtime_skips_decon_container_build(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")

        self.assertIn("decon_runtime_dir = '-1'", config_text)
        self.assertIn("id: decon_runtime_dir", package_text)
        self.assertIn("default: '-1'", package_text)
        self.assertIn("def isExternalRuntimeSupplied(value)", main_text)
        self.assertIn("if (isExternalRuntimeSupplied(params.decon_runtime_dir))", main_text)
        self.assertIn("file(params.decon_runtime_dir.toString(), checkIfExists: true)", main_text)
        self.assertIn("else {\n        BUILD_DECON_CONTAINER()", main_text)
        self.assertIn("text != 'true'", main_text)

    def test_decon_processes_accept_deskew_runtime_layout(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn('for candidate in decon_env deskew_env; do', modules_text)
        self.assertIn('runtime_env="${decon_runtime}/\\${candidate}"', modules_text)
        self.assertIn('export CONDA_PREFIX="\\${runtime_env}"', modules_text)


if __name__ == "__main__":
    unittest.main()
