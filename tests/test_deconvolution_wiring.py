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

    def test_light_sheet_profile_preserves_psf_mode(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("params.psf_mode = 'light_sheet'", config_text)
        self.assertIn("params.light_sheet_angle = 90", config_text)
        self.assertNotIn("decon_only", config_text)
        self.assertNotIn("deskew_backend", config_text)


if __name__ == "__main__":
    unittest.main()
