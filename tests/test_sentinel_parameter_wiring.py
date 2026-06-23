import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parameter_block(text, parameter_id):
    pattern = re.compile(
        rf"(?ms)^  - id: {re.escape(parameter_id)}\n"
        rf".*?(?=^  - id: |\Z)"
    )
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"Parameter {parameter_id!r} not found")
    return match.group(0)


class SentinelParameterWiringTests(unittest.TestCase):
    def test_astrocyte_optional_numeric_parameters_default_to_negative_one(self):
        text = (ROOT / "astrocyte_pkg.yml").read_text()
        optional_sentinel_parameters = [
            "dx",
            "dxy",
            "camera_pixel_size",
            "magnification",
            "detection_na",
            "illumination_na",
            "ni",
            "ns",
            "vram_gb",
        ]

        for parameter_id in optional_sentinel_parameters:
            with self.subTest(parameter_id=parameter_id):
                block = parameter_block(text, parameter_id)
                self.assertIn("default: -1", block)
                self.assertIn("Use -1", block)

    def test_nextflow_config_uses_negative_one_for_numeric_sentinels(self):
        text = (ROOT / "workflow/configs/nextflow.config").read_text()
        sentinel_defaults = [
            "cell_index",
            "dx",
            "dz",
            "wavelength",
            "ni",
            "ns",
            "ni0",
            "tg",
            "tg0",
            "ng",
            "ng0",
            "ti0",
        ]

        for parameter_id in sentinel_defaults:
            with self.subTest(parameter_id=parameter_id):
                self.assertRegex(text, rf"(?m)^\s*{parameter_id}\s*=\s*-1\b")

    def test_main_workflow_normalizes_sentinel_values_before_deskew(self):
        text = (ROOT / "workflow/main.nf").read_text()
        self.assertIn("def isSupplied(value)", text)
        self.assertIn("def requireSupplied(name, value, context)", text)
        self.assertIn("deskew_cell_index = optionalValue(params.cell_index)", text)
        self.assertIn("deskew_dx = requireSupplied('dx', params.dx, 'deskew runs')", text)

    def test_astrocyte_config_prepares_decon_container(self):
        text = (ROOT / "workflow/configs/astrocyte.config").read_text()
        self.assertRegex(text, r"(?m)^\s*params\.build_decon_container\s*=\s*true\b")

    def test_container_prep_can_build_with_fakeroot(self):
        text = (ROOT / "workflow/modules.nf").read_text()
        self.assertIn('singularity build --fakeroot decon_env.sif "${projectDir}/images/decon_env.def"', text)
        self.assertNotIn('workflow jobs cannot build containers as a normal user', text)
