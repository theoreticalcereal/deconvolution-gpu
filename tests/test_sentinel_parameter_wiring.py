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

    def test_container_prep_does_not_build_from_docker_at_runtime(self):
        text = (ROOT / "workflow/modules.nf").read_text()
        self.assertNotIn("singularity build", text)
        self.assertIn("ERROR: no usable deconvolution Singularity image found.", text)

    def test_container_prep_prefers_astrocyte_pulled_image(self):
        main_text = (ROOT / "workflow/main.nf").read_text()
        module_text = (ROOT / "workflow/modules.nf").read_text()
        self.assertIn("BUILD_DECON_CONTAINER(params.decon_container_image)", main_text)
        self.assertIn("val container_image", module_text)
        self.assertIn('if is_usable_sif "\\$container_image"; then', module_text)

    def test_astrocyte_package_uses_packaged_decon_sif(self):
        package_text = (ROOT / "astrocyte_pkg.yml").read_text()
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text()

        self.assertNotIn("workflow_containers:", package_text)
        self.assertNotIn("docker://git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution/decon_env:latest", package_text)
        self.assertIn('decon_container_image = "${projectDir}/images/decon_env.sif"', config_text)
        self.assertNotIn("git.biohpc.swmed.edu-5050-dean-lab-ctaslm2-deconvolution-decon_env-latest.img", config_text)
