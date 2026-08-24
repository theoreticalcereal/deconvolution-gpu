from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = ROOT / "astrocyte_pkg.yml"
GITLAB_CI_PATH = ROOT / ".gitlab-ci.yml"
CUSTOM_PROFILE_BUILDER_PATH = ROOT / "custom-profile-builder.html"


class CustomProfileBuilderTests(unittest.TestCase):
    def test_select_menu_labels_do_not_use_em_dashes(self):
        package = yaml.safe_load(PACKAGE_PATH.read_text(encoding="utf-8"))
        select_labels = [
            choice[1]
            for parameter in package["workflow_parameters"]
            if parameter["type"] == "select"
            for choice in parameter["choices"]
        ]

        self.assertFalse(any("—" in label for label in select_labels))

    def test_custom_profile_builder_is_gitlab_pages_entrypoint(self):
        ci = yaml.safe_load(GITLAB_CI_PATH.read_text(encoding="utf-8"))

        self.assertIn("pages", ci)
        self.assertEqual(ci["pages"]["stage"], "deploy")
        self.assertIn("custom-profile-builder.html public/index.html", "\n".join(ci["pages"]["script"]))
        self.assertEqual(ci["pages"]["artifacts"]["paths"], ["public"])

    def test_custom_profile_builder_exports_only_supported_custom_yaml_keys(self):
        html = CUSTOM_PROFILE_BUILDER_PATH.read_text(encoding="utf-8")

        self.assertIn("const CUSTOM_YAML_FIELDS", html)
        self.assertIn('download = "deconvolution-custom-profile.yml"', html)
        for key in ["wavelength", "dxy", "dz", "detection_na", "illumination_na", "iter"]:
            self.assertIn(f'key: "{key}"', html)
        for blocked_key in [
            "image_aggressiveness",
            "blind_backend",
            "cupy_fft_engine",
            "decon_backend",
            "output_format",
        ]:
            self.assertIsNone(re.search(rf'key:\s*"{blocked_key}"', html))


if __name__ == "__main__":
    unittest.main()
