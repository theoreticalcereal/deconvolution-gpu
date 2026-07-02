from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DECON = WORKSPACE / "deconvolution-gpu"
DESKEW = WORKSPACE / "deskew-gpu"


REQUIRED_PACKAGE_PATHS = [
    "docs/index.md",
    "test_data",
    "vizapp/.keep",
    "workflow/configs",
    "workflow/images",
    "workflow/lib",
    "workflow/output",
    "workflow/scripts",
    "workflow/main.nf",
    "astrocyte_pkg.yml",
    "CHANGES.md",
    "LICENSE.md",
    "README.md",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class PackageSplitContractTest(unittest.TestCase):
    def assert_astrocyte_package_shape(self, package_root: Path) -> None:
        for relative_path in REQUIRED_PACKAGE_PATHS:
            self.assertTrue(
                (package_root / relative_path).exists(),
                f"missing {package_root.name}/{relative_path}",
            )

        vizapp_entries = sorted(path.name for path in (package_root / "vizapp").iterdir())
        self.assertEqual([".keep"], vizapp_entries)

    def test_both_split_packages_have_astrocyte_shape(self):
        self.assert_astrocyte_package_shape(DECON)
        self.assert_astrocyte_package_shape(DESKEW)

    def test_deconvolution_package_contains_only_deconvolution_workflow(self):
        main_text = read(DECON / "workflow/main.nf")
        modules_text = read(DECON / "workflow/modules.nf")
        config_text = read(DECON / "workflow/configs/nextflow.config")
        package_text = read(DECON / "astrocyte_pkg.yml")

        self.assertIn("include { DECON }", main_text)
        self.assertIn("process DECON", modules_text)
        self.assertNotIn("DESKEW", main_text)
        self.assertNotIn("process DESKEW", modules_text)
        self.assertNotIn("decon_only", main_text)
        self.assertNotIn("decon_only", config_text)
        self.assertNotIn("decon_only", package_text)

    def test_deconvolution_package_has_no_vizapp_or_neuroglancer_functionality(self):
        checked_paths = [
            DECON / "astrocyte_pkg.yml",
            DECON / "README.md",
            DECON / "docs/index.md",
            DECON / "workflow/main.nf",
            DECON / "workflow/modules.nf",
            DECON / "workflow/configs/nextflow.config",
            DECON / "workflow/envs/decon-pip-requirements.txt",
        ]
        combined = "\n".join(read(path) for path in checked_paths)

        self.assertNotIn("vizapp_containers", combined)
        self.assertNotIn("vizapp_container_runscripts", combined)
        self.assertNotIn("CONVERT_TIFFS_TO_NEUROGLANCER", combined)
        self.assertNotIn("neuroglancer", combined.lower())

    def test_deskew_package_contains_only_deskew_workflow(self):
        main_text = read(DESKEW / "workflow/main.nf")
        modules_text = read(DESKEW / "workflow/modules.nf")
        config_text = read(DESKEW / "workflow/configs/nextflow.config")
        package_text = read(DESKEW / "astrocyte_pkg.yml")

        self.assertIn("include { DESKEW }", main_text)
        self.assertIn("process DESKEW", modules_text)
        self.assertNotIn("DECON", main_text)
        self.assertNotIn("process DECON", modules_text)
        self.assertNotIn("psf_mode", config_text)
        self.assertNotIn("psf_mode", package_text)


if __name__ == "__main__":
    unittest.main()
