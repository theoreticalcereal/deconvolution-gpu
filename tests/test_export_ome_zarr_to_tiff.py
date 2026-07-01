import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/export_ome_zarr_to_tiff.py"


def load_module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("export_ome_zarr_to_tiff", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExportOmeZarrToTiffTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_export_directory_writes_tiff_for_ome_zarr_outputs_and_copies_existing_tiffs(self):
        input_dir = self.root / "decon"
        output_dir = self.root / "deconvolved_tiff"
        zarr_dir = input_dir / "DB2_native.ome.zarr"
        zarr_dir.mkdir(parents=True)
        existing_tiff = input_dir / "DB2_existing.tif"
        existing_tiff.write_bytes(b"tiff")
        written = []

        with mock.patch.object(
            self.module,
            "open_ome_zarr_array",
            return_value=types.SimpleNamespace(shape=(2, 3, 4), dtype="uint16"),
        ), mock.patch.object(
            self.module.tifffile,
            "imwrite",
            side_effect=lambda path, array, **kwargs: written.append((pathlib.Path(path), array.shape, kwargs)),
        ):
            outputs = self.module.export_directory(input_dir, output_dir)

        self.assertEqual(
            [path.name for path in outputs],
            ["DB2_existing.tif", "DB2_native.tif"],
        )
        self.assertEqual(written[0][0], output_dir / "DB2_native.tif")
        self.assertEqual(written[0][1], (2, 3, 4))
        self.assertTrue(written[0][2]["bigtiff"])
        self.assertEqual((output_dir / "DB2_existing.tif").read_bytes(), b"tiff")

    def test_export_directory_rejects_unsupported_format(self):
        input_dir = self.root / "decon"
        input_dir.mkdir()
        (input_dir / "DB2_native.ome.zarr").mkdir()

        with self.assertRaisesRegex(ValueError, "Unsupported output format"):
            self.module.export_directory(input_dir, self.root / "out", output_format="czi")


if __name__ == "__main__":
    unittest.main()
