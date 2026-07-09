import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/ome_zarr_io.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ome_zarr_io", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OmeZarrNativeIoTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_discovers_tiffs_and_ome_zarr_directories_as_image_volumes(self):
        (self.root / "CH00_000002.ome.zarr").mkdir()
        (self.root / "CH00_000003.ozx").write_bytes(b"archive")
        (self.root / "CH00_000001.tif").write_bytes(b"placeholder")
        (self.root / "notes.txt").write_text("ignore")

        paths = self.module.discover_image_volumes(self.root)

        self.assertEqual(
            [path.name for path in paths],
            ["CH00_000001.tif", "CH00_000002.ome.zarr", "CH00_000003.ozx"],
        )

    def test_image_stem_strips_tiff_ome_zarr_and_ozx_suffixes(self):
        self.assertEqual(self.module.image_stem(pathlib.Path("sample.tiff")), "sample")
        self.assertEqual(self.module.image_stem(pathlib.Path("sample.ome.zarr")), "sample")
        self.assertEqual(self.module.image_stem(pathlib.Path("sample.OME.ZARR")), "sample")
        self.assertEqual(self.module.image_stem(pathlib.Path("sample.ozx")), "sample")
        self.assertEqual(self.module.image_stem(pathlib.Path("sample.OZX")), "sample")

    def test_zip_and_unzip_ozx_round_trip_ome_zarr_contents(self):
        source = self.root / "sample.ome.zarr"
        (source / "0").mkdir(parents=True)
        (source / ".zgroup").write_text('{"zarr_format": 2}\n')
        (source / "0" / ".zarray").write_text("{}\n")
        archive = self.root / "sample.ozx"
        target = self.root / "out" / "sample.ome.zarr"

        self.module.zip_ome_zarr_to_ozx(source, archive)
        extracted = self.module.unzip_ozx_to_ome_zarr(archive, target)

        self.assertEqual(extracted, target)
        self.assertEqual((target / ".zgroup").read_text(), '{"zarr_format": 2}\n')
        self.assertEqual((target / "0" / ".zarray").read_text(), "{}\n")

    def test_unzip_ozx_rejects_archive_members_outside_target(self):
        import zipfile

        archive = self.root / "bad.ozx"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../escape.txt", "bad")

        with self.assertRaisesRegex(ValueError, "Unsafe OZX archive member"):
            self.module.unzip_ozx_to_ome_zarr(archive, self.root / "out" / "bad.ome.zarr")

    def test_create_ome_zarr_array_writes_group_and_multiscales_metadata(self):
        calls = {}
        array_attrs = {}

        def fake_open(path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return types.SimpleNamespace(attrs=array_attrs)

        fake_zarr = types.SimpleNamespace(open=fake_open)
        output = self.root / "deskewed" / "CH00_000000.ome.zarr"

        with mock.patch.dict(sys.modules, {"zarr": fake_zarr}):
            self.module.create_ome_zarr_array(
                output,
                shape=(3, 5, 7),
                chunks=(1, 5, 7),
                dtype="uint16",
                layer_name="CH00_000000",
            )

        self.assertEqual(calls["path"], str(output / "0"))
        self.assertEqual(calls["kwargs"]["shape"], (3, 5, 7))
        self.assertEqual(calls["kwargs"]["chunks"], (1, 5, 7))
        self.assertEqual(calls["kwargs"]["dtype"], "uint16")
        self.assertEqual(json.loads((output / ".zgroup").read_text())["zarr_format"], 2)
        zattrs = json.loads((output / ".zattrs").read_text())
        self.assertEqual(zattrs["multiscales"][0]["name"], "CH00_000000")
        self.assertEqual(
            [dataset["path"] for dataset in zattrs["multiscales"][0]["datasets"]],
            ["0", "1", "2", "3", "4"],
        )
        self.assertEqual(
            [
                dataset["coordinateTransformations"][0]["scale"]
                for dataset in zattrs["multiscales"][0]["datasets"]
            ],
            [[1, 1, 1], [1, 2, 2], [1, 4, 4], [1, 8, 8], [1, 16, 16]],
        )
        self.assertEqual(
            zattrs["multiscales"][0]["axes"],
            [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        )
        self.assertEqual(array_attrs["_ARRAY_DIMENSIONS"], ["z", "y", "x"])

    def test_downsample_xy_uses_row_and_column_stride_slicing(self):
        fake_array = mock.MagicMock()

        self.module.downsample_xy(fake_array, 8)

        fake_array.__getitem__.assert_called_once_with((slice(None), slice(None, None, 8), slice(None, None, 8)))


if __name__ == "__main__":
    unittest.main()
