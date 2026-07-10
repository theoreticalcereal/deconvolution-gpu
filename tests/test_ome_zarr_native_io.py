import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
import zipfile
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
        with zipfile.ZipFile(archive, "r") as handle:
            self.assertTrue(handle.infolist())
            self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in handle.infolist()))

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

        class FakeBlosc:
            BITSHUFFLE = "bitshuffle"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_zarr = types.SimpleNamespace(open=fake_open)
        fake_numcodecs = types.SimpleNamespace(Blosc=FakeBlosc)
        output = self.root / "deskewed" / "CH00_000000.ome.zarr"

        with mock.patch.dict(sys.modules, {"zarr": fake_zarr, "numcodecs": fake_numcodecs}):
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

    def test_create_ome_zarr_array_uses_blosc_zstd_bitshuffle_compression(self):
        calls = {}
        array_attrs = {}

        class FakeBlosc:
            BITSHUFFLE = "bitshuffle"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def fake_open(path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return types.SimpleNamespace(attrs=array_attrs)

        fake_zarr = types.SimpleNamespace(open=fake_open)
        fake_numcodecs = types.SimpleNamespace(Blosc=FakeBlosc)
        output = self.root / "compressed.ome.zarr"

        with mock.patch.dict(sys.modules, {"zarr": fake_zarr, "numcodecs": fake_numcodecs}):
            self.module.create_ome_zarr_array(
                output,
                shape=(2, 4, 6),
                chunks=(1, 4, 6),
                dtype="uint16",
            )

        compressor = calls["kwargs"]["compressor"]
        self.assertIsInstance(compressor, FakeBlosc)
        self.assertEqual(
            compressor.kwargs,
            {"cname": "zstd", "clevel": 3, "shuffle": FakeBlosc.BITSHUFFLE},
        )

    def test_write_downsampled_pyramid_uses_blosc_zstd_bitshuffle_compression(self):
        calls = []

        class FakeBlosc:
            BITSHUFFLE = "bitshuffle"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeSource:
            shape = (2, 8, 8)
            chunks = (1, 4, 4)
            dtype = "uint16"

            def __getitem__(self, selection):
                return selection

        class FakeTarget:
            def __init__(self):
                self.attrs = {}
                self.writes = []

            def __setitem__(self, selection, value):
                self.writes.append((selection, value))

        def fake_open(path, **kwargs):
            calls.append((path, kwargs))
            if kwargs["mode"] == "r":
                return FakeSource()
            return FakeTarget()

        fake_zarr = types.SimpleNamespace(open=fake_open)
        fake_numcodecs = types.SimpleNamespace(Blosc=FakeBlosc)

        with mock.patch.dict(sys.modules, {"zarr": fake_zarr, "numcodecs": fake_numcodecs}):
            self.module.write_downsampled_pyramid(
                self.root / "sample.ome.zarr",
                max_downsample=4,
            )

        write_calls = [kwargs for _, kwargs in calls if kwargs["mode"] == "w"]
        self.assertEqual(len(write_calls), 2)
        for kwargs in write_calls:
            compressor = kwargs["compressor"]
            self.assertIsInstance(compressor, FakeBlosc)
            self.assertEqual(
                compressor.kwargs,
                {"cname": "zstd", "clevel": 3, "shuffle": FakeBlosc.BITSHUFFLE},
            )

    def test_downsample_xy_uses_row_and_column_stride_slicing(self):
        fake_array = mock.MagicMock()

        self.module.downsample_xy(fake_array, 8)

        fake_array.__getitem__.assert_called_once_with((slice(None), slice(None, None, 8), slice(None, None, 8)))


class OmeZarrNativeIoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _require_zarr_stack(self):
        try:
            import numpy as np
            import zarr
        except ImportError as exc:
            self.skipTest(f"OME-Zarr integration dependencies unavailable: {exc}")
        return np, zarr

    def test_write_read_pyramid_and_ozx_round_trip_uint16_losslessly(self):
        np, zarr = self._require_zarr_stack()
        data = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
        output = self.root / "uint16.ome.zarr"
        archive = self.root / "uint16.ozx"
        extracted = self.root / "extracted.ome.zarr"

        self.module.write_ome_zarr_array(output, data, chunks=(1, 4, 4), max_downsample=4)
        self.module.zip_ome_zarr_to_ozx(output, archive)
        self.module.unzip_ozx_to_ome_zarr(archive, extracted)

        self.assertTrue(np.array_equal(zarr.open(str(extracted / "0"), mode="r")[:], data))
        self.assertTrue(np.array_equal(zarr.open(str(extracted / "1"), mode="r")[:], data[:, ::2, ::2]))
        self.assertTrue(np.array_equal(zarr.open(str(extracted / "2"), mode="r")[:], data[:, ::4, ::4]))

    def test_write_read_float32_losslessly(self):
        np, zarr = self._require_zarr_stack()
        data = (np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4) / np.float32(7.0))
        output = self.root / "float32.ome.zarr"

        self.module.write_ome_zarr_array(output, data, chunks=(1, 4, 4), max_downsample=2)

        self.assertTrue(np.array_equal(zarr.open(str(output / "0"), mode="r")[:], data))


if __name__ == "__main__":
    unittest.main()
