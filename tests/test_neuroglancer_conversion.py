import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/convert_tiff_to_precomputed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("convert_tiff_to_precomputed", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Fake2DVolume:
    ndim = 2
    dtype = "uint16"
    shape = (5, 7)

    def reshape(self, shape):
        return Fake3DVolume(shape=shape)


class Fake3DVolume:
    ndim = 3
    dtype = "uint16"

    def __init__(self, shape=(3, 5, 7)):
        self.shape = shape
        self.transpose_axes = None

    def transpose(self, *axes):
        self.transpose_axes = axes
        return self

    def __getitem__(self, key):
        return self


class ConvertTiffToPrecomputedTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_tiff_volume_promotes_2d_tiff_to_single_slice_volume_in_auto_mode(self):
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake2DVolume())

        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            volume = self.module.load_tiff_volume(self.root / "DB2_2d.tif", volume_mode="auto")

        self.assertEqual(volume.ndim, 3)
        self.assertEqual(volume.shape, (1, 5, 7))

    def test_load_tiff_volume_accepts_2d_tiff_when_mode_is_2d(self):
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake2DVolume())

        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            volume = self.module.load_tiff_volume(self.root / "DB2_2d.tif", volume_mode="2d")

        self.assertEqual(volume.shape, (1, 5, 7))

    def test_load_tiff_volume_rejects_2d_tiff_when_mode_is_3d(self):
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake2DVolume())

        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            with self.assertRaisesRegex(self.module.ConversionError, "Expected a 3D TIFF volume"):
                self.module.load_tiff_volume(self.root / "DB2_2d.tif", volume_mode="3d")

    def test_load_tiff_volume_rejects_3d_tiff_when_mode_is_2d(self):
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake3DVolume())

        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            with self.assertRaisesRegex(self.module.ConversionError, "Expected a 2D TIFF image"):
                self.module.load_tiff_volume(self.root / "DB2_3d.tif", volume_mode="2d")

    def test_write_precomputed_writes_2d_tiff_as_one_z_slice(self):
        calls = {}

        class FakeCloudVolume:
            @staticmethod
            def create_new_info(**kwargs):
                calls["info"] = kwargs
                return {"created": True}

            def __init__(self, cloud_path, info, compress):
                calls["cloud_path"] = cloud_path
                calls["cloud_info"] = info
                calls["compress"] = compress

            def commit_info(self):
                calls["committed"] = True

            def __setitem__(self, key, value):
                calls["write_key"] = key
                calls["write_value"] = value

        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake2DVolume())
        fake_cloudvolume = types.SimpleNamespace(CloudVolume=FakeCloudVolume)

        output_layer_dir = self.root / "out" / "DB2_2d"
        with mock.patch.dict(
            sys.modules,
            {
                "tifffile": fake_tifffile,
                "cloudvolume": fake_cloudvolume,
            },
        ):
            layer_path = self.module.write_precomputed(
                self.root / "DB2_2d.tif",
                output_layer_dir,
                volume_mode="2d",
            )

        self.assertEqual(layer_path, output_layer_dir.resolve())
        self.assertEqual(calls["info"]["volume_size"], [7, 5, 1])
        self.assertEqual(calls["info"]["chunk_size"], [7, 5, 1])
        self.assertEqual(calls["write_key"], (slice(None), slice(None), slice(None)))
        self.assertTrue(calls["committed"])


if __name__ == "__main__":
    unittest.main()
