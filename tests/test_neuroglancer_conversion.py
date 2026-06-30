import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/convert_tiff_to_ome_zarr.py"


def load_module():
    spec = importlib.util.spec_from_file_location("convert_tiff_to_ome_zarr", SCRIPT_PATH)
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
        return FakeChunk(key)


class FakeChunk:
    def __init__(self, key):
        self.key = key

    def tobytes(self, order="C"):
        return repr((self.key, order)).encode("ascii")


class ConvertTiffToOmeZarrTests(unittest.TestCase):
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

    def test_load_tiff_volume_accepts_single_z_3d_tiff_when_mode_is_2d(self):
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake3DVolume(shape=(1, 5, 7)))

        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            volume = self.module.load_tiff_volume(self.root / "DB2_single_z.tif", volume_mode="2d")

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

    def test_write_ome_zarr_writes_2d_tiff_as_one_z_slice(self):
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake2DVolume())

        output_layer_dir = self.root / "out" / "DB2_2d.ome.zarr"
        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            layer_path = self.module.write_ome_zarr(
                self.root / "DB2_2d.tif",
                output_layer_dir,
                volume_mode="2d",
                max_levels=1,
            )

        self.assertEqual(layer_path, output_layer_dir.resolve())
        zattrs = self.module.json.loads((layer_path / ".zattrs").read_text())
        zarray = self.module.json.loads((layer_path / "0" / ".zarray").read_text())
        self.assertEqual(zarray["shape"], [1, 5, 7])
        self.assertEqual(zarray["chunks"], [1, 5, 7])
        self.assertEqual(zattrs["multiscales"][0]["datasets"][0]["path"], "0")
        self.assertTrue((layer_path / "0" / "0.0.0").is_file())

    def test_write_ome_zarr_writes_multiscale_3d_dataset_metadata(self):
        volume = Fake3DVolume(shape=(2, 4, 6))
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: volume)

        output_layer_dir = self.root / "out" / "DB2_final.ome.zarr"
        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}), mock.patch.object(
            self.module, "downsample_xy", return_value=Fake3DVolume(shape=(2, 2, 3))
        ):
            layer_path = self.module.write_ome_zarr(
                self.root / "DB2_final.tif",
                output_layer_dir,
                volume_mode="3d",
                max_levels=2,
            )

        zattrs = self.module.json.loads((layer_path / ".zattrs").read_text())
        multiscale = zattrs["multiscales"][0]
        self.assertEqual(multiscale["name"], "DB2_final")
        self.assertEqual([dataset["path"] for dataset in multiscale["datasets"]], ["0", "1"])
        self.assertEqual(self.module.json.loads((layer_path / "0" / ".zarray").read_text())["shape"], [2, 4, 6])
        self.assertEqual(self.module.json.loads((layer_path / "1" / ".zarray").read_text())["shape"], [2, 2, 3])
        self.assertTrue((layer_path / "1" / "0.0.0").is_file())

    def test_convert_directory_writes_zarr_manifest_sources(self):
        input_dir = self.root / "deconvolved"
        input_dir.mkdir()
        tiff_path = input_dir / "DB2_2d.tif"
        tiff_path.write_bytes(b"placeholder")
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake2DVolume())

        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            manifest_path = self.module.convert_directory(
                input_dir,
                self.root / "neuroglancer",
                volume_mode="2d",
                max_levels=1,
            )

        payload = self.module.json.loads(manifest_path.read_text())
        self.assertEqual(payload["layers"][0]["name"], "DB2_2d")
        source = payload["layers"][0]["source"]
        expected_layer_dir = (self.root / "neuroglancer" / "DB2_2d.ome.zarr").resolve()
        self.assertEqual(source, f"zarr://{expected_layer_dir.as_uri()}")
        self.assertNotIn("precomputed://", source)
        self.assertTrue((self.root / "neuroglancer" / "DB2_2d.ome.zarr" / ".zattrs").is_file())

    def test_convert_directory_can_write_manifest_separately_from_ome_zarr_data(self):
        input_dir = self.root / "deconvolved"
        input_dir.mkdir()
        (input_dir / "DB2_2d.tif").write_bytes(b"placeholder")
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake2DVolume())

        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            manifest_path = self.module.convert_directory(
                input_dir,
                self.root / "deconvolved_zarr",
                manifest_dir=self.root / "neuroglancer",
                volume_mode="2d",
                max_levels=1,
            )

        payload = self.module.json.loads(manifest_path.read_text())
        expected_layer_dir = (self.root / "deconvolved_zarr" / "DB2_2d.ome.zarr").resolve()
        self.assertEqual(manifest_path, (self.root / "neuroglancer" / "layers.json").resolve())
        self.assertEqual(payload["layers"][0]["source"], f"zarr://{expected_layer_dir.as_uri()}")
        self.assertTrue((expected_layer_dir / ".zattrs").is_file())
        self.assertFalse((self.root / "neuroglancer" / "DB2_2d.ome.zarr").exists())

    def test_convert_directory_writes_manifest_for_existing_ome_zarr_outputs(self):
        input_dir = self.root / "deconvolved"
        zarr_dir = input_dir / "DB2_native.ome.zarr"
        zarr_dir.mkdir(parents=True)
        (zarr_dir / ".zattrs").write_text("{}")
        (zarr_dir / ".zgroup").write_text("{}")
        (zarr_dir / "0").mkdir()
        (zarr_dir / "0" / ".zarray").write_text("{}")

        manifest_path = self.module.convert_directory(
            input_dir,
            self.root / "published_zarr",
            manifest_dir=self.root / "neuroglancer",
            volume_mode="3d",
            max_levels=1,
        )

        payload = self.module.json.loads(manifest_path.read_text())
        expected_layer_dir = (self.root / "published_zarr" / "DB2_native.ome.zarr").resolve()
        self.assertEqual(payload["layers"][0]["name"], "DB2_native")
        self.assertEqual(payload["layers"][0]["source"], f"zarr://{expected_layer_dir.as_uri()}")
        self.assertTrue((expected_layer_dir / ".zattrs").exists())
        self.assertTrue((expected_layer_dir / "0" / ".zarray").exists())

    def test_convert_directory_writes_browser_safe_file_uri_sources(self):
        input_dir = self.root / "deconvolved"
        output_dir = self.root / "neuroglancer output"
        input_dir.mkdir()
        (input_dir / "DB2_2d.tif").write_bytes(b"placeholder")
        fake_tifffile = types.SimpleNamespace(memmap=lambda path: Fake2DVolume())

        with mock.patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            manifest_path = self.module.convert_directory(
                input_dir,
                output_dir,
                volume_mode="2d",
                max_levels=1,
            )

        payload = self.module.json.loads(manifest_path.read_text())
        source = payload["layers"][0]["source"]
        self.assertEqual(source, f"zarr://{(output_dir / 'DB2_2d.ome.zarr').resolve().as_uri()}")
        self.assertNotIn(" ", source)

    def test_pyramid_shapes_downsample_xy_until_singleton_or_max_levels(self):
        shapes = self.module.pyramid_shapes((8, 1024, 512), max_levels=4)

        self.assertEqual(shapes, [(8, 1024, 512), (8, 512, 256), (8, 256, 128), (8, 128, 64)])


if __name__ == "__main__":
    unittest.main()
