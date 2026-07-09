import importlib.util
import io
import pathlib
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/normalize_input_to_ome_zarr.py"


def load_module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("normalize_input_to_ome_zarr", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NormalizeInputToOmeZarrTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_normalizes_supported_dynamic_input_formats(self):
        written = []
        loaders = {
            ".tif": mock.Mock(return_value=types.SimpleNamespace(shape=(1, 2, 3), dtype="uint16")),
            ".czi": mock.Mock(return_value=types.SimpleNamespace(shape=(2, 3, 4), dtype="uint16")),
            ".nd2": mock.Mock(return_value=types.SimpleNamespace(shape=(3, 4, 5), dtype="uint16")),
            ".lif": mock.Mock(return_value=types.SimpleNamespace(shape=(4, 5, 6), dtype="uint16")),
            ".h5": mock.Mock(return_value=types.SimpleNamespace(shape=(5, 6, 7), dtype="uint16")),
        }

        for suffix in loaders:
            (self.input_dir / f"sample_{suffix[1:]}{suffix}").write_bytes(b"placeholder")

        with mock.patch.dict(self.module.LOADERS_BY_SUFFIX, loaders, clear=True), mock.patch.object(
            self.module, "write_ome_zarr_array", side_effect=lambda path, array, layer_name: written.append(
                (path.name, array.shape, layer_name)
            )
        ):
            outputs = self.module.normalize_directory(self.input_dir, self.output_dir)

        self.assertEqual(
            [item[0] for item in written],
            [
                "sample_czi.ome.zarr",
                "sample_h5.ome.zarr",
                "sample_lif.ome.zarr",
                "sample_nd2.ome.zarr",
                "sample_tif.ome.zarr",
            ],
        )
        self.assertEqual(len(outputs), 5)
        for loader in loaders.values():
            loader.assert_called_once()

    def test_normalization_logs_discovery_and_per_input_progress(self):
        loader = mock.Mock(return_value=types.SimpleNamespace(shape=(1, 2, 3), dtype="uint16"))
        (self.input_dir / "sample.tif").write_bytes(b"placeholder")

        stdout = io.StringIO()
        with mock.patch.dict(self.module.LOADERS_BY_SUFFIX, {".tif": loader}, clear=True), mock.patch.object(
            self.module, "write_ome_zarr_array"
        ), redirect_stdout(stdout):
            self.module.normalize_directory(self.input_dir, self.output_dir)

        log = stdout.getvalue()
        self.assertIn("Input normalization: scanning", log)
        self.assertIn("Found 1 supported image input(s)", log)
        self.assertIn("Normalizing sample.tif -> sample.ome.zarr", log)
        self.assertIn("Loaded sample.tif: shape=(1, 2, 3), dtype=uint16", log)
        self.assertIn("Finished sample.ome.zarr", log)

    def test_existing_ome_zarr_input_is_copied_to_normalized_output(self):
        zarr_input = self.input_dir / "native.ome.zarr"
        zarr_input.mkdir()
        (zarr_input / ".zattrs").write_text("{}")
        (zarr_input / ".zgroup").write_text("{}")

        outputs = self.module.normalize_directory(self.input_dir, self.output_dir)

        self.assertEqual(outputs, [self.output_dir / "native.ome.zarr"])
        self.assertTrue((self.output_dir / "native.ome.zarr" / ".zattrs").exists())

    def test_ozx_input_is_unzipped_to_ome_zarr_without_conversion(self):
        ozx_input = self.input_dir / "native.ozx"
        ozx_input.write_bytes(b"archive")
        calls = []

        def fake_unzip(source, target):
            calls.append((source, target))
            target.mkdir(parents=True)
            (target / ".zgroup").write_text("{}")
            return target

        with (
            mock.patch.object(self.module, "unzip_ozx_to_ome_zarr", side_effect=fake_unzip),
            mock.patch.object(self.module, "write_ome_zarr_array") as write_ome_zarr_array,
        ):
            outputs = self.module.normalize_directory(self.input_dir, self.output_dir)

        self.assertEqual(outputs, [self.output_dir / "native.ome.zarr"])
        self.assertEqual(calls, [(ozx_input, self.output_dir / "native.ome.zarr")])
        write_ome_zarr_array.assert_not_called()

    def test_rejects_unsupported_input_with_clear_error(self):
        (self.input_dir / "sample.txt").write_text("unsupported")

        with self.assertRaisesRegex(FileNotFoundError, "No supported image inputs"):
            self.module.normalize_directory(self.input_dir, self.output_dir)

    def test_lif_loader_stacks_all_z_planes(self):
        class FakeFrame:
            def __init__(self, z):
                self.shape = (5, 7)
                self.z = z
                self.ndim = 2

        class FakeStack:
            dtype = "uint16"

            def __init__(self, frames):
                self.frames = frames
                self.shape = (len(frames), 5, 7)
                self.ndim = 3

        class FakeNp:
            newaxis = None

            @staticmethod
            def asarray(value):
                return value

            @staticmethod
            def squeeze(value):
                return value

            @staticmethod
            def stack(frames, axis=0):
                return FakeStack(frames)

        class FakeDims:
            z = 3

        class FakeImage:
            dims = FakeDims()

            def get_frame(self, z=0, t=0, c=0):
                return FakeFrame(z)

        class FakeLifFile:
            def __init__(self, path):
                self.path = path

            def get_iter_image(self):
                return iter([FakeImage()])

        fake_readlif_reader = types.SimpleNamespace(LifFile=FakeLifFile)

        with mock.patch.dict(
            sys.modules,
            {
                "numpy": FakeNp,
                "readlif": types.SimpleNamespace(reader=fake_readlif_reader),
                "readlif.reader": fake_readlif_reader,
            },
        ):
            volume = self.module.load_lif_volume(self.input_dir / "sample.lif")

        self.assertEqual(volume.shape, (3, 5, 7))

    def test_lif_loader_supports_frame_reader_without_channel_argument(self):
        class FakeFrame:
            shape = (5, 7)
            ndim = 2

        class FakeNp:
            newaxis = None

            @staticmethod
            def asarray(value):
                return value

            @staticmethod
            def squeeze(value):
                return value

            @staticmethod
            def stack(frames, axis=0):
                return types.SimpleNamespace(shape=(len(frames), 5, 7), ndim=3, dtype="uint16")

        class FakeDims:
            z = 2

        class FakeImage:
            dims = FakeDims()

            def get_frame(self, z=0, t=0):
                return FakeFrame()

        class FakeLifFile:
            def __init__(self, path):
                self.path = path

            def get_iter_image(self):
                return iter([FakeImage()])

        fake_readlif_reader = types.SimpleNamespace(LifFile=FakeLifFile)

        with mock.patch.dict(
            sys.modules,
            {
                "numpy": FakeNp,
                "readlif": types.SimpleNamespace(reader=fake_readlif_reader),
                "readlif.reader": fake_readlif_reader,
            },
        ):
            volume = self.module.load_lif_volume(self.input_dir / "sample.lif")

        self.assertEqual(volume.shape, (2, 5, 7))

    def test_coerce_volume_promotes_2d_and_rejects_incompatible_shapes(self):
        class FakeArray:
            def __init__(self, shape):
                self.shape = shape
                self.ndim = len(shape)

            def __getitem__(self, key):
                return FakeArray((1,) + self.shape)

        fake_np = types.SimpleNamespace(
            asarray=lambda value: value,
            squeeze=lambda value: FakeArray(tuple(axis for axis in value.shape if axis != 1)),
            newaxis=None,
            zeros=lambda shape: FakeArray(shape),
        )

        with mock.patch.dict(sys.modules, {"numpy": fake_np}):
            self.assertEqual(self.module.coerce_volume(fake_np.zeros((5, 7))).shape, (1, 5, 7))
            self.assertEqual(self.module.coerce_volume(fake_np.zeros((1, 3, 5, 7))).shape, (3, 5, 7))
            with self.assertRaisesRegex(ValueError, "Expected a 2-D or 3-D image volume"):
                self.module.coerce_volume(fake_np.zeros((2, 3, 4, 5)))


if __name__ == "__main__":
    unittest.main()
