import importlib.util
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "neuroloader.py"


def load_module():
    spec = importlib.util.spec_from_file_location("neuroloader", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NeuroloaderManifestTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)
        self.manifest_path = self.root / "workflow" / "output" / "neuroglancer" / "layers.json"
        self.manifest_path.parent.mkdir(parents=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def write_manifest(self, layers):
        self.manifest_path.write_text(json.dumps({"layers": layers}))

    def make_layer(self, name, root=None):
        layer_dir = (root or self.manifest_path.parent) / f"{name}.ome.zarr"
        layer_dir.mkdir(parents=True)
        (layer_dir / ".zgroup").write_text('{"zarr_format": 2}')
        (layer_dir / ".zattrs").write_text('{"multiscales": [{"datasets": [{"path": "0"}]}]}')
        (layer_dir / "0").mkdir()
        (layer_dir / "0" / ".zarray").write_text('{"zarr_format": 2}')
        return layer_dir

    def test_loads_valid_ome_zarr_manifest_and_rewrites_file_sources_to_http(self):
        layer_dir = self.make_layer("DB2_CH00_000000")
        self.write_manifest(
            [
                {
                    "name": "DB2_CH00_000000",
                    "source": f"zarr://file://{layer_dir}",
                }
            ]
        )

        layers = self.module.load_layers_manifest(self.root, "http://127.0.0.1:4141")

        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0]["name"], "DB2_CH00_000000")
        self.assertEqual(
            layers[0]["source"],
            "zarr://http://127.0.0.1:4141/f/workflow/output/neuroglancer/DB2_CH00_000000.ome.zarr",
        )

    def test_loads_encoded_file_uri_sources(self):
        layer_dir = self.make_layer(
            "DB2_CH00_000000",
            root=self.root / "workflow" / "work" / "with space" / "neuroglancer",
        )
        self.write_manifest(
            [
                {
                    "name": "DB2_CH00_000000",
                    "source": f"zarr://{layer_dir.resolve().as_uri()}",
                }
            ]
        )

        layers = self.module.load_layers_manifest(self.root, "http://127.0.0.1:4141")

        self.assertEqual(
            layers[0]["source"],
            "zarr://http://127.0.0.1:4141/f/workflow/work/with%20space/neuroglancer/DB2_CH00_000000.ome.zarr",
        )

    def test_rewrites_local_zarr_source_to_browser_http_source(self):
        layer_dir = self.make_layer(
            "DB2_CH00_000000",
            root=self.root / "workflow" / "work" / "with space" / "neuroglancer",
        )

        source = self.module.rewrite_local_zarr_source(
            f"zarr://{layer_dir.resolve().as_uri()}",
            self.manifest_path.parent,
            "DB2_CH00_000000",
            self.root,
            "http://127.0.0.1:4141",
        )

        self.assertEqual(
            source,
            "zarr://http://127.0.0.1:4141/f/workflow/work/with%20space/neuroglancer/DB2_CH00_000000.ome.zarr",
        )
        self.assertNotIn("file://", source)

    def test_work_dir_manifest_paths_are_rewritten_to_stable_published_output(self):
        stable_layer_dir = self.make_layer(
            "DB2_fused_skin_561",
            root=self.root / "workflow" / "output" / "deconvolved",
        )
        work_layer_dir = (
            self.root
            / "workflow"
            / "work"
            / "b8"
            / "1e1ddace4d36c4b97c8466c7a6aa03"
            / "deconvolved"
            / "DB2_fused_skin_561.ome.zarr"
        )
        work_layer_dir.mkdir(parents=True)
        (work_layer_dir / ".zgroup").write_text('{"zarr_format": 2}')
        (work_layer_dir / ".zattrs").write_text('{"multiscales": [{"datasets": [{"path": "0"}]}]}')
        (work_layer_dir / "0").mkdir()
        (work_layer_dir / "0" / ".zarray").write_text('{"zarr_format": 2}')
        self.write_manifest(
            [
                {
                    "name": "DB2_fused_skin_561",
                    "source": f"zarr://file://{work_layer_dir}",
                }
            ]
        )

        layers = self.module.load_layers_manifest(self.root, "http://127.0.0.1:5151")

        self.assertEqual(
            layers[0]["source"],
            "zarr://http://127.0.0.1:5151/f/workflow/output/deconvolved/DB2_fused_skin_561.ome.zarr",
        )
        self.assertTrue((stable_layer_dir / ".zattrs").exists())

    def test_missing_manifest_raises_clear_error(self):
        with self.assertRaisesRegex(self.module.NeuroloaderError, "layers.json is missing"):
            self.module.load_layers_manifest(self.root)

    def test_invalid_json_raises_clear_error(self):
        self.manifest_path.write_text("{invalid")

        with self.assertRaisesRegex(self.module.NeuroloaderError, "invalid JSON"):
            self.module.load_layers_manifest(self.root)

    def test_missing_ome_zarr_dataset_raises_clear_error(self):
        missing_dir = self.manifest_path.parent / "missing"
        self.write_manifest(
            [
                {
                    "name": "missing",
                    "source": f"zarr://file://{missing_dir}",
                }
            ]
        )

        with self.assertRaisesRegex(self.module.NeuroloaderError, "missing OME-Zarr dataset"):
            self.module.load_layers_manifest(self.root)

    def test_precomputed_source_is_rejected_with_clear_error(self):
        layer_dir = self.manifest_path.parent / "DB2_CH00_000000"
        self.write_manifest(
            [
                {
                    "name": "DB2_CH00_000000",
                    "source": f"precomputed://file://{layer_dir}",
                }
            ]
        )

        with self.assertRaisesRegex(self.module.NeuroloaderError, "precomputed sources are no longer supported"):
            self.module.load_layers_manifest(self.root)

    def test_layer_source_without_ome_zarr_metadata_raises_clear_error(self):
        layer_dir = self.manifest_path.parent / "DB2_CH00_000000.ome.zarr"
        layer_dir.mkdir()
        self.write_manifest(
            [
                {
                    "name": "DB2_CH00_000000",
                    "source": f"zarr://file://{layer_dir}",
                }
            ]
        )

        with self.assertRaisesRegex(self.module.NeuroloaderError, "missing OME-Zarr metadata"):
            self.module.load_layers_manifest(self.root)

    def test_layer_source_without_ome_zarr_suffix_raises_clear_error(self):
        layer_dir = self.manifest_path.parent / "DB2_CH00_000000.zarr"
        layer_dir.mkdir()
        (layer_dir / ".zgroup").write_text('{"zarr_format": 2}')
        (layer_dir / ".zattrs").write_text('{"multiscales": [{"datasets": [{"path": "0"}]}]}')
        (layer_dir / "0").mkdir()
        (layer_dir / "0" / ".zarray").write_text('{"zarr_format": 2}')
        self.write_manifest(
            [
                {
                    "name": "DB2_CH00_000000",
                    "source": f"zarr://file://{layer_dir}",
                }
            ]
        )

        with self.assertRaisesRegex(self.module.NeuroloaderError, "must point to a .ome.zarr directory"):
            self.module.load_layers_manifest(self.root)

    def test_layer_source_with_invalid_multiscales_metadata_raises_clear_error(self):
        layer_dir = self.make_layer("DB2_CH00_000000")
        (layer_dir / ".zattrs").write_text('{"not_multiscales": []}')
        self.write_manifest(
            [
                {
                    "name": "DB2_CH00_000000",
                    "source": f"zarr://file://{layer_dir}",
                }
            ]
        )

        with self.assertRaisesRegex(self.module.NeuroloaderError, "invalid OME-Zarr multiscales metadata"):
            self.module.load_layers_manifest(self.root)

    def test_create_viewer_uses_rewritten_layer_sources(self):
        layers = [
            {
                "name": "DB2_CH00_000000",
                "source": "zarr://http://127.0.0.1:4141/f/workflow/output/neuroglancer/DB2_CH00_000000.ome.zarr",
            }
        ]
        appended = []
        handlers = []

        class FakeState:
            def __init__(self):
                self.layers = self

            def append(self, **kwargs):
                appended.append(kwargs)

        class FakeTxn:
            def __enter__(self):
                return FakeState()

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeViewer:
            def txn(self):
                return FakeTxn()

            def __str__(self):
                return "http://127.0.0.1:4141/v/abc"

        class FakeApp:
            def add_handlers(self, host_pattern, handler_list):
                handlers.append((host_pattern, handler_list))

        def fake_image_layer(**kwargs):
            return kwargs

        fake_neuroglancer = type(
            "FakeNeuroglancer",
            (),
            {
                "set_server_bind_address": staticmethod(lambda **kwargs: None),
                "Viewer": FakeViewer,
                "ImageLayer": staticmethod(fake_image_layer),
                "server": type("Server", (), {"global_server": type("Global", (), {"app": FakeApp()})()})(),
            },
        )
        fake_web = type(
            "FakeWeb",
            (),
            {
                "StaticFileHandler": object,
                "RedirectHandler": object,
            },
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "neuroglancer": fake_neuroglancer,
                "tornado": types.SimpleNamespace(web=fake_web),
                "tornado.web": fake_web,
            },
        ):
            self.module.create_viewer(layers, 4141, self.root)

        self.assertEqual(appended[0]["name"], "DB2_CH00_000000")
        self.assertEqual(appended[0]["layer"]["source"], layers[0]["source"])
        self.assertIn("#uicontrol invlerp normalized", appended[0]["layer"]["shader"])
        self.assertIn("emitGrayscale(normalized())", appended[0]["layer"]["shader"])
        self.assertEqual(
            appended[0]["layer"]["shader_controls"],
            {"normalized": {"range": [0, 400]}},
        )
        self.assertTrue(any("/f/(.*)" in str(handler_list) for _, handler_list in handlers))


if __name__ == "__main__":
    unittest.main()
