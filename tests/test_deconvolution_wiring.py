from pathlib import Path
import importlib.util
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/decon_wrapper.py"
PSF_SCRIPT_PATH = ROOT / "workflow/scripts/psf_estimation.py"
WORKFLOW_CONTAINER_IMAGE = "git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.0"


class FakeArray:
    dtype = "float32"

    def __init__(self, shape):
        self.shape = shape
        self.size = 1

    def __getitem__(self, selection):
        if not isinstance(selection, tuple):
            selection = (selection,)

        new_shape = []
        for axis_size, axis_selection in zip(self.shape, selection):
            if isinstance(axis_selection, slice):
                start, stop, step = axis_selection.indices(axis_size)
                if step != 1:
                    raise AssertionError("FakeArray only supports unit-step slices")
                new_shape.append(max(0, stop - start))
            else:
                raise AssertionError(f"unsupported FakeArray selection: {axis_selection!r}")

        new_shape.extend(self.shape[len(selection):])
        return FakeArray(tuple(new_shape))

    def sum(self):
        return 1.0

    def astype(self, *args, **kwargs):
        return self


def load_decon_wrapper_with_fakes():
    def fake_pad(array, pad_width, mode=None):
        if mode != "edge":
            raise AssertionError(f"unexpected pad mode: {mode}")
        padded_shape = tuple(
            axis + before + after
            for axis, (before, after) in zip(array.shape, pad_width)
        )
        return FakeArray(padded_shape)

    fake_np = types.SimpleNamespace(
        float32="float32",
        uint16="uint16",
        ones=lambda shape, dtype=None: FakeArray(shape),
        zeros=lambda shape, dtype=None: FakeArray(shape),
        asarray=lambda array: array,
        clip=lambda array, *args, **kwargs: array,
        pad=fake_pad,
    )
    fake_dask_array = types.SimpleNamespace(
        from_array=lambda array, chunks=None: array,
        map_overlap=lambda func, array, **kwargs: func(array),
    )
    fake_dask = types.SimpleNamespace(array=fake_dask_array)
    fake_pycudadecon = types.SimpleNamespace(
        TemporaryOTF=object,
        RLContext=object,
        rl_decon=lambda *args, **kwargs: None,
    )
    fake_tifffile = types.SimpleNamespace(imwrite=lambda *args, **kwargs: None)
    fake_psf_estimation = types.SimpleNamespace(
        DEFAULT_BLIND_CHUNK_XY=256,
        DEFAULT_BLIND_Z_SLICES=128,
        DEFAULT_SNR_WEIGHT_CAP=100.0,
        estimate_psf_from_chunks=lambda **kwargs: FakeArray((3, 3, 3)),
        detect_vram_bytes=lambda: None,
        open_tiff_memmap=lambda path: FakeArray((2, 4, 4)),
        resolve_dxy=lambda *args, **kwargs: 0.168,
        resolve_chunk_xy=lambda *args, **kwargs: 64,
    )
    fake_psf_modes = types.SimpleNamespace(
        generate_psf_seed=lambda **kwargs: FakeArray((3, 3, 3)),
    )

    spec = importlib.util.spec_from_file_location("decon_wrapper", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "numpy": fake_np,
            "dask": fake_dask,
            "dask.array": fake_dask_array,
            "pycudadecon": fake_pycudadecon,
            "tifffile": fake_tifffile,
            "psf_estimation": fake_psf_estimation,
            "psf_modes": fake_psf_modes,
        },
    ):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def load_psf_estimation_with_fakes(zarr_calls, zarr_volume):
    class FakeNdarray:
        pass

    fake_np = types.SimpleNamespace(
        ndarray=FakeNdarray,
        dtype=lambda value: types.SimpleNamespace(itemsize=2),
    )
    fake_psfmodels = types.SimpleNamespace()
    fake_tifffile = types.SimpleNamespace(
        TiffFile=object,
        imread=lambda *args, **kwargs: None,
        imwrite=lambda *args, **kwargs: None,
        memmap=lambda *args, **kwargs: None,
    )

    def fake_open_ome_zarr_array(path, mode="r"):
        zarr_calls.append((Path(path), mode))
        return zarr_volume

    fake_ome_zarr_io = types.SimpleNamespace(
        is_ome_zarr_path=lambda path: str(path).lower().endswith(".ome.zarr"),
        open_ome_zarr_array=fake_open_ome_zarr_array,
    )

    spec = importlib.util.spec_from_file_location("psf_estimation", PSF_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "numpy": fake_np,
            "psfmodels": fake_psfmodels,
            "tifffile": fake_tifffile,
            "ome_zarr_io": fake_ome_zarr_io,
        },
    ):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class DeconvolutionWiringTest(unittest.TestCase):
    def test_decon_chunk_returns_original_chunk_shape_when_rl_context_shrinks_output(self):
        module = load_decon_wrapper_with_fakes()

        class FakeRLContext:
            def __init__(self, shape, *args, **kwargs):
                self.out_shape = (shape[0] - 10, shape[1] - 11, shape[2] - 11)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with (
            mock.patch.object(module, "RLContext", FakeRLContext),
            mock.patch.object(
                module,
                "rl_decon",
                side_effect=lambda *args, **kwargs: FakeArray(kwargs["output_shape"]),
            ),
        ):
            result = module._decon_chunk(
                FakeArray((550, 384, 384)),
                otf_path="fake.otf",
                dz=0.2,
                dxy=0.168,
                n_iters=10,
                total_chunks=1,
            )

        self.assertEqual(result.shape, (550, 384, 384))

    def test_main_wires_deconvolution_without_deskew_or_visualization(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")

        self.assertIn("include { STAGE_DECON_INPUT } from './modules'", main_text)
        self.assertIn("include { DECON } from './modules'", main_text)
        self.assertIn("include { EXPORT_OUTPUT_FORMAT } from './modules'", main_text)
        self.assertNotIn("BUILD_DECON_CONTAINER", main_text)
        self.assertNotIn("DESKEW", main_text)
        self.assertNotIn("CONVERT_TIFFS_TO_NEUROGLANCER", main_text)

    def test_modules_keep_decon_and_export_processes(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("process STAGE_DECON_INPUT", modules_text)
        self.assertIn("process DECON", modules_text)
        self.assertIn("process EXPORT_OUTPUT_FORMAT", modules_text)
        self.assertNotIn("process BUILD_DECON_CONTAINER", modules_text)
        self.assertNotIn("process DESKEW", modules_text)
        self.assertNotIn("process CONVERT_TIFFS_TO_NEUROGLANCER", modules_text)

    def test_matlab_stack_helpers_are_packaged_for_blind_psf_estimation(self):
        script_dir = ROOT / "workflow/scripts"

        self.assertTrue((script_dir / "readtiffstack.m").is_file())
        self.assertTrue((script_dir / "writetiffstack.m").is_file())

    def test_light_sheet_profile_preserves_psf_mode(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("params.psf_mode = 'light_sheet'", config_text)
        self.assertIn("params.light_sheet_angle = 90", config_text)
        self.assertNotIn("decon_only", config_text)
        self.assertNotIn("deskew_backend", config_text)

    def test_workflow_uses_prebuilt_singularity_container(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")

        self.assertIn(f"docker://{WORKFLOW_CONTAINER_IMAGE}", package_text)
        self.assertIn("workflow_containers:", package_text)
        self.assertIn("'singularity/3.9.9'", package_text)
        self.assertIn("'matlab/2024a'", package_text)
        self.assertNotIn("'cuda/11.8.0'", package_text)
        self.assertNotIn("id: decon_runtime_dir", package_text)
        self.assertNotIn("decon_runtime_dir", config_text)
        self.assertNotIn("build_decon_container", config_text)
        self.assertNotIn("isExternalRuntimeSupplied", main_text)
        self.assertNotIn("decon_container_ch", main_text)
        self.assertIn(f"def WORKFLOW_CONTAINER_IMAGE = '{WORKFLOW_CONTAINER_IMAGE}'", modules_text)
        self.assertIn("def CONTAINER_ENV_PREFIX = '/opt/conda/envs/app'", modules_text)
        self.assertIn("container WORKFLOW_CONTAINER_IMAGE", modules_text)
        self.assertNotIn(f"docker://{WORKFLOW_CONTAINER_IMAGE}", modules_text)

    def test_decon_processes_use_container_environment(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("module 'singularity/3.9.9'", modules_text)
        self.assertIn("module 'singularity/3.9.9:matlab/2024a'", modules_text)
        self.assertNotIn("module 'singularity/3.9.9:cuda/11.8.0:matlab/2024a'", modules_text)
        self.assertIn("containerOptions = '--nv'", modules_text)
        self.assertIn('export CONDA_PREFIX="${CONTAINER_ENV_PREFIX}"', modules_text)
        self.assertIn('export PATH="${CONTAINER_ENV_PREFIX}/bin:\\${PATH}"', modules_text)
        self.assertIn('export LD_LIBRARY_PATH="${CONTAINER_ENV_PREFIX}/lib:\\${LD_LIBRARY_PATH:-}"', modules_text)
        self.assertNotIn("path decon_runtime", modules_text)
        self.assertNotIn("runtime_env=", modules_text)

    def test_biohpc_config_binds_host_matlab_into_singularity(self):
        config_text = (ROOT / "workflow/configs/biohpc.config").read_text(encoding="utf-8")

        self.assertIn(
            "runOptions = '--bind /home1/apps/MATLAB:/home1/apps/MATLAB'",
            config_text,
        )

    def test_decon_process_publishes_native_db2_outputs(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn('publishDir "${params.output_dir}", mode: \'copy\', pattern: \'DB2_*.ozx\'', modules_text)
        self.assertIn('path "DB2_*.ozx", emit: decon_output', modules_text)
        self.assertIn("zip_ome_zarr_to_ozx", modules_text)
        self.assertIn("rm -rf DB2_*.ome.zarr", modules_text)

    def test_nextflow_storage_cleanup_is_enabled(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("cleanup = true", config_text)
        self.assertIn("scratch true", modules_text)

    def test_ozx_is_native_input_and_output_format(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")

        self.assertIn("output_formats = 'ozx'", config_text)
        self.assertIn("\\.ozx", package_text)
        self.assertIn("default: 'ozx'", package_text)
        self.assertIn("[ 'ozx', 'OZX zipped OME-Zarr output' ]", package_text)

    def test_ome_zarr_deconvolution_streams_directly_to_zarr_output(self):
        script_text = (ROOT / "workflow/scripts/decon_wrapper.py").read_text(encoding="utf-8")

        self.assertIn("def deconvolve_ome_zarr_to_zarr(", script_text)
        self.assertIn("deconvolve_ome_zarr_to_zarr(", script_text)
        self.assertNotIn("output = deconvolve_ome_zarr(\n", script_text)
        self.assertNotIn("write_ome_zarr_array(\n                out_name,", script_text)

    def test_downsampling_parameter_is_exposed_and_forwarded(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        script_text = (ROOT / "workflow/scripts/decon_wrapper.py").read_text(encoding="utf-8")

        self.assertIn("pyramid_max_downsample = 16", config_text)
        self.assertIn("id: pyramid_max_downsample", package_text)
        self.assertIn("type: select", package_text)
        self.assertIn("required: true", package_text)
        self.assertIn("default: '16'", package_text)
        self.assertIn("pyramid_max_downsample_flag = flag('pyramid_max_downsample', params.pyramid_max_downsample)", modules_text)
        self.assertIn("${pyramid_max_downsample_flag}", modules_text)
        self.assertIn('parser.add_argument("--pyramid_max_downsample"', script_text)
        self.assertIn("max_downsample=args.pyramid_max_downsample", script_text)

    def test_ome_zarr_psf_estimation_does_not_materialize_full_volume_to_tiff(self):
        module = load_decon_wrapper_with_fakes()

        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            zarr_path = image_dir / "sample.ome.zarr"
            zarr_path.mkdir()
            psf_calls = []

            def fake_estimate_psf_from_chunks(**kwargs):
                psf_calls.append(kwargs)
                return FakeArray((3, 3, 3))

            def fake_imwrite(path, data):
                path = Path(path)
                if path.name != "estimated_psf.tif":
                    raise AssertionError(f"unexpected full-volume TIFF materialization: {path}")

            argv = [
                "decon_wrapper.py",
                "--image_path", str(image_dir),
                "--dxy", "0.168",
                "--dz", "0.2",
                "--wavelength", "0.595",
                "--detection_na", "0.7",
                "--ni", "1.33333",
                "--ns", "1.33333",
                "--psf_size_z", "3",
                "--psf_size_xy", "3",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(module, "discover_image_volumes", return_value=[zarr_path]),
                mock.patch.object(module, "open_ome_zarr_array", return_value=FakeArray((2, 4, 4))),
                mock.patch.object(module, "estimate_psf_from_chunks", side_effect=fake_estimate_psf_from_chunks),
                mock.patch.object(module, "deconvolve_ome_zarr_to_zarr"),
                mock.patch.object(module, "imwrite", side_effect=fake_imwrite),
            ):
                module.main()

        self.assertEqual(psf_calls[0]["image_path"], str(zarr_path))

    def test_psf_estimation_opens_ome_zarr_source_without_tiff_memmap(self):
        zarr_volume = FakeArray((2, 4, 4))
        zarr_calls = []
        module = load_psf_estimation_with_fakes(zarr_calls, zarr_volume)

        with mock.patch.object(module, "open_tiff_memmap") as open_tiff_memmap:
            volume = module.open_psf_source(Path("sample.ome.zarr"))

        self.assertIs(volume, zarr_volume)
        self.assertEqual(zarr_calls, [(Path("sample.ome.zarr"), "r")])
        open_tiff_memmap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
