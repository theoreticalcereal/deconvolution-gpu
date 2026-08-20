from pathlib import Path
import importlib.util
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "workflow/scripts/decon_wrapper.py"
PSF_SCRIPT_PATH = ROOT / "workflow/scripts/psf_estimation.py"
PSF_MODES_PATH = ROOT / "workflow/scripts/psf_modes.py"
WORKFLOW_CONTAINER_IMAGE = "git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.2"


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
        prod=np.prod,
    )
    fake_dask_array = types.SimpleNamespace(
        from_array=lambda array, chunks=None: array,
        map_overlap=lambda func, array, **kwargs: func(array),
    )
    fake_dask = types.SimpleNamespace(array=fake_dask_array)
    fake_blind_rl = types.SimpleNamespace(
        deconvolve_with_cucim=lambda chunk, psf, n_iters: chunk,
    )
    fake_petakit_rl = types.SimpleNamespace(
        restore_uint16_cupy=lambda chunk, psf, n_iters, **kwargs: chunk,
        restore_uint16_petakit_cpu=lambda chunk, psf, n_iters, **kwargs: chunk,
    )
    fake_tifffile = types.SimpleNamespace(imwrite=lambda *args, **kwargs: None)
    fake_psf_estimation = types.SimpleNamespace(
        DEFAULT_BLIND_ITERS=8,
        DEFAULT_BLIND_CHUNK_XY=256,
        DEFAULT_BLIND_LATENT_UPDATE_PERIOD=2,
        DEFAULT_BLIND_MAX_TILES=16,
        DEFAULT_BLIND_Z_SLICES=128,
        DEFAULT_SNR_WEIGHT_CAP=100.0,
        estimate_psf_from_chunks=lambda **kwargs: FakeArray((3, 3, 3)),
        detect_vram_bytes=lambda: None,
        normalize_blind_backend=lambda blind_backend, cupy_fft_engine="scout": (
            ("cupy", "scout")
            if blind_backend == "scout"
            else ("cupy", "cupyx")
            if blind_backend == "cupyx"
            else (blind_backend, cupy_fft_engine)
        ),
        open_tiff_memmap=lambda path: FakeArray((2, 4, 4)),
        resolve_dxy=lambda *args, **kwargs: 0.168,
        resolve_chunk_xy=lambda *args, **kwargs: 64,
    )
    fake_psf_modes = types.SimpleNamespace(
        generate_psf_seed=lambda **kwargs: FakeArray((3, 3, 3)),
        load_fixed_psf=lambda path: FakeArray((3, 3, 3)),
        load_psf_seed=lambda path, shape: FakeArray(shape),
    )

    spec = importlib.util.spec_from_file_location("decon_wrapper", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "numpy": fake_np,
            "dask": fake_dask,
            "dask.array": fake_dask_array,
            "blind_rl": fake_blind_rl,
            "petakit_rl": fake_petakit_rl,
            "tifffile": fake_tifffile,
            "psf_estimation": fake_psf_estimation,
            "psf_modes": fake_psf_modes,
        },
    ):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def load_decon_wrapper():
    script_dir = str(SCRIPT_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "decon_wrapper_integration_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
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


def load_psf_estimation():
    spec = importlib.util.spec_from_file_location(
        "psf_estimation_selection_test", PSF_SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_psf_modes():
    spec = importlib.util.spec_from_file_location(
        "psf_modes_external_seed_test", PSF_MODES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DeconvolutionWiringTest(unittest.TestCase):
    def test_blind_psf_refinement_uses_tuned_shared_default(self):
        psf_module = load_psf_estimation()
        wrapper_text = SCRIPT_PATH.read_text(encoding="utf-8")
        psf_text = PSF_SCRIPT_PATH.read_text(encoding="utf-8")
        matlab_runner = (
            ROOT / "workflow/scripts/run_matlab_reference_comparison.sh"
        ).read_text(encoding="utf-8")
        petakit_runner = (
            ROOT / "workflow/scripts/run_petakit_reference_psf_comparison.sh"
        ).read_text(encoding="utf-8")

        self.assertEqual(psf_module.DEFAULT_BLIND_ITERS, 8)
        self.assertIn("default=DEFAULT_BLIND_ITERS", psf_text)
        self.assertIn("default=DEFAULT_BLIND_ITERS", wrapper_text)
        self.assertIn("WF_BLIND_ITERS=${WF_BLIND_ITERS:-8}", matlab_runner)
        self.assertIn("WF_BLIND_ITERS=${WF_BLIND_ITERS:-8}", petakit_runner)

    def test_decon_output_contract_accepts_direct_tiff_or_ozx(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        wrapper_text = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("def resolveOutputSelection", main_text)
        self.assertIn("output_settings.output_format", main_text)
        self.assertIn('val  output_format', modules_text)
        self.assertIn('val  pyramid_max_downsample', modules_text)
        self.assertIn('path "DB2_*.{ozx,tif,tiff}", emit: decon_output', modules_text)
        self.assertIn("output_format_flag = flag('output_format', output_format)", modules_text)
        self.assertIn("${output_format_flag}", modules_text)
        self.assertIn('parser.add_argument("--output_format"', wrapper_text)

    def test_materialized_tiff_result_writes_direct_tiff_when_requested(self):
        module = load_decon_wrapper_with_fakes()
        restored = FakeArray((2, 4, 4))

        with (
            mock.patch.object(module, "imwrite") as imwrite,
            mock.patch.object(module, "write_ome_zarr_array") as write_zarr,
        ):
            output_path = module._write_materialized_decon_output(
                restored,
                Path("sample.tiff"),
                {},
                output_format="tiff",
                max_downsample=1,
            )

        self.assertEqual(output_path, Path("DB2_sample.tif"))
        imwrite.assert_called_once_with(Path("DB2_sample.tif"), restored)
        write_zarr.assert_not_called()

    def test_materialized_tiff_result_writes_ome_zarr_for_ozx_when_requested(self):
        module = load_decon_wrapper_with_fakes()
        restored = FakeArray((2, 4, 4))

        with (
            mock.patch.object(module, "imwrite") as imwrite,
            mock.patch.object(module, "write_ome_zarr_array") as write_zarr,
        ):
            output_path = module._write_materialized_decon_output(
                restored,
                Path("sample.tiff"),
                {},
                output_format="ozx",
                max_downsample=4,
            )

        self.assertEqual(output_path, Path("DB2_sample.ome.zarr"))
        imwrite.assert_not_called()
        write_zarr.assert_called_once_with(
            Path("DB2_sample.ome.zarr"),
            restored,
            layer_name="DB2_sample",
            max_downsample=4,
        )

    def test_blind_max_tiles_parameter_is_hidden_and_forwarded(self):
        config_text = (
            ROOT / "workflow/configs/nextflow.config"
        ).read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        wrapper_text = SCRIPT_PATH.read_text(encoding="utf-8")
        psf_text = PSF_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("blind_max_tiles = 16", config_text)
        self.assertNotIn("id: blind_max_tiles", package_text)
        self.assertIn(
            "blind_max_tiles_flag = flag('blind_max_tiles', params.blind_max_tiles)",
            modules_text,
        )
        self.assertIn("${blind_max_tiles_flag}", modules_text)
        self.assertIn('parser.add_argument("--blind_max_tiles"', wrapper_text)
        self.assertIn("blind_max_tiles=args.blind_max_tiles", wrapper_text)
        self.assertIn('parser.add_argument("--blind_max_tiles"', psf_text)
        self.assertIn("blind_max_tiles=args.blind_max_tiles", psf_text)

    def test_blind_latent_update_period_parameter_is_hidden_and_forwarded(self):
        config_text = (
            ROOT / "workflow/configs/nextflow.config"
        ).read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        wrapper_text = SCRIPT_PATH.read_text(encoding="utf-8")
        psf_text = PSF_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("blind_latent_update_period = 2", config_text)
        self.assertNotIn("id: blind_latent_update_period", package_text)
        self.assertIn(
            "blind_latent_update_period_flag = flag('blind_latent_update_period', params.blind_latent_update_period)",
            modules_text,
        )
        self.assertIn("${blind_latent_update_period_flag}", modules_text)
        self.assertIn('parser.add_argument("--blind_latent_update_period"', wrapper_text)
        self.assertIn("blind_latent_update_period=args.blind_latent_update_period", wrapper_text)
        self.assertIn('parser.add_argument("--blind_latent_update_period"', psf_text)
        self.assertIn("blind_latent_update_period=args.blind_latent_update_period", psf_text)
        self.assertIn('"blind_latent_update_period": blind_latent_update_period', psf_text)

    def test_cache_key_separates_representative_and_full_grid_psfs(self):
        module = load_psf_estimation()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.tiff"
            image_path.write_bytes(b"image")
            common = {
                "image_path": image_path,
                "psf_seed": np.ones((3, 3, 3), dtype=np.float32),
                "n_iters": 20,
                "chunk_xy": 160,
                "pad_xy": 32,
                "pad_z": 20,
                "script_dir": ROOT / "workflow/scripts",
                "merge_mode": "snr_weighted_mean",
                "snr_weight_cap": 100.0,
                "z_window": (0, 128),
                "blind_backend": "cupy",
                "blind_peak_normalization": "none",
                "blind_peak_gamma_max": 2.5,
                "tile_selection_strategy": "spatial_snr_v1",
                "blind_latent_update_period": 2,
            }

            representative = module._psf_cache_key(
                **common, blind_max_tiles=16
            )
            full_grid = module._psf_cache_key(**common, blind_max_tiles=0)

        self.assertNotEqual(representative, full_grid)

    def test_cache_key_separates_scout_psf_settings(self):
        module = load_psf_estimation()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.tiff"
            image_path.write_bytes(b"image")
            common = {
                "image_path": image_path,
                "psf_seed": np.ones((3, 3, 3), dtype=np.float32),
                "n_iters": 20,
                "chunk_xy": 160,
                "pad_xy": 32,
                "pad_z": 20,
                "script_dir": ROOT / "workflow/scripts",
                "merge_mode": "snr_weighted_mean",
                "snr_weight_cap": 100.0,
                "z_window": (0, 128),
                "blind_backend": "cupy",
                "blind_peak_normalization": "none",
                "blind_peak_gamma_max": 2.5,
                "blind_latent_update_period": 2,
                "blind_max_tiles": 16,
                "tile_selection_strategy": "spatial_snr_v1",
                "coarse_region_rows": 4,
                "coarse_region_columns": 4,
                "coarse_region_limit": 8,
            }

            direct = module._psf_cache_key(
                **common,
                cupy_fft_engine="cupyx",
                adaptive_scout_iters=2,
                adaptive_keep_tiles=4,
            )
            scout_more_tiles = module._psf_cache_key(
                **common,
                cupy_fft_engine="scout",
                adaptive_scout_iters=2,
                adaptive_keep_tiles=6,
            )

        self.assertNotEqual(direct, scout_more_tiles)

    def test_adaptive_scout_keeps_consistent_psf_shapes(self):
        module = load_psf_estimation()
        origins = [(0, 0, 1, 1), (0, 1, 1, 2), (0, 2, 1, 3)]
        first = np.array([[[0.9, 0.1, 0.0]]], dtype=np.float32)
        second = np.array([[[0.8, 0.2, 0.0]]], dtype=np.float32)
        outlier = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)

        kept_origins, kept_weights, scout_seed = module._select_adaptive_scout_tiles(
            origins,
            [first, second, outlier],
            [1.0, 1.0, 100.0],
            keep_tiles=2,
            snr_weight_cap=100.0,
        )

        self.assertEqual(kept_origins, origins[:2])
        self.assertEqual(kept_weights, [1.0, 1.0])
        self.assertGreater(module._psf_shape_similarity(first, second), 0.9)
        self.assertLess(module._psf_shape_similarity(first, outlier), 0.1)
        np.testing.assert_allclose(scout_seed.sum(), 1.0, rtol=1e-6)

    def test_adaptive_scout_refinement_continues_from_consensus_seed(self):
        module = load_psf_estimation()
        initial_seed = np.full((1, 1, 3), 1.0 / 3.0, dtype=np.float32)
        first = np.array([[[0.9, 0.1, 0.0]]], dtype=np.float32)
        second = np.array([[[0.8, 0.2, 0.0]]], dtype=np.float32)
        final = np.array([[[0.7, 0.3, 0.0]]], dtype=np.float32)
        origins = [(0, 0, 1, 1), (0, 1, 1, 2)]

        with mock.patch.object(
            module,
            "_run_blind_tile_pass",
            side_effect=[([first, second], [1.0, 1.0]), ([final], [1.0])],
        ) as run_pass:
            module._run_blind_tile_adaptive_cupyx_pass(
                np.ones((1, 1, 2), dtype=np.float32),
                initial_seed,
                origins,
                pad_xy=0,
                pad_z=0,
                n_iters=4,
                script_dir=ROOT / "workflow/scripts",
                max_workers=1,
                prefetch_chunks=1,
                matlab_workers=1,
                matlab_threads=1,
                matlab_bin="matlab",
                matlab_timeout=1,
                blind_peak_normalization="none",
                blind_peak_gamma_max=2.5,
                blind_latent_update_period=1,
                snr_weight_cap=100.0,
                cupy_pool_trim_bytes=None,
                adaptive_scout_iters=2,
                adaptive_keep_tiles=2,
            )

        expected_consensus = module._merge_weighted_psfs(
            [first, second], [1.0, 1.0], 100.0
        )
        np.testing.assert_allclose(
            run_pass.call_args_list[1].args[1], expected_consensus
        )

    def test_external_psf_seed_is_center_fitted_and_normalized(self):
        from tifffile import imwrite

        module = load_psf_modes()
        source = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)

        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "calibrated_psf.tif"
            imwrite(seed_path, source)
            seed = module.load_psf_seed(seed_path, (5, 5, 5))

        expected = source[1:6, 2:7, 3:8]
        expected /= expected.sum()
        self.assertEqual(seed.shape, (5, 5, 5))
        self.assertEqual(seed.dtype, np.float32)
        np.testing.assert_allclose(seed, expected, rtol=1e-6)
        self.assertAlmostEqual(float(seed.sum()), 1.0, places=6)

    def test_fixed_psf_preserves_native_support_and_normalizes(self):
        from tifffile import imwrite

        module = load_psf_modes()
        source = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)

        with tempfile.TemporaryDirectory() as tmpdir:
            psf_path = Path(tmpdir) / "fixed_psf.tif"
            imwrite(psf_path, source)
            psf = module.load_fixed_psf(psf_path)

        expected = source / source.sum()
        self.assertEqual(psf.shape, source.shape)
        self.assertEqual(psf.dtype, np.float32)
        np.testing.assert_allclose(psf, expected, rtol=1e-6)
        self.assertAlmostEqual(float(psf.sum()), 1.0, places=6)

    def test_external_psf_seed_is_wired_through_workflow_and_comparison(self):
        config_text = (
            ROOT / "workflow/configs/nextflow.config"
        ).read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        wrapper_text = SCRIPT_PATH.read_text(encoding="utf-8")
        runner_text = (
            ROOT / "workflow/scripts/run_matlab_reference_comparison.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("psf_seed_path = ''", config_text)
        self.assertIn(
            "psf_seed_path_flag = flag('psf_seed_path', params.psf_seed_path)",
            modules_text,
        )
        self.assertIn("${psf_seed_path_flag}", modules_text)
        self.assertIn('parser.add_argument("--psf_seed_path"', wrapper_text)
        self.assertIn("load_psf_seed(", wrapper_text)
        self.assertIn("WF_PSF_SEED_PATH", runner_text)
        self.assertIn('--psf_seed_path "${workflow_psf_seed_path}"', runner_text)

    def test_fixed_psf_is_wired_as_a_blind_estimation_bypass(self):
        config_text = (
            ROOT / "workflow/configs/nextflow.config"
        ).read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        wrapper_text = SCRIPT_PATH.read_text(encoding="utf-8")
        runner_text = (
            ROOT / "workflow/scripts/run_matlab_reference_comparison.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("fixed_psf_path = ''", config_text)
        self.assertIn(
            "fixed_psf_path_flag = flag('fixed_psf_path', params.fixed_psf_path)",
            modules_text,
        )
        self.assertIn("${fixed_psf_path_flag}", modules_text)
        self.assertIn('parser.add_argument("--fixed_psf_path"', wrapper_text)
        self.assertIn("if args.fixed_psf_path:", wrapper_text)
        self.assertIn("Skipping blind PSF estimation", wrapper_text)
        self.assertIn("WF_FIXED_PSF_PATH", runner_text)
        self.assertIn("WF_DECON_ITERS", runner_text)
        self.assertIn('--iter "${WF_DECON_ITERS}"', runner_text)
        self.assertIn("REFERENCE_RUN_DIR", runner_text)

    def test_scout_parameters_keep_simple_astrocyte_surface_and_hidden_defaults(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")
        package_data = yaml.safe_load(package_text)
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        wrapper_text = SCRIPT_PATH.read_text(encoding="utf-8")
        psf_text = PSF_SCRIPT_PATH.read_text(encoding="utf-8")

        for expected in (
            "blind_backend = 'cupy'",
            "cupy_fft_engine = 'scout'",
            "adaptive_scout_iters = 2",
            "adaptive_keep_tiles = 4",
            "tile_selection_strategy = 'spatial_snr_v1'",
            "coarse_region_rows = 4",
            "coarse_region_columns = 4",
            "coarse_region_limit = 8",
        ):
            self.assertIn(expected, config_text)

        exposed_param_ids = [
            parameter["id"] for parameter in package_data["workflow_parameters"]
        ]
        self.assertEqual(
            exposed_param_ids,
            [
                "input",
                "microscope_profile",
                "config_file",
                "wavelength",
                "dz",
                "image_aggressiveness",
                "output_selection",
            ],
        )

        for param_id in (
            "cupy_fft_engine",
            "adaptive_scout_iters",
            "adaptive_keep_tiles",
            "tile_selection_strategy",
            "coarse_region_rows",
            "coarse_region_columns",
            "coarse_region_limit",
        ):
            self.assertIn(f"{param_id}_flag = flag('{param_id}', params.{param_id})", modules_text)
            self.assertIn(f"${{{param_id}_flag}}", modules_text)
            self.assertIn(f'parser.add_argument("--{param_id}"', wrapper_text)
            self.assertIn(f'parser.add_argument("--{param_id}"', psf_text)
            self.assertIn(f"{param_id}=args.{param_id}", wrapper_text)
            self.assertIn(f"{param_id}=args.{param_id}", psf_text)

        for param_id in (
            "blind_iters",
            "chunk_xy",
            "blind_max_tiles",
            "adaptive_scout_iters",
            "adaptive_keep_tiles",
            "tile_selection_strategy",
            "coarse_region_rows",
            "coarse_region_columns",
            "coarse_region_limit",
            "blind_peak_normalization",
            "blind_peak_gamma_max",
            "blind_latent_update_period",
            "blind_workers",
            "matlab_workers",
            "matlab_threads",
            "matlab_timeout",
            "prefetch_chunks",
            "blind_z_slices",
            "snr_weight_cap",
            "decon_chunk_xy",
            "decon_workers",
            "vram_gb",
        ):
            self.assertNotIn(f"id: {param_id}", package_text)

        workflow_parameters_text = package_text.split("workflow_parameters:", 1)[1]
        self.assertNotIn("title:", workflow_parameters_text)
        self.assertIn("id: image_aggressiveness", package_text)

    def test_scout_defaults_are_documented_for_test_run(self):
        params_text = (ROOT / "params.yml").read_text(encoding="utf-8")
        profiles_text = (ROOT / "docs/profiles-and-parameters.md").read_text(encoding="utf-8")
        process_text = (ROOT / "docs/psf-estimation-process.md").read_text(encoding="utf-8")
        troubleshooting_text = (ROOT / "docs/outputs-and-troubleshooting.md").read_text(encoding="utf-8")

        for expected in (
            "blind_backend: cupy",
            "cupy_fft_engine: scout",
        ):
            self.assertIn(expected, params_text)
        for hidden_param in (
            "blind_iters",
            "chunk_xy",
            "blind_max_tiles",
            "adaptive_scout_iters",
            "adaptive_keep_tiles",
            "tile_selection_strategy",
            "coarse_region_rows",
            "coarse_region_columns",
            "coarse_region_limit",
            "blind_peak_normalization",
            "blind_peak_gamma_max",
            "blind_latent_update_period",
            "blind_workers",
            "prefetch_chunks",
            "blind_z_slices",
            "snr_weight_cap",
            "matlab_workers",
            "matlab_threads",
            "matlab_timeout",
            "decon_chunk_xy",
            "decon_workers",
            "vram_gb",
        ):
            self.assertNotIn(f"{hidden_param}:", params_text)

        self.assertIn("`image-aggressiveness`", profiles_text)
        self.assertIn("acquisition YAML", profiles_text)
        self.assertIn("default scout path", process_text)
        self.assertIn("direct `cupyx`", process_text)
        self.assertIn("Advanced scout tuning", troubleshooting_text)
        self.assertIn("hidden from Astrocyte", troubleshooting_text)

    def test_scout_backend_alias_normalizes_to_cupy_scout_mode(self):
        module = load_psf_estimation()

        self.assertEqual(
            module.normalize_blind_backend("scout", "cupyx"),
            ("cupy", "scout"),
        )
        self.assertEqual(
            module.normalize_blind_backend("cupyx", "scout"),
            ("cupy", "cupyx"),
        )
        self.assertEqual(
            module.normalize_blind_backend("cupy", "scout"),
            ("cupy", "scout"),
        )

    def test_decon_wrapper_accepts_legacy_scout_backend_argument(self):
        wrapper_text = SCRIPT_PATH.read_text(encoding="utf-8")
        psf_text = PSF_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('choices=("matlab", "cupy", "scout", "cupyx")', wrapper_text)
        self.assertIn('choices=("matlab", "cupy", "scout", "cupyx")', psf_text)
        self.assertIn("normalize_blind_backend(", wrapper_text)
        self.assertIn("normalize_blind_backend(", psf_text)
        self.assertIn("args.blind_backend", wrapper_text)
        self.assertIn("args.blind_backend", psf_text)

    def test_backend_executor_workers_honor_matlab_parallelism(self):
        module = load_psf_estimation()

        self.assertEqual(
            module.resolve_backend_executor_workers(
                "matlab", blind_workers=1, matlab_workers=24
            ),
            24,
        )
        self.assertEqual(
            module.resolve_backend_executor_workers(
                "matlab", blind_workers=8, matlab_workers=2
            ),
            8,
        )
        self.assertEqual(
            module.resolve_backend_executor_workers(
                "cupy", blind_workers=24, matlab_workers=24
            ),
            1,
        )

    def test_representative_tiles_choose_best_candidate_per_spatial_region(self):
        module = load_psf_estimation()
        scores = np.array(
            [
                [1, 2, 3, 9],
                [4, 8, 7, 6],
                [3, 2, 8, 1],
                [9, 4, 2, 7],
            ],
            dtype=np.float32,
        )
        volume = scores[np.newaxis, :, :]
        tiles = [
            (row, column, row + 1, column + 1)
            for row in range(4)
            for column in range(4)
        ]

        with mock.patch.object(
            module,
            "_snr_weight",
            side_effect=lambda core, weight_cap: float(core[0, 0, 0]),
        ):
            selected = module._select_representative_tiles(
                volume, tiles, max_tiles=4, snr_weight_cap=100.0
            )

        self.assertEqual(
            selected,
            [(0, 3, 1, 4), (1, 1, 2, 2), (2, 2, 3, 3), (3, 0, 4, 1)],
        )

    def test_representative_tiles_resolve_equal_scores_by_coordinate(self):
        module = load_psf_estimation()
        volume = np.ones((1, 4, 4), dtype=np.float32)
        tiles = [
            (row, column, row + 1, column + 1)
            for row in range(4)
            for column in range(4)
        ]

        with mock.patch.object(module, "_snr_weight", return_value=1.0):
            selected = module._select_representative_tiles(
                volume, tiles, max_tiles=4, snr_weight_cap=100.0
            )

        self.assertEqual(
            selected,
            [(0, 0, 1, 1), (0, 2, 1, 3), (2, 0, 3, 1), (2, 2, 3, 3)],
        )

    def test_representative_tiles_respect_limit_on_tall_candidate_grid(self):
        module = load_psf_estimation()
        volume = np.ones((1, 8, 1), dtype=np.float32)
        tiles = [(row, 0, row + 1, 1) for row in range(8)]

        with mock.patch.object(module, "_snr_weight", return_value=1.0):
            selected = module._select_representative_tiles(
                volume, tiles, max_tiles=4, snr_weight_cap=100.0
            )

        self.assertEqual(len(selected), 4)
        self.assertEqual(selected, [(0, 0, 1, 1), (2, 0, 3, 1),
                                    (4, 0, 5, 1), (6, 0, 7, 1)])

    def test_representative_tiles_keep_full_grid_without_scoring_when_limit_is_zero(self):
        module = load_psf_estimation()
        tiles = [(0, 0, 1, 1), (0, 1, 1, 2)]

        with mock.patch.object(module, "_snr_weight") as snr_weight:
            selected = module._select_representative_tiles(
                np.ones((1, 1, 2), dtype=np.float32),
                tiles,
                max_tiles=0,
                snr_weight_cap=100.0,
            )

        self.assertEqual(selected, tiles)
        snr_weight.assert_not_called()

    def test_decon_chunk_forwards_background_and_returns_original_shape(self):
        module = load_decon_wrapper_with_fakes()
        calls = []

        def fake_restore(chunk, psf, n_iters, **kwargs):
            calls.append(kwargs)
            return FakeArray((540, 373, 373))

        with mock.patch.object(
            module,
            "restore_uint16_cupy",
            side_effect=fake_restore,
        ):
            result = module._decon_chunk(
                FakeArray((550, 384, 384)),
                psf=FakeArray((31, 31, 31)),
                n_iters=10,
                background=17.0,
                total_chunks=1,
            )

        self.assertEqual(result.shape, (550, 384, 384))
        self.assertEqual(calls, [{"background": 17.0}])

    def test_psf_halo_matches_petakit_large_file_convention_on_all_axes(self):
        module = load_decon_wrapper_with_fakes()

        self.assertEqual(module._psf_halo((101, 61, 31)), (56, 36, 21))

    def test_deconvolution_source_bypasses_overlap_for_one_block(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("if total_chunks == 1:", source)
        self.assertIn("da.map_blocks(", source)
        self.assertIn("depth={0: halo_z, 1: halo_y, 2: halo_x}", source)
        self.assertIn('boundary="none"', source)
        self.assertIn("allow_rechunk=False", source)

    def test_balanced_chunks_have_no_remainder_smaller_than_halo(self):
        module = load_decon_wrapper_with_fakes()

        chunks = module._balanced_axis_chunks(500, 256, 36)

        self.assertEqual(sum(chunks), 500)
        self.assertLessEqual(max(chunks), 256)
        self.assertGreaterEqual(min(chunks), 36)
        self.assertEqual(module._balanced_axis_chunks(8, 4, 7), (8,))

    def test_vram_chunk_fit_accounts_for_expanded_halo(self):
        module = load_decon_wrapper_with_fakes()
        available = int(80 * 600_000 / 0.55)

        core = module._fit_core_chunks_to_vram(
            (100, 256, 256),
            (100, 512, 512),
            (20, 30, 30),
            available,
        )
        expanded = module._expanded_chunk_shape(
            core, (100, 512, 512), (20, 30, 30)
        )

        self.assertLessEqual(np.prod(expanded) * 80, available * 0.55)
        self.assertLess(core[1], 256)

    def test_deconvolve_volume_returns_kernel_intensities_without_remapping(self):
        module = load_decon_wrapper_with_fakes()
        expected = FakeArray((2, 4, 4))

        class FakeProcessed:
            def compute(self, **kwargs):
                return expected

        with mock.patch.object(
            module,
            "_build_deconvolution_graph",
            return_value=(FakeProcessed(), 1),
        ):
            actual = module.deconvolve_volume(
                FakeArray((2, 4, 4)),
                "sample.tif",
                FakeArray((1, 1, 1)),
                3,
                0.2,
                0.1,
                0.5,
                1.0,
                1.33,
                background=9.0,
            )

        self.assertIs(actual, expected)

    def test_restoration_path_no_longer_references_cucim_or_range_matching(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("deconvolve_with_cucim", source)
        self.assertNotIn("_match_input_intensity_range", source)
        self.assertNotIn("_match_block_intensity_range", source)
        self.assertIn("restore_uint16_cupy", source)

    def test_chunk_fallback_clamps_halo_to_small_volume_axes(self):
        module = load_decon_wrapper()
        volume = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
        psf = np.ones((3, 3, 3), dtype=np.float32)

        with (
            mock.patch.object(module, "detect_vram_bytes", return_value=None),
            mock.patch.object(
                module,
                "restore_uint16_cupy",
                side_effect=lambda chunk, psf, n_iters, **kwargs: np.asarray(
                    chunk, dtype=np.uint16
                ),
            ),
        ):
            actual = module.deconvolve_volume(
                volume,
                "small.tif",
                psf,
                1,
                0.2,
                0.1,
                0.5,
                1.0,
                1.33,
                background=2.0,
                chunk_xy=4,
            )

        np.testing.assert_array_equal(actual, volume)

    def test_main_wires_deconvolution_without_deskew_or_visualization(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")

        self.assertIn("include { STAGE_DECON_INPUT } from './modules'", main_text)
        self.assertIn("include { STAGE_DECON_TIFF_INPUT } from './modules'", main_text)
        self.assertIn("include { DECON } from './modules'", main_text)
        self.assertIn("include { EXPORT_OUTPUT_FORMAT } from './modules'", main_text)
        self.assertNotIn("BUILD_DECON_CONTAINER", main_text)
        self.assertNotIn("DESKEW", main_text)
        self.assertNotIn("CONVERT_TIFFS_TO_NEUROGLANCER", main_text)

    def test_modules_keep_decon_and_export_processes(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("process STAGE_DECON_INPUT", modules_text)
        self.assertIn("process STAGE_DECON_TIFF_INPUT", modules_text)
        self.assertIn("process DECON", modules_text)
        self.assertIn("process EXPORT_OUTPUT_FORMAT", modules_text)
        self.assertNotIn("process BUILD_DECON_CONTAINER", modules_text)
        self.assertNotIn("process DESKEW", modules_text)
        self.assertNotIn("process CONVERT_TIFFS_TO_NEUROGLANCER", modules_text)

    def test_tiff_inputs_bypass_ome_zarr_normalization(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("def isTiffInputPattern(inputPattern)", main_text)
        self.assertIn("input_patterns.every { input_pattern -> isTiffInputPattern(input_pattern) }", main_text)
        self.assertIn("STAGE_DECON_TIFF_INPUT(input_files_ch)", main_text)
        self.assertIn("STAGE_DECON_INPUT(input_files_ch)", main_text)
        self.assertIn("process STAGE_DECON_TIFF_INPUT", modules_text)
        self.assertIn("ln -s", modules_text)
        self.assertIn("input_tiff", modules_text)
        self.assertNotIn("normalize_input_to_ome_zarr.py \\\n        --input input_tiff", modules_text)

    def test_matlab_stack_helpers_are_packaged_for_blind_psf_estimation(self):
        script_dir = ROOT / "workflow/scripts"

        self.assertTrue((script_dir / "readtiffstack.m").is_file())
        self.assertTrue((script_dir / "writetiffstack.m").is_file())

    def test_cupy_backend_persists_float_seed_before_spawning_worker(self):
        source = PSF_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("str(psf_seed_path)", source)
        self.assertIn("np.asarray(psf_seed, dtype=np.float32)", source)
        self.assertIn('photometric="minisblack"', source)

    def test_cupy_blind_backend_uses_direct_array_estimation(self):
        psf_source = PSF_SCRIPT_PATH.read_text(encoding="utf-8")
        blind_source = (ROOT / "workflow/scripts/blind_rl.py").read_text(encoding="utf-8")

        self.assertIn("def estimate_psf_array_cupy(", blind_source)
        self.assertIn("def clear_cupy_memory(", blind_source)
        self.assertIn("def trim_cupy_memory_pool(", blind_source)
        self.assertIn("def _run_cupy_deconvblind_array(", psf_source)
        self.assertIn("estimate_psf_array_cupy(", psf_source)
        self.assertIn("clear_plan_cache=False", psf_source)
        self.assertIn("free_memory_pool=False", psf_source)
        self.assertIn("trim_cupy_memory_pool", psf_source)
        self.assertNotIn(
            "isolated spawned GPU process",
            psf_source,
        )

    def test_cupy_tile_estimation_skips_chunk_tiff_roundtrip(self):
        module = load_psf_estimation()
        volume = np.ones((3, 8, 8), dtype=np.float32)
        psf_seed = np.ones((3, 3, 3), dtype=np.float32)
        psf_seed /= psf_seed.sum()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(
                    module,
                    "_write_chunk",
                    side_effect=AssertionError("CuPy path should not write chunk TIFFs"),
                ) as write_chunk,
                mock.patch.object(
                    module,
                    "_run_cupy_deconvblind_array",
                    return_value=psf_seed.copy(),
                ) as run_cupy,
            ):
                idx, psf_chunk, weight, error = module._estimate_one_tile(
                    0,
                    1,
                    volume,
                    (0, 0, 8, 8),
                    psf_seed,
                    pad_xy=0,
                    pad_z=0,
                    n_iters=2,
                    script_dir=ROOT / "workflow/scripts",
                    tmpdir=Path(tmpdir),
                    backend="cupy",
                    backend_lock=None,
                    matlab_threads=1,
                    matlab_bin="matlab",
                    matlab_timeout=1,
                    blind_peak_normalization="none",
                    blind_peak_gamma_max=2.5,
                    blind_latent_update_period=2,
                    snr_weight_cap=100.0,
                )

        self.assertEqual(idx, 0)
        self.assertIsNone(error)
        self.assertGreater(weight, 0)
        np.testing.assert_allclose(psf_chunk, psf_seed, rtol=1e-6, atol=1e-8)
        write_chunk.assert_not_called()
        run_cupy.assert_called_once()

    def test_cupy_array_estimation_reuses_pool_between_tiles(self):
        module = load_psf_estimation()
        psf_seed = np.ones((3, 3, 3), dtype=np.float32)
        psf_seed /= psf_seed.sum()
        fake_blind_rl = types.SimpleNamespace(
            estimate_psf_array_cupy=mock.Mock(return_value=psf_seed.copy()),
            trim_cupy_memory_pool=mock.Mock(return_value=False),
        )

        with mock.patch.dict(sys.modules, {"blind_rl": fake_blind_rl}):
            result = module._run_cupy_deconvblind_array(
                np.ones((3, 8, 8), dtype=np.float32),
                psf_seed,
                n_iters=2,
                pad_z=0,
                blind_peak_normalization="none",
                blind_peak_gamma_max=2.5,
                blind_latent_update_period=2,
                cupy_pool_trim_bytes=1234,
            )

        np.testing.assert_allclose(result, psf_seed)
        fake_blind_rl.estimate_psf_array_cupy.assert_called_once()
        _, kwargs = fake_blind_rl.estimate_psf_array_cupy.call_args
        self.assertFalse(kwargs["clear_plan_cache"])
        self.assertFalse(kwargs["free_memory_pool"])
        self.assertEqual(kwargs["latent_update_period"], 2)
        fake_blind_rl.trim_cupy_memory_pool.assert_called_once_with(1234)

    def test_cupy_blind_sizing_models_fft_workspace_and_restarts_after_oom(self):
        source = PSF_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("def resolve_cupy_blind_chunk_xy(", source)
        self.assertIn("DEFAULT_CUPY_FFT_BYTES_PER_VOXEL", source)
        self.assertIn("from scipy.fft import next_fast_len", source)
        self.assertIn("def _is_cupy_out_of_memory(", source)
        self.assertIn("discarding the partial pass and retrying all tiles", source)
        self.assertIn("cupy_gpu_cap=1", source)

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
        self.assertIn("containerOptions = { params.image_aggressiveness == 'high'", modules_text)
        self.assertIn("'--nv -B /home1/apps/MATLAB:/home1/apps/MATLAB'", modules_text)
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

        self.assertIn('publishDir "${params.output_dir}", mode: \'copy\', pattern: \'DB2_*.{ozx,tif,tiff}\'', modules_text)
        self.assertIn(
            'path "DB2_*.{ozx,tif,tiff}", emit: decon_output',
            modules_text,
        )
        self.assertIn("zip_ome_zarr_to_ozx", modules_text)
        self.assertIn("mkdir -p ${shell_quote(publishRoot)}", modules_text)
        self.assertIn('cp -f "estimated_psf.tif" ${shell_quote(publishRoot)}/', modules_text)
        self.assertIn('cp -f "\\$output_archive" ${shell_quote(publishRoot)}/', modules_text)
        self.assertIn('cp -f "\\$output_tiff" ${shell_quote(publishRoot)}/', modules_text)
        self.assertLess(
            modules_text.index('cp -f "\\$output_archive" ${shell_quote(publishRoot)}/'),
            modules_text.index("rm -rf DB2_*.ome.zarr"),
        )

    def test_nextflow_storage_cleanup_is_enabled(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("cleanup = true", config_text)
        self.assertIn("scratch true", modules_text)

    def test_slurm_jobs_start_from_stable_workflow_directory(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("def slurmChdirOption = \"--chdir=${baseDir}\"", config_text)
        self.assertIn("clusterOptions = slurmChdirOption", config_text)
        self.assertIn("queue = { params.image_aggressiveness == 'high' ? '256GBv1' : 'GPUp40' }", config_text)
        self.assertIn("clusterOptions = { params.image_aggressiveness == 'high'", config_text)
        self.assertIn("--gres=gpu:1", config_text)

    def test_stage_decon_input_uses_super_queue(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        match = re.search(
            r"withName:\s*STAGE_DECON_INPUT\s*\{(?P<body>[^}]*)\}",
            config_text,
        )

        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertIn("queue = 'super'", body)
        self.assertNotIn("cpus", body)
        self.assertNotIn("memory", body)

    def test_ozx_is_native_input_and_output_format(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")

        self.assertIn("output_formats = 'ozx'", config_text)
        self.assertIn("output_dir = './output'", config_text)
        self.assertIn("\\.ozx", package_text)
        self.assertNotIn("id: output_formats", package_text)

    def test_ome_zarr_deconvolution_streams_directly_to_zarr_output(self):
        script_text = (ROOT / "workflow/scripts/decon_wrapper.py").read_text(encoding="utf-8")

        self.assertIn("def deconvolve_ome_zarr_to_zarr(", script_text)
        self.assertIn("deconvolve_ome_zarr_to_zarr(", script_text)
        self.assertNotIn("output = deconvolve_ome_zarr(\n", script_text)
        self.assertNotIn("write_ome_zarr_array(\n                out_name,", script_text)

    def test_downsampling_parameter_is_exposed_and_forwarded(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        script_text = (ROOT / "workflow/scripts/decon_wrapper.py").read_text(encoding="utf-8")

        self.assertIn("output_selection = 'ozx_1x'", config_text)
        self.assertIn("pyramid_max_downsample = 1", config_text)
        self.assertIn("id: output_selection", package_text)
        self.assertIn("[ 'tiff', 'TIFF (1x)' ]", package_text)
        self.assertIn("def resolveOutputSelection", main_text)
        self.assertIn("output_settings.pyramid_max_downsample", main_text)
        self.assertIn("pyramid_max_downsample_flag = flag('pyramid_max_downsample', pyramid_max_downsample)", modules_text)
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

    def test_decon_wrapper_uses_ni_when_ns_is_negative_one(self):
        module = load_decon_wrapper_with_fakes()

        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            zarr_path = image_dir / "sample.ome.zarr"
            zarr_path.mkdir()
            psf_seed_calls = []

            def fake_generate_psf_seed(**kwargs):
                psf_seed_calls.append(kwargs)
                return FakeArray((3, 3, 3))

            argv = [
                "decon_wrapper.py",
                "--image_path", str(image_dir),
                "--dxy", "0.168",
                "--dz", "0.2",
                "--wavelength", "0.595",
                "--detection_na", "0.7",
                "--ni", "1.33333",
                "--ns", "-1",
                "--psf_size_z", "3",
                "--psf_size_xy", "3",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(module, "discover_image_volumes", return_value=[zarr_path]),
                mock.patch.object(module, "open_ome_zarr_array", return_value=FakeArray((2, 4, 4))),
                mock.patch.object(module, "generate_psf_seed", side_effect=fake_generate_psf_seed),
                mock.patch.object(module, "deconvolve_ome_zarr_to_zarr"),
            ):
                module.main()

        self.assertEqual(psf_seed_calls[0]["ni"], 1.33333)
        self.assertEqual(psf_seed_calls[0]["ns"], 1.33333)

    def test_decon_wrapper_uses_ni_when_ns_is_omitted(self):
        module = load_decon_wrapper_with_fakes()

        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            zarr_path = image_dir / "sample.ome.zarr"
            zarr_path.mkdir()
            psf_seed_calls = []

            def fake_generate_psf_seed(**kwargs):
                psf_seed_calls.append(kwargs)
                return FakeArray((3, 3, 3))

            argv = [
                "decon_wrapper.py",
                "--image_path", str(image_dir),
                "--dxy", "0.168",
                "--dz", "0.2",
                "--wavelength", "0.595",
                "--detection_na", "0.7",
                "--ni", "1.33333",
                "--psf_size_z", "3",
                "--psf_size_xy", "3",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(module, "discover_image_volumes", return_value=[zarr_path]),
                mock.patch.object(module, "open_ome_zarr_array", return_value=FakeArray((2, 4, 4))),
                mock.patch.object(module, "generate_psf_seed", side_effect=fake_generate_psf_seed),
                mock.patch.object(module, "deconvolve_ome_zarr_to_zarr"),
            ):
                module.main()

        self.assertEqual(psf_seed_calls[0]["ni"], 1.33333)
        self.assertEqual(psf_seed_calls[0]["ns"], 1.33333)

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
