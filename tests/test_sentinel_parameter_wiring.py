import re
import sys
import tempfile
import types
import unittest
import yaml
from importlib import util
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


DESKEW_WRAPPER_PATH = ROOT / "workflow/scripts/deskew_wrapper.py"
DESKEW_WRAPPER_SPEC = util.spec_from_file_location("deskew_wrapper", DESKEW_WRAPPER_PATH)
deskew_wrapper = util.module_from_spec(DESKEW_WRAPPER_SPEC)
DESKEW_WRAPPER_SPEC.loader.exec_module(deskew_wrapper)


def load_decon_wrapper_with_stubs():
    stubs = {
        "dask": types.ModuleType("dask"),
        "dask.array": types.ModuleType("dask.array"),
        "numpy": types.ModuleType("numpy"),
        "pycudadecon": types.SimpleNamespace(
            TemporaryOTF=object,
            RLContext=object,
            rl_decon=lambda *args, **kwargs: None,
        ),
        "psf_estimation": types.SimpleNamespace(
            DEFAULT_BLIND_CHUNK_XY=512,
            DEFAULT_BLIND_Z_SLICES=0,
            DEFAULT_SNR_WEIGHT_CAP=10.0,
            estimate_psf_from_chunks=lambda *args, **kwargs: None,
            detect_vram_bytes=lambda: 0,
            open_tiff_memmap=lambda *args, **kwargs: None,
            resolve_dxy=lambda *args, **kwargs: None,
            resolve_chunk_xy=lambda *args, **kwargs: None,
        ),
        "psf_modes": types.SimpleNamespace(
            generate_psf_seed=lambda *args, **kwargs: None,
        ),
        "tifffile": types.SimpleNamespace(imwrite=lambda *args, **kwargs: None),
    }
    stubs["dask"].array = stubs["dask.array"]
    stubs["numpy"].ndarray = object
    with mock.patch.dict(sys.modules, stubs):
        spec = util.spec_from_file_location(
            "decon_wrapper_for_tests",
            ROOT / "workflow/scripts/decon_wrapper.py",
        )
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def parameter_block(text, parameter_id):
    pattern = re.compile(
        rf"(?ms)^  - id: {re.escape(parameter_id)}\n"
        rf".*?(?=^  - id: |\Z)"
    )
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"Parameter {parameter_id!r} not found")
    return match.group(0)


class SentinelParameterWiringTests(unittest.TestCase):
    def test_astrocyte_optional_numeric_parameters_default_to_negative_one(self):
        text = (ROOT / "astrocyte_pkg.yml").read_text()
        optional_sentinel_parameters = [
            "dx",
            "dxy",
            "camera_pixel_size",
            "magnification",
            "detection_na",
            "illumination_na",
            "ni",
            "ns",
            "vram_gb",
        ]

        for parameter_id in optional_sentinel_parameters:
            with self.subTest(parameter_id=parameter_id):
                block = parameter_block(text, parameter_id)
                self.assertIn("default: -1", block)
                self.assertIn("Use -1", block)

    def test_nextflow_config_uses_negative_one_for_numeric_sentinels(self):
        text = (ROOT / "workflow/configs/nextflow.config").read_text()
        sentinel_defaults = [
            "cell_index",
            "dx",
            "dz",
            "wavelength",
            "ni",
            "ns",
            "ni0",
            "tg",
            "tg0",
            "ng",
            "ng0",
            "ti0",
        ]

        for parameter_id in sentinel_defaults:
            with self.subTest(parameter_id=parameter_id):
                self.assertRegex(text, rf"(?m)^\s*{parameter_id}\s*=\s*-1\b")

    def test_dataset_params_files_only_use_declared_workflow_parameters(self):
        astrocyte_text = (ROOT / "astrocyte_pkg.yml").read_text()
        parameter_ids = set(re.findall(r"(?m)^  - id: ([A-Za-z0-9_]+)$", astrocyte_text))

        for params_path in ROOT.glob("*.params.yml"):
            with self.subTest(params_path=params_path.name):
                params = yaml.safe_load(params_path.read_text()) or {}
                unknown_keys = sorted(set(params) - parameter_ids)
                self.assertEqual(unknown_keys, [])

    def test_main_workflow_normalizes_sentinel_values_before_deskew(self):
        text = (ROOT / "workflow/main.nf").read_text()
        self.assertIn("def isSupplied(value)", text)
        self.assertIn("def requireSupplied(name, value, context)", text)
        self.assertIn("deskew_cell_index = optionalValue(params.cell_index)", text)
        self.assertIn("deskew_dx = requireSupplied('dx', params.dx, 'deskew runs')", text)

    def test_decon_process_passes_configured_matlab_binary(self):
        text = (ROOT / "workflow/modules.nf").read_text()
        self.assertIn('matlab_bin="${params.matlab_bin', text)
        self.assertIn('--matlab_bin "\\$matlab_bin"', text)

    def test_deskew_process_uses_chunked_python_runtime(self):
        main_text = (ROOT / "workflow/main.nf").read_text()
        modules_text = (ROOT / "workflow/modules.nf").read_text()
        self.assertIn("path decon_runtime", modules_text)
        self.assertIn("chunked_deskew.py", modules_text)
        self.assertIn("export CONDA_PREFIX=\"${decon_runtime}/decon_env\"", modules_text)
        self.assertIn("deskew_workers", modules_text)
        self.assertIn("deskew_prefetch", modules_text)
        self.assertIn("decon_container_ch", main_text)
        self.assertIn("DESKEW(", main_text)

    def test_chunked_deskew_parallelizes_page_computation(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text()
        script_text = (ROOT / "workflow/scripts/chunked_deskew.py").read_text()
        self.assertRegex(config_text, r"(?m)^\s*deskew_workers\s*=\s*32\b")
        self.assertRegex(config_text, r"(?m)^\s*deskew_prefetch\s*=\s*64\b")
        self.assertIn("ThreadPoolExecutor", script_text)
        self.assertIn("FIRST_COMPLETED", script_text)
        self.assertIn("pending_pages", script_text)
        self.assertIn("write_buffer", script_text)

    def test_deskew_wrapper_uses_configured_matlab_binary(self):
        with mock.patch.object(deskew_wrapper.subprocess, "run") as run:
            deskew_wrapper.run_deskew(
                "/input",
                "",
                "",
                "0.167",
                "0.2",
                "45",
                "1",
                ".",
                matlab_bin="/home1/apps/MATLAB/R2024a/bin/matlab",
            )

        command = run.call_args.args[0]
        self.assertEqual(command[0], "/home1/apps/MATLAB/R2024a/bin/matlab")
        self.assertEqual(command[1], "-batch")

    def test_deskew_processes_discovered_tiffs_without_channel_timepoint_names(self):
        text = (ROOT / "workflow/scripts/deskew.m").read_text()
        self.assertIn("for fileIdx = 1:numel(tifFiles)", text)
        self.assertNotIn("No CH##_###### TIFF files found", text)
        self.assertNotIn("Missing expected file", text)

    def test_decon_processes_discovered_tiffs_without_channel_timepoint_names(self):
        text = (ROOT / "workflow/scripts/decon_wrapper.py").read_text()
        self.assertIn("discover_image_volumes(image_dir)", text)
        self.assertIn("deconvolve_tiff(", text)
        self.assertIn("deconvolve_ome_zarr(", text)
        self.assertNotIn("No CH##_######", text)

    def test_deskew_preserves_original_input_filename_metadata_for_decon(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text()
        chunked_deskew_text = (ROOT / "workflow/scripts/chunked_deskew.py").read_text()

        self.assertIn("original_filenames.tsv", modules_text)
        self.assertIn("original_filenames.tsv", chunked_deskew_text)
        self.assertIn("shutil.copy2", chunked_deskew_text)

    def test_selected_inputs_are_normalized_to_ome_zarr_before_processing(self):
        main_text = (ROOT / "workflow/main.nf").read_text()
        modules_text = (ROOT / "workflow/modules.nf").read_text()

        self.assertIn("STAGE_DECON_INPUT(input_tiffs_ch, decon_container_ch)", main_text)
        self.assertIn('path "input_zarr", emit: decon_input_dir', modules_text)
        self.assertIn("normalize_input_to_ome_zarr.py", modules_text)
        self.assertIn("--output input_zarr", modules_text)

    def test_decon_names_outputs_from_original_filename_metadata(self):
        text = (ROOT / "workflow/scripts/decon_wrapper.py").read_text()

        self.assertIn("def _load_original_name_map", text)
        self.assertIn("def _decon_output_name", text)
        self.assertIn("def _decon_ome_zarr_output_name", text)
        self.assertIn("original_filenames.tsv", text)
        self.assertIn("out_name = _decon_output_name(image_path, original_name_map)", text)
        self.assertIn("out_name = _decon_ome_zarr_output_name(image_path, original_name_map)", text)

        decon_wrapper = load_decon_wrapper_with_stubs()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            (image_dir / "original_filenames.tsv").write_text(
                "CH00_000000.tif\toriginal sample 561.tiff\n"
            )

            original_name_map = decon_wrapper._load_original_name_map(image_dir)

        self.assertEqual(
            decon_wrapper._decon_output_name(Path("CH00_000000.tif"), original_name_map),
            "DB2_original sample 561.tif",
        )
        self.assertEqual(
            decon_wrapper._decon_output_name(Path("CH00_000001.tif"), original_name_map),
            "DB2_CH00_000001.tif",
        )
        self.assertEqual(
            decon_wrapper._decon_ome_zarr_output_name(Path("CH00_000001.ome.zarr"), original_name_map),
            "DB2_CH00_000001.ome.zarr",
        )

    def test_deskew_handles_single_page_tiffs(self):
        deskew_text = (ROOT / "workflow/scripts/deskew.m").read_text()
        writer_text = (ROOT / "workflow/scripts/writetiffstack.m").read_text()
        self.assertIn("if zsize == 1", deskew_text)
        self.assertIn("Single-slice TIFF detected", deskew_text)
        self.assertIn("if ismatrix(image)", writer_text)

    def test_deskew_reader_handles_imagej_single_ifd_stacks(self):
        text = (ROOT / "workflow/scripts/readtiffstack.m").read_text()
        self.assertIn("parseImageJStackDepth", text)
        self.assertIn("readImageJSingleIFDStack", text)
        self.assertIn("images=", text)
        self.assertIn("firstStripOffset", text)
        self.assertIn("fread(fid, expectedPixels", text)
        self.assertIn("NumberImages = imageJDepth", text)

    def test_decon_promotes_single_page_tiffs_to_one_slice_volumes(self):
        text = (ROOT / "workflow/scripts/psf_estimation.py").read_text()
        self.assertIn("def ensure_3d_volume", text)
        self.assertIn("if volume.ndim == 2", text)
        self.assertIn("return volume[np.newaxis, :, :]", text)
        self.assertIn("def adapt_psf_seed_to_volume", text)
        self.assertIn("Adapted PSF seed shape", text)
        self.assertIn("Single-slice blind volume detected; disabling Z padding", text)
        self.assertIn("psf_chunk = ensure_3d_volume(imread(str(psf_out_path)))", text)
        self.assertIn("failure_details.append", text)

    def test_gpu_decon_uses_available_cpu_allocation(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text()
        modules_text = (ROOT / "workflow/modules.nf").read_text()
        self.assertRegex(config_text, r"(?m)^\s*blind_workers\s*=\s*72\b")
        self.assertRegex(config_text, r"(?m)^\s*matlab_workers\s*=\s*24\b")
        self.assertRegex(config_text, r"(?m)^\s*cpus\s*=\s*72\b")
        self.assertIn("memory         = '256 GB'", config_text)
        self.assertIn("cpus 72", modules_text)
        self.assertIn("memory '256 GB'", modules_text)

    def test_auto_decon_chunking_is_capped_for_wide_single_slice_inputs(self):
        text = (ROOT / "workflow/scripts/decon_wrapper.py").read_text()
        self.assertIn("def _auto_decon_max_xy", text)
        self.assertIn("target_bytes = vram_bytes * 0.05 / workers", text)
        self.assertIn("memory_multiplier = 2048.0", text)
        self.assertIn("max_xy=_auto_decon_max_xy", text)

    def test_astrocyte_config_prepares_decon_container(self):
        wrapper_text = (ROOT / "workflow/configs/astrocyte.config").read_text()
        text = (ROOT / "workflow/configs/biohpc.config").read_text()

        self.assertIn("includeConfig 'biohpc.config'", wrapper_text)
        self.assertRegex(text, r"(?m)^\s*params\.build_decon_container\s*=\s*true\b")

    def test_container_prep_builds_conda_libmamba_env_each_run(self):
        main_text = (ROOT / "workflow/main.nf").read_text()
        text = (ROOT / "workflow/modules.nf").read_text()
        self.assertIn("BUILD_DECON_CONTAINER()", main_text)
        self.assertNotIn("module '", text)
        self.assertNotIn("module load", text)
        self.assertIn("conda-libmamba-solver", text)
        self.assertIn('--file "${projectDir}/envs/decon-conda.txt"', text)
        self.assertIn('decon_runtime/decon_env/bin/python -m pip install -r "${projectDir}/envs/decon-pip-requirements.txt"', text)
        self.assertNotIn('env create -y --solver=libmamba -p decon_runtime/decon_env -f "${projectDir}/../environment.yml"', text)
        self.assertNotIn("mamba env create", text)
        self.assertNotIn("decon_env.tar.gz", text)
        self.assertNotIn("decon_env.sif", text)

        conda_text = (ROOT / "workflow/envs/decon-conda.txt").read_text()
        pip_text = (ROOT / "workflow/envs/decon-pip-requirements.txt").read_text()
        self.assertIn("pycudadecon=0.5.1", conda_text)
        self.assertIn("cudatoolkit=11.8", conda_text)
        self.assertIn("dask", pip_text)
        self.assertIn("psfmodels", pip_text)
        self.assertNotIn("imagecodecs", pip_text)
        self.assertNotIn("tifffile", pip_text)

    def test_neuroglancer_dependencies_are_declared(self):
        environment_text = (ROOT / "environment.yml").read_text()
        conda_text = (ROOT / "workflow/envs/decon-conda.txt").read_text()
        pip_text = (ROOT / "workflow/envs/decon-pip-requirements.txt").read_text()

        self.assertNotRegex(environment_text, r"(?m)^\s*-\s+cloud-volume\b")
        self.assertNotRegex(conda_text, r"(?m)^cloud-volume\b")
        self.assertNotRegex(pip_text, r"(?m)^cloud-volume\b")
        for dependency in ("numpy", "tifffile", "zarr", "numcodecs"):
            self.assertRegex(conda_text, rf"(?m)^{dependency}\b")
            self.assertRegex(environment_text, rf"(?m)^\s*-\s+{dependency}\b")
        self.assertRegex(environment_text, r"(?m)^\s*-\s+neuroglancer\b")
        self.assertRegex(pip_text, r"(?m)^neuroglancer\b")

    def test_dynamic_input_formats_are_declared_and_normalized_to_ome_zarr(self):
        astrocyte_text = (ROOT / "astrocyte_pkg.yml").read_text()
        normalizer_text = (ROOT / "workflow/scripts/normalize_input_to_ome_zarr.py").read_text()
        conda_text = (ROOT / "workflow/envs/decon-conda.txt").read_text()
        pip_text = (ROOT / "workflow/envs/decon-pip-requirements.txt").read_text()

        block = parameter_block(astrocyte_text, "input")
        for suffix in ("tif", "tiff", "czi", "nd2", "lif", "h5", "hdf5", "ome\\\\.zarr"):
            self.assertIn(suffix, block)
        for loader in ("load_tiff_volume", "load_czi_volume", "load_nd2_volume", "load_lif_volume", "load_hdf5_volume"):
            self.assertIn(loader, normalizer_text)
        self.assertRegex(conda_text, r"(?m)^h5py\b")
        for dependency in ("aicsimageio", "nd2", "readlif"):
            self.assertRegex(pip_text, rf"(?m)^{dependency}\b")

    def test_nextflow_wires_neuroglancer_conversion_after_decon(self):
        main_text = (ROOT / "workflow/main.nf").read_text()
        modules_text = (ROOT / "workflow/modules.nf").read_text()

        self.assertIn("include { CONVERT_TIFFS_TO_NEUROGLANCER } from './modules'", main_text)
        self.assertIn("process CONVERT_TIFFS_TO_NEUROGLANCER", modules_text)
        self.assertEqual(main_text.count("CONVERT_TIFFS_TO_NEUROGLANCER("), 2)
        for branch_call in (
            "DECON(\n            decon_input_ch",
            "DECON(\n            DESKEW.out.deskewed_path",
        ):
            decon_index = main_text.index(branch_call)
            convert_index = main_text.index("CONVERT_TIFFS_TO_NEUROGLANCER(", decon_index)
            self.assertGreater(convert_index, decon_index)

    def test_neuroglancer_conversion_publishes_zarr_data_under_deconvolved_contract(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text()

        self.assertNotIn("pattern: '{deconvolved,neuroglancer}'", modules_text)
        self.assertNotIn('path "deconvolved", emit: ome_zarr_output', modules_text)
        self.assertIn("workflow.launchDir", modules_text)
        self.assertIn('--output "${outputPrefix}/deconvolved"', modules_text)
        self.assertIn('--manifest-output "${outputPrefix}/neuroglancer"', modules_text)

    def test_neuroglancer_vizapp_files_are_packaged(self):
        astrocyte_text = (ROOT / "astrocyte_pkg.yml").read_text()

        self.assertTrue((ROOT / "workflow/scripts/convert_tiff_to_ome_zarr.py").exists())
        self.assertFalse((ROOT / "workflow/scripts/convert_tiff_to_precomputed.py").exists())
        self.assertFalse((ROOT / "dockerFileCode/Dockerfile").exists())
        self.assertFalse((ROOT / "vizapp/run_container.sh").exists())
        self.assertTrue((ROOT / "vizapp/run_neuroglancer.sh").exists())
        self.assertTrue((ROOT / "vizapp/neuroloader.py").exists())
        self.assertIn("vizapp_containers:", astrocyte_text)
        self.assertIn("docker://hello-world", astrocyte_text)
        self.assertIn("vizapp_container_runscripts:", astrocyte_text)
        self.assertIn("- run_neuroglancer.sh", astrocyte_text)

        runscript_text = (ROOT / "vizapp/run_neuroglancer.sh").read_text()
        self.assertIn("module load neuroglancer/2.40.1", runscript_text)

    def test_neuroglancer_data_mode_is_user_selectable_and_wired_to_converter(self):
        astrocyte_text = (ROOT / "astrocyte_pkg.yml").read_text()
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text()
        modules_text = (ROOT / "workflow/modules.nf").read_text()

        block = parameter_block(astrocyte_text, "neuroglancer_data_mode")
        self.assertIn("type: select", block)
        self.assertIn("default: 'auto'", block)
        self.assertIn("[ 'auto'", block)
        self.assertIn("[ '2d'", block)
        self.assertIn("[ '3d'", block)
        self.assertRegex(config_text, r"(?m)^\s*neuroglancer_data_mode\s*=\s*'auto'")
        self.assertIn("convert_tiff_to_ome_zarr.py", modules_text)
        self.assertIn('--volume-mode "${params.neuroglancer_data_mode}"', modules_text)

    def test_output_formats_defaults_to_native_ome_zarr(self):
        astrocyte_text = (ROOT / "astrocyte_pkg.yml").read_text()
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text()

        block = parameter_block(astrocyte_text, "output_formats")
        self.assertIn("type: string", block)
        self.assertIn("default: 'ome_zarr'", block)
        self.assertIn("ome_zarr,tiff", block)
        self.assertRegex(config_text, r"(?m)^\s*output_formats\s*=\s*'ome_zarr'")

    def test_decon_uses_conda_runtime_without_container_or_sif(self):
        text = (ROOT / "workflow/modules.nf").read_text()
        self.assertNotIn("container { decon_container.toString() }", text)
        self.assertNotIn("singularity exec", text)
        self.assertIn('export CONDA_PREFIX="${decon_runtime}/decon_env"', text)
        self.assertIn('export PATH="\\${CONDA_PREFIX}/bin:\\${PATH}"', text)
        self.assertIn('for candidate in "\\${matlab_bin}" matlab /home1/apps/MATLAB/R2024a/bin/matlab', text)
        self.assertIn('--matlab_bin "\\$matlab_bin"', text)

    def test_astrocyte_package_declares_only_vizapp_dummy_container(self):
        package_text = (ROOT / "astrocyte_pkg.yml").read_text()
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text()
        biohpc_text = (ROOT / "workflow/configs/biohpc.config").read_text()

        self.assertIn("nextflow_config: 'biohpc.config'", package_text)
        self.assertIn("container: 'singularity'", package_text)
        self.assertIn("singularity_version: '3.9.9'", package_text)
        self.assertIn("singularity {", biohpc_text)
        self.assertIn("cacheDir = \"$baseDir/images/singularity\"", biohpc_text)
        self.assertIn("  - 'matlab/2024a'", package_text)
        self.assertIn("  - 'anaconda3/2023.09-0'", package_text)
        self.assertNotIn("  - 'mamba/2.3.0'", package_text)
        self.assertNotIn("workflow_containers:", package_text)
        self.assertIn("vizapp_containers:", package_text)
        self.assertIn("docker://hello-world", package_text)
        self.assertNotIn("docker://git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution/decon_env:latest", package_text)
        self.assertNotIn("decon_container_image", config_text)
        self.assertNotIn("decon_conda_env_archive", config_text)
