import re
import unittest
from importlib import util
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


DESKEW_WRAPPER_PATH = ROOT / "workflow/scripts/deskew_wrapper.py"
DESKEW_WRAPPER_SPEC = util.spec_from_file_location("deskew_wrapper", DESKEW_WRAPPER_PATH)
deskew_wrapper = util.module_from_spec(DESKEW_WRAPPER_SPEC)
DESKEW_WRAPPER_SPEC.loader.exec_module(deskew_wrapper)


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

    def test_main_workflow_normalizes_sentinel_values_before_deskew(self):
        text = (ROOT / "workflow/main.nf").read_text()
        self.assertIn("def isSupplied(value)", text)
        self.assertIn("def requireSupplied(name, value, context)", text)
        self.assertIn("deskew_cell_index = optionalValue(params.cell_index)", text)
        self.assertIn("deskew_dx = requireSupplied('dx', params.dx, 'deskew runs')", text)

    def test_deskew_process_passes_configured_matlab_binary(self):
        text = (ROOT / "workflow/modules.nf").read_text()
        self.assertIn('--matlab_bin "${params.matlab_bin}"', text)

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
        self.assertIn('glob(str(image_dir / "*.tif"))', text)
        self.assertIn('glob(str(image_dir / "*.tiff"))', text)
        self.assertNotIn("No CH##_######", text)

    def test_deskew_handles_single_page_tiffs(self):
        deskew_text = (ROOT / "workflow/scripts/deskew.m").read_text()
        writer_text = (ROOT / "workflow/scripts/writetiffstack.m").read_text()
        self.assertIn("if zsize == 1", deskew_text)
        self.assertIn("Single-slice TIFF detected", deskew_text)
        self.assertIn("if ismatrix(image)", writer_text)

    def test_decon_promotes_single_page_tiffs_to_one_slice_volumes(self):
        text = (ROOT / "workflow/scripts/psf_estimation.py").read_text()
        self.assertIn("def ensure_3d_volume", text)
        self.assertIn("if volume.ndim == 2", text)
        self.assertIn("return volume[np.newaxis, :, :]", text)
        self.assertIn("def adapt_psf_seed_to_volume", text)
        self.assertIn("Adapted PSF seed shape", text)
        self.assertIn("Single-slice blind volume detected; disabling Z padding", text)
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

    def test_astrocyte_config_prepares_decon_container(self):
        text = (ROOT / "workflow/configs/astrocyte.config").read_text()
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

    def test_decon_uses_conda_runtime_without_container_or_sif(self):
        text = (ROOT / "workflow/modules.nf").read_text()
        self.assertNotIn("container { decon_container.toString() }", text)
        self.assertNotIn("singularity exec", text)
        self.assertIn('export CONDA_PREFIX="${decon_runtime}/decon_env"', text)
        self.assertIn('export PATH="\\${CONDA_PREFIX}/bin:\\${PATH}"', text)
        self.assertIn('for candidate in "\\${matlab_bin}" matlab /home1/apps/MATLAB/R2024a/bin/matlab', text)
        self.assertIn('--matlab_bin "\\$matlab_bin"', text)

    def test_astrocyte_package_does_not_declare_external_container(self):
        package_text = (ROOT / "astrocyte_pkg.yml").read_text()
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text()

        self.assertNotIn("container: 'singularity'", package_text)
        self.assertIn("  - 'matlab/2024a'", package_text)
        self.assertIn("  - 'anaconda3/2023.09-0'", package_text)
        self.assertNotIn("  - 'mamba/2.3.0'", package_text)
        self.assertNotIn("workflow_containers:", package_text)
        self.assertNotIn("docker://git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution/decon_env:latest", package_text)
        self.assertNotIn("decon_container_image", config_text)
        self.assertNotIn("decon_conda_env_archive", config_text)
