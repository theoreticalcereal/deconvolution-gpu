from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = ROOT / "astrocyte_pkg.yml"
WRAPPER_PATH = ROOT / "workflow" / "scripts" / "decon_wrapper.py"
PETAKIT_PATH = ROOT / "workflow" / "scripts" / "petakit_rl.py"


def load_module(name: str, path: Path):
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    # The lightweight test environment intentionally does not contain the
    # production-only psfmodels dependency. Parameter intake never invokes it.
    with mock.patch.dict(sys.modules, {"psfmodels": types.SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class AstrocyteParameterIntakeTests(unittest.TestCase):
    def test_astrocyte_form_exposes_microscope_profile_and_optional_acquisition_values(self):
        package = yaml.safe_load(PACKAGE_PATH.read_text(encoding="utf-8"))
        parameters = package["workflow_parameters"]

        self.assertEqual(
            [parameter["id"] for parameter in parameters],
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
        self.assertEqual(parameters[1]["type"], "select")
        self.assertTrue(parameters[1]["required"])
        self.assertEqual(
            parameters[1]["choices"],
            [
                ["auto", "Infer from optional acquisition YAML"],
                ["upright_aslm_36x_ri_1_33", "Upright ASLM — 36x — RI 1.33"],
                ["benchtop_mesospim_4x_ri_1_56", "BenchTop MesoSPIM — 4x — RI 1.56"],
                ["benchtop_mesospim_10x_ri_1_56", "BenchTop MesoSPIM — 10x — RI 1.56"],
                ["benchtop_mesospim_10x_ri_1_52", "BenchTop MesoSPIM — 10x — RI 1.52"],
                ["ctaslm_v3_50x_ri_1_56", "ctASLM v3 — 50x — RI 1.56"],
                ["ctaslm_v3_50x_ri_1_52", "ctASLM v3 — 50x — RI 1.52"],
                ["multiscale_low_res_0_63x_ri_1_56", "Multiscale - Low Res — 0.63x — RI 1.56"],
                ["multiscale_low_res_1x_ri_1_56", "Multiscale - Low Res — 1x — RI 1.56"],
                ["multiscale_low_res_2x_ri_1_56", "Multiscale - Low Res — 2x — RI 1.56"],
                ["multiscale_low_res_3x_ri_1_56", "Multiscale - Low Res — 3x — RI 1.56"],
                ["multiscale_low_res_4x_ri_1_56", "Multiscale - Low Res — 4x — RI 1.56"],
                ["multiscale_low_res_5x_ri_1_56", "Multiscale - Low Res — 5x — RI 1.56"],
                ["multiscale_low_res_6x_ri_1_56", "Multiscale - Low Res — 6x — RI 1.56"],
                ["multiscale_high_res_38x_ri_1_56", "Multiscale - High Res — 38x — RI 1.56"],
                ["multiscale_high_res_37x_ri_1_52", "Multiscale - High Res — 37x — RI 1.52"],
                ["custom", "Custom — provide deconvolution parameters YAML"],
            ],
        )
        self.assertEqual(parameters[2]["type"], "files")
        self.assertFalse(parameters[2]["required"])
        self.assertIn(r"\.yml", parameters[2]["regex"])
        self.assertIn(r"\.yaml", parameters[2]["regex"])
        self.assertNotIn(r"\.json", parameters[2]["regex"])
        self.assertEqual(
            parameters[5]["choices"],
            [
                ["low", "Low — fastest GPU processing"],
                ["medium", "Medium — balanced GPU processing"],
                ["high", "High — maximum-accuracy CPU processing"],
            ],
        )
        self.assertEqual(parameters[6]["type"], "select")
        self.assertEqual(parameters[6]["default"], "ozx_1x")
        self.assertEqual(
            parameters[6]["choices"],
            [
                ["ozx_1x", "1x"],
                ["ozx_2x", "2x"],
                ["ozx_4x", "4x"],
                ["ozx_8x", "8x"],
                ["ozx_16x", "16x"],
                ["tiff", "TIFF (1x)"],
            ],
        )

    def test_optional_acquisition_yaml_infers_profile_wavelength_and_z_spacing(self):
        wrapper = load_module("decon_wrapper_parameter_intake", WRAPPER_PATH)
        configuration = {
            "Saving": {"prefix": "38x_", "solvent": "BABB"},
            "MicroscopeState": {
                "microscope_name": "Nanoscale",
                "step_size": -0.2,
                "channels": {
                    "channel_3": {"is_selected": True, "laser": "642nm"},
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "acquisition.yml"
            config_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
            args = wrapper.parse_workflow_arguments(
                [
                    "--image_path",
                    str(tmp_path / "image_input"),
                    "--config_file",
                    str(config_path),
                    "--microscope_profile",
                    "auto",
                    "--image_aggressiveness",
                    "high",
                ]
            )

        self.assertEqual(args.microscope_profile, "multiscale_high_res_38x_ri_1_56")
        self.assertEqual(args.detection_na, 0.753)
        self.assertEqual(args.illumination_na, 0.753)
        self.assertEqual(args.dxy, 0.171)
        self.assertEqual(args.ni, 1.56)
        self.assertEqual(args.ns, 1.56)
        self.assertEqual(args.wavelength, 0.642)
        self.assertEqual(args.dz, 0.2)
        self.assertEqual(args.blind_backend, "matlab")
        self.assertEqual(args.decon_backend, "petakit")

    def test_explicit_profile_and_acquisition_values_override_yaml_inference(self):
        wrapper = load_module("decon_wrapper_parameter_precedence", WRAPPER_PATH)
        configuration = {
            "Saving": {"prefix": "38x_", "solvent": "BABB"},
            "MicroscopeState": {
                "microscope_name": "Nanoscale",
                "step_size": -0.2,
                "channels": {
                    "channel_3": {"is_selected": True, "laser": "642nm"},
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "acquisition.yml"
            config_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
            args = wrapper.parse_workflow_arguments(
                [
                    "--image_path",
                    str(tmp_path / "image_input"),
                    "--config_file",
                    str(config_path),
                    "--microscope_profile",
                    "benchtop_mesospim_4x_ri_1_56",
                    "--wavelength",
                    "0.561",
                    "--dz",
                    "0.3",
                    "--image_aggressiveness",
                    "medium",
                ]
            )

        self.assertEqual(args.microscope_profile, "benchtop_mesospim_4x_ri_1_56")
        self.assertEqual(args.detection_na, 0.25)
        self.assertEqual(args.illumination_na, 0.1)
        self.assertEqual(args.dxy, 1.609)
        self.assertEqual(args.wavelength, 0.561)
        self.assertEqual(args.dz, 0.3)

    def test_custom_profile_applies_flat_deconvolution_parameter_yaml(self):
        wrapper = load_module("decon_wrapper_custom_parameter_intake", WRAPPER_PATH)
        configuration = {
            "iter": 7,
            "background": 3,
            "wavelength": 0.61,
            "dxy": 0.104,
            "dz": 0.3,
            "detection_na": 1.1,
            "illumination_na": 0.19,
            "ni": 1.33,
            "ns": 1.33,
            "vram_gb": 40,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "custom.yml"
            config_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
            args = wrapper.parse_workflow_arguments(
                [
                    "--image_path",
                    str(tmp_path / "image_input"),
                    "--config_file",
                    str(config_path),
                    "--microscope_profile",
                    "custom",
                    "--image_aggressiveness",
                    "high",
                ]
            )

        self.assertEqual(args.microscope_profile, "custom")
        self.assertEqual(args.iter, 7)
        self.assertEqual(args.background, 3)
        self.assertEqual(args.wavelength, 0.61)
        self.assertEqual(args.dxy, 0.104)
        self.assertEqual(args.dz, 0.3)
        self.assertEqual(args.detection_na, 1.1)
        self.assertEqual(args.illumination_na, 0.19)
        self.assertEqual(args.ni, 1.33)
        self.assertEqual(args.ns, 1.33)
        self.assertEqual(args.blind_backend, "matlab")
        self.assertEqual(args.decon_backend, "petakit")
        self.assertIsNone(args.vram_gb)

    def test_custom_yaml_cannot_override_the_output_dropdown(self):
        wrapper = load_module("decon_wrapper_custom_output_policy", WRAPPER_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "custom.yml"
            config_path.write_text("output_format: tiff\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output_format"):
                wrapper.parse_workflow_arguments(
                    [
                        "--image_path",
                        str(tmp_path / "image_input"),
                        "--config_file",
                        str(config_path),
                        "--microscope_profile",
                        "custom",
                        "--image_aggressiveness",
                        "medium",
                    ]
                )

    def test_mode_resolver_maps_each_aggressiveness_level_to_its_required_engines(self):
        wrapper = load_module("decon_wrapper_mode_resolver", WRAPPER_PATH)

        for mode, expected in {
            "low": {"blind_backend": "cupy", "cupy_fft_engine": "scout", "decon_backend": "cupy"},
            "medium": {"blind_backend": "cupy", "cupy_fft_engine": "cupyx", "decon_backend": "cupy", "blind_max_tiles": 0},
            "high": {"blind_backend": "matlab", "cupy_fft_engine": "scout", "decon_backend": "petakit"},
        }.items():
            with self.subTest(mode=mode):
                self.assertEqual(wrapper.resolve_image_aggressiveness(mode), expected)

    def test_high_mode_petakit_cpu_restoration_matches_the_petakit_reference_core(self):
        petakit = load_module("petakit_rl_cpu_mode", PETAKIT_PATH)
        observed = np.array([[[3, 9, 6], [8, 2, 4]]], dtype=np.uint16)
        psf = np.array([[[1, 2, 1]]], dtype=np.float32)

        restored = petakit.restore_uint16_petakit_cpu(
            observed, psf, n_iters=3, background=1
        )
        expected = petakit.petakit_simplified_rl(
            observed, psf, n_iters=3, background=1, xp=np
        )
        expected = np.floor(np.clip(expected, 0, 65535) + 0.5).astype(np.uint16)

        np.testing.assert_array_equal(restored, expected)


if __name__ == "__main__":
    unittest.main()
