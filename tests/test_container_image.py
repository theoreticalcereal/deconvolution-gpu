from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "workflow" / "images" / "deconvolution-gpu"
IMAGE = "git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.0"


class DeconvolutionContainerImageTests(unittest.TestCase):
    def test_image_context_builds_expected_app_environment(self):
        dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
        environment = (IMAGE_ROOT / "environment.yml").read_text(encoding="utf-8")
        build_script = (IMAGE_ROOT / "build-deconvolution-image.sh").read_text(encoding="utf-8")
        check_script = (IMAGE_ROOT / "check-deployment-container.sh").read_text(encoding="utf-8")

        self.assertIn("FROM docker.io/continuumio/miniconda3:", dockerfile)
        self.assertIn("conda env create --name app --file /tmp/environment.yml", dockerfile)
        self.assertIn("ENV PATH=/opt/conda/envs/app/bin:/opt/conda/bin:$PATH", dockerfile)

        for dependency in (
            "python=3.10",
            "cudatoolkit=11.8",
            "cudadecon=0.7.0",
            "pycudadecon=0.5.1",
            "numpy",
            "numba",
            "tifffile=2025.5.10",
            "zarr=2.18.3",
            "numcodecs",
            "h5py",
            "aicsimageio",
            "nd2",
            "readlif",
            "psfmodels",
            "antspyx",
            "dask-jobqueue",
        ):
            self.assertIn(dependency, environment)

        self.assertIn("readonly IMAGE=${REGISTRY}/${IMAGE_NAME}:${TAG}", build_script)
        self.assertIn(IMAGE, build_script)
        self.assertIn("REGISTRY_USERNAME", build_script)
        self.assertIn("REGISTRY_PASSWORD", build_script)
        self.assertIn("podman", build_script)
        self.assertIn("NO_CACHE", build_script)
        self.assertIn("--no-cache", build_script)
        self.assertIn("singularity exec", build_script)

        self.assertIn(IMAGE, check_script)
        self.assertIn("id -un", check_script)
        self.assertIn("SINGULARITY_DOCKER_USERNAME", check_script)
        self.assertIn("singularity inspect", check_script)


if __name__ == "__main__":
    unittest.main()
