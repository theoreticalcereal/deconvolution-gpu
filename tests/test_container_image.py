from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "workflow" / "images" / "deconvolution-gpu"
IMAGE = "git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deconvolution:0.1.2"


class DeconvolutionContainerImageTests(unittest.TestCase):
    def test_image_context_builds_expected_app_environment(self):
        dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
        environment = (IMAGE_ROOT / "environment.yml").read_text(encoding="utf-8")
        gpu_environment = (IMAGE_ROOT / "gpu-environment.yml").read_text(encoding="utf-8")
        requirements = (IMAGE_ROOT / "requirements.txt").read_text(encoding="utf-8")
        constraints = (IMAGE_ROOT / "constraints.txt").read_text(encoding="utf-8")
        build_script = (IMAGE_ROOT / "build-deconvolution-image.sh").read_text(encoding="utf-8")
        check_script = (IMAGE_ROOT / "check-deployment-container.sh").read_text(encoding="utf-8")

        self.assertIn("FROM docker.io/continuumio/miniconda3:", dockerfile)
        self.assertIn("COPY conda-linux-64.lock requirements.txt constraints.txt /tmp/", dockerfile)
        self.assertIn("conda create --yes --name app --file /tmp/conda-linux-64.lock", dockerfile)
        self.assertIn("python -m pip install", dockerfile)
        self.assertIn("--constraint /tmp/constraints.txt", dockerfile)
        self.assertIn("python -m pip check", dockerfile)
        self.assertIn("scientific stack versions and binary imports verified", dockerfile)
        self.assertIn("ENV PATH=/opt/conda/envs/app/bin:/opt/conda/bin:$PATH", dockerfile)

        for dependency in (
            "python=3.10",
            "cudatoolkit=11.8",
            "numpy",
            "numba",
            "tifffile=2025.5.10",
            "zarr=2.18.3",
            "numcodecs",
            "h5py",
            "imagecodecs",
            "scipy",
            "dask-jobqueue",
        ):
            self.assertIn(dependency, environment)

        for dependency in (
            "antspyx",
            "psfmodels",
            "aicsimageio",
            "nd2",
            "readlif",
            "tifffile==2025.5.10",
        ):
            self.assertIn(dependency, requirements)

        for constraint in (
            "numpy==1.26.4",
            "numba==0.59.1",
            "scipy==1.15.2",
            "scikit-image==0.19.3",
            "tifffile==2025.5.10",
            "zarr==2.18.3",
            "imagecodecs==2021.11.20",
        ):
            self.assertIn(constraint, constraints)

        for dependency in (
            "cuda-version=11.8",
            "cupy=13.6.0",
        ):
            self.assertIn(dependency, gpu_environment)

        self.assertNotIn("cucim", gpu_environment)
        self.assertNotIn("rapidsai", gpu_environment)
        self.assertIn("nodefaults", environment)
        self.assertIn("nodefaults", gpu_environment)
        self.assertIn("readonly IMAGE=${REGISTRY}/${IMAGE_NAME}:${TAG}", build_script)
        self.assertIn(IMAGE, build_script)
        self.assertIn("REGISTRY_USERNAME", build_script)
        self.assertIn("REGISTRY_PASSWORD", build_script)
        self.assertIn("podman", build_script)
        self.assertIn("NO_CACHE", build_script)
        self.assertIn("--no-cache", build_script)
        self.assertIn("singularity exec", build_script)
        self.assertIn("module load mamba/2.3.0", build_script)
        self.assertIn("HEARTBEAT_SECONDS", build_script)
        self.assertIn("run_with_heartbeat", build_script)
        self.assertIn("still running (pid=%s)", build_script)
        self.assertIn('run_with_heartbeat "Podman image build"', build_script)
        self.assertIn('run_with_heartbeat "Singularity image verification"', build_script)
        self.assertIn("mamba list", build_script)
        self.assertIn("--explicit", build_script)
        self.assertIn("printf '@EXPLICIT\\n'", build_script)
        self.assertIn("(cucim|pycudadecon|cudadecon)-", build_script)
        self.assertIn('!= "@EXPLICIT"', build_script)
        self.assertIn("grep -q '/cupy-'", build_script)
        self.assertNotIn("! grep -q '/cucim-'", build_script)
        self.assertIn(
            "grep -Eq '/(cucim|pycudadecon|cudadecon)-'", build_script
        )
        self.assertNotIn(
            "import numpy, scipy, numba, zarr, tifffile, dask, pycudadecon",
            build_script,
        )
        self.assertIn("conda-linux-64.lock", build_script)
        self.assertIn("SOURCE_ENV_PREFIX", build_script)
        self.assertIn("source environment versions verified", build_script)
        self.assertIn("source environment version mismatch", build_script)
        self.assertIn("'tifffile': '2022.10.10'", build_script)
        self.assertNotIn("mamba env create", build_script)
        self.assertNotIn("mamba install", build_script)
        self.assertIn("from cupyx.scipy.signal import fftconvolve", build_script)
        self.assertIn("from cupy.fft import fftn, ifftn", build_script)
        self.assertNotIn("import cucim", build_script)
        self.assertNotIn("cucim.__version__", build_script)
        self.assertIn('numpy.__version__ == \\"1.26.4\\"', build_script)
        self.assertIn("import numpy, scipy, numba, zarr, tifffile, dask, pandas", build_script)

        self.assertIn(IMAGE, check_script)
        self.assertIn("id -un", check_script)
        self.assertIn("SINGULARITY_DOCKER_USERNAME", check_script)
        self.assertIn("singularity inspect", check_script)
        self.assertIn("from cupyx.scipy.signal import fftconvolve", check_script)
        self.assertIn("from cupy.fft import fftn, ifftn", check_script)
        self.assertNotIn("import cucim", check_script)
        self.assertIn('numpy.__version__ == \\"1.26.4\\"', check_script)
        self.assertIn("import numpy, scipy, pandas", check_script)
        self.assertIn("MATLAB_BIND=${MATLAB_BIND:-/home1/apps/MATLAB:/home1/apps/MATLAB}", check_script)
        self.assertIn('--bind "${MATLAB_BIND}"', check_script)
        self.assertIn("command -v matlab", check_script)
        self.assertIn("/home1/apps/MATLAB/R2024a/bin/matlab", check_script)
        self.assertIn("matlab -batch", check_script)


if __name__ == "__main__":
    unittest.main()
