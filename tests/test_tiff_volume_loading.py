import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import tifffile
except ModuleNotFoundError as exc:
    np = None
    tifffile = None
    MISSING_DEPENDENCY = exc.name
else:
    MISSING_DEPENDENCY = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "workflow/scripts"


class TiffVolumeLoadingTests(unittest.TestCase):
    def test_single_page_tiff_loads_as_one_slice_volume(self):
        if MISSING_DEPENDENCY:
            self.skipTest(f"missing test dependency: {MISSING_DEPENDENCY}")

        import sys

        sys.path.insert(0, str(SCRIPTS))
        try:
            try:
                from psf_estimation import open_tiff_memmap
            except ModuleNotFoundError as exc:
                self.skipTest(f"missing decon dependency: {exc.name}")
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single_page.tif"
            image = np.arange(12, dtype=np.uint16).reshape(3, 4)
            tifffile.imwrite(path, image)

            volume = open_tiff_memmap(path)

            self.assertEqual(volume.shape, (1, 3, 4))
            np.testing.assert_array_equal(np.asarray(volume[0]), image)


if __name__ == "__main__":
    unittest.main()
