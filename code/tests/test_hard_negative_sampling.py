"""Check that hard-negative centres obey the locked training-only definition."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hard_negative_sampling import hard_negative_centers
from tiff_io import normalize_image


class HardNegativeSamplingTests(unittest.TestCase):
    def test_centres_are_bright_distant_labelled_background(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.full((128, 128), 100, dtype=np.uint16)
            image[90:105, 90:105] = 2000
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[20:35, 20:35] = 255
            image_path = Path(directory) / "input.tif"
            mask_path = Path(directory) / "mask.tif"
            tifffile.imwrite(image_path, image)
            tifffile.imwrite(mask_path, mask)

            centres = hard_negative_centers(
                str(image_path), str(mask_path), 1.0, 99.8
            )
            self.assertGreater(len(centres), 0)
            foreground = mask != 0
            normalized = normalize_image(image)
            residual = normalized - gaussian_filter(normalized, sigma=12.0)
            background = ~foreground
            ys, xs = centres.T
            self.assertTrue(np.all(background[ys, xs]))
            self.assertTrue(np.all(distance_transform_edt(background)[ys, xs] > 10))
            self.assertTrue(np.all(normalized[ys, xs] >= np.percentile(
                normalized[background], 90
            )))
            self.assertTrue(np.all(residual[ys, xs] >= np.percentile(
                residual[background], 90
            )))

    def test_no_foreground_returns_uniform_fallback_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.tif"
            mask_path = Path(directory) / "mask.tif"
            tifffile.imwrite(image_path, np.ones((32, 32), dtype=np.uint16))
            tifffile.imwrite(mask_path, np.zeros((32, 32), dtype=np.uint8))
            self.assertEqual(len(hard_negative_centers(
                str(image_path), str(mask_path), 1.0, 99.8
            )), 0)


if __name__ == "__main__":
    unittest.main()
