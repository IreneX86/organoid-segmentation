"""The Nuc illumination candidate changes intensity without changing labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patch_dataset import RandomPatchSequence, _augment_spatial_illumination
from tiff_io import ImageLabelPair


class NucSpatialIlluminationTests(unittest.TestCase):
    def test_field_is_bounded_and_reproducible(self) -> None:
        image = np.full((64, 64), 0.5, dtype=np.float32)
        first = _augment_spatial_illumination(image, np.random.default_rng(7), 0.12)
        second = _augment_spatial_illumination(image, np.random.default_rng(7), 0.12)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(first - image))), 0.12 + 1e-6)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    @patch("patch_dataset._load_normalized_pair")
    def test_only_candidate_image_changes(self, load_pair) -> None:
        axis = np.linspace(0.2, 0.8, 96, dtype=np.float32)
        image = np.tile(axis[None, :], (96, 1))
        mask = np.zeros((96, 96), dtype=np.float32)
        mask[20:60, 30:70] = 1
        load_pair.return_value = (image, mask)
        pair = ImageLabelPair(Path("input.tif"), Path("mask.tif"))
        settings = dict(
            pairs=[pair], batch_size=2, patch_size=64, steps_per_epoch=2,
            lower_percentile=1, upper_percentile=99.8,
            positive_patch_probability=0.6, random_seed=42, augment=True,
        )
        baseline = RandomPatchSequence(**settings)
        candidate = RandomPatchSequence(
            **settings, spatial_illumination_probability=1.0,
            spatial_illumination_max_offset=0.12,
        )
        base_images, base_masks = baseline[0]
        changed_images, changed_masks = candidate[0]
        np.testing.assert_array_equal(base_images, baseline[0][0])
        np.testing.assert_array_equal(base_masks, changed_masks)
        self.assertGreater(float(np.max(np.abs(base_images - changed_images))), 0.0)
        np.testing.assert_array_equal(changed_images, candidate[0][0])
        baseline.on_epoch_end()
        candidate.on_epoch_end()
        np.testing.assert_array_equal(baseline[0][1], candidate[0][1])

    def test_invalid_candidate_settings_rejected(self) -> None:
        pair = ImageLabelPair(Path("input.tif"), Path("mask.tif"))
        with self.assertRaisesRegex(ValueError, "positive maximum offset"):
            RandomPatchSequence(
                [pair], batch_size=1, patch_size=64, steps_per_epoch=1,
                lower_percentile=1, upper_percentile=99.8,
                spatial_illumination_probability=0.5,
            )


if __name__ == "__main__":
    unittest.main()
