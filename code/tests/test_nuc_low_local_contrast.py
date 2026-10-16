"""The optional Nuc contrast transform changes training images only."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuc_train_internal import parse_args, validate_low_contrast_settings
from patch_dataset import (
    RandomPatchSequence,
    _augment_low_local_contrast,
    gaussian_blur_reflect,
)
from tiff_io import ImageLabelPair


def _spatial_reflect_reference(image: np.ndarray, sigma: float) -> np.ndarray:
    """Slow separable Gaussian with explicit symmetric boundary padding."""

    radius = math.ceil(4 * sigma)
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (positions / sigma) ** 2)
    weights /= weights.sum()
    height, width = image.shape
    padded_y = np.pad(image, ((radius, radius), (0, 0)), mode="symmetric")
    vertical = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        vertical[row] = np.sum(
            padded_y[row : row + len(weights)] * weights[:, None], axis=0
        )
    padded_x = np.pad(vertical, ((0, 0), (radius, radius)), mode="symmetric")
    result = np.empty_like(vertical)
    for column in range(width):
        result[:, column] = np.sum(
            padded_x[:, column : column + len(weights)] * weights[None, :], axis=1
        )
    return result.astype(np.float32)


class NucLowLocalContrastTests(unittest.TestCase):
    def test_fft_gaussian_matches_spatial_reflect_reference(self) -> None:
        image = np.random.default_rng(10).random((17, 19), dtype=np.float32)
        for sigma in (2.0, 64.0):
            with self.subTest(sigma=sigma):
                actual = gaussian_blur_reflect(image, sigma)
                reference = _spatial_reflect_reference(image, sigma)
                np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=0)
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(actual.shape, image.shape)
                self.assertAlmostEqual(float(actual.mean()), float(image.mean()), places=6)

    def test_transform_uses_locked_equation_and_seed(self) -> None:
        image = np.random.default_rng(11).random((25, 27), dtype=np.float32)
        sigma = 5.0
        alpha = float(np.random.default_rng(23).uniform(0.5, 1.0))
        low_pass = _spatial_reflect_reference(image, sigma)
        expected = np.clip(low_pass + alpha * (image - low_pass), 0, 1)
        actual = _augment_low_local_contrast(
            image, np.random.default_rng(23), sigma, 0.5
        )
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
        np.testing.assert_array_equal(
            actual, _augment_low_local_contrast(
                image, np.random.default_rng(23), sigma, 0.5
            ),
        )
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0)
        self.assertLessEqual(float(actual.max()), 1)

    @patch("patch_dataset._load_normalized_pair")
    def test_mask_and_baseline_draws_remain_paired(self, load_pair) -> None:
        axis = np.linspace(0.1, 0.9, 96, dtype=np.float32)
        image = np.tile(axis[None, :], (96, 1))
        mask = np.zeros((96, 96), dtype=np.float32)
        mask[20:70, 25:75] = 1
        load_pair.return_value = (image, mask)
        pair = ImageLabelPair(Path("input.tif"), Path("mask.tif"))
        settings = dict(
            pairs=[pair], batch_size=3, patch_size=64, steps_per_epoch=2,
            lower_percentile=1, upper_percentile=99.8,
            positive_patch_probability=0.6, random_seed=42, augment=True,
        )
        baseline = RandomPatchSequence(**settings)
        candidate = RandomPatchSequence(
            **settings, low_contrast_probability=1.0,
            low_contrast_sigma=64.0, low_contrast_alpha_min=0.5,
        )
        for _ in range(2):
            for batch_index in range(2):
                baseline_images, baseline_masks = baseline[batch_index]
                candidate_images, candidate_masks = candidate[batch_index]
                np.testing.assert_array_equal(baseline_masks, candidate_masks)
                np.testing.assert_array_equal(
                    baseline_images, baseline[batch_index][0]
                )
                np.testing.assert_array_equal(
                    candidate_images, candidate[batch_index][0]
                )
                self.assertGreater(
                    float(np.max(np.abs(baseline_images - candidate_images))), 0
                )
            baseline.on_epoch_end()
            candidate.on_epoch_end()

    @patch("patch_dataset.gaussian_blur_reflect", side_effect=AssertionError("blur called"))
    @patch("patch_dataset._load_normalized_pair")
    def test_default_off_never_calls_blur(self, load_pair, blur) -> None:
        load_pair.return_value = (
            np.full((64, 64), 0.5, dtype=np.float32),
            np.zeros((64, 64), dtype=np.float32),
        )
        sequence = RandomPatchSequence(
            [ImageLabelPair(Path("input.tif"), Path("mask.tif"))],
            batch_size=1, patch_size=64, steps_per_epoch=1,
            lower_percentile=1, upper_percentile=99.8,
        )
        sequence[0]
        blur.assert_not_called()

    def test_nuc_entry_rejects_unlocked_enabled_settings(self) -> None:
        validate_low_contrast_settings(0.0, 64.0, 0.5)
        validate_low_contrast_settings(0.5, 64.0, 0.5)
        for settings in ((0.4, 64, 0.5), (0.5, 32, 0.5), (0.5, 64, 0.4)):
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, "locked"):
                    validate_low_contrast_settings(*settings)
        with self.assertRaisesRegex(ValueError, "allowed ranges"):
            validate_low_contrast_settings(0.5, float("nan"), 0.5)

    def test_cli_accepts_frozen_candidate_flags(self) -> None:
        with patch("sys.argv", [
            "nuc_train_internal.py", "--project-root", "/tmp/project",
            "--split-manifest", "/tmp/split.json", "--output-dir", "/tmp/run",
            "--low-contrast-probability", "0.5", "--low-contrast-sigma", "64",
            "--low-contrast-alpha-min", "0.5",
        ]):
            args = parse_args()
        self.assertEqual(
            (args.low_contrast_probability, args.low_contrast_sigma,
             args.low_contrast_alpha_min), (0.5, 64.0, 0.5),
        )

    @patch("patch_dataset._load_normalized_pair")
    def test_half_probability_is_deterministic_per_patch(self, load_pair) -> None:
        axis = np.linspace(0.1, 0.9, 96, dtype=np.float32)
        load_pair.return_value = (
            np.tile(axis[None, :], (96, 1)),
            np.ones((96, 96), dtype=np.float32),
        )
        settings = dict(
            pairs=[ImageLabelPair(Path("input.tif"), Path("mask.tif"))],
            batch_size=8, patch_size=64, steps_per_epoch=1,
            lower_percentile=1, upper_percentile=99.8,
            random_seed=42, augment=True,
        )
        baseline = RandomPatchSequence(**settings)
        candidate = RandomPatchSequence(
            **settings, low_contrast_probability=0.5,
        )
        baseline_images, baseline_masks = baseline[0]
        candidate_images, candidate_masks = candidate[0]
        np.testing.assert_array_equal(baseline_masks, candidate_masks)
        decisions = []
        for item_index in range(8):
            rng = np.random.default_rng(np.random.SeedSequence([
                42, 0, 0, item_index, 0x4E55434C,
            ]))
            expected_change = rng.random() < 0.5
            changed = not np.array_equal(
                baseline_images[item_index], candidate_images[item_index]
            )
            self.assertEqual(changed, expected_change)
            decisions.append(changed)
        self.assertTrue(any(decisions))
        self.assertTrue(any(not changed for changed in decisions))


if __name__ == "__main__":
    unittest.main()
