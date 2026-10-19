"""TensorFlow-free checks for the reviewed nucleus-mask postprocessing policy."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuc_postprocess import postprocess_nucleus_masks


CONNECT8 = np.ones((3, 3), dtype=bool)
REPO = Path(__file__).resolve().parents[2]


class NucPostprocessTests(unittest.TestCase):
    def test_empty_mask_is_a_valid_true_negative(self) -> None:
        empty = np.zeros((37, 51), dtype=np.uint8)
        result, report = postprocess_nucleus_masks(empty, empty)
        self.assertEqual(result.shape, empty.shape)
        self.assertEqual(result.dtype, np.bool_)
        self.assertFalse(result.any())
        self.assertEqual(report["raw_components"], 0)
        self.assertEqual(report["components_after_split"], 0)
        self.assertEqual(report["accepted_splits"], [])
        json.dumps(report)  # The CLI may save this report directly.

    def test_rejects_grayscale_or_unpaired_high_core(self) -> None:
        low = np.zeros((32, 49), dtype=np.uint8)
        high = low.copy()
        low[2, 3] = 128
        with self.assertRaisesRegex(ValueError, "binary values|only 0/1"):
            postprocess_nucleus_masks(low, high)
        low[2, 3] = 0
        high[2, 3] = 255
        with self.assertRaisesRegex(ValueError, "subset"):
            postprocess_nucleus_masks(low, high)
        with self.assertRaisesRegex(ValueError, "identical pixel grids"):
            postprocess_nucleus_masks(low, np.zeros((31, 49), dtype=bool))

    def test_small_component_removed_and_hole_filled(self) -> None:
        low = np.zeros((96, 128), dtype=bool)
        low[15:55, 12:52] = True
        low[29:38, 26:35] = False
        low[70:78, 70:78] = True  # 64 pixels is below the raw 500 px cutoff.
        result, report = postprocess_nucleus_masks(low, np.zeros_like(low))
        self.assertFalse(result[72, 72])
        self.assertTrue(result[33, 30])
        self.assertEqual(report["raw_components"], 2)
        self.assertEqual(report["retained_components"], 1)
        self.assertEqual(report["components_after_split"], 1)
        self.assertFalse(np.any(ndimage.binary_fill_holes(result) & ~result))

    def test_closing_does_not_join_two_independent_nuclei(self) -> None:
        low = np.zeros((100, 120), dtype=bool)
        low[20:65, 15:45] = True
        low[20:65, 49:79] = True
        result, report = postprocess_nucleus_masks(low, np.zeros_like(low))
        _, count = ndimage.label(result, CONNECT8)
        self.assertEqual(count, 2)
        self.assertEqual(report["components_before_split"], 2)
        self.assertEqual(report["components_after_split"], 2)
        self.assertTrue(result[35, 30])
        self.assertTrue(result[35, 64])

    def test_two_high_cores_split_one_component_at_image_edge(self) -> None:
        low = np.zeros((120, 180), dtype=bool)
        low[15:100, 0:160] = True  # Touches the left image boundary.
        high = np.zeros_like(low)
        high[30:80, 0:25] = True
        high[30:80, 115:145] = True
        result, report = postprocess_nucleus_masks(low, high)
        labels, count = ndimage.label(result, CONNECT8)
        self.assertEqual(count, 2)
        self.assertEqual(len(report["accepted_splits"]), 1)
        self.assertFalse(np.any(result & ~low))
        self.assertFalse(np.any(high & ~result))
        self.assertNotEqual(labels[50, 10], labels[50, 130])
        self.assertTrue(np.all(np.bincount(labels.ravel())[1:] >= 500))
        self.assertFalse(np.any(ndimage.binary_fill_holes(result) & ~result))

    def test_three_strong_cores_are_ambiguous_and_unchanged(self) -> None:
        low = np.zeros((110, 240), dtype=bool)
        low[15:95, 5:225] = True
        high = np.zeros_like(low)
        for x0 in (20, 90, 160):
            high[30:70, x0 : x0 + 25] = True
        result, report = postprocess_nucleus_masks(low, high)
        self.assertEqual(report["accepted_splits"], [])
        self.assertEqual(len(report["ambiguous_multiple_cores"]), 1)
        self.assertEqual(report["components_after_split"], 1)
        # The pre-split median may round outer corners, but an ambiguous
        # multi-core component must receive no additional split-side edits.
        cleaned_without_cores, _ = postprocess_nucleus_masks(low, np.zeros_like(low))
        np.testing.assert_array_equal(result, cleaned_without_cores)

    def test_invalid_split_is_skipped_without_changing_cleaned_component(self) -> None:
        low = np.zeros((120, 180), dtype=bool)
        low[15:100, 0:160] = True
        high = np.zeros_like(low)
        high[30:80, 0:25] = True
        high[30:80, 115:145] = True
        cleaned_without_cores, _ = postprocess_nucleus_masks(low, np.zeros_like(low))
        with patch("nuc_postprocess._split_pair", side_effect=ValueError("unsafe boundary")):
            result, report = postprocess_nucleus_masks(low, high)
        np.testing.assert_array_equal(result, cleaned_without_cores)
        self.assertEqual(report["accepted_splits"], [])
        self.assertEqual(len(report["skipped_two_core_candidates"]), 1)
        self.assertIn("unsafe boundary", report["skipped_two_core_candidates"][0]["reason"])

    def test_regression_against_reviewed_ten_image_preview(self) -> None:
        root = REPO / "runpod_results/nuc_threshold003_preview_20261019"
        high_root = REPO / "runpod_results/nuc_fluo_preview_20261016/output10"
        manifest_path = root / "curved_separated_10_manifest.json"
        if not manifest_path.is_file():
            self.skipTest("Ignored local review archive is unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest["files"]) != 10:
            self.fail("The reviewed golden manifest must list exactly ten images")
        for index, row in enumerate(manifest["files"], start=1):
            name = f"fluo_{index:03d}.tif"
            self.assertEqual(row["file"], name)
            low_path = root / ("raw_0p03" if index <= 2 else "raw_0p03_more8") / name
            high_path = high_root / name
            reviewed_path = root / "min500_close4_smooth_curved_separated_10" / name
            if not all(path.is_file() for path in (low_path, high_path, reviewed_path)):
                self.skipTest("Ignored local source and preview TIFFs are unavailable")
            with self.subTest(filename=name):
                result, report = postprocess_nucleus_masks(
                    tifffile.imread(low_path), tifffile.imread(high_path)
                )
                np.testing.assert_array_equal(result, tifffile.imread(reviewed_path) > 0)
                self.assertEqual(
                    report["components_after_split"], row["components_after"]
                )
                self.assertEqual(
                    len(report["accepted_splits"]), len(row["accepted_splits"])
                )


if __name__ == "__main__":
    unittest.main()
