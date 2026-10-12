"""Synthetic checks for LD validation-only error diagnostics."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_validation_error_analysis import analyse_case, analyse_validation_runs


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LDValidationErrorAnalysisTests(unittest.TestCase):
    def test_counts_boundary_and_distant_errors_without_instance_claim(self) -> None:
        image = np.arange(256, dtype=np.uint16).reshape(16, 16)
        truth = np.zeros((16, 16), dtype=np.uint8)
        truth[3:5, 3:5] = 1
        probability = np.zeros((16, 16), dtype=np.float32)
        probability[3, 3:5] = 0.9
        probability[4, 3] = 0.9
        probability[3, 5] = 0.9  # False positive one pixel from GT.
        probability[15, 15] = 0.9  # False positive more than ten pixels away.

        result = analyse_case(image, truth, probability, threshold=0.5)
        self.assertEqual((result["tp_pixels"], result["fp_pixels"], result["fn_pixels"]), (3, 2, 1))
        self.assertAlmostEqual(result["dice"], 2 / 3)
        self.assertAlmostEqual(result["precision"], 3 / 5)
        self.assertAlmostEqual(result["recall"], 3 / 4)
        self.assertEqual(result["fp_within_2px_of_gt"], 1)
        self.assertEqual(result["fp_farther_than_10px_from_gt"], 1)
        self.assertEqual(result["fn_within_2px_of_gt_boundary"], 1)
        self.assertEqual(result["small_component_count_proxy"], 1)
        self.assertEqual(result["small_component_detected_half_proxy"], 1)

    def test_empty_case_and_probability_validation(self) -> None:
        image = np.zeros((8, 8), dtype=np.uint16)
        truth = np.zeros_like(image, dtype=np.uint8)
        probability = np.zeros_like(image, dtype=np.float32)
        result = analyse_case(image, truth, probability, threshold=0.5)
        self.assertEqual(result["dice"], 1.0)
        self.assertIsNone(result["small_component_recall_any_proxy"])
        self.assertIsNone(result["fp_intensity_median_normalised"])
        probability[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            analyse_case(image, truth, probability, threshold=0.5)

    def test_uses_only_declared_validation_artifacts_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir, label_dir = root / "input", root / "labels"
            validation = root / "runs" / "ld" / "fold_0" / "validation_artifacts"
            (validation / "probabilities").mkdir(parents=True)
            input_dir.mkdir(); label_dir.mkdir()
            name = "ld_001.tif"
            image = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
            truth = np.zeros((16, 16), dtype=np.uint8); truth[2:6, 2:6] = 1
            probability = truth.astype(np.float32) * 0.8
            tifffile.imwrite(input_dir / name, image)
            tifffile.imwrite(label_dir / name, truth)
            tifffile.imwrite(validation / "probabilities" / name, probability)
            label_manifest = root / "labels.json"
            label_manifest.write_text(json.dumps({
                "rule_id": "ld_antialias_midpoint_v1",
                "label_semantics": {"background": 0, "foreground": 1},
                "files": [{"filename": name,
                           "source_image_sha256": _hash(input_dir / name),
                           "derived_mask_sha256": _hash(label_dir / name)}],
            }))
            (validation.parent / "split.json").write_text(json.dumps({
                "target": "ld", "fold": 0, "validation": [name],
            }))
            (validation / "manifest.json").write_text(json.dumps({
                "target": "ld", "threshold": 0.5, "image_count": 1,
                "files": [{"filename": name,
                           "probability_sha256": _hash(validation / "probabilities" / name)}],
            }))
            # An invalid sentinel in the outer-test directory must have no
            # effect: this tool is incapable of reading that tree.
            outer = validation.parent / "outer_test"
            outer.mkdir()
            (outer / "test_per_image.csv").write_bytes(b"unreadable,outer,test\xff")
            output = root / "analysis"
            report = analyse_validation_runs(root / "runs", input_dir, label_dir,
                                             label_manifest, output, folds=(0,),
                                             expected_images=1)
            self.assertEqual(report["all_validation_images"]["image_count"], 1)
            self.assertTrue((output / "validation_errors_per_image.csv").is_file())
            self.assertTrue((output / "validation_error_summary.json").is_file())
            self.assertGreater((output / "validation_error_montage.png").stat().st_size, 0)
            self.assertIn("Exploratory", report["selection_warning"])
            with self.assertRaises(FileExistsError):
                analyse_validation_runs(root / "runs", input_dir, label_dir,
                                        label_manifest, output, folds=(0,),
                                        expected_images=1)
            tifffile.imwrite(validation / "probabilities" / name,
                             np.zeros_like(probability))
            with self.assertRaisesRegex(ValueError, "probability changed"):
                analyse_validation_runs(root / "runs", input_dir, label_dir,
                                        label_manifest, root / "tampered_analysis",
                                        folds=(0,), expected_images=1)


if __name__ == "__main__":
    unittest.main()
