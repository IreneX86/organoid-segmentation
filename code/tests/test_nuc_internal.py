"""Nuc split and empty-mask reporting checks without TensorFlow."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuc_train_internal import (
    _save_validation_results,
    choose_nuc_threshold,
    main,
    validate_frozen_split,
)
from tiff_io import ImageLabelPair, pair_tiff_directories


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NucInternalTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_dir = self.root / "Nuc_input"
        self.label_dir = self.root / "Nuc_output"
        self.input_dir.mkdir()
        self.label_dir.mkdir()
        self.names = [f"nuc_{index:03d}.tif" for index in range(1, 5)]
        for index, name in enumerate(self.names):
            image = np.full((32, 32), 100 + index, dtype=np.uint16)
            label = np.zeros((32, 32), dtype=np.uint8)
            if index in (0, 2):
                label[8:16, 8:16] = 255
            tifffile.imwrite(self.input_dir / name, image)
            tifffile.imwrite(self.label_dir / name, label)
        self.pairs = pair_tiff_directories(self.input_dir, self.label_dir)
        self.manifest_path = self.root / "split.json"
        self.manifest = {
            "version": 1,
            "target": "nuc",
            "purpose": "same_batch_internal_diagnostic",
            "train": self.names[:2],
            "validation": self.names[2:],
            "true_negative_filenames": [self.names[1], self.names[3]],
            "input_sha256": {
                name: _sha256(self.input_dir / name) for name in self.names
            },
            "label_sha256": {
                name: _sha256(self.label_dir / name) for name in self.names
            },
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_frozen_split_accepts_exact_sources_and_true_negatives(self) -> None:
        _, training, validation, report = validate_frozen_split(
            self.manifest_path, self.pairs
        )
        self.assertEqual([pair.filename for pair in training], self.names[:2])
        self.assertEqual([pair.filename for pair in validation], self.names[2:])
        self.assertEqual(
            [row["filename"] for row in report if row["foreground_fraction"] == 0],
            [self.names[1], self.names[3]],
        )

    def test_rejects_overlapping_split(self) -> None:
        self.manifest["validation"][0] = self.names[0]
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "uniquely cover"):
            validate_frozen_split(self.manifest_path, self.pairs)

    def test_rejects_changed_input(self) -> None:
        tifffile.imwrite(
            self.input_dir / self.names[0], np.zeros((32, 32), dtype=np.uint16)
        )
        with self.assertRaisesRegex(ValueError, "input source hash mismatch"):
            validate_frozen_split(self.manifest_path, self.pairs)

    def test_rejects_unlisted_empty_mask(self) -> None:
        self.manifest["true_negative_filenames"] = [self.names[1]]
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "true negatives differ"):
            validate_frozen_split(self.manifest_path, self.pairs)

    def test_reports_nonempty_dice_and_empty_false_positives_separately(self) -> None:
        positive_truth = np.zeros((8, 8), dtype=np.uint8)
        positive_truth[2:4, 2:4] = 1
        positive_probability = positive_truth.astype(np.float32) * 0.6
        negative_truth = np.zeros((8, 8), dtype=np.uint8)
        negative_probability = np.zeros((8, 8), dtype=np.float32)
        negative_probability[0, 0] = 0.9
        predictions = [
            (ImageLabelPair(Path(self.names[0]), Path(self.names[0])),
             positive_probability, positive_truth),
            (ImageLabelPair(Path(self.names[1]), Path(self.names[1])),
             negative_probability, negative_truth),
        ]
        (self.root / "results").mkdir()
        summary = _save_validation_results(predictions, 0.75, self.root / "results")
        self.assertEqual(summary["nonempty_validation"]["fixed_050"]["dice_mean"], 1.0)
        self.assertEqual(summary["nonempty_validation"]["selected"]["dice_mean"], 0.0)
        self.assertEqual(
            summary["true_negative_validation"]["fixed_050"]["false_positive_pixels_total"], 1
        )
        self.assertEqual(
            summary["true_negative_validation"]["selected"]["false_positive_pixels_total"], 1
        )
        with (self.root / "results" / "validation_per_image.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[1]["truth_group"], "true_negative")
        self.assertEqual(rows[1]["fixed_050_false_positive_pixels"], "1")
        self.assertEqual(rows[1]["selected_false_positive_pixels"], "1")
        self.assertEqual(rows[1]["fixed_050_dice"], "")

    def test_threshold_tie_prefers_fixed_point_five(self) -> None:
        truth = np.zeros((8, 8), dtype=np.uint8)
        truth[2:4, 2:4] = 1
        probability = truth.astype(np.float32)
        selected, candidates = choose_nuc_threshold([
            (ImageLabelPair(Path(self.names[0]), Path(self.names[0])),
             probability, truth),
        ])
        self.assertEqual(selected, 0.5)
        self.assertEqual(len(candidates), 17)

    def test_threshold_equidistant_tie_prefers_lower(self) -> None:
        truth = np.zeros((8, 8), dtype=np.uint8)
        truth[0, 0] = 1
        truth[0, 1] = 1
        probability = np.zeros((8, 8), dtype=np.float32)
        probability[0, 0] = 0.56  # True positive above 0.55.
        probability[0, 1] = 0.47  # True positive only below 0.50.
        probability[1, 0] = 0.52  # False positive at 0.50, gone by 0.55.
        probability[1, 1] = 0.48  # False positive only below 0.50.
        selected, _ = choose_nuc_threshold([
            (ImageLabelPair(Path(self.names[0]), Path(self.names[0])),
             probability, truth),
        ])
        self.assertEqual(selected, 0.45)

    def test_rejects_output_inside_raw_or_labelled_data(self) -> None:
        for destination in (
            self.root / "label_ref" / "Nuc" / "outputs",
            self.root / "organoid_data" / "outputs",
        ):
            with self.subTest(destination=destination):
                arguments = [
                    "nuc_train_internal.py", "--project-root", str(self.root),
                    "--split-manifest", str(self.manifest_path),
                    "--output-dir", str(destination),
                ]
                with patch.object(sys, "argv", arguments):
                    with self.assertRaisesRegex(ValueError, "outside raw and labelled data"):
                        main()
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
