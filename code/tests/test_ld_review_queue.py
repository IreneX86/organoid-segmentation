"""Small synthetic checks for read-only LD human-review triage."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_review_queue import build_review_queue


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LDReviewQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.predictions = self.root / "predictions"
        self.reference = self.root / "reference.json"
        self.inputs.mkdir(); self.predictions.mkdir()
        (self.predictions / "probabilities").mkdir()
        reference_files = [{"filename": f"ref_{index:03d}.tif", "shape": [32, 32],
                            "foreground_pixels_50pct": 15 + index}
                           for index in range(36)]
        self.reference.write_text(json.dumps({
            "rule_id": "ld_antialias_midpoint_v1", "pair_count": 36,
            "label_semantics": {"background": 0, "foreground": 1},
            "files": reference_files,
        }))
        self.names = ("ld_A.tif", "ld_B.tif", "ld_C.tif")
        records = []
        for index, name in enumerate(self.names):
            image = (np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
                     + index * 200)
            probability = np.full((32, 32), 0.03, dtype=np.float32)
            if index == 0:
                probability[5:17, 5:17] = 0.91  # Unusually large predicted area.
            elif index == 2:
                probability[12:17, 12:17] = 0.52  # Low-confidence boundary.
            mask = (probability >= 0.5).astype(np.uint8) * 255
            tifffile.imwrite(self.inputs / name, image)
            tifffile.imwrite(self.predictions / name, mask)
            tifffile.imwrite(self.predictions / "probabilities" / name, probability)
            records.append({
                "filename": name, "height": 32, "width": 32,
                "input_sha256": _hash(self.inputs / name),
                "mask_sha256": _hash(self.predictions / name),
                "probability_sha256": _hash(self.predictions / "probabilities" / name),
                "foreground_fraction": float((mask != 0).mean()),
            })
        self.manifest = {
            "target": "ld", "status": "complete",
            "semantic_labels_internal": {"background": 0, "foreground": 1},
            "probabilities_saved": True, "threshold": 0.5,
            "foreground_value": 255, "image_count": len(records), "files": records,
        }
        self._save_manifest()

    def _save_manifest(self) -> None:
        (self.predictions / "prediction_manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def test_ranks_uncertainty_and_foreground_outliers_without_label_edits(self) -> None:
        original_hashes = {_name: _hash(self.inputs / _name) for _name in self.names}
        destination = self.root / "review"
        summary = build_review_queue(self.predictions, self.inputs, self.reference,
                                     destination, contact_sheet=True, top_k=2,
                                     crop_size=16)
        self.assertEqual(summary["prediction_images"], 3)
        self.assertIn("no ground truth read", summary["scope"])
        with (destination / "review_queue.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual([int(row["priority_rank"]) for row in rows], [1, 2, 3])
        self.assertEqual({row["filename"] for row in rows}, set(self.names))
        self.assertEqual(next(row for row in rows if row["filename"] == "ld_B.tif")
                         ["foreground_outlier_direction"], "below_p05")
        self.assertGreater(float(next(row for row in rows if row["filename"] == "ld_C.tif")
                                 ["boundary_uncertainty_fraction"]), 0.25)
        from PIL import Image
        with Image.open(destination / "contact_sheet.png") as sheet:
            self.assertEqual(sheet.size, (48, 124))
        self.assertEqual(len(summary["contact_sheet_crops_native_coordinates"]), 2)
        for name in self.names:
            self.assertEqual(_hash(self.inputs / name), original_hashes[name])
        with self.assertRaises(FileExistsError):
            build_review_queue(self.predictions, self.inputs, self.reference,
                               destination)

    def test_tampered_probability_or_missing_saved_probabilities_fail_closed(self) -> None:
        tifffile.imwrite(self.predictions / "probabilities" / self.names[0],
                         np.zeros((32, 32), dtype=np.float32))
        destination = self.root / "review"
        with self.assertRaisesRegex(ValueError, "probability_sha256 differs"):
            build_review_queue(self.predictions, self.inputs, self.reference,
                               destination)
        self.assertFalse(destination.exists())
        self.manifest["probabilities_saved"] = False
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "--save-probabilities"):
            build_review_queue(self.predictions, self.inputs, self.reference,
                               destination)


if __name__ == "__main__":
    unittest.main()
