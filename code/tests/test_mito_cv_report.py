"""Synthetic checks for outer-fold isolation and complete OOF accounting."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mito_cv_report import summarize_cv


class MitoCVReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fold_ids = {f"mask_{index:03d}.tif": index % 5 for index in range(1, 37)}
        for fold in range(5):
            directory = self.root / f"fold_{fold}"
            directory.mkdir()
            split = {
                "target": "mito", "fold": fold, "num_folds": 5,
                "fold_ids": fold_ids,
                "test": [name for name, group in fold_ids.items() if group == fold],
                "validation": [name for name, group in fold_ids.items() if group == (fold + 1) % 5],
                "train": [name for name, group in fold_ids.items() if group not in {fold, (fold + 1) % 5}],
            }
            (directory / "split.json").write_text(json.dumps(split), encoding="utf-8")
            with (directory / "test_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["filename", "dice", "iou", "precision", "recall", "foreground_fraction_true"])
                writer.writeheader()
                for name in split["test"]:
                    writer.writerow({"filename": name, "dice": .8, "iou": 2/3, "precision": .8, "recall": .8,
                                     "foreground_fraction_true": .01 + int(name[5:8]) / 1000})
            with (directory / "threshold_policy_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["filename"] + [f"{policy}_{metric}"
                    for policy in ("validation", "fixed_050") for metric in ("dice", "iou", "precision", "recall")])
                writer.writeheader()
                for name in split["test"]:
                    writer.writerow({"filename": name, **{f"{policy}_{metric}": (.6666666666666666 if metric == "iou" else .8)
                        for policy in ("validation", "fixed_050") for metric in ("dice", "iou", "precision", "recall")}})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_five_fold_report(self) -> None:
        report = summarize_cv(self.root)
        self.assertEqual(report["image_count"], 36)
        self.assertAlmostEqual(report["pooled_per_image"]["dice"]["mean"], .8)
        self.assertEqual(len(report["per_image"]), 36)

    def test_rejects_outer_test_leakage(self) -> None:
        path = self.root / "fold_0" / "split.json"
        split = json.loads(path.read_text(encoding="utf-8"))
        split["train"].append(split["test"][0])
        path.write_text(json.dumps(split), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlap"):
            summarize_cv(self.root)

    def test_rejects_missing_fold_report(self) -> None:
        (self.root / "fold_4" / "test_per_image.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            summarize_cv(self.root)

    def test_rejects_changed_threshold_export(self) -> None:
        path = self.root / "fold_0" / "threshold_policy_per_image.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["validation_dice"] = "0.2"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "disagree"):
            summarize_cv(self.root)


if __name__ == "__main__":
    unittest.main()
