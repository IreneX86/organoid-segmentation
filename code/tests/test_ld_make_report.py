"""No-GPU checks for the reporting-only LD figure and Markdown generator."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_cv_report import _statistics
from ld_make_report import create_report


class LDMakeReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cv = self.root / "cv_report"
        self.cv.mkdir()
        self.rows: list[dict[str, str | int | float]] = []
        for index in range(36):
            score = 0.55 + index * 0.01
            self.rows.append({
                "filename": f"ld_{index + 1:03d}.tif", "fold": index % 5,
                "dice": score, "iou": score / (2 - score),
                "precision": score + 0.02, "recall": score - 0.02,
                "foreground_fraction_true": 0.0001 + index * 0.0001,
                "foreground_fraction_predicted": 0.00015 + index * 0.0001,
                "fixed_050_dice": score - 0.01,
                "fixed_050_iou": (score - 0.01) / (2 - score + 0.01),
                "fixed_050_precision": score + 0.01,
                "fixed_050_recall": score - 0.03,
            })
        metrics = ("dice", "iou", "precision", "recall")
        self.report = {
            "target": "ld", "image_count": 36, "fold_count": 5,
            "settings": {
                "label_manifest_sha256": "l" * 64,
                "fold_map_sha256": "f" * 64,
                "code_sha256": {"code/ld_train.py": "c" * 64},
            },
            "oof": {name: _statistics([float(row[name]) for row in self.rows])
                    for name in (*metrics, "foreground_fraction_true",
                                 "foreground_fraction_predicted")},
            "fixed_050_oof": {
                name: _statistics([float(row[f"fixed_050_{name}"])
                                   for row in self.rows]) for name in metrics
            },
            "folds": [],
        }
        for fold in range(5):
            members = [row for row in self.rows if row["fold"] == fold]
            self.report["folds"].append({
                "fold": fold, "n": len(members), "threshold": 0.45,
                **{name: _statistics([float(row[name]) for row in members])
                   for name in metrics},
            })
        with (self.cv / "oof_per_image.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)
        (self.cv / "oof_summary.json").write_text(json.dumps(self.report))

    def test_creates_figures_and_report_without_raw_pixels(self) -> None:
        destination = self.root / "reports" / "ld_baseline"
        destination.parent.mkdir()
        with patch("ld_make_report.aggregate_cv", return_value=(self.rows, self.report)):
            report_path = create_report(self.root, self.cv, destination)
        self.assertEqual(report_path, (destination / "README.md").resolve())
        self.assertTrue((destination / "oof_per_image.csv").exists())
        for name in ("dice_vs_foreground.svg", "fold_dice_distribution.svg",
                     "threshold_policy_difference.svg"):
            svg = (destination / name).read_text()
            self.assertIn("<svg", svg)
            self.assertIn("<title", svg)
            self.assertIn("</svg>", svg)
            ElementTree.fromstring(svg)
        markdown = report_path.read_text()
        self.assertIn("36 distinct held-out", markdown)
        self.assertIn("must not be used to choose", markdown)
        self.assertIn("0.10%", markdown)
        self.assertIn("0.7250", markdown)  # Mean of synthetic Dice 0.55..0.90.
        with patch("ld_make_report.aggregate_cv", return_value=(self.rows, self.report)):
            with self.assertRaises(FileExistsError):
                create_report(self.root, self.cv, destination)

    def test_refuses_tampered_cv_metrics_before_writing(self) -> None:
        path = self.cv / "oof_per_image.csv"
        text = path.read_text().replace("0.55,", "0.99,", 1)
        path.write_text(text)
        destination = self.root / "reports" / "ld_baseline"
        destination.parent.mkdir()
        with patch("ld_make_report.aggregate_cv", return_value=(self.rows, self.report)):
            with self.assertRaisesRegex(ValueError, "Saved CV dice changed"):
                create_report(self.root, self.cv, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
