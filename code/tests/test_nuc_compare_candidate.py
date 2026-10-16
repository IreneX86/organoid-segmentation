"""Paired Nuc comparison checks using synthetic metric CSVs only."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuc_compare_candidate import compare_validation_csv, main


COLUMNS = ["filename", "truth_group"]
for _policy in ("fixed_050", "selected"):
    COLUMNS.extend([
        f"{_policy}_false_positive_pixels",
        f"{_policy}_dice",
        f"{_policy}_iou",
        f"{_policy}_precision",
        f"{_policy}_recall",
    ])


def metric_row(
    name: str,
    group: str,
    fixed: float | None,
    selected: float | None,
    fixed_fp: int,
    selected_fp: int,
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "filename": name,
        "truth_group": group,
        "fixed_050_false_positive_pixels": fixed_fp,
        "selected_false_positive_pixels": selected_fp,
    }
    for policy, score in (("fixed_050", fixed), ("selected", selected)):
        for metric in ("dice", "iou", "precision", "recall"):
            row[f"{policy}_{metric}"] = "" if score is None else score
    return row


class NucCandidateComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline_path = self.root / "baseline.csv"
        self.candidate_path = self.root / "candidate.csv"
        self.baseline = [
            metric_row("nuc_001.tif", "nonempty", 0.8, 0.82, 4, 5),
            metric_row("nuc_002.tif", "nonempty", 0.6, 0.62, 7, 8),
            metric_row("nuc_005.tif", "true_negative", None, None, 5, 6),
        ]
        self.candidate = [
            metric_row("nuc_005.tif", "true_negative", None, None, 3, 9),
            metric_row("nuc_002.tif", "nonempty", 0.7, 0.72, 5, 6),
            metric_row("nuc_001.tif", "nonempty", 0.9, 0.91, 2, 3),
        ]
        self.write_rows()

    def write_rows(self) -> None:
        for path, rows in (
            (self.baseline_path, self.baseline),
            (self.candidate_path, self.candidate),
        ):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

    def compare(self) -> tuple[dict[str, object], list[dict[str, object]]]:
        return compare_validation_csv(
            self.baseline_path, self.candidate_path,
            expected_nonempty=2, expected_true_negative=1,
        )

    def test_pairs_by_filename_and_keeps_empty_truth_out_of_dice(self) -> None:
        summary, rows = self.compare()
        self.assertEqual([row["filename"] for row in rows], [
            "nuc_001.tif", "nuc_002.tif", "nuc_005.tif",
        ])
        self.assertEqual(summary["nonempty_image_count"], 2)
        self.assertEqual(summary["true_negative_image_count"], 1)
        fixed = summary["policies"]["fixed_050"]
        self.assertAlmostEqual(fixed["nonempty"]["dice"]["baseline_mean"], 0.7)
        self.assertAlmostEqual(fixed["nonempty"]["dice"]["candidate_mean"], 0.8)
        self.assertAlmostEqual(fixed["nonempty"]["dice"]["paired_delta_mean"], 0.1)
        self.assertEqual(fixed["nonempty"]["dice"]["images_improved"], 2)
        self.assertEqual(fixed["true_negative"]["baseline_false_positive_pixels_total"], 5)
        self.assertEqual(fixed["true_negative"]["candidate_false_positive_pixels_total"], 3)
        self.assertEqual(fixed["true_negative"]["paired_false_positive_pixels_delta_total"], -2)
        selected = summary["policies"]["selected"]
        self.assertAlmostEqual(selected["nonempty"]["dice"]["paired_delta_mean"], 0.095)
        self.assertEqual(selected["true_negative"]["paired_false_positive_pixels_delta_total"], 3)
        self.assertEqual(rows[2]["fixed_050_paired_dice_delta"], "")
        self.assertEqual(rows[2]["fixed_050_paired_false_positive_pixels_delta"], -2)

    def test_rejects_different_names_and_truth_group(self) -> None:
        self.candidate[0]["filename"] = "nuc_999.tif"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "filenames differ"):
            self.compare()
        self.candidate[0]["filename"] = "nuc_005.tif"
        self.candidate[0]["truth_group"] = "nonempty"
        self.candidate[0]["fixed_050_dice"] = 0.5
        self.candidate[0]["fixed_050_iou"] = 0.5
        self.candidate[0]["fixed_050_precision"] = 0.5
        self.candidate[0]["fixed_050_recall"] = 0.5
        self.candidate[0]["selected_dice"] = 0.5
        self.candidate[0]["selected_iou"] = 0.5
        self.candidate[0]["selected_precision"] = 0.5
        self.candidate[0]["selected_recall"] = 0.5
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "truth_group differ"):
            self.compare()

    def test_rejects_duplicate_name_and_invalid_metrics(self) -> None:
        self.candidate.append(self.candidate[1].copy())
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "duplicate filename"):
            self.compare()
        self.candidate.pop()
        self.candidate[1]["fixed_050_dice"] = "NaN"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Out-of-range dice"):
            self.compare()

    def test_rejects_numeric_metric_on_true_negative(self) -> None:
        self.candidate[0]["fixed_050_dice"] = 1.0
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "must have blank dice"):
            self.compare()

    def test_cli_writes_json_and_csv_once(self) -> None:
        output_json = self.root / "out" / "comparison.json"
        output_csv = self.root / "out" / "per_image.csv"
        arguments = [
            "nuc_compare_candidate.py",
            "--baseline-csv", str(self.baseline_path),
            "--candidate-csv", str(self.candidate_path),
            "--output-json", str(output_json),
            "--output-csv", str(output_csv),
            "--expected-nonempty", "2",
            "--expected-true-negative", "1",
        ]
        with patch.object(sys, "argv", arguments):
            main()
        summary = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(summary["nonempty_image_count"], 2)
        with output_csv.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)
        with patch.object(sys, "argv", arguments):
            with self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
