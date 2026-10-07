"""Verify a locked OOF summary cannot silently change rejected folds."""

from __future__ import annotations

import csv
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mito_postdecision_summary import render_aggregate_svg, summarize_locked  # noqa: E402


class LockedPostDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline.csv"
        self.decisions = self.root / "decisions.json"
        self.candidate = self.root / "selected_oof.json"
        baseline_rows = []
        chosen_rows = []
        for index in range(36):
            name = f"case_{index:02d}.tif"
            fold = index % 5
            selected = fold == 3
            fraction = 0.04 if index < 7 else 0.08
            baseline = {
                "filename": name, "fold": fold,
                "foreground_fraction_true": fraction,
                "dice": 0.8 + index * 0.001,
                "iou": 0.67 + index * 0.001,
                "precision": 0.78 + index * 0.001,
                "recall": 0.85 + index * 0.001,
            }
            baseline_rows.append(baseline)
            chosen = {
                "filename": name, "fold": fold,
                "selected_candidate": selected,
                "foreground_fraction_true": fraction,
            }
            for metric, change in (
                ("dice", 0.01), ("iou", 0.008),
                ("precision", 0.02), ("recall", -0.005),
            ):
                chosen[f"baseline_{metric}"] = baseline[metric]
                chosen[metric] = baseline[metric] + (change if selected else 0.0)
                chosen[f"delta_{metric}"] = chosen[metric] - baseline[metric]
            chosen_rows.append(chosen)
        with self.baseline.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
            writer.writeheader()
            writer.writerows(baseline_rows)
        self.decisions.write_text(json.dumps({
            "protocol": "mito_focal_dice_validation_gate_v1",
            "variant": "focal_dice",
            "decisions": [{"fold": fold, "selected_candidate": fold == 3}
                          for fold in range(5)],
        }))
        self.payload = {
            "protocol": "mito_validation_gated_candidate_oof_v1",
            "variant": "focal_dice",
            "image_count": 36,
            "selected_candidate_folds": 1,
            "per_image": chosen_rows,
            "folds": [{
                "fold": fold,
                "image_count": sum(row["fold"] == fold for row in chosen_rows),
                "selected_candidate": fold == 3,
                "dice": {"mean": statistics.mean(
                    row["dice"] for row in chosen_rows if row["fold"] == fold
                )},
            } for fold in range(5)],
            "metrics": {metric: {"mean": statistics.mean(
                row[metric] for row in chosen_rows
            )} for metric in ("dice", "iou", "precision", "recall")},
            "paired_delta": {metric: {"mean": statistics.mean(
                row[f"delta_{metric}"] for row in chosen_rows
            )} for metric in ("dice", "iou", "precision", "recall")},
        }
        self._write_candidate()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_candidate(self) -> None:
        self.candidate.write_text(json.dumps(self.payload), encoding="utf-8")

    def _summarize(self) -> dict[str, object]:
        return summarize_locked(
            self.decisions, self.candidate, self.baseline, "focal_dice"
        )

    def test_paired_aggregate_and_safe_svg(self) -> None:
        report = self._summarize()
        self.assertEqual(report["selected_candidate_folds"], 1)
        self.assertAlmostEqual(
            report["overall"]["metrics"]["dice"]["paired_delta"]["mean"],
            7 * 0.01 / 36,
        )
        self.assertAlmostEqual(
            report["low_foreground_le_5pct"]["metrics"]["dice"]["paired_delta"]["mean"],
            0.01 / 7,
        )
        self.assertAlmostEqual(
            report["overall"]["metrics"]["dice"]["selected_policy"]["min"], 0.8
        )
        self.assertEqual(
            report["precision_recall_tradeoff"]["precision_up_recall_down_image_count"], 7
        )
        self.assertEqual(report["fold_consistency"]["improved_mean_dice"], 1)
        svg = render_aggregate_svg(report)
        self.assertIn("<svg", svg)
        self.assertNotIn("case_", svg)
        self.assertNotIn(".tif", svg)
        self.assertEqual(ElementTree.fromstring(svg).tag,
                         "{http://www.w3.org/2000/svg}svg")

    def test_rejects_unselected_outer_score_instead_of_baseline_fallback(self) -> None:
        self.payload["per_image"][0]["dice"] += 0.001
        self.payload["per_image"][0]["delta_dice"] += 0.001
        self._write_candidate()
        with self.assertRaisesRegex(ValueError, "unselected dice"):
            self._summarize()

    def test_rejects_conflicting_fold_decision(self) -> None:
        self.payload["per_image"][0]["selected_candidate"] = True
        self._write_candidate()
        with self.assertRaisesRegex(ValueError, "Outer-fold/decision mismatch"):
            self._summarize()


if __name__ == "__main__":
    unittest.main()
