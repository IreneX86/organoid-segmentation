"""Synthetic, no-TIFF tests for reporting-only paired LD OOF figures."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_component_figures import render_paired_figures
from ld_make_report import FOLD_COLORS


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class LDComponentFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paired_csv = self.root / "paired.csv"
        self.baseline_csv = self.root / "baseline.csv"
        self.audit_csv = self.root / "audit.csv"
        self.policy = self.root / "global_method_policy.json"
        self.destination = self.root / "figures"
        self.policy.write_text(json.dumps({
            "policy_version": "ld_global_validation_method_v1",
            "target": "ld", "selected_method": "baseline",
            "selection_basis": "predeclared_five_fold_inner_validation_only",
            "outer_test_used_for_selection": False, "image_count": 36,
            "fold_gate_records": [
                {"fold": fold, "gate_selected":
                 "candidate" if fold in {1, 3} else "baseline"}
                for fold in range(5)
            ],
        }), encoding="utf-8")
        self.baseline: list[dict[str, object]] = []
        self.paired: list[dict[str, object]] = []
        self.audit: list[dict[str, object]] = []
        for index in range(36):
            name = f"ld_{index + 1:03d}.tif"
            fold = index % 5
            choice = "candidate" if fold in {1, 3} else "baseline"
            base = .55 + .006 * index
            selected = base + (.02 if choice == "candidate" else 0)
            ratio = .00008 + .00004 * index
            baseline_row: dict[str, object] = {
                "filename": name, "fold": fold, "foreground_fraction_true": ratio,
                "dice": base, "iou": base / (2 - base),
                "precision": base + .04, "recall": base - .04,
            }
            row: dict[str, object] = {
                "filename": name, "fold": fold, "selected": choice,
                "foreground_fraction_true": ratio,
            }
            for metric in ("dice", "iou", "precision", "recall"):
                old = float(baseline_row[metric])
                new = selected if metric == "dice" else old + (
                    .01 if choice == "candidate" else 0)
                row[f"baseline_{metric}"] = old
                row[f"selected_{metric}"] = new
                row[f"paired_{metric}_delta"] = new - old
            audit_row = {
                "filename": name, "fold": fold, "fold_gate_selected": choice,
                **{f"baseline_{metric}": row[f"baseline_{metric}"]
                   for metric in ("dice", "iou", "precision", "recall")},
                **{f"selected_{metric}": row[f"selected_{metric}"]
                   for metric in ("dice", "iou", "precision", "recall")},
                "baseline_fp_pixels": 20, "selected_fp_pixels": 18,
                "paired_fp_pixels_delta": -2,
                "baseline_fn_pixels": 30, "selected_fn_pixels": 33,
                "paired_fn_pixels_delta": 3,
            }
            self.baseline.append(baseline_row)
            self.paired.append(row)
            self.audit.append(audit_row)
        _write_csv(self.baseline_csv, self.baseline)
        _write_csv(self.paired_csv, self.paired)
        _write_csv(self.audit_csv, self.audit)

    def test_renders_deterministic_figures_without_tiff(self) -> None:
        files = render_paired_figures(self.paired_csv, self.baseline_csv,
                                      self.policy, self.destination, self.audit_csv)
        self.assertEqual({path.name for path in files}, {
            "paired_dice_change.svg", "fold_mean_dice.svg",
            "dice_vs_foreground_paired.svg", "paired_fp_fn_change.svg",
        })
        rendered = {path.name: path.read_text(encoding="utf-8") for path in files}
        for svg in rendered.values():
            ElementTree.fromstring(svg)
            self.assertIn("<title", svg)
            self.assertNotIn(".tif</image>", svg)
        self.assertIn("descriptive", rendered["paired_dice_change.svg"])
        self.assertIn("ld_001.tif", rendered["paired_dice_change.svg"])
        fold_svg = rendered["fold_mean_dice.svg"]
        self.assertIn("Fold-gated (fold colors)", fold_svg)
        for fold, color in enumerate(FOLD_COLORS):
            self.assertIn(f'<rect x="{790 + 4 * fold}" y="129" width="4" '
                          f'height="12" fill="{color}"/>', fold_svg)
        with self.assertRaises(FileExistsError):
            render_paired_figures(self.paired_csv, self.baseline_csv,
                                  self.policy, self.destination, self.audit_csv)

    def test_rejects_unlocked_policy_before_creating_output(self) -> None:
        self.policy.write_text(json.dumps({"selected_method": "candidate"}))
        with self.assertRaisesRegex(ValueError, "method policy"):
            render_paired_figures(self.paired_csv, self.baseline_csv,
                                  self.policy, self.destination)
        self.assertFalse(self.destination.exists())

    def test_rejects_changed_baseline_score_or_delta(self) -> None:
        self.paired[0]["baseline_dice"] = .99
        _write_csv(self.paired_csv, self.paired)
        with self.assertRaisesRegex(ValueError, "Paired baseline_dice"):
            render_paired_figures(self.paired_csv, self.baseline_csv,
                                  self.policy, self.destination)
        self.assertFalse(self.destination.exists())
        self.paired[0]["baseline_dice"] = self.baseline[0]["dice"]
        self.paired[0]["paired_dice_delta"] = .5
        _write_csv(self.paired_csv, self.paired)
        with self.assertRaisesRegex(ValueError, "dice delta"):
            render_paired_figures(self.paired_csv, self.baseline_csv,
                                  self.policy, self.destination)

    def test_rejects_paired_row_that_breaks_locked_fold_gate(self) -> None:
        self.paired[0]["selected"] = "candidate"
        _write_csv(self.paired_csv, self.paired)
        with self.assertRaisesRegex(ValueError, "locked fold gate"):
            render_paired_figures(self.paired_csv, self.baseline_csv,
                                  self.policy, self.destination)
        self.assertFalse(self.destination.exists())

    def test_rejects_audit_not_matching_paired_scores(self) -> None:
        self.audit[4]["selected_dice"] = .9
        _write_csv(self.audit_csv, self.audit)
        with self.assertRaisesRegex(ValueError, "audit selected_dice"):
            render_paired_figures(self.paired_csv, self.baseline_csv,
                                  self.policy, self.destination, self.audit_csv)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
