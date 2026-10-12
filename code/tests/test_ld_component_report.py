"""Synthetic reporting checks for the predeclared LD fold-local gate."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_component_report import _candidate_test_rows, aggregate_gated_candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GatedCandidateReportTests(unittest.TestCase):
    def test_candidate_outer_tiffs_are_hashed_and_match_probability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "fold_0"
            final = run / "outer_test"
            name = "ld_001.tif"
            final.mkdir(parents=True)
            (run / "selection.lock.json").write_text("{}")
            (run / "decision.lock.json").write_text("{}")
            (run / "threshold.json").write_text(json.dumps({"threshold": 0.6}))
            probability = np.asarray([[0.1, 0.8], [0.6, 0.2]], dtype=np.float32)
            paths = {
                "probability_sha256": final / "probabilities" / name,
                "validation_mask_sha256": final / "predictions" / "validation" / name,
                "fixed_050_mask_sha256": final / "predictions" / "fixed_050" / name,
            }
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(paths["probability_sha256"], probability)
            for policy, cutoff in (("validation_mask_sha256", 0.6),
                                   ("fixed_050_mask_sha256", 0.5)):
                tifffile.imwrite(
                    paths[policy], (probability >= cutoff).astype(np.uint8) * 255
                )
            row = {
                "filename": name, "dice": 0.5, "iou": 1 / 3,
                "precision": 0.5, "recall": 0.5,
                "foreground_fraction_true": 0.5,
                "foreground_fraction_predicted": 0.5,
            }
            metrics = final / "test_per_image.csv"
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            summary = final / "test_summary.json"
            summary.write_text(json.dumps({"image_count": 1}))
            policy = final / "threshold_policy_per_image.csv"
            policy_row = {"filename": name}
            for metric in ("dice", "iou", "precision", "recall"):
                policy_row[f"validation_{metric}"] = row[metric]
                policy_row[f"fixed_050_{metric}"] = row[metric]
            with policy.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(policy_row))
                writer.writeheader()
                writer.writerow(policy_row)
            manifest = {
                "candidate": "component_uniform_positive_center_v1",
                "fold": 0,
                "test_filenames": [name],
                "validation_threshold": 0.6,
                "selection_lock_sha256": _sha256(run / "selection.lock.json"),
                "decision_lock_sha256": _sha256(run / "decision.lock.json"),
                "files": [{
                    "filename": name, "shape": [2, 2],
                    **{key: _sha256(path) for key, path in paths.items()},
                }],
                "test_per_image_sha256": _sha256(metrics),
                "test_summary_sha256": _sha256(summary),
                "threshold_policy_per_image_sha256": _sha256(policy),
            }
            (final / "evaluation_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(_candidate_test_rows(run, 0, {name})[0]["dice"], 0.5)
            tifffile.imwrite(paths["probability_sha256"], np.zeros((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "probability TIFF changed"):
                _candidate_test_rows(run, 0, {name})

    def test_uses_candidate_only_in_validation_selected_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_root = root / "baseline"
            candidate_root = root / "candidate"
            baseline_rows = []
            decisions = []
            for fold in range(5):
                directory = candidate_root / "ld" / f"fold_{fold}"
                directory.mkdir(parents=True)
                decision = {
                    "fold": fold,
                    "selected": "candidate" if fold == 0 else "baseline",
                    "mean_paired_dice_delta_050": 0.02 if fold == 0 else 0.0,
                }
                (directory / "decision.lock.json").write_text(json.dumps(decision))
                decisions.append(decision)
            for index in range(36):
                row = {"fold": index % 5, "filename": f"ld_{index:03d}.tif",
                       "foreground_fraction_true": 0.002,
                       "foreground_fraction_predicted": 0.002}
                for metric in ("dice", "iou", "precision", "recall"):
                    row[metric] = 0.5
                    row[f"fixed_050_{metric}"] = 0.5
                baseline_rows.append(row)

            def synthetic_test_rows(_run: Path, fold: int, names: set[str]) -> list[dict]:
                self.assertEqual(fold, 0)
                self.assertTrue(names)
                output = []
                for name in names:
                    row = {"filename": name, "foreground_fraction_true": 0.002,
                           "foreground_fraction_predicted": 0.002}
                    for metric in ("dice", "iou", "precision", "recall"):
                        row[metric] = 0.6
                        row[f"fixed_050_{metric}"] = 0.6
                    output.append(row)
                return output

            def synthetic_gate(_base: Path, candidate: Path, *_: object) -> dict:
                fold = int(candidate.name.removeprefix("fold_"))
                return decisions[fold]

            with (mock.patch("ld_component_report.compute_gate",
                             side_effect=synthetic_gate),
                  mock.patch("ld_component_report.aggregate_cv",
                             return_value=(baseline_rows, {"oof": {"dice": {"mean": 0.5}}})),
                  mock.patch("ld_component_report._candidate_test_rows",
                             side_effect=synthetic_test_rows)):
                rows, report = aggregate_gated_candidate(
                    baseline_root, candidate_root,
                    root / "fold_map.json", root / "manifest.json", root / "labels",
                )
            self.assertEqual(len(rows), 36)
            self.assertEqual(report["candidate_fold_count"], 1)
            for row in rows:
                expected = 0.6 if row["fold"] == 0 else 0.5
                self.assertAlmostEqual(row["selected_dice"], expected)
                self.assertEqual(row["selected"],
                                 "candidate" if row["fold"] == 0 else "baseline")


if __name__ == "__main__":
    unittest.main()
