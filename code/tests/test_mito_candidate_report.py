"""A validation-gated report must account for each frozen outer-test image."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mito_candidate_report import summarize_candidate


class CandidateReportTests(unittest.TestCase):
    def test_all_baseline_choices_reproduce_frozen_dice(self) -> None:
        baseline = Path(__file__).resolve().parents[2] / "baseline/mito_five_fold_20261001"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decisions = []
            for fold in range(5):
                candidate_dir = root / "mito" / f"fold_{fold}"
                candidate_dir.mkdir(parents=True)
                (candidate_dir / "split.json").write_bytes(
                    (baseline / f"fold_{fold}" / "split.json").read_bytes()
                )
                decisions.append({"fold": fold, "selected_candidate": False})
            (root / "candidate_decisions.json").write_text(json.dumps({
                "protocol": "mito_one_factor_validation_gate_v1",
                "variant": "sampling_035", "decisions": decisions,
            }), encoding="utf-8")
            result = summarize_candidate(baseline, root)
        self.assertEqual(result["image_count"], 36)
        self.assertEqual(result["selected_candidate_folds"], 0)
        self.assertAlmostEqual(result["metrics"]["dice"]["mean"], 0.8543, places=4)
        self.assertEqual(result["paired_delta"]["dice"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
