"""The sampling gate must use validation scores and reject early test scoring."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mito_sampling_gate import select_folds


class MitoSamplingGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        for fold in range(5):
            base = self.baseline / f"fold_{fold}"
            cand = self.candidate / "mito" / f"fold_{fold}"
            base.mkdir(parents=True)
            cand.mkdir(parents=True)
            split = {"target": "mito", "fold": fold, "num_folds": 5,
                     "fold_ids": {f"image_{index}.tif": index for index in range(5)}}
            for directory in (base, cand):
                (directory / "split.json").write_text(json.dumps(split), encoding="utf-8")
            (cand / "experiment_config.json").write_text(json.dumps({
                "target": "mito", "fold": fold,
                "positive_patch_probability": 0.35,
                "outer_test_deferred": True,
            }), encoding="utf-8")
            (cand / "baseline_unet.keras").write_bytes(b"synthetic model")
            (cand / "threshold.json").write_text('{"threshold": 0.5}', encoding="utf-8")
            for directory, score in ((base, 0.8), (cand, 0.82 if fold == 0 else 0.805)):
                with (directory / "training_history.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["epoch", "val_full_dice"])
                    writer.writeheader()
                    writer.writerow({"epoch": 0, "val_full_dice": score})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selects_candidate_only_after_margin(self) -> None:
        decisions = select_folds(self.baseline, self.candidate)["decisions"]
        self.assertEqual(decisions[0]["selected_positive_patch_probability"], 0.35)
        self.assertTrue(all(
            decision["selected_positive_patch_probability"] == 0.6
            for decision in decisions[1:]
        ))

    def test_rejects_candidate_outer_test_scored_early(self) -> None:
        (self.candidate / "mito" / "fold_2" / "test_per_image.csv").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evaluated before"):
            select_folds(self.baseline, self.candidate)


if __name__ == "__main__":
    unittest.main()
