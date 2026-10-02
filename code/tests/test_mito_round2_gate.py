"""Verify the locked validation decision logic without opening any TIFFs."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mito_round2_gate import decide


def rows(values, sparse=(0,)):
    return [
        {"filename": f"image_{i:02d}.tif", "dice": value,
         "foreground_fraction_true": 0.03 if i in sparse else 0.08}
        for i, value in enumerate(values)
    ]


class Round2GateTests(unittest.TestCase):
    def test_accepts_sparse_gain_without_mean_penalty(self):
        baseline = rows([0.78, 0.86, 0.86, 0.86])
        candidate = rows([0.80, 0.855, 0.855, 0.855])
        self.assertTrue(decide(baseline, candidate)["selected_candidate"])

    def test_rejects_large_individual_loss(self):
        baseline = rows([0.78, 0.86, 0.86, 0.86])
        candidate = rows([0.80, 0.83, 0.89, 0.89])
        self.assertFalse(decide(baseline, candidate)["selected_candidate"])

    def test_rejects_sparse_deterioration(self):
        baseline = rows([0.78, 0.86, 0.86, 0.86])
        candidate = rows([0.77, 0.87, 0.87, 0.87])
        self.assertFalse(decide(baseline, candidate)["selected_candidate"])


if __name__ == "__main__":
    unittest.main()
