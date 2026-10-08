"""Check coordinate preservation and proposal semantics without model weights."""

import unittest
from pathlib import Path

import numpy as np
from mito_sam_feasibility import boxes_from_proposal, metrics, origins, select_cases


class SamFeasibilityTests(unittest.TestCase):
    def test_all_native_pixels_covered(self):
        coverage = np.zeros((1200, 1200), dtype=np.uint8)
        for y in origins(1200):
            for x in origins(1200):
                coverage[y : y + 512, x : x + 512] += 1
        self.assertGreaterEqual(coverage.min(), 1)
        self.assertEqual(origins(1200)[-1] + 512, 1200)

    def test_xy_order_padding_and_tiny_component_exclusion(self):
        proposal = np.zeros((32, 40), dtype=bool)
        proposal[1:6, 20:24] = True
        proposal[25, 2] = True
        np.testing.assert_array_equal(boxes_from_proposal(proposal), [[16, 0, 28, 10]])

    def test_empty_proposal(self):
        self.assertEqual(boxes_from_proposal(np.zeros((32, 32), bool)).shape, (0, 4))

    def test_known_counts(self):
        gt = np.zeros((32, 32), bool)
        gt[10:12, 10:12] = True
        pred = gt.copy()
        pred[30, 30] = True
        result = metrics(pred, gt)
        self.assertEqual(
            (result["tp"], result["fp"], result["fn"], result["distant_fp"]), (4, 1, 0, 1)
        )
        self.assertAlmostEqual(result["dice"], 8 / 9)

    def test_four_distinct_frozen_strata(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "baseline/mito_five_fold_20261001/oof_per_image.csv"
        )
        rows = select_cases(path)
        self.assertEqual(len({r["filename"] for r in rows}), 4)
        self.assertEqual(len({r["stratum"] for r in rows}), 4)


if __name__ == "__main__":
    unittest.main()
