"""Focused CPU-only checks for the predeclared fold-local LD validation gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_highlight_gate import _gate_checks, small_component_counts


def row(name: str, *, dice: float, precision: float, count: int, hits: int) -> dict:
    return {
        "filename": name, "dice": dice, "precision": precision,
        "small_component_count_proxy": count,
        "small_component_detected_any_proxy": hits,
    }


class SmallComponentTests(unittest.TestCase):
    def test_diagonal_pixels_are_one_eight_connected_region(self) -> None:
        truth = np.zeros((8, 8), dtype=bool)
        truth[2, 2] = truth[3, 3] = True
        pred = np.zeros_like(truth)
        pred[3, 3] = True
        self.assertEqual(small_component_counts(truth, pred), (1, 1))

    def test_region_larger_than_25_pixels_is_excluded(self) -> None:
        truth = np.zeros((10, 10), dtype=bool)
        truth[2:8, 2:8] = True
        pred = truth.copy()
        self.assertEqual(small_component_counts(truth, pred), (0, 0))


class FoldGateTests(unittest.TestCase):
    def test_ordinary_component_gain_passes_four_conditions(self) -> None:
        baseline = [row("a.tif", dice=.80, precision=.80, count=10, hits=4)]
        candidate = [row("a.tif", dice=.795, precision=.78, count=10, hits=5)]
        checks, observed = _gate_checks(baseline, candidate)
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(observed["small_component_overlap_fraction_delta"], .10)

    def test_ceiling_requires_no_regression_and_dice_gain(self) -> None:
        baseline = [row("a.tif", dice=.70, precision=.80, count=7, hits=7)]
        candidate = [row("a.tif", dice=.71, precision=.80, count=7, hits=7)]
        checks, observed = _gate_checks(baseline, candidate)
        self.assertTrue(all(checks.values()))
        self.assertEqual(observed["baseline_component_ceiling_rule_applies"], 1)
        candidate[0]["dice"] = .705
        checks, _ = _gate_checks(baseline, candidate)
        self.assertFalse(checks["small_component_ceiling_aware_condition"])
        candidate[0]["dice"] = .72
        candidate[0]["small_component_detected_any_proxy"] = 6
        checks, _ = _gate_checks(baseline, candidate)
        self.assertFalse(checks["small_component_ceiling_aware_condition"])

    def test_zero_components_is_inconclusive(self) -> None:
        baseline = [row("a.tif", dice=.70, precision=.80, count=0, hits=0)]
        candidate = [row("a.tif", dice=.80, precision=.80, count=0, hits=0)]
        checks, observed = _gate_checks(baseline, candidate)
        self.assertFalse(checks["small_component_ceiling_aware_condition"])
        self.assertIsNone(observed["small_component_overlap_fraction_delta"])

    def test_single_image_drop_and_precision_loss_fail(self) -> None:
        baseline = [
            row("a.tif", dice=.8, precision=.8, count=10, hits=4),
            row("b.tif", dice=.8, precision=.8, count=10, hits=4),
        ]
        candidate = [
            row("a.tif", dice=.69, precision=.7, count=10, hits=6),
            row("b.tif", dice=.92, precision=.8, count=10, hits=5),
        ]
        checks, _ = _gate_checks(baseline, candidate)
        self.assertFalse(checks["no_single_image_dice_drop_more_than_0_10"])
        self.assertFalse(checks["mean_precision_drop_no_more_than_0_03"])


if __name__ == "__main__":
    unittest.main()
