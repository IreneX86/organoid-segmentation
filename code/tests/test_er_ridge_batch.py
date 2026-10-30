"""Safety checks for field-specific ER artifact tracking across Z."""

import numpy as np
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er_ridge_batch_predict import exclude_temporally_confirmed_artifact  # noqa: E402


def _audit(status: str, bbox: list[int] | None = None, z: int = 0) -> dict:
    return {
        "source_z_zero_based": z,
        "strong": {"artifact_status": status, "artifact_group_bbox_yxyx": bbox},
        "final_pixels": 0,
    }


class ArtifactFallbackTests(unittest.TestCase):
    def test_isolated_tracker_miss_uses_matching_neighbors(self) -> None:
        mask = np.zeros((3, 40, 40), dtype=bool)
        mask[1, 5:15, 5:15] = True
        mask[1, 25:30, 25:30] = True
        audits = [
            _audit("matched", [3, 3, 16, 16], 0),
            _audit("unresolved_keep_candidates", z=1),
            _audit("matched", [3, 3, 16, 16], 2),
        ]
        repairs = exclude_temporally_confirmed_artifact(mask, audits, padding=0)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]["removed_pixels"], 100)
        self.assertFalse(mask[1, 5:15, 5:15].any())
        self.assertTrue(mask[1, 25:30, 25:30].all())

    def test_uncertain_neighbor_boxes_never_delete(self) -> None:
        mask = np.ones((3, 40, 40), dtype=bool)
        audits = [
            _audit("matched", [0, 0, 10, 10], 0),
            _audit("unresolved_keep_candidates", z=1),
            _audit("matched", [20, 20, 30, 30], 2),
        ]
        self.assertEqual(exclude_temporally_confirmed_artifact(mask, audits), [])
        self.assertTrue(mask.all())
