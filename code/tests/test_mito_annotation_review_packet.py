"""Protocol-level checks for the local blinded annotation packet builder."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mito_annotation_review_packet import (  # noqa: E402
    CROP_CENTER,
    MIN_SEPARATION,
    SEED,
    _candidate_masks,
    _crop,
    _pick_hash_order,
    Case,
    select_source_images,
)


class ReviewPacketTests(unittest.TestCase):
    def test_image_draw_uses_only_oof_values_and_sha_order(self) -> None:
        oof = []
        audit = []
        for index in range(36):
            name = f"mask_{index + 1:03d}.tif"
            foreground = 0.01 if index < 7 else 0.08
            dice = 0.5 + index * 0.005
            oof.append({
                "filename": name, "fold": str(index % 5),
                "foreground_fraction_true": str(foreground), "dice": str(dice),
            })
            audit.append({
                "filename": name, "fold": str(index % 5),
                "foreground_fraction": str(foreground), "dice": str(dice),
            })
        selected = select_source_images(oof, audit)
        self.assertEqual(len(selected), 8)
        for foreground_group in ("low_foreground", "higher_foreground"):
            for dice_group in ("lower_dice", "higher_dice"):
                cell = [row for row in selected if
                        row.foreground_group == foreground_group
                        and row.dice_group == dice_group]
                self.assertEqual(len(cell), 2)
                self.assertEqual(
                    [row.filename for row in cell],
                    sorted(
                        [row.filename for row in cell],
                        key=lambda name: hashlib.sha256(
                            f"{SEED}|{name}".encode()
                        ).hexdigest(),
                    ),
                )

    def test_pixel_draw_enforces_spacing(self) -> None:
        eligible = np.zeros((140, 140), dtype=bool)
        eligible[10, 10] = True
        eligible[12, 12] = True
        eligible[100, 100] = True
        points = _pick_hash_order(eligible, "mask_001.tif", "tp", [], 3)
        self.assertEqual(len(points), 2)
        self.assertGreaterEqual(
            sum((a - b) ** 2 for a, b in zip(points[0], points[1])),
            MIN_SEPARATION ** 2,
        )

    def test_crop_keeps_target_center_with_blank_padding(self) -> None:
        array = np.zeros((400, 400), dtype=np.uint8)
        array[3, 4] = 250
        crop = _crop(array, 3, 4, fill=128)
        self.assertEqual(crop.shape, (320, 320))
        self.assertEqual(crop[CROP_CENTER, CROP_CENTER], 250)
        self.assertEqual(crop[0, 0], 128)

    def test_candidates_require_annotated_background_and_fixed_distance(self) -> None:
        shape = (400, 400)
        truth = np.zeros(shape, dtype=bool)
        truth[200, 200] = True
        predicted = truth.copy()
        predicted[100, 100] = True
        predicted[300, 300] = True
        predicted[200, 201] = True
        normalized = np.zeros(shape, dtype=np.float32)
        residual = np.zeros(shape, dtype=np.float32)
        normalized[100, 100] = 0.9
        residual[100, 100] = 0.9
        normalized[350, 350] = 0.91
        residual[350, 350] = 0.9
        case = Case(
            normalized=normalized, truth=truth, predicted=predicted,
            residual=residual,
            distance_to_gt=np.hypot(*np.indices(shape) - np.array([200, 200])[:, None, None]),
            intensity_p90=0.5, residual_p90=0.5,
        )
        masks = _candidate_masks(case, bright_fp_intensity=0.9)
        self.assertTrue(masks["distant_fp_bright"][100, 100])
        self.assertTrue(masks["distant_fp_other"][300, 300])
        self.assertTrue(masks["bright_tn"][350, 350])
        self.assertTrue(masks["boundary_fp"][200, 201])
        self.assertTrue(masks["tp"][200, 200])
        self.assertFalse(masks["bright_tn"][100, 100])


if __name__ == "__main__":
    unittest.main()
