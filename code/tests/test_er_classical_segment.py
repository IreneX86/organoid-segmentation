"""Small, reversible checks for the exploratory ER batch pipeline."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


MODULE_PATH = Path(__file__).resolve().parents[1] / "er_classical_segment.py"
SPEC = importlib.util.spec_from_file_location("er_classical_segment", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ERBatchTests(unittest.TestCase):
    def test_reviewed_pilot_is_pixel_identical_when_available(self) -> None:
        project = MODULE_PATH.parents[1]
        pilot = project / "outputs" / "er_pilot_20261027"
        required = [pilot / name for name in (
            "ER_z037.tif", "fluo_z037.tif", "ER_mask_z037_denoised_v2.tif"
        )]
        if not all(path.is_file() for path in required):
            self.skipTest("Ignored local pilot TIFFs are unavailable")
        rois = json.loads((project / "experiments" / "er_pilot_z037_review_rois.json").read_text())
        ellipses = rois["MCF-7_20260814/cell_20260814_23298_NZ_50"]["36"]
        mask, audit = MODULE.segment_er_slice(
            tifffile.imread(required[0]),
            tifffile.imread(required[1]),
            review_ellipses=tuple(ellipses),
        )
        np.testing.assert_array_equal(mask, tifffile.imread(required[2]))
        self.assertEqual(audit["final_pixels"], 46810)

    def test_same_shape_binary_output_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "input" / "group with space " / "cell_1"
            cell.mkdir(parents=True)
            image = np.zeros((2, 64, 64), dtype=np.uint16)
            image[:, 16:48, 16:48] = 900
            tifffile.imwrite(cell / "ER.tif", image, imagej=True)
            tifffile.imwrite(cell / "fluo.tif", image, imagej=True)
            results = MODULE.run_batch(root / "input", root / "output")
            self.assertEqual(len(results), 1)
            mask = tifffile.imread(results[0])
            self.assertEqual(mask.shape, image.shape)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
            with self.assertRaises(FileExistsError):
                MODULE.run_batch(root / "input", root / "output")
            np.testing.assert_array_equal(tifffile.imread(cell / "ER.tif"), image)

    def test_rejects_mismatched_stack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "input" / "cell_1"
            cell.mkdir(parents=True)
            tifffile.imwrite(cell / "ER.tif", np.zeros((2, 64, 64), dtype=np.uint16), imagej=True)
            tifffile.imwrite(cell / "fluo.tif", np.zeros((3, 64, 64), dtype=np.uint16), imagej=True)
            with self.assertRaises(ValueError):
                MODULE.run_batch(root / "input", root / "output")


if __name__ == "__main__":
    unittest.main()
