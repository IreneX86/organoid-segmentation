"""Regression checks for the selected ER topology-safe batch method."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er_coherent_fiber import refine_coherent_fiber  # noqa: E402
from er_coherent_fiber_batch import run_batch  # noqa: E402
from verify_er_coherent_fiber_batch import verify  # noqa: E402


class CoherentFiberTests(unittest.TestCase):
    def test_refinement_preserves_old_center_and_components(self) -> None:
        old = np.zeros((100, 100), dtype=bool)
        old[40:60, 20:70] = True
        current = np.zeros_like(old)
        current[48, 10:80] = True
        adjacent = current.copy()
        result, audit = refine_coherent_fiber(current, old, adjacent)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.isin(result, (0, 255)).all())
        self.assertTrue(np.all(result[48, 20:70] == 255))
        self.assertEqual(label(result > 0, np.ones((3, 3)))[1], 1)
        self.assertGreater(audit["candidate_pixels"], 0)

    def test_global_z_neighbors_and_exclusive_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cell = root / "inputs" / "cell_a"
            base_folder = root / "bases" / "cell_a"  # flat base layout
            cell.mkdir(parents=True)
            base_folder.mkdir(parents=True)
            yy, xx = np.ogrid[:96, :96]
            ring = ((xx - 48) ** 2 + (yy - 48) ** 2 <= 26 ** 2) & (
                (xx - 48) ** 2 + (yy - 48) ** 2 >= 14 ** 2
            )
            er = np.full((3, 96, 96), 30, dtype=np.uint16)
            er[:, ring] = 400
            fluo = np.full_like(er, 250)
            base = np.broadcast_to(ring, er.shape).astype(np.uint8) * 255
            for path, value in ((cell / "ER.tif", er), (cell / "fluo.tif", fluo),
                                (base_folder / "ER_mask.tif", base)):
                tifffile.imwrite(path, value, imagej=True)
            selected = run_batch(root / "inputs", root / "bases", root / "selected",
                                 z_start=1, z_stop=2)
            whole = run_batch(root / "inputs", root / "bases", root / "whole")
            self.assertEqual(verify(root / "inputs", root / "whole")["z_slices"], 3)
            self.assertTrue(np.array_equal(tifffile.imread(selected[0]),
                                           tifffile.imread(whole[0])[1]))
            manifest = json.loads((root / "selected" / "cell_a" / "manifest.json").read_text())
            self.assertEqual(manifest["slice_audits"][0]["neighbor_z_zero_based"], [0, 2])
            self.assertEqual(manifest["source_z_start_zero_based"], 1)
            self.assertEqual(manifest["outputs"]["ER_mask_line"]["shape"], [96, 96])
            with self.assertRaises(FileExistsError):
                run_batch(root / "inputs", root / "bases", root / "selected")
            with self.assertRaises(ValueError):
                run_batch(root / "inputs", root / "bases", root / "bad",
                          profile_map={"wrong": "spheroid_default"})
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
