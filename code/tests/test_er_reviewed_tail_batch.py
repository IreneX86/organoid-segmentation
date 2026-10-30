"""Regression checks for review-guided tails with different Z-stack depths."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er_reviewed_tail_batch import run_reviewed_tail  # noqa: E402


class ReviewedTailBatchTests(unittest.TestCase):
    def _fixture(self, root: Path):
        inputs, bases = root / "inputs", root / "bases"
        source_paths = []
        yy, xx = np.ogrid[:96, :96]
        ring = ((xx-48)**2 + (yy-48)**2 <= 26**2) & ((xx-48)**2 + (yy-48)**2 >= 16**2)
        for cell, depth in (("cell_a", 4), ("cell_b", 5)):
            source, base = inputs / cell, bases / cell
            source.mkdir(parents=True)
            base.mkdir(parents=True)
            er = np.full((depth, 96, 96), 25, np.uint16)
            er[:, ring] = 300
            fluo = np.full_like(er, 30)
            fluo[:, 15:81, 15:81] = 350
            mask = np.repeat((ring.astype(np.uint8) * 255)[None], depth, axis=0)
            for path, array in ((source / "ER.tif", er), (source / "fluo.tif", fluo),
                                (base / "ER_mask.tif", mask)):
                tifffile.imwrite(path, array, imagej=True)
                source_paths.append(path)
        config = {"cells": {
            "cell_a": {"z_zero_based": [2, 3], "review": {"exclude_ellipses": [
                {"center_xy": [48, 48], "radius_xy": [35, 35]}]}},
            "cell_b": {"z_zero_based": [3, 4], "review": {"rescue_ellipses": [
                {"center_xy": [48, 48], "radius_xy": [35, 35]}]}},
        }}
        config_path = root / "review.json"
        config_path.write_text(json.dumps(config))
        return inputs, bases, source_paths, config_path

    def test_selected_tails_use_real_cell_review_keys_and_keep_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, bases, source_paths, config = self._fixture(root)
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths]
            output = root / "new_results"
            result = run_reviewed_tail(inputs, bases, config, output, tail_slices=2)
            self.assertEqual(len(result), 2)
            self.assertEqual(json.loads((output / "batch_complete.json").read_text())["slices"], 4)
            for cell, start in (("cell_a", 2), ("cell_b", 3)):
                manifest = json.loads((output / cell / "manifest.json").read_text())
                self.assertTrue(manifest["manual_review_applied"])
                self.assertEqual(manifest["source_z_start_zero_based"], start)
                bright = tifffile.imread(output / cell / "ER_mask_bright.tif")
                line = tifffile.imread(output / cell / "ER_mask_line.tif")
                self.assertEqual(bright.shape, (2, 96, 96))
                self.assertTrue(np.all(line <= bright))
            excluded = json.loads((output / "cell_a" / "manifest.json").read_text())
            self.assertGreater(excluded["slice_audits"][0]["review_excluded_pixels"], 0)
            self.assertEqual(
                hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths]
            )
            with self.assertRaises(FileExistsError):
                run_reviewed_tail(inputs, bases, config, output, tail_slices=2)

    def test_wrong_global_z_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, bases, _, config_path = self._fixture(root)
            config = json.loads(config_path.read_text())
            config["cells"]["cell_b"]["z_zero_based"] = [2, 3]
            config_path.write_text(json.dumps(config))
            output = root / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "Review Z indices"):
                run_reviewed_tail(inputs, bases, config_path, output, tail_slices=2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
