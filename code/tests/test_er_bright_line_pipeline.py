"""Focused regression checks for the reusable ER bright-line batch entry point."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er_bright_line_pipeline import (  # noqa: E402
    ERBrightLineSettings,
    ellipse_region,
    narrow_existing_lines,
    refine_lines,
    run_batch,
    settings_for_profile,
)


class ERBrightLineTests(unittest.TestCase):
    def test_profiles_keep_historical_default_and_opt_in_dim_candidate(self) -> None:
        self.assertEqual(settings_for_profile("spheroid_default"), ERBrightLineSettings())
        dim = settings_for_profile("dim_organoid_candidate")
        self.assertLess(dim.minimum_highpass, ERBrightLineSettings().minimum_highpass)
        self.assertEqual(dim.fleck_min_area, ERBrightLineSettings().fleck_min_area)
        with self.assertRaises(ValueError):
            settings_for_profile("unknown")

    def test_line_refinement_only_removes_pixels_and_preserves_components(self) -> None:
        y, x = np.ogrid[:96, :96]
        left = (x-27)**2 + (y-48)**2 <= 13**2
        right = (x-67)**2 + (y-48)**2 <= 13**2
        narrow_bridge = (44 <= y) & (y <= 52) & (27 <= x) & (x <= 67)
        original = left | right | narrow_bridge
        refined, _ = refine_lines(original, half_width=4)
        self.assertTrue(np.all(refined <= original))
        self.assertLess(refined.sum(), original.sum())
        self.assertEqual(label(refined, np.ones((3, 3)))[1], 1)
        thinner, _ = narrow_existing_lines(original, refined, half_width=3)
        self.assertTrue(np.all(thinner <= refined))
        self.assertEqual(label(thinner, np.ones((3, 3)))[1], 1)

    def test_review_ellipse_is_only_a_coarse_region(self) -> None:
        roi = ellipse_region((80, 80), [{"center_xy": [40, 30], "radius_xy": [12, 8]}])
        self.assertTrue(roi[30, 40])
        self.assertFalse(roi[10, 40])
        with self.assertRaises(ValueError):
            ellipse_region((80, 80), [{"center_xy": [40, 30], "radius_xy": [0, 8]}])

    def test_batch_keeps_tiff_shape_and_never_overwrites_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root/"inputs"/"cell_a"
            bases = root/"bases"/"cell_a"
            inputs.mkdir(parents=True)
            bases.mkdir(parents=True)
            y, x = np.ogrid[:96, :96]
            ring = (x-48)**2 + (y-48)**2 <= 24**2
            hole = (x-48)**2 + (y-48)**2 < 13**2
            mask = ring & ~hole
            er = np.full((2, 96, 96), 25, dtype=np.uint16)
            er[:, mask] = 250
            fluo = np.full_like(er, 30)
            fluo[:, 18:78, 18:78] = 300
            base = np.stack([mask, mask]).astype(np.uint8)*255
            paths = [inputs/"ER.tif", inputs/"fluo.tif", bases/"ER_mask.tif"]
            for path, array in zip(paths, (er, fluo, base)):
                tifffile.imwrite(path, array, imagej=True)
            original_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            output = root/"new_output"
            results = run_batch(root/"inputs", output, base_mask_root=root/"bases", settings=ERBrightLineSettings())
            self.assertEqual(len(results), 1)
            bright = tifffile.imread(results[0].with_name("ER_mask_bright.tif"))
            line = tifffile.imread(results[0])
            self.assertEqual(bright.shape, er.shape)
            self.assertEqual(line.shape, er.shape)
            with tifffile.TiffFile(results[0]) as stack_file:
                self.assertEqual(stack_file.series[0].axes, "ZYX")
            self.assertTrue(np.isin(bright, [0, 255]).all())
            self.assertTrue(np.isin(line, [0, 255]).all())
            self.assertTrue(np.all(line <= bright))
            self.assertEqual(original_hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            with self.assertRaises(FileExistsError):
                run_batch(root/"inputs", output, base_mask_root=root/"bases")
            one_slice = root/"one_slice"
            selected = run_batch(root/"inputs", one_slice, base_mask_root=root/"bases", z_start=1, z_stop=2)
            self.assertEqual(tifffile.imread(selected[0]).shape, er.shape[1:])
            selected_manifest = json.loads((one_slice/"cell_a"/"manifest.json").read_text())
            self.assertEqual(selected_manifest["output_shape"], list(er.shape[1:]))
            self.assertEqual(selected_manifest["source_z_start_zero_based"], 1)
            self.assertEqual(selected_manifest["profile"], "spheroid_default")
            with self.assertRaises(ValueError):
                run_batch(root/"inputs", root/"invalid_map", base_mask_root=root/"bases",
                          profile_map={"other_cell": "dim_organoid_candidate"})
            self.assertFalse((root/"invalid_map").exists())
            with self.assertRaises(ValueError):
                run_batch(root/"inputs", root/"invalid_name", base_mask_root=root/"bases",
                          profile_map={"cell_a": "unknown"})
            self.assertFalse((root/"invalid_name").exists())
            mapped = root/"mapped"
            run_batch(root/"inputs", mapped, base_mask_root=root/"bases", z_start=1, z_stop=2,
                      profile_map={"cell_a": "dim_organoid_candidate"})
            mapped_manifest = json.loads((mapped/"cell_a"/"manifest.json").read_text())
            self.assertEqual(mapped_manifest["profile"], "dim_organoid_candidate")
            self.assertEqual(mapped_manifest["settings"]["minimum_highpass"], 6.0)
            thinner = root/"mapped_thinner"
            run_batch(root/"inputs", thinner, base_mask_root=root/"bases", z_start=1, z_stop=2,
                      profile_map={"cell_a": "dim_organoid_candidate"}, line_half_width_override=3)
            thin_manifest = json.loads((thinner/"cell_a"/"manifest.json").read_text())
            self.assertEqual(thin_manifest["settings"]["line_half_width"], 3)
            self.assertTrue(np.array_equal(tifffile.imread(mapped/"cell_a"/"ER_mask_bright.tif"),
                                           tifffile.imread(thinner/"cell_a"/"ER_mask_bright.tif")))
            mapped_line = tifffile.imread(mapped/"cell_a"/"ER_mask_line.tif") > 0
            thinner_line = tifffile.imread(thinner/"cell_a"/"ER_mask_line.tif") > 0
            self.assertTrue(np.all(thinner_line <= mapped_line))
            self.assertEqual(label(mapped_line, np.ones((3, 3)))[1],
                             label(thinner_line, np.ones((3, 3)))[1])
            with self.assertRaises(ValueError):
                run_batch(root/"inputs", root/"bad_thinner", base_mask_root=root/"bases",
                          line_half_width_override=0)
            self.assertFalse((root/"bad_thinner").exists())

    def test_single_2d_pair_runs_without_base_mask(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root/"inputs"
            inputs.mkdir()
            y, x = np.ogrid[:128, :128]
            ring = ((x-64)**2 + (y-64)**2 <= 30**2) & ((x-64)**2 + (y-64)**2 >= 19**2)
            er = np.full((128, 128), 25, dtype=np.uint16)
            er[ring] = 300
            fluo = np.full_like(er, 30)
            fluo[15:113, 15:113] = 350
            tifffile.imwrite(inputs/"ER.tif", er)
            tifffile.imwrite(inputs/"fluo.tif", fluo)
            output = root/"result"
            result = run_batch(inputs, output)
            self.assertEqual(len(result), 1)
            self.assertEqual(tifffile.imread(result[0]).shape, er.shape)
            self.assertEqual(tifffile.imread(output/"ER_mask_bright.tif").shape, er.shape)


if __name__ == "__main__":
    unittest.main()
