"""Conservative synthetic tests for exploratory manual tissue ROI and gate."""

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

from tissue_roi import _check_explicit_pair, apply_gate, derive_tissue_roi, prepare


def annulus(size: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:size, :size]
    radius = np.hypot(yy - size // 2, xx - size // 2)
    fluo = np.zeros((size, size), dtype=np.uint16)
    org = np.zeros_like(fluo)
    bk = np.zeros_like(fluo)
    org[(radius >= 13) & (radius <= 24)] = 65535
    bk[radius > 24] = 65535
    return fluo, org, bk


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TissueRoiTests(unittest.TestCase):
    def test_rejects_wrong_cell_or_slice_pairing(self) -> None:
        fluo = Path("/tmp/cell_001/fluo/fluo_094.tif")
        wrong_cell = Path("/tmp/cell_002/Mito_label/Mito_094.tif")
        wrong_slice = Path("/tmp/cell_001/Mito_label/Mito_095.tif")
        with self.assertRaisesRegex(ValueError, "same cell directory"):
            _check_explicit_pair(fluo, wrong_cell)
        with self.assertRaisesRegex(ValueError, "slice numbers do not match"):
            _check_explicit_pair(fluo, wrong_slice)
        self.assertEqual(
            _check_explicit_pair(fluo, Path("/tmp/cell_001/Mito_label/Mito_094.tif")),
            "cell_001",
        )

    def test_annulus_has_lumen_and_zero_gate_in_center(self) -> None:
        fluo, org, bk = annulus()
        tissue, classes, qc = derive_tissue_roi(fluo, org, bk)
        self.assertEqual(tissue.dtype, np.uint8)
        self.assertEqual(tissue.shape, fluo.shape)
        self.assertEqual(set(np.unique(tissue).tolist()), {0, 255})
        self.assertEqual(classes[40, 40], 2)
        self.assertEqual(classes[0, 0], 0)
        self.assertEqual(classes[40, 60], 1)
        self.assertEqual(tissue[40, 40], 0)
        self.assertEqual(qc["lumen_fraction_of_enclosed_neither"], 1.0)

    def test_antialias_midpoint_and_overlap_uncertainty(self) -> None:
        fluo, org, bk = annulus()
        org[40, 60] = 32767  # Exactly at midpoint: not tissue.
        org[39, 60] = 32768  # Above midpoint: tissue.
        bk[39, 60] = 65535   # Conflict remains tissue gate, uncertain class.
        tissue, classes, qc = derive_tissue_roi(fluo, org, bk)
        self.assertEqual(tissue[40, 60], 0)
        self.assertEqual(tissue[39, 60], 255)
        self.assertEqual(classes[39, 60], 255)
        self.assertEqual(qc["org_bk_overlap_pixels"], 1)
        self.assertGreaterEqual(qc["uncertain_pixels"], 1)

    def test_missing_lumen_and_major_conflict_fail_closed(self) -> None:
        fluo, org, bk = annulus()
        bk[org > 0] = 65535
        with self.assertRaisesRegex(ValueError, "Major org/bk conflict"):
            derive_tissue_roi(fluo, org, bk)
        fluo, org, bk = annulus()
        bk[org == 0] = 65535  # No unclaimed central lumen remains.
        with self.assertRaisesRegex(ValueError, "No dominant enclosed"):
            derive_tissue_roi(fluo, org, bk)

    def test_tiff_pipeline_is_provenanced_no_clobber_and_no_new_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fluo, org, bk = annulus()
            bk[40, 60] = 65535  # Contested tissue boundary remains in the gate.
            org[40, 59] = 0  # Another uncertain annotation gap remains in the gate.
            sources = root / "sources"
            sources.mkdir()
            fluo_path = sources / "fluo_094.tif"
            org_path = sources / "org.tif"
            bk_path = sources / "bk.tif"
            for path, image in ((fluo_path, fluo), (org_path, org), (bk_path, bk)):
                tifffile.imwrite(path, image, metadata=None)
            original_hashes = [sha256(path) for path in (fluo_path, org_path, bk_path)]
            tissue_path = root / "derived" / "tissue_094.tif"
            class_path = root / "derived" / "classes_094.tif"
            prep_manifest = root / "derived" / "prepare.json"
            prep = prepare(fluo_path, org_path, bk_path,
                           tissue_path, class_path, prep_manifest)
            self.assertEqual(prep["source_names"]["fluo"], "fluo_094.tif")
            self.assertEqual(json.loads(prep_manifest.read_text())["gate_policy"],
                             "remove_outside_and_lumen_preserve_uncertain_v1")
            with self.assertRaises(FileExistsError):
                prepare(fluo_path, org_path, bk_path, tissue_path, class_path, prep_manifest)

            prediction = np.zeros_like(fluo, dtype=np.uint8)
            prediction[40, 40] = 255  # Lumen debris must disappear.
            prediction[0, 0] = 255    # Outside artifact must disappear.
            prediction[40, 60] = 255  # Tissue object must remain.
            prediction[40, 59] = 255  # Unclassified enclosed gap must remain.
            pred_path = root / "prediction" / "Mito_094.tif"
            pred_path.parent.mkdir()
            tifffile.imwrite(pred_path, prediction, metadata=None)
            out = root / "gated" / "Mito_094.tif"
            out_manifest = root / "gated" / "gate.json"
            with self.assertRaisesRegex(ValueError, "cross-channel-alignment-confirmed"):
                apply_gate(fluo_path, pred_path, tissue_path, class_path,
                           prep_manifest, out, out_manifest)
            result = apply_gate(fluo_path, pred_path, tissue_path, class_path,
                                prep_manifest, out, out_manifest,
                                cross_channel_alignment_confirmed=True)
            gated = tifffile.imread(out)
            self.assertEqual(gated.shape, fluo.shape)
            self.assertEqual(gated.dtype, np.uint8)
            self.assertEqual(gated[40, 40], 0)
            self.assertEqual(gated[0, 0], 0)
            self.assertEqual(gated[40, 60], 255)
            self.assertEqual(gated[40, 59], 255)
            self.assertEqual(tifffile.imread(class_path)[40, 60], 255)
            self.assertEqual(tifffile.imread(class_path)[40, 59], 255)
            self.assertFalse(np.any((gated > 0) & (prediction == 0)))
            self.assertEqual(result["removed_prediction_pixels"], 2)
            self.assertEqual(result["retained_uncertain_pixels"], 2)
            self.assertTrue(result["cross_channel_alignment_confirmed_by_analyst"])
            self.assertEqual([sha256(path) for path in (fluo_path, org_path, bk_path)],
                             original_hashes)
            with self.assertRaises(FileExistsError):
                apply_gate(fluo_path, pred_path, tissue_path, class_path,
                           prep_manifest, out, out_manifest,
                           cross_channel_alignment_confirmed=True)

    def test_rejects_shape_name_values_and_stale_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fluo, org, bk = annulus()
            files = [root / name for name in ("fluo_094.tif", "org.tif", "bk.tif")]
            for path, image in zip(files, (fluo, org, bk)):
                tifffile.imwrite(path, image, metadata=None)
            tissue = root / "roi" / "tissue.tif"
            classes = root / "roi" / "classes.tif"
            manifest = root / "roi" / "prepare.json"
            prepare(*files, tissue, classes, manifest)
            pred_dir = root / "pred"
            pred_dir.mkdir()
            pred = pred_dir / "fluo_094.tif"
            tifffile.imwrite(pred, np.full(fluo.shape, 128, dtype=np.uint8), metadata=None)
            with self.assertRaisesRegex(ValueError, "only uint8 0/255"):
                apply_gate(files[0], pred, tissue, classes, manifest,
                           root / "out" / pred.name, root / "out" / "manifest.json")
            tifffile.imwrite(pred, np.zeros((79, 80), dtype=np.uint8), metadata=None)
            with self.assertRaisesRegex(ValueError, "Pixel-grid mismatch"):
                apply_gate(files[0], pred, tissue, classes, manifest,
                           root / "out" / pred.name, root / "out" / "manifest.json")
            tifffile.imwrite(pred, np.zeros_like(fluo, dtype=np.uint8), metadata=None)
            with self.assertRaisesRegex(ValueError, "exact prediction filename"):
                apply_gate(files[0], pred, tissue, classes, manifest,
                           root / "out" / "renamed.tif", root / "out" / "manifest.json")
            modified_tissue = tifffile.imread(tissue)
            modified_tissue[40, 40] = 255
            tifffile.imwrite(tissue, modified_tissue, metadata=None)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                apply_gate(files[0], pred, tissue, classes, manifest,
                           root / "out" / pred.name, root / "out" / "manifest.json")
            with self.assertRaisesRegex(ValueError, "Pixel-grid mismatch"):
                derive_tissue_roi(fluo, org[:-1], bk)
            with self.assertRaisesRegex(ValueError, "2D uint16"):
                derive_tissue_roi(fluo, org.astype(np.uint8), bk)
            multipage = root / "multipage.tif"
            with tifffile.TiffWriter(multipage) as writer:
                writer.write(fluo, metadata=None)
                writer.write(fluo, metadata=None)
            with self.assertRaisesRegex(ValueError, "one independent 2D TIFF plane"):
                prepare(multipage, files[1], files[2],
                        root / "out" / "roi.tif", root / "out" / "classes.tif",
                        root / "out" / "prepare.json")

    def test_same_name_fluo_prediction_needs_no_cross_channel_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fluo, org, bk = annulus()
            cell = root / "cell_001"
            paths = [cell / "fluo" / "fluo_094.tif", root / "org.tif", root / "bk.tif"]
            paths[0].parent.mkdir(parents=True)
            for path, image in zip(paths, (fluo, org, bk)):
                tifffile.imwrite(path, image, metadata=None)
            tissue = root / "roi" / "tissue.tif"
            classes = root / "roi" / "classes.tif"
            prep_manifest = root / "roi" / "prepare.json"
            prepare(*paths, tissue, classes, prep_manifest)
            prediction = np.zeros(fluo.shape, dtype=np.uint8)
            prediction[40, 40] = 255  # Lumen.
            prediction[40, 60] = 255  # Tissue.
            prediction_path = cell / "fluo_label" / "fluo_094.tif"
            prediction_path.parent.mkdir()
            tifffile.imwrite(prediction_path, prediction, metadata=None)
            output = root / "gated" / prediction_path.name
            result = apply_gate(paths[0], prediction_path, tissue, classes,
                                prep_manifest, output, root / "gated" / "gate.json")
            gated = tifffile.imread(output)
            self.assertEqual(int(gated[40, 40]), 0)
            self.assertEqual(int(gated[40, 60]), 255)
            self.assertFalse(result["cross_channel_alignment_confirmed_by_analyst"])


if __name__ == "__main__":
    unittest.main()
