"""Protect the LD label rule across Photoshop and ImageJ TIFF encodings."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_prepare_labels import binarize_antialiased_mask, prepare_labels


class LDLabelPreparationTests(unittest.TestCase):
    def test_uses_per_file_midpoint_without_erasing_255_encoded_masks(self) -> None:
        photoshop = np.array([[0, 100, 32000, 32768, 65535]], dtype=np.uint16)
        imagej = np.array([[0, 127, 128, 255, 0]], dtype=np.uint16)
        photoshop_binary, photoshop_threshold = binarize_antialiased_mask(photoshop)
        imagej_binary, imagej_threshold = binarize_antialiased_mask(imagej)
        self.assertEqual(photoshop_threshold, 32768)
        self.assertEqual(imagej_threshold, 128)
        np.testing.assert_array_equal(photoshop_binary, [[0, 0, 0, 1, 1]])
        np.testing.assert_array_equal(imagej_binary, [[0, 0, 1, 1, 0]])

    def test_writes_separate_verified_labels_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            mask_dir = root / "manual"
            output_dir = root / "derived"
            input_dir.mkdir()
            mask_dir.mkdir()
            image = np.full((4, 4), 1000, dtype=np.uint16)
            mask = np.zeros((4, 4), dtype=np.uint16)
            mask[1, 1] = 65535
            mask[1, 2] = 1000  # Antialiased fringe does not become foreground.
            tifffile.imwrite(input_dir / "ld_001.tif", image)
            tifffile.imwrite(mask_dir / "ld_001.tif", mask)
            original_bytes = (mask_dir / "ld_001.tif").read_bytes()

            manifest = prepare_labels(input_dir, mask_dir, output_dir, expected_pairs=1)
            derived = tifffile.imread(output_dir / "ld_001.tif")
            self.assertEqual(derived.dtype, np.uint8)
            self.assertEqual(int(derived.sum()), 1)
            self.assertEqual((mask_dir / "ld_001.tif").read_bytes(), original_bytes)
            item = manifest["files"][0]
            self.assertEqual(
                item["source_mask_sha256"], hashlib.sha256(original_bytes).hexdigest()
            )
            self.assertTrue((output_dir / "manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                prepare_labels(input_dir, mask_dir, output_dir, expected_pairs=1)

    def test_rejects_unrecognized_mask_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "white level"):
            binarize_antialiased_mask(np.array([[0, 3000]], dtype=np.uint16))


if __name__ == "__main__":
    unittest.main()
