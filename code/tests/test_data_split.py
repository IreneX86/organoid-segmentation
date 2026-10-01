"""Check that candidate runs reuse the frozen image-level fold assignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_split import split_from_fold_ids
from tiff_io import ImageLabelPair


class FrozenFoldSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = [
            ImageLabelPair(Path(f"image_{index}.tif"), Path(f"mask_{index}.tif"))
            for index in range(5)
        ]
        self.fold_ids = {pair.filename: index for index, pair in enumerate(self.pairs)}

    def test_assigns_test_validation_and_training_by_filename(self) -> None:
        split = split_from_fold_ids(self.pairs, self.fold_ids, fold=2, num_folds=5)
        self.assertEqual([pair.filename for pair in split.test], ["image_2.tif"])
        self.assertEqual([pair.filename for pair in split.validation], ["image_3.tif"])
        self.assertEqual(len(split.train), 3)

    def test_rejects_missing_filename(self) -> None:
        self.fold_ids.pop("image_4.tif")
        with self.assertRaisesRegex(ValueError, "filenames differ"):
            split_from_fold_ids(self.pairs, self.fold_ids, fold=2, num_folds=5)

    def test_rejects_invalid_fold_index(self) -> None:
        self.fold_ids["image_4.tif"] = 5
        with self.assertRaisesRegex(ValueError, "invalid fold index"):
            split_from_fold_ids(self.pairs, self.fold_ids, fold=2, num_folds=5)


if __name__ == "__main__":
    unittest.main()
