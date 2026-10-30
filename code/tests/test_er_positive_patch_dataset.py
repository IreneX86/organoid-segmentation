"""Regression tests for partial-label semantics, not biological accuracy."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er_positive_patch_dataset import encode, supervision, require_two_classes

class PartialLabelTests(unittest.TestCase):
    def test_unmarked_is_ignored(self):
        labels = encode(np.array([[0, 255], [255, 0]], dtype=np.uint8))
        np.testing.assert_array_equal(labels, [[255, 1], [1, 255]])
        target, weight = supervision(labels)
        np.testing.assert_array_equal(target, [[0, 1], [1, 0]])
        np.testing.assert_array_equal(weight, target)

    def test_negative_is_not_unknown(self):
        labels = np.array([0, 1, 255], dtype=np.uint8)
        target, weight = supervision(labels)
        np.testing.assert_array_equal(target, [0, 1, 0])
        np.testing.assert_array_equal(weight, [1, 1, 0])
        require_two_classes(labels)

    def test_positive_only_training_rejected(self):
        with self.assertRaises(ValueError):
            require_two_classes(np.array([1, 255], dtype=np.uint8))
        with self.assertRaises(ValueError):
            supervision(np.array([2], dtype=np.uint8))

if __name__ == '__main__':
    unittest.main()
