"""Check the focal ablation's numerical behavior and gradient stability."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import tensorflow as tf
    from losses import binary_focal_dice_loss
except ImportError:
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is installed in the Runpod training environment")
class FocalDiceLossTests(unittest.TestCase):
    def test_hard_false_positive_is_penalized_and_gradient_is_finite(self):
        truth = tf.constant([[[[1.0], [0.0]], [[0.0], [0.0]]]])
        good = tf.constant([[[[0.9], [0.05]], [[0.05], [0.05]]]])
        hard_fp = tf.Variable([[[[0.9], [0.95]], [[0.05], [0.05]]]])
        with tf.GradientTape() as tape:
            hard_loss = binary_focal_dice_loss(truth, hard_fp)
        gradient = tape.gradient(hard_loss, hard_fp)
        self.assertGreater(float(hard_loss), float(binary_focal_dice_loss(truth, good)))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
        self.assertGreater(float(gradient[0, 0, 1, 0]), 0.0)

    def test_perfect_prediction_has_no_nonfinite_terms(self):
        truth = tf.constant([[[[1.0], [0.0]]]])
        self.assertTrue(bool(tf.math.is_finite(binary_focal_dice_loss(truth, truth))))


if __name__ == "__main__":
    unittest.main()
