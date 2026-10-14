"""Verify the LD loss ablation changes BCE weighting, not the Dice term."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import tensorflow as tf
    from losses import (
        binary_crossentropy_dice_loss,
        binary_focal_dice_loss,
        soft_dice_coefficient,
    )
except ImportError:
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is required for numerical loss checks")
class LDFocalLossTests(unittest.TestCase):
    def test_gamma_zero_is_baseline_bce_and_gamma_two_reuses_same_dice(self) -> None:
        true = tf.constant([[[[1.0], [0.0]], [[1.0], [0.0]]]])
        predicted = tf.constant([[[[0.7], [0.2]], [[0.15], [0.95]]]])
        epsilon = tf.keras.backend.epsilon()
        safe = tf.clip_by_value(predicted, epsilon, 1.0 - epsilon)
        pixel_bce = -(true * tf.math.log(safe)
                      + (1.0 - true) * tf.math.log(1.0 - safe))
        p_true = true * predicted + (1.0 - true) * (1.0 - predicted)
        shared_dice_loss = 1.0 - soft_dice_coefficient(true, predicted)

        # Gamma zero cancels the modulating factor, so the local reference
        # must reproduce exactly the frozen BCE + Dice objective.
        gamma_zero = tf.reduce_mean(pixel_bce) + shared_dice_loss
        self.assertAlmostEqual(float(gamma_zero),
                               float(binary_crossentropy_dice_loss(true, predicted)),
                               places=6)
        gamma_two = tf.reduce_mean(tf.square(1.0 - p_true) * pixel_bce)
        focal_only = binary_focal_dice_loss(true, predicted) - shared_dice_loss
        self.assertAlmostEqual(float(focal_only), float(gamma_two), places=6)

    def test_extreme_probabilities_have_finite_loss_and_gradient(self) -> None:
        true = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]])
        predicted = tf.Variable([[[[1e-6], [1.0 - 1e-6]],
                                   [[0.4], [0.6]]]])
        with tf.GradientTape() as tape:
            value = binary_focal_dice_loss(true, predicted)
        gradient = tape.gradient(value, predicted)
        self.assertTrue(bool(tf.math.is_finite(value)))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))


if __name__ == "__main__":
    unittest.main()
