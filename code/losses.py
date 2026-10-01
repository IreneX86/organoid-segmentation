"""Keras losses for binary semantic segmentation."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="organoid_segmentation")
def soft_dice_coefficient(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """Compute soft Dice over each image, then average the batch."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = tuple(range(1, len(y_true.shape)))
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    return tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))


@tf.keras.utils.register_keras_serializable(package="organoid_segmentation")
def binary_crossentropy_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combine pixelwise binary cross-entropy and region-overlap Dice loss."""

    binary_crossentropy = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )
    return binary_crossentropy + (1.0 - soft_dice_coefficient(y_true, y_pred))

