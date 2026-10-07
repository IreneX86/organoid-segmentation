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


@tf.keras.utils.register_keras_serializable(package="organoid_segmentation")
def binary_focal_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Symmetric gamma-2 focal BCE plus the unchanged baseline soft-Dice term.

    This is a single Mito ablation. There is no class-alpha or tunable focal
    weight: difficult pixels receive more emphasis solely through ``p_t``.
    Clipping protects logarithms without changing the model output itself.
    """

    true = tf.cast(y_true, tf.float32)
    predicted = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    safe = tf.clip_by_value(predicted, epsilon, 1.0 - epsilon)
    pixel_bce = -(true * tf.math.log(safe)
                  + (1.0 - true) * tf.math.log(1.0 - safe))
    p_true_class = true * predicted + (1.0 - true) * (1.0 - predicted)
    focal_bce = tf.reduce_mean(tf.square(1.0 - p_true_class) * pixel_bce)
    return focal_bce + (1.0 - soft_dice_coefficient(true, predicted))


@tf.keras.utils.register_keras_serializable(package="organoid_segmentation")
def binary_crossentropy_tversky_06_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """BCE plus mild precision-aware Tversky loss for one Mito ablation.

    Alpha=0.6 penalizes false positives a little more than beta=0.4 penalizes
    missed foreground. The baseline BCE and all other training settings stay
    unchanged; this is one predeclared alternative, not a parameter search.
    """

    true = tf.cast(y_true, tf.float32)
    predicted = tf.cast(y_pred, tf.float32)
    axes = tuple(range(1, len(true.shape)))
    true_positive = tf.reduce_sum(true * predicted, axis=axes)
    false_positive = tf.reduce_sum((1.0 - true) * predicted, axis=axes)
    false_negative = tf.reduce_sum(true * (1.0 - predicted), axis=axes)
    smooth = 1e-6
    tversky = (true_positive + smooth) / (
        true_positive + 0.6 * false_positive + 0.4 * false_negative + smooth
    )
    binary_crossentropy = tf.reduce_mean(tf.keras.losses.binary_crossentropy(true, predicted))
    return binary_crossentropy + (1.0 - tf.reduce_mean(tversky))
