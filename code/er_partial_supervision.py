"""Conservative background proposals and masked loss for incomplete ER labels.

Automatic negatives remain weak labels, never certified manual ground truth.
"""
import numpy as np


def exterior_background(er, fluo, positive):
    """Use full planes, never cropped patch edges, to identify exterior candidates.

    Fixed rule: both smoothed channels below their plane's 20th percentile,
    connected to image edge, eroded four pixels, >12 pixels from teacher ER.
    Enclosed dark nuclei cannot qualify solely because they are dark.
    """
    from scipy import ndimage as ndi
    if er.shape != fluo.shape or er.shape != positive.shape or er.ndim != 2:
        raise ValueError("Aligned 2D planes required")
    if not np.isfinite(er).all() or not np.isfinite(fluo).all():
        raise ValueError("Nonfinite source")
    smooth = [ndi.gaussian_filter(a.astype(np.float32), 2) for a in (er, fluo)]
    thresholds = [float(np.percentile(a, 20)) for a in smooth]
    low = (smooth[0] < thresholds[0]) & (smooth[1] < thresholds[1])
    labels, _ = ndi.label(low)
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    exterior = np.isin(labels, edge_ids[edge_ids != 0])
    exterior = ndi.binary_erosion(exterior, iterations=4, border_value=0)
    # No positive pixels means the distance transform has no meaningful seed.
    if positive.any():
        exterior &= ndi.distance_transform_edt(~positive.astype(bool)) > 12
    return exterior & ~positive.astype(bool), thresholds


def masked_bce_dice(labels, predictions):
    """TF loss, labels 0 negative / 1 positive / 255 ignore, same NHWC shape.

    Unknown pixels contribute neither BCE nor either Dice term. Empty batches
    contribute zero; training should separately enforce both supervised classes.
    """
    import tensorflow as tf
    y = tf.cast(labels, tf.float32)
    p = tf.clip_by_value(tf.cast(predictions, tf.float32), 1e-7, 1-1e-7)
    tf.debugging.assert_equal(tf.shape(y), tf.shape(p))
    tf.debugging.assert_equal(tf.reduce_all((y == 0) | (y == 1) | (y == 255)), True)
    valid = tf.cast(y != 255, tf.float32)
    target = tf.cast(y == 1, tf.float32)
    n = tf.reduce_sum(valid)
    bce = tf.math.divide_no_nan(tf.reduce_sum(valid *
          (-target * tf.math.log(p) - (1-target)*tf.math.log1p(-p))), n)
    dice = (2*tf.reduce_sum(valid*target*p)+1e-6) / (
        tf.reduce_sum(valid*target)+tf.reduce_sum(valid*p)+1e-6)
    return tf.where(n > 0, bce + 1-dice, 0.)
