"""Two-dimensional U-Net baseline for binary semantic segmentation."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def _convolution_block(
    inputs: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    outputs = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    outputs = layers.Conv2D(filters, 3, padding="same", activation="relu")(outputs)
    outputs = layers.BatchNormalization()(outputs)
    if dropout_rate > 0:
        outputs = layers.Dropout(dropout_rate)(outputs)
    return outputs


def build_unet_2d(input_channels: int = 1, base_filters: int = 32) -> Model:
    """Build a U-Net that accepts any H×W divisible by 16.

    Training uses fixed-size patches, while inference uses overlapping patches.
    Keeping the spatial dimensions dynamic avoids any whole-image resize.
    """

    inputs = tf.keras.Input(shape=(None, None, input_channels), name="image")

    encoder_1 = _convolution_block(inputs, base_filters)
    encoder_2 = _convolution_block(layers.MaxPooling2D()(encoder_1), base_filters * 2)
    encoder_3 = _convolution_block(layers.MaxPooling2D()(encoder_2), base_filters * 4)
    encoder_4 = _convolution_block(layers.MaxPooling2D()(encoder_3), base_filters * 8)
    bottleneck = _convolution_block(
        layers.MaxPooling2D()(encoder_4), base_filters * 16, dropout_rate=0.4
    )

    decoder_4 = layers.UpSampling2D(interpolation="bilinear")(bottleneck)
    decoder_4 = _convolution_block(
        layers.Concatenate()([decoder_4, encoder_4]), base_filters * 8, dropout_rate=0.3
    )
    decoder_3 = layers.UpSampling2D(interpolation="bilinear")(decoder_4)
    decoder_3 = _convolution_block(
        layers.Concatenate()([decoder_3, encoder_3]), base_filters * 4, dropout_rate=0.2
    )
    decoder_2 = layers.UpSampling2D(interpolation="bilinear")(decoder_3)
    decoder_2 = _convolution_block(
        layers.Concatenate()([decoder_2, encoder_2]), base_filters * 2, dropout_rate=0.1
    )
    decoder_1 = layers.UpSampling2D(interpolation="bilinear")(decoder_2)
    decoder_1 = _convolution_block(
        layers.Concatenate()([decoder_1, encoder_1]), base_filters
    )

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="foreground_probability")(
        decoder_1
    )
    return Model(inputs=inputs, outputs=outputs, name="unet_2d_baseline")


def build_unet_2d_three_pool(input_channels: int = 1, base_filters: int = 32) -> Model:
    """Mito candidate with one fewer pooling stage and an unchanged 3×3 block.

    A typical labelled Mito pixel is within roughly two pixels of an edge.
    Keeping the deepest feature map at 1/8 rather than 1/16 resolution tests
    whether the baseline's fourth downsampling loses thin-structure detail.
    The remaining blocks, skip connections, and output grid are unchanged.
    """

    inputs = tf.keras.Input(shape=(None, None, input_channels), name="image")
    encoder_1 = _convolution_block(inputs, base_filters)
    encoder_2 = _convolution_block(layers.MaxPooling2D()(encoder_1), base_filters * 2)
    encoder_3 = _convolution_block(layers.MaxPooling2D()(encoder_2), base_filters * 4)
    bottleneck = _convolution_block(
        layers.MaxPooling2D()(encoder_3), base_filters * 8, dropout_rate=0.4
    )
    decoder_3 = _convolution_block(
        layers.Concatenate()([
            layers.UpSampling2D(interpolation="bilinear")(bottleneck), encoder_3,
        ]),
        base_filters * 4,
        dropout_rate=0.2,
    )
    decoder_2 = _convolution_block(
        layers.Concatenate()([
            layers.UpSampling2D(interpolation="bilinear")(decoder_3), encoder_2,
        ]),
        base_filters * 2,
        dropout_rate=0.1,
    )
    decoder_1 = _convolution_block(
        layers.Concatenate()([
            layers.UpSampling2D(interpolation="bilinear")(decoder_2), encoder_1,
        ]),
        base_filters,
    )
    output = layers.Conv2D(
        1, 1, activation="sigmoid", name="foreground_probability"
    )(decoder_1)
    return Model(inputs=inputs, outputs=output, name="unet_2d_three_pool")
