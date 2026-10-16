"""Online, geometry-preserving patch sampling for 2D U-Net training."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from tiff_io import ImageLabelPair, load_validated_pair, normalize_image


@lru_cache(maxsize=16)
def _load_normalized_pair(
    image_path: str,
    label_path: str,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    pair = ImageLabelPair(image_path=Path(image_path), label_path=Path(label_path))
    image, mask = load_validated_pair(pair)
    return (
        normalize_image(image, lower_percentile, upper_percentile),
        mask.astype(np.float32, copy=False),
    )


def _pad_to_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pad_height = max(0, patch_size - image.shape[0])
    pad_width = max(0, patch_size - image.shape[1])
    if pad_height == 0 and pad_width == 0:
        return image, mask
    image_mode = "reflect" if min(image.shape) > 1 else "edge"
    padding = ((0, pad_height), (0, pad_width))
    return (
        np.pad(image, padding, mode=image_mode),
        np.pad(mask, padding, mode="constant", constant_values=0),
    )


def _sample_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    rng: np.random.Generator,
    positive_patch_probability: float,
    hard_negative_fraction: float = 0.0,
    hard_negative_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_patch(image, mask, patch_size)
    max_y = image.shape[0] - patch_size
    max_x = image.shape[1] - patch_size

    # The default branch preserves the frozen baseline's random draw exactly.
    draw = rng.random()
    use_positive_center = mask.any() and draw < positive_patch_probability
    if use_positive_center:
        foreground_y, foreground_x = np.nonzero(mask)
        selected = int(rng.integers(len(foreground_y)))
        center_y = int(foreground_y[selected])
        center_x = int(foreground_x[selected])
        y0 = int(np.clip(center_y - patch_size // 2, 0, max_y))
        x0 = int(np.clip(center_x - patch_size // 2, 0, max_x))
    elif (hard_negative_fraction > 0
          and draw < positive_patch_probability + hard_negative_fraction
          and hard_negative_centers is not None
          and len(hard_negative_centers)):
        center_y, center_x = hard_negative_centers[
            int(rng.integers(len(hard_negative_centers)))
        ]
        y0 = int(np.clip(int(center_y) - patch_size // 2, 0, max_y))
        x0 = int(np.clip(int(center_x) - patch_size // 2, 0, max_x))
    else:
        y0 = int(rng.integers(max_y + 1))
        x0 = int(rng.integers(max_x + 1))

    image_patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size].copy()
    mask_patch = mask[y0 : y0 + patch_size, x0 : x0 + patch_size].copy()
    return image_patch, mask_patch


def _augment_patch(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply conservative augmentation without changing image/label alignment."""

    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    rotation_count = int(rng.integers(4))
    image = np.rot90(image, rotation_count)
    mask = np.rot90(mask, rotation_count)

    if rng.random() < 0.5:
        gamma = float(rng.uniform(0.85, 1.20))
        image = np.power(np.clip(image, 0.0, 1.0), gamma)
    if rng.random() < 0.5:
        scale = float(rng.uniform(0.90, 1.10))
        offset = float(rng.uniform(-0.05, 0.05))
        image = np.clip(image * scale + offset, 0.0, 1.0)
    if rng.random() < 0.35:
        sigma = float(rng.uniform(0.0, 0.025))
        noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    return (
        np.ascontiguousarray(image, dtype=np.float32),
        np.ascontiguousarray(mask, dtype=np.float32),
    )


def _augment_spatial_illumination(
    image: np.ndarray,
    rng: np.random.Generator,
    max_offset: float,
) -> np.ndarray:
    """Add a bounded, zero-centred illumination plane to a normalized image.

    Only image intensity changes; the mask and all pixel coordinates stay fixed.
    The two slopes each contribute at most half of ``max_offset``, so the
    combined change at any corner cannot exceed ``max_offset``.
    """

    slope_x, slope_y = rng.uniform(
        -max_offset / 2, max_offset / 2, size=2
    ).astype(np.float32)
    x = np.linspace(-1.0, 1.0, image.shape[1], dtype=np.float32)
    y = np.linspace(-1.0, 1.0, image.shape[0], dtype=np.float32)
    illumination = slope_x * x[None, :] + slope_y * y[:, None]
    return np.ascontiguousarray(np.clip(image + illumination, 0.0, 1.0), dtype=np.float32)


@lru_cache(maxsize=8)
def _gaussian_reflect_transfer(
    height: int, width: int, sigma: float,
) -> np.ndarray:
    """Frequency response for a Gaussian on a half-sample-reflected grid."""

    frequency_y = np.fft.fftfreq(2 * height)[:, None]
    frequency_x = np.fft.rfftfreq(2 * width)[None, :]
    transfer = np.exp(
        -2.0 * (np.pi * sigma) ** 2 * (frequency_y**2 + frequency_x**2)
    )
    transfer.flags.writeable = False
    return transfer


def gaussian_blur_reflect(image: np.ndarray, sigma: float = 64.0) -> np.ndarray:
    """Blur a 2D patch with reflect boundaries using only NumPy FFT.

    A symmetric 2x2 tiling gives the same half-sample reflect boundary as
    ``scipy.ndimage.gaussian_filter(mode="reflect")``. The Fourier Gaussian
    avoids a costly 513-tap spatial kernel at sigma 64. This path needs no
    optional package beyond the existing NumPy dependency.
    """

    if image.ndim != 2 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian blur needs a 2D image and positive finite sigma")
    height, width = image.shape
    mirrored = np.pad(
        np.asarray(image, dtype=np.float32),
        ((0, height), (0, width)), mode="symmetric",
    )
    spectrum = np.fft.rfftn(mirrored, axes=(0, 1))
    spectrum *= _gaussian_reflect_transfer(height, width, float(sigma))
    blurred = np.fft.irfftn(spectrum, s=mirrored.shape, axes=(0, 1))
    return np.ascontiguousarray(blurred[:height, :width], dtype=np.float32)


def _augment_low_local_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    alpha_min: float,
) -> np.ndarray:
    """Dampen detail while preserving the patch's Gaussian low-frequency part."""

    blurred = gaussian_blur_reflect(image, sigma)
    alpha = float(rng.uniform(alpha_min, 1.0))
    return np.ascontiguousarray(
        np.clip(blurred + alpha * (image - blurred), 0.0, 1.0),
        dtype=np.float32,
    )


class RandomPatchSequence(tf.keras.utils.Sequence):
    """Generate fresh random training patches for every epoch."""

    def __init__(
        self,
        pairs: list[ImageLabelPair],
        batch_size: int,
        patch_size: int,
        steps_per_epoch: int,
        lower_percentile: float,
        upper_percentile: float,
        positive_patch_probability: float = 0.6,
        hard_negative_fraction: float = 0.0,
        random_seed: int = 42,
        augment: bool = True,
        spatial_illumination_probability: float = 0.0,
        spatial_illumination_max_offset: float = 0.0,
        low_contrast_probability: float = 0.0,
        low_contrast_sigma: float = 64.0,
        low_contrast_alpha_min: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if not pairs:
            raise ValueError("RandomPatchSequence requires at least one image/label pair")
        if batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch_size and steps_per_epoch must be positive")
        if not 0 <= positive_patch_probability <= 1:
            raise ValueError("positive_patch_probability must be between 0 and 1")
        if not 0 <= hard_negative_fraction <= 1 - positive_patch_probability:
            raise ValueError("hard-negative fraction must fit the non-positive branch")
        if not 0 <= spatial_illumination_probability <= 1:
            raise ValueError("spatial illumination probability must be between 0 and 1")
        if not 0 <= spatial_illumination_max_offset <= 1:
            raise ValueError("spatial illumination offset must be between 0 and 1")
        if spatial_illumination_probability and not spatial_illumination_max_offset:
            raise ValueError("spatial illumination needs a positive maximum offset")
        if not 0 <= low_contrast_probability <= 1:
            raise ValueError("low contrast probability must be between 0 and 1")
        if not np.isfinite(low_contrast_sigma) or low_contrast_sigma <= 0:
            raise ValueError("low contrast sigma must be positive and finite")
        if not np.isfinite(low_contrast_alpha_min) or not 0 <= low_contrast_alpha_min <= 1:
            raise ValueError("low contrast alpha minimum must be between 0 and 1")

        self.pairs = pairs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.steps_per_epoch = steps_per_epoch
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.positive_patch_probability = positive_patch_probability
        self.hard_negative_fraction = hard_negative_fraction
        if hard_negative_fraction:
            # SciPy is an opt-in dependency; the frozen baseline needs none.
            from hard_negative_sampling import hard_negative_centers
            self._hard_negative_centers = hard_negative_centers
        self.random_seed = random_seed
        self.augment = augment
        self.spatial_illumination_probability = spatial_illumination_probability
        self.spatial_illumination_max_offset = spatial_illumination_max_offset
        self.low_contrast_probability = low_contrast_probability
        self.low_contrast_sigma = low_contrast_sigma
        self.low_contrast_alpha_min = low_contrast_alpha_min
        self.epoch_index = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(
            self.random_seed + self.epoch_index * self.steps_per_epoch + batch_index
        )
        images = np.empty(
            (self.batch_size, self.patch_size, self.patch_size, 1), dtype=np.float32
        )
        masks = np.empty_like(images)

        for item_index in range(self.batch_size):
            pair = self.pairs[int(rng.integers(len(self.pairs)))]
            image, mask = _load_normalized_pair(
                str(pair.image_path),
                str(pair.label_path),
                self.lower_percentile,
                self.upper_percentile,
            )
            centers = (
                self._hard_negative_centers(
                    str(pair.image_path), str(pair.label_path),
                    self.lower_percentile, self.upper_percentile,
                )
                if self.hard_negative_fraction else None
            )
            image_patch, mask_patch = _sample_patch(
                image,
                mask,
                self.patch_size,
                rng,
                self.positive_patch_probability,
                self.hard_negative_fraction,
                centers,
            )
            if self.augment:
                image_patch, mask_patch = _augment_patch(image_patch, mask_patch, rng)
                if self.spatial_illumination_probability:
                    # A separate stream leaves every baseline crop, geometric transform,
                    # and intensity draw unchanged for a paired comparison.
                    illumination_rng = np.random.default_rng(np.random.SeedSequence([
                        self.random_seed, self.epoch_index, batch_index,
                        item_index, 0x4E5543,
                    ]))
                    if illumination_rng.random() < self.spatial_illumination_probability:
                        image_patch = _augment_spatial_illumination(
                            image_patch, illumination_rng,
                            self.spatial_illumination_max_offset,
                        )
                if self.low_contrast_probability:
                    # This stream cannot perturb the frozen crop or earlier
                    # augmentation draws used by the baseline and other targets.
                    contrast_rng = np.random.default_rng(np.random.SeedSequence([
                        self.random_seed, self.epoch_index, batch_index,
                        item_index, 0x4E55434C,
                    ]))
                    if contrast_rng.random() < self.low_contrast_probability:
                        image_patch = _augment_low_local_contrast(
                            image_patch, contrast_rng, self.low_contrast_sigma,
                            self.low_contrast_alpha_min,
                        )
            images[item_index, ..., 0] = image_patch
            masks[item_index, ..., 0] = mask_patch
        return images, masks

    def on_epoch_end(self) -> None:
        self.epoch_index += 1
