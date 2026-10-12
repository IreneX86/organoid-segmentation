"""TensorFlow training sequence for the LD component-uniform candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ld_component_sampling import component_pixels, sample_component_uniform_patch
from patch_dataset import RandomPatchSequence, _augment_patch, _load_normalized_pair
from tiff_io import read_binary_mask


class ComponentUniformPatchSequence(RandomPatchSequence):
    """Baseline sequence with only its positive-center distribution changed."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        if kwargs.get("hard_negative_fraction", 0.0) != 0.0:
            raise ValueError("Hard-negative sampling is outside this LD protocol")
        super().__init__(*args, **kwargs)
        # `self.pairs` contains only the training split supplied by the fold
        # trainer. Validation and outer-test labels are not accessed here.
        self.components = {
            pair.filename: component_pixels(read_binary_mask(Path(pair.label_path)))
            for pair in self.pairs
        }

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
                str(pair.image_path), str(pair.label_path),
                self.lower_percentile, self.upper_percentile,
            )
            image_patch, mask_patch = sample_component_uniform_patch(
                image, mask, self.components[pair.filename],
                self.patch_size, rng, self.positive_patch_probability,
            )
            if self.augment:
                image_patch, mask_patch = _augment_patch(image_patch, mask_patch, rng)
            images[item_index, ..., 0] = image_patch
            masks[item_index, ..., 0] = mask_patch
        return images, masks
