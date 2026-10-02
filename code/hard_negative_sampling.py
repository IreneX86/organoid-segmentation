"""Training-fold-only bright-background centres for the locked Mito ablation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from tiff_io import ImageLabelPair, load_validated_pair, normalize_image


@lru_cache(maxsize=64)
def hard_negative_centers(
    image_path: str,
    label_path: str,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    """Return centres satisfying the fixed protocol for one training pair.

    The caller must pass only the current outer fold's training pairs. Both
    percentile cutoffs use this image's labelled background, in native pixels.
    """

    pair = ImageLabelPair(Path(image_path), Path(label_path))
    raw_image, mask = load_validated_pair(pair)
    image = normalize_image(raw_image, lower_percentile, upper_percentile)
    background = mask == 0
    if not np.any(background) or not np.any(~background):
        return np.empty((0, 2), dtype=np.int32)
    intensity_cutoff = np.percentile(image[background], 90)
    residual = image - gaussian_filter(image, sigma=12.0)
    residual_cutoff = np.percentile(residual[background], 90)
    eligible = (
        background
        & (distance_transform_edt(background) > 10.0)
        & (image >= intensity_cutoff)
        & (residual >= residual_cutoff)
    )
    return np.argwhere(eligible).astype(np.int32, copy=False)
