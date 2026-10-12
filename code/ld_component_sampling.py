"""LD-only component-uniform positive crop sampling.

The frozen baseline sampler is untouched. This candidate changes only the
distribution of the center pixel in the existing positive-crop branch.
Connected regions are semantic-mask proxies, not individual droplet IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class ComponentPixels:
    """Foreground pixels grouped by eight-connected component in one mask."""

    coordinates: np.ndarray  # (N, 2), int32, ordered by component
    offsets: np.ndarray  # (component_count + 1,), int64

    @property
    def count(self) -> int:
        return len(self.offsets) - 1


def component_pixels(mask: np.ndarray) -> ComponentPixels:
    """Build a compact index from one training mask, including tiny regions."""

    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional binary mask")
    binary = np.asarray(mask, dtype=bool)
    labeled, count = ndi.label(binary, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return ComponentPixels(
            coordinates=np.empty((0, 2), dtype=np.int32),
            offsets=np.zeros(1, dtype=np.int64),
        )
    rows, columns = np.nonzero(binary)
    ids = labeled[rows, columns]
    order = np.argsort(ids, kind="stable")
    sizes = np.bincount(ids, minlength=count + 1)[1:]
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    coordinates = np.stack((rows[order], columns[order]), axis=1).astype(np.int32)
    if offsets[-1] != len(coordinates) or np.any(np.diff(offsets) == 0):
        raise RuntimeError("Component pixel index is inconsistent")
    return ComponentPixels(coordinates=coordinates, offsets=offsets)


def sample_component_center(
    components: ComponentPixels, rng: np.random.Generator
) -> tuple[int, int]:
    """Give each nonempty component equal probability, then a uniform pixel."""

    if components.count == 0:
        raise ValueError("Cannot draw a positive center from an empty mask")
    component = int(rng.integers(components.count))
    position = int(rng.integers(components.offsets[component], components.offsets[component + 1]))
    row, column = components.coordinates[position]
    return int(row), int(column)


def _pad_to_patch(
    image: np.ndarray, mask: np.ndarray, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the baseline's right/bottom padding for undersized images."""

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


def sample_component_uniform_patch(
    image: np.ndarray,
    mask: np.ndarray,
    components: ComponentPixels,
    patch_size: int,
    rng: np.random.Generator,
    positive_patch_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the frozen baseline crop geometry and 0.60/0.40 branch split."""

    if not 0 <= positive_patch_probability <= 1:
        raise ValueError("positive_patch_probability must be in [0, 1]")
    image, mask = _pad_to_patch(image, mask, patch_size)
    max_y = image.shape[0] - patch_size
    max_x = image.shape[1] - patch_size
    draw = rng.random()
    if components.count and draw < positive_patch_probability:
        center_y, center_x = sample_component_center(components, rng)
        y0 = int(np.clip(center_y - patch_size // 2, 0, max_y))
        x0 = int(np.clip(center_x - patch_size // 2, 0, max_x))
    else:
        y0 = int(rng.integers(max_y + 1))
        x0 = int(rng.integers(max_x + 1))
    return (
        image[y0 : y0 + patch_size, x0 : x0 + patch_size].copy(),
        mask[y0 : y0 + patch_size, x0 : x0 + patch_size].copy(),
    )
