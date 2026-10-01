"""Strict TIFF input, label validation, normalization, and mask output."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class ImageLabelPair:
    """A raw 2D image and its same-named semantic mask."""

    image_path: Path
    label_path: Path

    @property
    def filename(self) -> str:
        return self.image_path.name


def list_tiff_files(directory: Path) -> list[Path]:
    """List TIFF files in deterministic, case-insensitive filename order."""

    if not directory.is_dir():
        raise FileNotFoundError(f"TIFF directory does not exist: {directory}")
    return sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in TIFF_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def pair_tiff_directories(image_dir: Path, label_dir: Path) -> list[ImageLabelPair]:
    """Pair input and label TIFFs by their complete filename.

    A mismatch is treated as a data error. Silent skipping would make an
    experiment difficult to reproduce and can hide annotation mistakes.
    """

    image_files = {path.name: path for path in list_tiff_files(image_dir)}
    label_files = {path.name: path for path in list_tiff_files(label_dir)}

    image_only = sorted(image_files.keys() - label_files.keys())
    label_only = sorted(label_files.keys() - image_files.keys())
    if image_only or label_only:
        details: list[str] = []
        if image_only:
            details.append(f"missing labels: {image_only}")
        if label_only:
            details.append(f"missing inputs: {label_only}")
        raise ValueError("Input/label filenames do not match; " + "; ".join(details))
    if not image_files:
        raise ValueError(f"No TIFF files found in {image_dir}")

    return [
        ImageLabelPair(image_files[name], label_files[name])
        for name in sorted(image_files, key=str.lower)
    ]


def read_tiff_2d(path: Path) -> np.ndarray:
    """Read one independent 2D TIFF without selecting or stacking Z slices."""

    array = np.asarray(tifffile.imread(path))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected one 2D image in {path}, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected numeric TIFF pixels in {path}, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"TIFF contains NaN or infinite values: {path}")
    return array


def read_binary_mask(path: Path) -> np.ndarray:
    """Read a semantic mask and convert its two levels to uint8 0/1.

    Both 0/1 and 0/255 files are accepted. Any two-level numeric encoding is
    accepted, but masks with more than two distinct values are rejected rather
    than guessed.
    """

    mask = read_tiff_2d(path)
    values = np.unique(mask)
    if values.size > 2:
        preview = values[:10].tolist()
        raise ValueError(
            f"Semantic mask must have at most two values: {path}; "
            f"found {values.size}, starting with {preview}"
        )
    if values.size == 1:
        return np.full(mask.shape, int(values[0] != 0), dtype=np.uint8)
    return (mask == values[-1]).astype(np.uint8)


def load_validated_pair(pair: ImageLabelPair) -> tuple[np.ndarray, np.ndarray]:
    """Load one image/mask pair and enforce exact pixel-grid agreement."""

    image = read_tiff_2d(pair.image_path)
    mask = read_binary_mask(pair.label_path)
    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch for {pair.filename}: image={image.shape}, mask={mask.shape}"
        )
    return image, mask


def validate_pairs(pairs: list[ImageLabelPair]) -> list[dict[str, int | float | str]]:
    """Fully inspect a dataset before training and return a serializable report."""

    report = []
    for pair in pairs:
        image, mask = load_validated_pair(pair)
        report.append(
            {
                "filename": pair.filename,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "image_dtype": str(image.dtype),
                "image_min": float(image.min()),
                "image_max": float(image.max()),
                "foreground_fraction": float(mask.mean()),
            }
        )
    return report


def normalize_image(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.8,
) -> np.ndarray:
    """Robustly normalize a 2D intensity image to float32 [0, 1]."""

    image_float = image.astype(np.float32, copy=False)
    low, high = np.percentile(image_float, [lower_percentile, upper_percentile])
    if high <= low:
        return np.zeros(image_float.shape, dtype=np.float32)
    normalized = (image_float - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def write_mask_tiff(
    output_path: Path,
    binary_mask: np.ndarray,
    foreground_value: int = 255,
) -> None:
    """Atomically save a 2D semantic mask as uint8 0/1 or 0/255 TIFF."""

    if binary_mask.ndim != 2:
        raise ValueError(f"Output mask must be 2D, got {binary_mask.shape}")
    if foreground_value not in {1, 255}:
        raise ValueError("foreground_value must be 1 or 255")
    unique_values = np.unique(binary_mask)
    if not set(unique_values.tolist()).issubset({0, 1}):
        raise ValueError(f"Internal output mask must contain only 0/1, got {unique_values}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = binary_mask.astype(np.uint8, copy=False) * foreground_value
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    tifffile.imwrite(
        temporary_path,
        encoded,
        photometric="minisblack",
        metadata=None,
    )
    os.replace(temporary_path, output_path)
