"""Reproducible image-level folds for independent 2D TIFF samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiff_io import ImageLabelPair, read_binary_mask


@dataclass(frozen=True)
class FoldSplit:
    train: list[ImageLabelPair]
    validation: list[ImageLabelPair]
    test: list[ImageLabelPair]


def _balanced_fold_ids(
    pairs: list[ImageLabelPair],
    num_folds: int,
    random_seed: int,
) -> dict[str, int]:
    """Distribute foreground-sparse and foreground-dense images across folds."""

    if num_folds < 3:
        raise ValueError("num_folds must be at least 3 to create train/val/test sets")
    if len(pairs) < num_folds:
        raise ValueError(f"Need at least {num_folds} images, found {len(pairs)}")

    rng = np.random.default_rng(random_seed)
    records = []
    for pair in pairs:
        foreground_fraction = float(read_binary_mask(pair.label_path).mean())
        records.append((foreground_fraction, float(rng.random()), pair.filename))
    records.sort()

    # A serpentine assignment balances the foreground distribution better than
    # taking contiguous blocks from the sorted list.
    fold_cycle = list(range(num_folds)) + list(range(num_folds - 1, -1, -1))
    return {
        filename: fold_cycle[index % len(fold_cycle)]
        for index, (_, _, filename) in enumerate(records)
    }


def build_fold_split(
    pairs: list[ImageLabelPair],
    fold: int,
    num_folds: int = 5,
    random_seed: int = 42,
) -> tuple[FoldSplit, dict[str, int]]:
    """Use one fold for testing, the next for validation, and the rest for training."""

    if not 0 <= fold < num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")
    fold_ids = _balanced_fold_ids(pairs, num_folds, random_seed)
    return split_from_fold_ids(pairs, fold_ids, fold, num_folds), fold_ids


def split_from_fold_ids(
    pairs: list[ImageLabelPair],
    fold_ids: dict[str, int],
    fold: int,
    num_folds: int,
) -> FoldSplit:
    """Apply a frozen fold map without recalculating it from manual masks."""

    if not 0 <= fold < num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")
    filenames = {pair.filename for pair in pairs}
    if set(fold_ids) != filenames:
        raise ValueError("Frozen fold map filenames differ from the paired TIFFs")
    if any(not isinstance(index, int) or not 0 <= index < num_folds for index in fold_ids.values()):
        raise ValueError("Frozen fold map contains an invalid fold index")
    validation_fold = (fold + 1) % num_folds

    train = [
        pair for pair in pairs if fold_ids[pair.filename] not in {fold, validation_fold}
    ]
    validation = [pair for pair in pairs if fold_ids[pair.filename] == validation_fold]
    test = [pair for pair in pairs if fold_ids[pair.filename] == fold]
    return FoldSplit(train=train, validation=validation, test=test)
