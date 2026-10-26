"""Exploratory Golgi U-Net training with a fixed organoid-level holdout.

The 14 manual labels come from only two spheroids. This script never uses the
older U-Net predictions as training labels and makes no independent-test claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from evaluation import FullImageValidationCallback, metric_rows_from_predictions, predict_pairs, write_metric_report
from losses import binary_crossentropy_dice_loss, soft_dice_coefficient
from model import build_unet_2d
from patch_dataset import RandomPatchSequence
from tiff_io import pair_tiff_directories, validate_pairs


# Source attribution is locked by exact raw TIFF SHA256 in the run manifest.
VALIDATION = {"mask_029.tif", "mask_052.tif", "mask_075.tif"}  # cell_16472_NZ_150
TRAIN = {
    "mask_004.tif", "mask_005.tif", "mask_010.tif", "mask_020.tif",
    "mask_033.tif", "mask_038.tif", "mask_046.tif", "mask_076.tif",
    "mask_098.tif", "mask_111.tif", "mask_112.tif",
}  # cell_16471_NZ_150


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive directory creation prevents an accidental overwrite on rerun.
    args.output_dir.mkdir(parents=True, exist_ok=False)

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != 1:
        raise RuntimeError(f"Expected exactly one GPU, found {physical}")
    tf.config.experimental.set_memory_growth(physical[0], True)

    pairs = pair_tiff_directories(args.input_dir, args.label_dir)
    names = {pair.filename for pair in pairs}
    if names != TRAIN | VALIDATION:
        raise ValueError(f"Unexpected Golgi pairing: missing={sorted((TRAIN | VALIDATION)-names)}, extra={sorted(names-(TRAIN | VALIDATION))}")
    train = [pair for pair in pairs if pair.filename in TRAIN]
    validation = [pair for pair in pairs if pair.filename in VALIDATION]
    report = validate_pairs(pairs)
    with (args.output_dir / "dataset_report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    config = {
        "purpose": "exploratory organoid-grouped Golgi pilot; no independent test",
        "train_organoid": "cell_16471_NZ_150",
        "validation_organoid": "cell_16472_NZ_150",
        "train": sorted(TRAIN), "validation": sorted(VALIDATION),
        "sha256": {pair.filename: {"input": sha256(pair.image_path), "mask": sha256(pair.label_path)} for pair in pairs},
        "model": "2D U-Net base_filters=32", "loss": "BCE + soft Dice",
        "patch_size": 512, "overlap": 128, "batch_size": 4,
        "steps_per_epoch": 40, "max_epochs": 60, "positive_patch_probability": 0.6,
        "normalization_percentiles": [1.0, 99.8], "learning_rate": 1e-4,
        "checkpoint_monitor": "full-image validation Dice at threshold 0.5",
        "early_stopping_patience": 12, "seed": 42,
        "python": sys.version, "tensorflow": tf.__version__, "numpy": np.__version__,
    }
    with (args.output_dir / "config.json").open("x") as stream:
        json.dump(config, stream, indent=2)

    sequence = RandomPatchSequence(
        train, batch_size=4, patch_size=512, steps_per_epoch=40,
        lower_percentile=1.0, upper_percentile=99.8,
        positive_patch_probability=0.6, random_seed=42, augment=True,
    )
    model = build_unet_2d()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=binary_crossentropy_dice_loss,
        metrics=[soft_dice_coefficient],
    )
    checkpoint = args.output_dir / "golgi_unet.keras"
    callbacks = [
        FullImageValidationCallback(validation, 512, 128, 4, 1.0, 99.8, threshold=0.5),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_full_dice", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_full_dice", mode="max", factor=0.5, patience=6, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_full_dice", mode="max", patience=12, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(args.output_dir / "training_history.csv")),
    ]
    model.fit(sequence, epochs=60, callbacks=callbacks, verbose=2)

    best = tf.keras.models.load_model(checkpoint, compile=False)
    predictions = predict_pairs(best, validation, 512, 128, 4, 1.0, 99.8)
    rows = metric_rows_from_predictions(predictions, threshold=0.5)
    summary = write_metric_report(args.output_dir, "validation_fixed_050", rows)
    # Validation metrics are exploratory because there is only one held-out organoid.
    print("GOLGI_PILOT_COMPLETE", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
