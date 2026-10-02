"""Train and evaluate the geometry-preserving 2D U-Net baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_split import build_fold_split, split_from_fold_ids
from evaluation import (
    FullImageValidationCallback,
    choose_threshold,
    metric_rows_from_predictions,
    predict_pairs,
    write_metric_report,
)
from losses import (
    binary_crossentropy_dice_loss,
    binary_crossentropy_tversky_06_loss,
    soft_dice_coefficient,
)
from model import build_unet_2d, build_unet_2d_three_pool
from patch_dataset import RandomPatchSequence
from pipeline_config import PipelineConfig, get_target_spec
from tiff_io import pair_tiff_directories, validate_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an independent 2D semantic-segmentation U-Net."
    )
    parser.add_argument("--target", required=True, choices=("mito", "ld", "nuc"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory containing label_ref and code (default: project root).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mito-positive-patch-probability",
        type=float,
        default=0.6,
        help="Mito-only foreground-centred patch probability (baseline: 0.6).",
    )
    parser.add_argument(
        "--mito-loss",
        choices=("bce_dice", "bce_tversky_06"),
        default="bce_dice",
        help="Mito-only loss ablation; baseline is bce_dice.",
    )
    parser.add_argument(
        "--mito-architecture",
        choices=("baseline", "three_pool"),
        default="baseline",
        help="Mito-only thin-structure ablation; the default remains frozen U-Net.",
    )
    parser.add_argument(
        "--expected-mito-fold-map",
        type=Path,
        help="Mito-only frozen baseline split.json; fail if fold assignments differ.",
    )
    parser.add_argument(
        "--require-new-run-dir",
        action="store_true",
        help="Fail if the target fold output directory already exists.",
    )
    parser.add_argument(
        "--defer-mito-outer-test",
        action="store_true",
        help="Train/calibrate Mito using validation only; score outer test later.",
    )
    return parser.parse_args()


def _write_split_manifest(
    path: Path,
    target: str,
    fold: int,
    num_folds: int,
    fold_ids: dict[str, int],
    train_names: list[str],
    validation_names: list[str],
    test_names: list[str],
) -> None:
    manifest = {
        "target": target,
        "fold": fold,
        "num_folds": num_folds,
        "fold_ids": fold_ids,
        "train": train_names,
        "validation": validation_names,
        "test": test_names,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if not 0 <= args.mito_positive_patch_probability <= 1:
        raise ValueError("mito-positive-patch-probability must be between 0 and 1")
    if args.target != "mito" and (
        args.mito_positive_patch_probability != 0.6
        or args.mito_loss != "bce_dice"
        or args.mito_architecture != "baseline"
        or args.expected_mito_fold_map is not None
        or args.defer_mito_outer_test
    ):
        raise ValueError("Mito experiment options cannot be used for LD or Nuc")
    if args.defer_mito_outer_test and args.expected_mito_fold_map is None:
        raise ValueError("Deferred Mito outer test requires a frozen fold map")
    if args.mito_loss != "bce_dice" and args.mito_positive_patch_probability != 0.6:
        raise ValueError("Do not combine the Mito loss and sampling ablations")
    if args.mito_architecture != "baseline" and (
        args.mito_loss != "bce_dice"
        or args.mito_positive_patch_probability != 0.6
    ):
        raise ValueError("Do not combine the Mito architecture with prior ablations")
    config = PipelineConfig(
        patch_size=args.patch_size,
        overlap=args.overlap,
        random_seed=args.seed,
    )
    config.validate()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    target_spec = get_target_spec(args.target)
    project_root = args.project_root.resolve()
    pairs = pair_tiff_directories(
        project_root / target_spec.input_relative_dir,
        project_root / target_spec.label_relative_dir,
    )
    if args.expected_mito_fold_map is not None:
        # Apply the already frozen map directly. Recomputing it would read
        # outer-test mask foreground fractions during candidate setup.
        expected = json.loads(args.expected_mito_fold_map.read_text(encoding="utf-8"))
        if expected.get("target") != "mito" or expected.get("num_folds") != args.num_folds:
            raise ValueError("Expected fold map is not a matching Mito split")
        fold_ids = expected["fold_ids"]
        split = split_from_fold_ids(pairs, fold_ids, args.fold, args.num_folds)
    else:
        split, fold_ids = build_fold_split(
            pairs,
            fold=args.fold,
            num_folds=args.num_folds,
            random_seed=args.seed,
        )
    # During the candidate's validation-only stage, do not even open the
    # outer-test mask. The frozen fold map assigns it using filenames alone.
    report_pairs = split.train + split.validation if args.defer_mito_outer_test else pairs
    dataset_report = validate_pairs(report_pairs)

    run_dir = args.output_root.resolve() / args.target / f"fold_{args.fold}"
    if args.require_new_run_dir and run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.target == "mito":
        # Record the single experimental factor next to each fold's split and metrics.
        with (run_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target": args.target,
                    "fold": args.fold,
                    "num_folds": args.num_folds,
                    "seed": args.seed,
                    "positive_patch_probability": args.mito_positive_patch_probability,
                    "loss": args.mito_loss,
                    "architecture": args.mito_architecture,
                    "patch_size": args.patch_size,
                    "overlap": args.overlap,
                    "batch_size": args.batch_size,
                    "steps_per_epoch": args.steps_per_epoch,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "outer_test_deferred": args.defer_mito_outer_test,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
    with (run_dir / "dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_report, handle, indent=2)
    _write_split_manifest(
        run_dir / "split.json",
        args.target,
        args.fold,
        args.num_folds,
        fold_ids,
        [pair.filename for pair in split.train],
        [pair.filename for pair in split.validation],
        [pair.filename for pair in split.test],
    )

    train_sequence = RandomPatchSequence(
        split.train,
        batch_size=args.batch_size,
        patch_size=config.patch_size,
        steps_per_epoch=args.steps_per_epoch,
        lower_percentile=config.lower_percentile,
        upper_percentile=config.upper_percentile,
        positive_patch_probability=args.mito_positive_patch_probability,
        random_seed=config.random_seed,
        augment=True,
    )

    model = (
        build_unet_2d_three_pool()
        if args.mito_architecture == "three_pool"
        else build_unet_2d()
    )
    selected_loss = (
        binary_crossentropy_tversky_06_loss
        if args.mito_loss == "bce_tversky_06"
        else binary_crossentropy_dice_loss
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=selected_loss,
        metrics=[soft_dice_coefficient],
    )

    best_model_path = run_dir / "baseline_unet.keras"
    validation_callback = FullImageValidationCallback(
        split.validation,
        tile_size=config.patch_size,
        overlap=config.overlap,
        prediction_batch_size=args.prediction_batch_size,
        lower_percentile=config.lower_percentile,
        upper_percentile=config.upper_percentile,
        threshold=config.threshold,
    )
    callbacks: list[tf.keras.callbacks.Callback] = [
        validation_callback,
        tf.keras.callbacks.ModelCheckpoint(
            str(best_model_path),
            monitor="val_full_dice",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_full_dice",
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_full_dice",
            mode="max",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_history.csv")),
    ]
    model.fit(train_sequence, epochs=args.epochs, callbacks=callbacks)

    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    validation_predictions = predict_pairs(
        best_model,
        split.validation,
        config.patch_size,
        config.overlap,
        args.prediction_batch_size,
        config.lower_percentile,
        config.upper_percentile,
    )
    threshold, threshold_candidates = choose_threshold(validation_predictions)
    with (run_dir / "threshold.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"threshold": threshold, "candidates": threshold_candidates},
            handle,
            indent=2,
        )

    if args.defer_mito_outer_test:
        # Inner-validation model choice happens before outer-test evaluation.
        print("MITO_OUTER_TEST_DEFERRED: no outer-test metric was computed")
        return

    test_predictions = predict_pairs(
        best_model,
        split.test,
        config.patch_size,
        config.overlap,
        args.prediction_batch_size,
        config.lower_percentile,
        config.upper_percentile,
    )
    test_rows = metric_rows_from_predictions(test_predictions, threshold)
    summary = write_metric_report(run_dir, "test", test_rows)
    print(f"Best validation threshold: {threshold:.2f}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
