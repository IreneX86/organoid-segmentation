"""Train a Nuc 2D prototype with same-batch internal validation only.

The frozen manifest names every training and validation TIFF and pins the raw
input/annotation bytes. Related views may occur on both sides of this split;
these results are diagnostics for model development, not an independent test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile

from pipeline_config import PipelineConfig, get_target_spec
from segmentation_metrics import calculate_binary_metrics, summarize_metric_rows
from tiff_io import (
    ImageLabelPair,
    pair_tiff_directories,
    validate_pairs,
    write_mask_tiff,
)


SOURCE_FILES = (
    "code/nuc_train_internal.py",
    "code/evaluation.py",
    "code/losses.py",
    "code/model/__init__.py",
    "code/model/unet.py",
    "code/patch_dataset.py",
    "code/pipeline_config.py",
    "code/segmentation_metrics.py",
    "code/sliding_window.py",
    "code/tiff_io.py",
)
PURPOSE = "same_batch_internal_diagnostic"
LOW_CONTRAST_LOCKED_SETTINGS = (0.5, 64.0, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_frozen_split(
    manifest_path: Path,
    pairs: list[ImageLabelPair],
) -> tuple[dict[str, object], list[ImageLabelPair], list[ImageLabelPair], list[dict]]:
    """Check all names, source hashes, and true-negative labels before fitting."""

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Nuc split manifest must be a JSON object")
    if (manifest.get("version") != 1 or manifest.get("target") != "nuc"
            or manifest.get("purpose") != PURPOSE):
        raise ValueError("Expected a version-1 Nuc same-batch internal split")

    names = {pair.filename for pair in pairs}
    train_names = manifest.get("train")
    validation_names = manifest.get("validation")
    if (not isinstance(train_names, list) or not isinstance(validation_names, list)
            or not train_names or not validation_names
            or any(not isinstance(name, str) for name in train_names + validation_names)
            or len(set(train_names)) != len(train_names)
            or len(set(validation_names)) != len(validation_names)
            or set(train_names) & set(validation_names)
            or set(train_names) | set(validation_names) != names):
        raise ValueError("Frozen Nuc train/validation names must uniquely cover every pair")

    input_hashes = manifest.get("input_sha256")
    label_hashes = manifest.get("label_sha256")
    if (not isinstance(input_hashes, dict) or set(input_hashes) != names
            or not isinstance(label_hashes, dict) or set(label_hashes) != names):
        raise ValueError("Frozen Nuc source hashes must cover every paired TIFF")
    for pair in pairs:
        if input_hashes[pair.filename] != _sha256(pair.image_path):
            raise ValueError(f"Nuc input source hash mismatch: {pair.filename}")
        if label_hashes[pair.filename] != _sha256(pair.label_path):
            raise ValueError(f"Nuc label source hash mismatch: {pair.filename}")

    report = validate_pairs(pairs)
    empty_names = {
        str(item["filename"]) for item in report if item["foreground_fraction"] == 0
    }
    expected_empty = manifest.get("true_negative_filenames")
    if (not isinstance(expected_empty, list)
            or any(not isinstance(name, str) for name in expected_empty)
            or len(set(expected_empty)) != len(expected_empty)
            or set(expected_empty) != empty_names):
        raise ValueError("Frozen Nuc true negatives differ from the actual empty masks")

    by_name = {pair.filename: pair for pair in pairs}
    training = [by_name[name] for name in train_names]
    validation = [by_name[name] for name in validation_names]
    nonempty_names = names - empty_names
    if not nonempty_names.intersection(train_names):
        raise ValueError("Nuc training split needs at least one nonempty mask")
    if not nonempty_names.intersection(validation_names):
        raise ValueError("Nuc validation split needs at least one nonempty mask")
    return manifest, training, validation, report


def choose_nuc_threshold(
    predictions: list[tuple[ImageLabelPair, np.ndarray, np.ndarray]],
) -> tuple[float, list[dict[str, float]]]:
    """Use nonempty Dice; ties prefer 0.50 proximity and then lower cutoff."""

    if not predictions or any(not np.any(truth) for _, _, truth in predictions):
        raise ValueError("Nuc threshold selection needs only nonempty validation masks")
    candidates: list[dict[str, float]] = []
    for index in range(17):
        threshold = round(0.10 + 0.05 * index, 2)
        rows = [
            {"filename": pair.filename,
             **calculate_binary_metrics(truth, probability >= threshold).to_dict()}
            for pair, probability, truth in predictions
        ]
        summary = summarize_metric_rows(rows)
        candidates.append({
            "threshold": threshold,
            "nonempty_dice_mean": summary["dice_mean"],
            "nonempty_iou_mean": summary["iou_mean"],
        })
    chosen = max(
        candidates,
        key=lambda item: (
            item["nonempty_dice_mean"],
            -abs(round(item["threshold"] * 100) - 50),
            -round(item["threshold"] * 100),
        ),
    )
    return chosen["threshold"], candidates


def validate_low_contrast_settings(
    probability: float, sigma: float, alpha_min: float,
) -> None:
    """Allow the default-off path or the predeclared Nuc candidate only."""

    if (not np.isfinite(probability) or not 0 <= probability <= 1
            or not np.isfinite(sigma) or sigma <= 0
            or not np.isfinite(alpha_min) or not 0 <= alpha_min <= 1):
        raise ValueError("Nuc low contrast settings are outside their allowed ranges")
    if probability and (probability, sigma, alpha_min) != LOW_CONTRAST_LOCKED_SETTINGS:
        raise ValueError("Nuc low contrast candidate must use locked 0.5/64/0.5 settings")


def _save_validation_results(
    predictions: list[tuple[ImageLabelPair, np.ndarray, np.ndarray]],
    threshold: float,
    output_dir: Path,
) -> dict[str, object]:
    """Keep foreground overlap and empty-mask false positives separate."""

    artifact_dir = output_dir / "validation_artifacts"
    artifact_dir.mkdir(exist_ok=False)
    all_rows: list[dict[str, object]] = []
    nonempty_rows: dict[str, list[dict[str, float | str]]] = {
        "fixed_050": [], "selected": [],
    }
    empty_fp_pixels: dict[str, list[int]] = {"fixed_050": [], "selected": []}
    for pair, probability, truth in predictions:
        if probability.shape != truth.shape or not np.isfinite(probability).all():
            raise ValueError(f"Invalid full-grid Nuc prediction: {pair.filename}")
        probability_path = artifact_dir / "probabilities" / pair.filename
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(probability_path, probability.astype(np.float32), metadata=None)
        is_empty = not bool(np.any(truth))
        row: dict[str, object] = {
            "filename": pair.filename,
            "truth_group": "true_negative" if is_empty else "nonempty",
        }
        for policy, cutoff in (("fixed_050", 0.5), ("selected", threshold)):
            predicted = (probability >= cutoff).astype(np.uint8)
            write_mask_tiff(
                artifact_dir / "predictions" / policy / pair.filename, predicted
            )
            fp_pixels = int(np.logical_and(truth == 0, predicted != 0).sum())
            row[f"{policy}_false_positive_pixels"] = fp_pixels
            if is_empty:
                empty_fp_pixels[policy].append(fp_pixels)
                row[f"{policy}_dice"] = ""
                row[f"{policy}_iou"] = ""
                row[f"{policy}_precision"] = ""
                row[f"{policy}_recall"] = ""
            else:
                metrics = calculate_binary_metrics(truth, predicted)
                nonempty_rows[policy].append({
                    "filename": pair.filename, **metrics.to_dict(),
                })
                for metric in ("dice", "iou", "precision", "recall"):
                    row[f"{policy}_{metric}"] = getattr(metrics, metric)
        all_rows.append(row)

    with (output_dir / "validation_per_image.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    nonempty = {
        policy: summarize_metric_rows(rows)
        for policy, rows in nonempty_rows.items()
    }
    empty = {
        policy: {
            "image_count": len(counts),
            "false_positive_pixels_total": int(sum(counts)),
            "false_positive_pixels_mean": (
                float(np.mean(counts)) if counts else None
            ),
            "images_with_any_false_positive": sum(count > 0 for count in counts),
        }
        for policy, counts in empty_fp_pixels.items()
    }
    summary: dict[str, object] = {
        "target": "nuc",
        "purpose": PURPOSE,
        "interpretation": "same-batch internal diagnostic; not independent generalization",
        "threshold": threshold,
        "nonempty_validation": nonempty,
        "true_negative_validation": empty,
    }
    _write_json(output_dir / "validation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--positive-patch-probability", type=float, default=0.6)
    parser.add_argument("--spatial-illumination-probability", type=float, default=0.0)
    parser.add_argument("--spatial-illumination-max-offset", type=float, default=0.0)
    parser.add_argument("--low-contrast-probability", type=float, default=0.0)
    parser.add_argument("--low-contrast-sigma", type=float, default=64.0)
    parser.add_argument("--low-contrast-alpha-min", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.batch_size < 1 or args.prediction_batch_size < 1
            or args.steps_per_epoch < 1 or args.epochs < 1
            or args.learning_rate <= 0
            or not 0 <= args.positive_patch_probability <= 1
            or not 0 <= args.spatial_illumination_probability <= 1
            or not 0 <= args.spatial_illumination_max_offset <= 1
            or (args.spatial_illumination_probability > 0
                and args.spatial_illumination_max_offset == 0)):
        raise ValueError("Nuc training parameters are outside their allowed ranges")
    validate_low_contrast_settings(
        args.low_contrast_probability, args.low_contrast_sigma,
        args.low_contrast_alpha_min,
    )
    if args.low_contrast_probability and args.spatial_illumination_probability:
        raise ValueError("Nuc low contrast and spatial illumination candidates cannot be combined")
    config = PipelineConfig(
        patch_size=args.patch_size, overlap=args.overlap, random_seed=args.seed
    )
    config.validate()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing Nuc run: {output_dir}")
    project_root = args.project_root.resolve()
    target = get_target_spec("nuc")
    input_dir = project_root / target.input_relative_dir
    label_dir = project_root / target.label_relative_dir
    protected_roots = [
        (project_root / "label_ref").resolve(),
        (project_root / "organoid_data").resolve(),
    ]
    if any(output_dir == root or root in output_dir.parents for root in protected_roots):
        raise ValueError("Nuc output directory must be outside raw and labelled data")
    pairs = pair_tiff_directories(input_dir, label_dir)
    split_path = args.split_manifest.resolve()
    _, training, validation, report = validate_frozen_split(split_path, pairs)
    foreground_by_name = {
        str(item["filename"]): float(item["foreground_fraction"]) for item in report
    }
    nonempty_validation = [
        pair for pair in validation if foreground_by_name[pair.filename] > 0
    ]

    # TensorFlow is imported only after all source and split checks pass.
    import tensorflow as tf

    from evaluation import (
        FullImageValidationCallback,
        predict_pairs,
    )
    from losses import binary_crossentropy_dice_loss, soft_dice_coefficient
    from model import build_unet_2d
    from patch_dataset import RandomPatchSequence

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    code_root = Path(__file__).resolve().parent.parent
    code_hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        source = code_root / name
        expected = _sha256(source)
        destination = output_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha256(destination) != expected:
            raise RuntimeError(f"Nuc source snapshot changed during copy: {name}")
        code_hashes[name] = expected
    shutil.copyfile(split_path, output_dir / "split.json")
    _write_json(output_dir / "dataset_report.json", {
        "pairs": report,
        "train_count": len(training),
        "validation_count": len(validation),
        "train_true_negative_count": sum(
            foreground_by_name[pair.filename] == 0 for pair in training
        ),
        "validation_true_negative_count": sum(
            foreground_by_name[pair.filename] == 0 for pair in validation
        ),
    })
    _write_json(output_dir / "experiment_config.json", {
        "target": "nuc", "purpose": PURPOSE, "split_manifest_sha256": _sha256(split_path),
        "code_sha256": code_hashes,
        "project_root": str(project_root), "input_dir": str(input_dir),
        "label_dir": str(label_dir), "seed": args.seed,
        "architecture": "unet_2d_baseline", "loss": "bce_dice",
        "patch_size": config.patch_size, "overlap": config.overlap,
        "batch_size": args.batch_size,
        "prediction_batch_size": args.prediction_batch_size,
        "steps_per_epoch": args.steps_per_epoch, "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "positive_patch_probability": args.positive_patch_probability,
        "spatial_illumination_probability": args.spatial_illumination_probability,
        "spatial_illumination_max_offset": args.spatial_illumination_max_offset,
        "low_contrast_probability": args.low_contrast_probability,
        "low_contrast_sigma": args.low_contrast_sigma,
        "low_contrast_alpha_min": args.low_contrast_alpha_min,
        "low_contrast_alpha_max": 1.0,
        "low_contrast_boundary": "reflect",
        "lower_percentile": config.lower_percentile,
        "upper_percentile": config.upper_percentile,
        "checkpoint_selection": "nonempty_validation_full_image_dice_at_0.5",
        "threshold_selection": "nonempty_validation_mean_per_image_dice",
        "python": platform.python_version(), "tensorflow": tf.__version__,
    })

    sequence = RandomPatchSequence(
        training, batch_size=args.batch_size, patch_size=config.patch_size,
        steps_per_epoch=args.steps_per_epoch,
        lower_percentile=config.lower_percentile,
        upper_percentile=config.upper_percentile,
        positive_patch_probability=args.positive_patch_probability,
        spatial_illumination_probability=args.spatial_illumination_probability,
        spatial_illumination_max_offset=args.spatial_illumination_max_offset,
        low_contrast_probability=args.low_contrast_probability,
        low_contrast_sigma=args.low_contrast_sigma,
        low_contrast_alpha_min=args.low_contrast_alpha_min,
        random_seed=args.seed, augment=True,
    )
    model = build_unet_2d()  # Fresh Nuc initialization; no Mito or LD weights.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=binary_crossentropy_dice_loss,
        metrics=[soft_dice_coefficient],
    )
    checkpoint = output_dir / "nuc_unet.keras"
    callbacks: list[tf.keras.callbacks.Callback] = [
        FullImageValidationCallback(
            nonempty_validation, tile_size=config.patch_size,
            overlap=config.overlap,
            prediction_batch_size=args.prediction_batch_size,
            lower_percentile=config.lower_percentile,
            upper_percentile=config.upper_percentile,
            threshold=0.5,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_full_dice", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_full_dice", mode="max", factor=0.5,
            patience=8, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_full_dice", mode="max", patience=20,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
    ]
    model.fit(sequence, epochs=args.epochs, callbacks=callbacks)
    if not checkpoint.is_file():
        raise RuntimeError("Nuc validation-selected checkpoint was not saved")
    with (output_dir / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    if not history or any("val_full_dice" not in row for row in history):
        raise RuntimeError("Nuc training history lacks full-image validation Dice")
    checkpoint_scores = np.asarray(
        [float(row["val_full_dice"]) for row in history], dtype=np.float64
    )
    if not np.isfinite(checkpoint_scores).all():
        raise RuntimeError("Nuc checkpoint selection metrics are not finite")
    best_record = history[int(np.argmax(checkpoint_scores))]
    best_epoch = int(best_record["epoch"]) + 1
    best_dice = float(best_record["val_full_dice"])

    best_model = tf.keras.models.load_model(checkpoint, compile=False)
    predictions = predict_pairs(
        best_model, validation, config.patch_size, config.overlap,
        args.prediction_batch_size, config.lower_percentile,
        config.upper_percentile,
    )
    nonempty_predictions = [
        item for item in predictions if foreground_by_name[item[0].filename] > 0
    ]
    threshold, candidates = choose_nuc_threshold(nonempty_predictions)
    summary = _save_validation_results(predictions, threshold, output_dir)
    _write_json(output_dir / "threshold.json", {
        "threshold": threshold,
        "selection_basis": "nonempty_validation_mean_per_image_dice",
        "tie_break": "closest_to_0.50_then_lower",
        "candidates": candidates,
        "selected_nonempty_dice_mean": summary["nonempty_validation"]["selected"]["dice_mean"],
        "fixed_050_nonempty_dice_mean": summary["nonempty_validation"]["fixed_050"]["dice_mean"],
        "best_epoch_1based": best_epoch,
        "checkpoint_metric": "val_full_dice_on_nonempty_at_0.50",
        "checkpoint_metric_value": best_dice,
        "checkpoint_sha256": _sha256(checkpoint),
    })
    _write_json(output_dir / "run_manifest.json", {
        "target": "nuc", "purpose": PURPOSE,
        "best_epoch_1based": best_epoch,
        "checkpoint_metric": "val_full_dice_on_nonempty_at_0.50",
        "checkpoint_metric_value": best_dice,
        "split_sha256": _sha256(output_dir / "split.json"),
        "checkpoint_sha256": _sha256(checkpoint),
        "history_sha256": _sha256(output_dir / "training_history.csv"),
        "threshold_sha256": _sha256(output_dir / "threshold.json"),
        "validation_summary_sha256": _sha256(output_dir / "validation_summary.json"),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
