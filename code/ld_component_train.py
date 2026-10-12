"""Train one LD component-uniform sampling candidate without opening outer test.

This is a separate candidate entry point. The frozen LD baseline remains
unchanged, including its sampler and source-code hash manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile

from data_split import FoldSplit, split_from_fold_ids
from pipeline_config import PipelineConfig
from segmentation_metrics import calculate_binary_metrics, summarize_metric_rows
from tiff_io import (
    ImageLabelPair,
    pair_tiff_directories,
    read_tiff_2d,
    validate_pairs,
    write_mask_tiff,
)


CODE_RELATIVE_PATHS = (
    "code/data_split.py", "code/evaluation.py", "code/losses.py",
    "code/patch_dataset.py", "code/pipeline_config.py",
    "code/segmentation_metrics.py", "code/sliding_window.py",
    "code/tiff_io.py", "code/model/__init__.py", "code/model/unet.py",
    "code/ld_prepare_labels.py", "code/ld_make_fold_map.py",
    "code/ld_train.py", "code/ld_evaluate.py", "code/ld_cv_report.py",
    "code/ld_component_sampling.py", "code/ld_component_train.py",
    "code/ld_component_sequence.py",
    "code/ld_component_gate.py", "code/ld_component_evaluate.py",
    "code/ld_component_report.py",
    "experiments/ld_component_uniform_sampling_protocol.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256_manifest() -> dict[str, str]:
    """Hash every local source file that defines this LD experiment."""
    project_root = Path(__file__).resolve().parent.parent
    return {name: _sha256(project_root / name) for name in CODE_RELATIVE_PATHS}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    # Exclusive creation makes accidental re-use of a partially completed run
    # visible instead of silently changing a prior model-selection decision.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _save_validation_artifacts(
    predictions: list[tuple[ImageLabelPair, np.ndarray, np.ndarray]],
    threshold: float,
    run_dir: Path,
) -> Path:
    """Archive full-resolution validation outputs, never outer-test outputs."""
    if not predictions:
        raise ValueError("Validation artifacts require at least one image")
    destination = run_dir / "validation_artifacts"
    destination.mkdir(parents=True, exist_ok=False)
    manifest_files: list[dict[str, object]] = []
    selected_rows: list[dict[str, float | str]] = []
    policy_rows: list[dict[str, float | str]] = []
    for pair, probability, truth in predictions:
        image_shape = read_tiff_2d(pair.image_path).shape
        if probability.shape != image_shape or truth.shape != image_shape:
            raise ValueError(f"Validation grid differs from input for {pair.filename}")
        if not np.isfinite(probability).all():
            raise ValueError(f"Non-finite validation probabilities for {pair.filename}")
        probability_path = destination / "probabilities" / pair.filename
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_probability = probability.astype(np.float32)
        tifffile.imwrite(probability_path, encoded_probability, metadata=None)
        saved = read_tiff_2d(probability_path)
        if saved.shape != image_shape or not np.array_equal(saved, encoded_probability):
            raise RuntimeError(f"Validation probability read-back failed: {pair.filename}")

        file_record: dict[str, object] = {
            "filename": pair.filename,
            "shape": list(image_shape),
            "probability_sha256": _sha256(probability_path),
        }
        comparison: dict[str, float | str] = {"filename": pair.filename}
        for policy, cutoff in (("validation", threshold), ("fixed_050", 0.5)):
            predicted = (probability >= cutoff).astype(np.uint8)
            mask_path = destination / "predictions" / policy / pair.filename
            write_mask_tiff(mask_path, predicted)
            saved_mask = read_tiff_2d(mask_path)
            if (saved_mask.shape != image_shape
                    or not np.array_equal(saved_mask, predicted * 255)):
                raise RuntimeError(f"Validation mask read-back failed: {pair.filename}")
            file_record[f"{policy}_mask_sha256"] = _sha256(mask_path)
            metrics = calculate_binary_metrics(truth, predicted)
            if policy == "validation":
                selected_rows.append({"filename": pair.filename, **metrics.to_dict()})
            for metric in ("dice", "iou", "precision", "recall"):
                comparison[f"{policy}_{metric}"] = getattr(metrics, metric)
        policy_rows.append(comparison)
        manifest_files.append(file_record)

    with (destination / "validation_per_image.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    _write_json(destination / "validation_summary.json", summarize_metric_rows(selected_rows))
    with (destination / "threshold_policy_per_image.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(policy_rows[0]))
        writer.writeheader()
        writer.writerows(policy_rows)
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, {
        "target": "ld", "threshold": threshold,
        "image_count": len(predictions), "files": manifest_files,
    })
    return manifest_path


def _read_frozen_map(path: Path, filenames: set[str], num_folds: int) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("target") != "ld" or data.get("num_folds") != num_folds:
        raise ValueError("Frozen fold map must identify LD and the requested fold count")
    fold_ids = data.get("fold_ids")
    if not isinstance(fold_ids, dict) or set(fold_ids) != filenames:
        raise ValueError("Frozen fold-map filenames differ from paired LD TIFFs")
    if any(type(index) is not int or not 0 <= index < num_folds for index in fold_ids.values()):
        raise ValueError("Frozen fold map contains an invalid fold index")
    if set(fold_ids.values()) != set(range(num_folds)):
        raise ValueError("Every outer fold must contain at least one image")
    groups = data.get("groups")
    if groups is not None:
        if not isinstance(groups, dict) or set(groups) != filenames:
            raise ValueError("Frozen image groups must cover exactly the paired TIFFs")
        group_folds: dict[str, int] = {}
        for filename, group in groups.items():
            group = str(group)
            if group in group_folds and group_folds[group] != fold_ids[filename]:
                raise ValueError(f"Near-duplicate group {group!r} crosses outer folds")
            group_folds[group] = fold_ids[filename]
    suspect_groups = data.get("suspect_groups", [])
    if not isinstance(suspect_groups, list):
        raise ValueError("Frozen suspect_groups must be a list")
    for group in suspect_groups:
        if (not isinstance(group, list) or len(group) < 2
                or any(name not in filenames for name in group)):
            raise ValueError("Frozen suspect group contains unknown or too few images")
        if len({fold_ids[name] for name in group}) != 1:
            raise ValueError("A near-duplicate suspect group crosses outer folds")
    input_hashes = data.get("input_sha256")
    if input_hashes is not None and (
        not isinstance(input_hashes, dict) or set(input_hashes) != filenames
    ):
        raise ValueError("Frozen input SHA256 map must cover exactly the paired TIFFs")
    return fold_ids


def _validate_train_validation_only(
    split: FoldSplit, records: dict[str, dict[str, object]]
) -> list[dict[str, int | float | str]]:
    """Check fit/validation masks while leaving outer-test mask bytes unread."""

    for pair in split.train + split.validation:
        if _sha256(pair.label_path) != records[pair.filename]["derived_mask_sha256"]:
            raise ValueError(f"Derived label changed since preparation: {pair.filename}")
    return validate_pairs(split.train + split.validation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True,
                        help="Versioned, binary LD labels; never original Photoshop masks.")
    parser.add_argument("--label-manifest", type=Path, required=True,
                        help="Provenance manifest for the derived 0/1 labels.")
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--positive-patch-probability", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Import TensorFlow only for actual training. The frozen-map preflight and
    # its unit tests remain lightweight and do not initialize a GPU.
    import tensorflow as tf

    from evaluation import FullImageValidationCallback, choose_threshold, predict_pairs
    from losses import binary_crossentropy_dice_loss, soft_dice_coefficient
    from model import build_unet_2d
    from ld_component_sequence import ComponentUniformPatchSequence

    if args.num_folds != 5:
        raise ValueError("The LD study uses a frozen five-fold outer split")
    if not 0 <= args.fold < args.num_folds:
        raise ValueError("fold must be in [0, 4]")
    if not 0 <= args.positive_patch_probability <= 1:
        raise ValueError("positive-patch-probability must be in [0, 1]")
    if args.batch_size < 1 or args.prediction_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.steps_per_epoch < 1 or args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("steps, epochs and learning rate must be positive")
    # Any change to these values is a different experiment, not the single
    # factor predeclared in the prospective component-sampling protocol.
    fixed = {
        "patch_size": 512, "overlap": 128, "batch_size": 4,
        "prediction_batch_size": 4, "steps_per_epoch": 100,
        "epochs": 100, "learning_rate": 1e-4,
        "positive_patch_probability": 0.6, "seed": 42,
    }
    if any(getattr(args, name) != expected for name, expected in fixed.items()):
        raise ValueError("Candidate scientific settings differ from the locked LD protocol")

    config = PipelineConfig(patch_size=args.patch_size, overlap=args.overlap,
                            random_seed=args.seed)
    config.validate()
    input_dir = args.input_dir.resolve()
    label_dir = args.label_dir.resolve()
    original_mask_sibling = (input_dir.parent / "LD_output").resolve()
    if input_dir == label_dir or label_dir == original_mask_sibling:
        raise ValueError("Use an independent derived-label directory, not original LD_output")
    label_manifest = args.label_manifest.resolve()
    fold_map = args.fold_map.resolve()
    if not label_manifest.is_file() or not fold_map.is_file():
        raise FileNotFoundError("Both the label provenance manifest and fold map are required")

    # pair_tiff_directories only lists names. The outer-test label contents are
    # never read here, including during dataset validation and fold assignment.
    pairs = pair_tiff_directories(input_dir, label_dir)
    fold_ids = _read_frozen_map(fold_map, {pair.filename for pair in pairs}, args.num_folds)
    frozen_data = json.loads(fold_map.read_text(encoding="utf-8"))
    if "input_sha256" in frozen_data:
        for pair in pairs:
            if _sha256(pair.image_path) != frozen_data["input_sha256"][pair.filename]:
                raise ValueError(f"Input TIFF changed since fold-map freezing: {pair.filename}")
    split = split_from_fold_ids(pairs, fold_ids, args.fold, args.num_folds)
    if not split.train or not split.validation or not split.test:
        raise ValueError("Every fold needs non-empty train, validation and test sets")
    derived = json.loads(label_manifest.read_text(encoding="utf-8"))
    if (derived.get("rule_id") != "ld_antialias_midpoint_v1"
            or derived.get("label_semantics") != {"background": 0, "foreground": 1}):
        raise ValueError("Expected the frozen LD midpoint 0/1 label definition")
    manifest_files = derived.get("files")
    if not isinstance(manifest_files, list):
        raise ValueError("Derived-label manifest has no file records")
    records = {item["filename"]: item for item in manifest_files}
    if len(records) != len(manifest_files) or set(records) != set(fold_ids):
        raise ValueError("Derived-label manifest filenames do not match the frozen dataset")
    # The only label-content preflight intentionally excludes split.test.
    dataset_report = _validate_train_validation_only(split, records)
    code_sha256 = _code_sha256_manifest()

    run_dir = args.output_root.resolve() / "ld" / f"fold_{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=False)
    # Preserve a small, exact source snapshot alongside the checkpoint so a
    # downloaded run remains auditable even if the working tree later changes.
    project_root = Path(__file__).resolve().parent.parent
    for name, expected in code_sha256.items():
        destination = run_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_root / name, destination)
        if _sha256(destination) != expected:
            raise RuntimeError(f"Source changed while archiving this run: {name}")
    seed = args.seed + args.fold
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    experiment_config: dict[str, object] = {
        "target": "ld", "fold": args.fold, "num_folds": args.num_folds,
        "input_dir": str(input_dir), "label_dir": str(label_dir),
        "label_manifest": str(label_manifest),
        "label_manifest_sha256": _sha256(label_manifest),
        "fold_map": str(fold_map), "fold_map_sha256": _sha256(fold_map),
        "code_sha256": code_sha256,
        "seed_base": args.seed, "seed": seed,
        "architecture": "unet_2d_baseline",
        "candidate": "component_uniform_positive_center_v1",
        "protocol_sha256": code_sha256[
            "experiments/ld_component_uniform_sampling_protocol.md"
        ],
        "loss": "bce_dice", "patch_size": config.patch_size,
        "overlap": config.overlap, "batch_size": args.batch_size,
        "prediction_batch_size": args.prediction_batch_size,
        "steps_per_epoch": args.steps_per_epoch, "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "positive_patch_probability": args.positive_patch_probability,
        "positive_center_policy": "uniform_8_connected_training_component_then_pixel",
        "lower_percentile": config.lower_percentile,
        "upper_percentile": config.upper_percentile,
        "checkpoint_selection": "validation_full_image_dice_at_0.5",
        "threshold_selection": "validation_mean_per_image_dice",
        "outer_test_deferred": True,
    }
    _write_json(run_dir / "experiment_config.json", experiment_config)
    _write_json(run_dir / "split.json", {
        "target": "ld", "fold": args.fold, "num_folds": args.num_folds,
        "fold_ids": fold_ids,
        "train": [pair.filename for pair in split.train],
        "validation": [pair.filename for pair in split.validation],
        "test": [pair.filename for pair in split.test],
    })
    _write_json(run_dir / "dataset_report.json", {"train_validation": dataset_report})

    sequence = ComponentUniformPatchSequence(
        split.train, batch_size=args.batch_size, patch_size=config.patch_size,
        steps_per_epoch=args.steps_per_epoch,
        lower_percentile=config.lower_percentile,
        upper_percentile=config.upper_percentile,
        positive_patch_probability=args.positive_patch_probability,
        random_seed=seed, augment=True,
    )
    component_counts_path = run_dir / "training_component_counts.json"
    _write_json(component_counts_path, {
        "scope": "training_masks_only",
        "eight_connected_components_per_image": {
            filename: table.count for filename, table in sequence.components.items()
        },
    })
    model = build_unet_2d()  # Always initialize LD weights from scratch.
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  loss=binary_crossentropy_dice_loss,
                  metrics=[soft_dice_coefficient])
    checkpoint = run_dir / "component_unet.keras"
    callbacks: list[tf.keras.callbacks.Callback] = [
        FullImageValidationCallback(
            split.validation, tile_size=config.patch_size, overlap=config.overlap,
            prediction_batch_size=args.prediction_batch_size,
            lower_percentile=config.lower_percentile,
            upper_percentile=config.upper_percentile,
            threshold=config.threshold,
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
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_history.csv")),
    ]
    model.fit(sequence, epochs=args.epochs, callbacks=callbacks)
    if not checkpoint.is_file():
        raise RuntimeError("No validation-selected checkpoint was saved")

    best_model = tf.keras.models.load_model(checkpoint, compile=False)
    validation_predictions = predict_pairs(
        best_model, split.validation, config.patch_size, config.overlap,
        args.prediction_batch_size, config.lower_percentile,
        config.upper_percentile,
    )
    threshold, candidates = choose_threshold(validation_predictions)
    threshold_path = run_dir / "threshold.json"
    _write_json(threshold_path, {"threshold": threshold, "candidates": candidates})
    validation_manifest = _save_validation_artifacts(
        validation_predictions, threshold, run_dir
    )
    # This lock is checked by the separate outer-test command. It is produced
    # before any outer-test mask is opened and binds the model and threshold.
    _write_json(run_dir / "selection.lock.json", {
        "target": "ld", "fold": args.fold, "fold_map_sha256": _sha256(fold_map),
        "label_manifest_sha256": _sha256(label_manifest),
        "code_sha256": code_sha256,
        "training_component_counts_sha256": _sha256(component_counts_path),
        "experiment_config_sha256": _sha256(run_dir / "experiment_config.json"),
        "split_sha256": _sha256(run_dir / "split.json"),
        "checkpoint_sha256": _sha256(checkpoint),
        "threshold_sha256": _sha256(threshold_path),
        "validation_artifacts_manifest_sha256": _sha256(validation_manifest),
        "threshold": threshold,
    })
    print(f"LD_COMPONENT_VALIDATION_LOCKED fold={args.fold} threshold={threshold:.2f}")
    print("LD_OUTER_TEST_DEFERRED: no outer-test label was opened or scored")


if __name__ == "__main__":
    main()
