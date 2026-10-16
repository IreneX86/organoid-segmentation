"""Paired Nuc validation under one frozen synthetic low-contrast perturbation.

This is a same-batch diagnostic, not a checkpoint/threshold selection step or
an independent estimate of target-domain accuracy. The 34 validation masks
are used only to measure predictions from already-frozen checkpoints. No
target ``organoid_data`` image or label is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from nuc_train_internal import validate_frozen_split
from pipeline_config import get_target_spec
from segmentation_metrics import calculate_binary_metrics
from sliding_window import predict_sliding_window
from tiff_io import (
    ImageLabelPair,
    normalize_image,
    pair_tiff_directories,
    read_binary_mask,
    read_tiff_2d,
)


# These are the original Nuc split and baseline checkpoint, frozen before the
# low-contrast candidate existed. In particular, this script never searches
# validation scores to choose a different checkpoint.
FROZEN_SPLIT_SHA256 = "eb1e1be161865e3e92cf93e30dcac7a8287462a670a4068aa97f17a525c8bc54"
FROZEN_BASELINE_SHA256 = "c1191f00e327d6f73acc73f202a3a8e44895cb669bbfa2c6a1f3f664cc986b9e"
PURPOSE = "same_batch_synthetic_contrast_stress_diagnostic"
THRESHOLD = 0.50
SIGMA_PIXELS = 64.0
ALPHA = 0.50
TILE_SIZE = 512
OVERLAP = 128
PREDICTION_BATCH_SIZE = 4
NORMALIZATION_PERCENTILES = (1.0, 99.8)
METRICS = ("dice", "iou", "precision", "recall")
CSV_COLUMNS = (
    "filename", "truth_group",
    "baseline_dice", "baseline_iou", "baseline_precision", "baseline_recall",
    "baseline_false_positive_pixels",
    "candidate_dice", "candidate_iou", "candidate_precision", "candidate_recall",
    "candidate_false_positive_pixels",
    "paired_dice_delta", "paired_iou_delta", "paired_precision_delta",
    "paired_recall_delta",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_provenance(
    split_manifest: Path,
    baseline_model: Path,
    candidate_model: Path,
) -> dict[str, str]:
    """Refuse altered sources/checkpoints or a non-protocol candidate run."""

    split_hash = _sha256(split_manifest)
    if split_hash != FROZEN_SPLIT_SHA256:
        raise ValueError("Nuc split is not the frozen internal split")
    baseline_hash = _sha256(baseline_model)
    if baseline_hash != FROZEN_BASELINE_SHA256:
        raise ValueError("Baseline checkpoint differs from the frozen Nuc model")
    candidate_hash = _sha256(candidate_model)
    if candidate_hash == baseline_hash:
        raise ValueError("Candidate checkpoint is the frozen baseline checkpoint")

    # The candidate model must come directly from a completed Nuc training
    # run. Checking the saved run metadata prevents a same-name stale model or
    # a changed split from being evaluated as the preregistered candidate.
    run_dir = candidate_model.parent
    run_manifest_path = run_dir / "run_manifest.json"
    config_path = run_dir / "experiment_config.json"
    saved_split_path = run_dir / "split.json"
    run_manifest = _read_json(run_manifest_path)
    config = _read_json(config_path)
    if (run_manifest.get("target") != "nuc"
            or run_manifest.get("purpose") != "same_batch_internal_diagnostic"
            or _sha256(saved_split_path) != split_hash
            or run_manifest.get("split_sha256") != split_hash
            or run_manifest.get("checkpoint_sha256") != candidate_hash
            or config.get("split_manifest_sha256") != split_hash):
        raise ValueError("Candidate checkpoint/split provenance does not match")
    expected_config: dict[str, object] = {
        "target": "nuc",
        "purpose": "same_batch_internal_diagnostic",
        "architecture": "unet_2d_baseline",
        "loss": "bce_dice",
        "patch_size": TILE_SIZE,
        "overlap": OVERLAP,
        "batch_size": 4,
        "steps_per_epoch": 100,
        "epochs": 100,
        "learning_rate": 1e-4,
        "positive_patch_probability": 0.60,
        "seed": 42,
        "lower_percentile": NORMALIZATION_PERCENTILES[0],
        "upper_percentile": NORMALIZATION_PERCENTILES[1],
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
        "checkpoint_selection": "nonempty_validation_full_image_dice_at_0.5",
        "threshold_selection": "nonempty_validation_mean_per_image_dice",
        "low_contrast_probability": 0.50,
        "low_contrast_sigma": SIGMA_PIXELS,
        "low_contrast_alpha_min": 0.50,
        "low_contrast_alpha_max": 1.00,
        "low_contrast_boundary": "reflect",
        "spatial_illumination_probability": 0.0,
        "spatial_illumination_max_offset": 0.0,
    }
    for key, expected in expected_config.items():
        if config.get(key, 0.0 if key == "spatial_illumination_probability" else None) != expected:
            raise ValueError(
                f"Candidate experiment_config {key} differs from locked protocol"
            )
    return {
        "split_manifest_sha256": split_hash,
        "baseline_checkpoint_sha256": baseline_hash,
        "candidate_checkpoint_sha256": candidate_hash,
        "candidate_run_manifest_sha256": _sha256(run_manifest_path),
        "candidate_experiment_config_sha256": _sha256(config_path),
        "baseline_model": str(baseline_model.resolve()),
        "candidate_model": str(candidate_model.resolve()),
    }


def _stress_image(normalized_image: np.ndarray) -> np.ndarray:
    """Apply the one predeclared full-image perturbation after normalization."""

    from patch_dataset import gaussian_blur_reflect

    image = np.asarray(normalized_image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Stress input must be a finite 2D normalized image")
    blur = gaussian_blur_reflect(image, sigma=SIGMA_PIXELS)
    if blur.shape != image.shape or not np.isfinite(blur).all():
        raise ValueError("Gaussian blur returned an invalid image")
    stressed = blur + ALPHA * (image - blur)
    return np.ascontiguousarray(np.clip(stressed, 0.0, 1.0), dtype=np.float32)


def _predict_model(
    model: object,
    validation: list[ImageLabelPair],
) -> dict[str, dict[str, float | int | str]]:
    """Evaluate one frozen model, without looking at the other model's scores."""

    rows: dict[str, dict[str, float | int | str]] = {}
    for pair in validation:
        image = read_tiff_2d(pair.image_path)
        truth = read_binary_mask(pair.label_path).astype(bool)
        if image.shape != truth.shape:
            raise ValueError(f"Image/mask grid mismatch: {pair.filename}")
        normalized = normalize_image(image, *NORMALIZATION_PERCENTILES)
        stressed = _stress_image(normalized)
        probability = predict_sliding_window(
            model, stressed, tile_size=TILE_SIZE, overlap=OVERLAP,
            prediction_batch_size=PREDICTION_BATCH_SIZE,
        )
        if (probability.shape != truth.shape
                or not np.isfinite(probability).all()):
            raise ValueError(f"Invalid full-grid prediction: {pair.filename}")
        predicted = probability >= THRESHOLD
        fp_count = int(np.logical_and(~truth, predicted).sum())
        if not truth.any():
            rows[pair.filename] = {
                "truth_group": "true_negative", "false_positive_pixels": fp_count,
            }
        else:
            metrics = calculate_binary_metrics(truth, predicted)
            rows[pair.filename] = {
                "truth_group": "nonempty", "false_positive_pixels": fp_count,
                **{name: float(getattr(metrics, name)) for name in METRICS},
            }
    return rows


def _distribution(values: list[float]) -> dict[str, float]:
    if not values or not all(np.isfinite(values)):
        raise ValueError("Expected finite, nonempty metric values")
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "p10": float(np.percentile(values, 10)),
    }


def _combine_results(
    baseline: dict[str, dict[str, float | int | str]],
    candidate: dict[str, dict[str, float | int | str]],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    if baseline.keys() != candidate.keys():
        raise ValueError("Baseline/candidate validation filenames differ")
    if len(baseline) != 34 or set(baseline) & {"nuc_005.tif"} != {"nuc_005.tif"}:
        raise ValueError("Expected exactly the 34 frozen Nuc validation images")
    rows: list[dict[str, object]] = []
    for name in sorted(baseline):
        old, new = baseline[name], candidate[name]
        if old["truth_group"] != new["truth_group"]:
            raise ValueError(f"Validation truth groups differ for {name}")
        row: dict[str, object] = {
            "filename": name, "truth_group": old["truth_group"],
            "baseline_false_positive_pixels": old["false_positive_pixels"],
            "candidate_false_positive_pixels": new["false_positive_pixels"],
        }
        for metric in METRICS:
            if old["truth_group"] == "nonempty":
                left = float(old[metric])
                right = float(new[metric])
                row[f"baseline_{metric}"] = left
                row[f"candidate_{metric}"] = right
                row[f"paired_{metric}_delta"] = right - left
            else:
                row[f"baseline_{metric}"] = ""
                row[f"candidate_{metric}"] = ""
                row[f"paired_{metric}_delta"] = ""
        rows.append(row)

    positives = [row for row in rows if row["truth_group"] == "nonempty"]
    negatives = [row for row in rows if row["truth_group"] == "true_negative"]
    if (len(positives) != 33 or len(negatives) != 1
            or negatives[0]["filename"] != "nuc_005.tif"):
        raise ValueError("Frozen Nuc validation groups must be 33 nonempty + nuc_005")
    metric_summary: dict[str, object] = {}
    for metric in METRICS:
        old_values = [float(row[f"baseline_{metric}"]) for row in positives]
        new_values = [float(row[f"candidate_{metric}"]) for row in positives]
        deltas = [new - old for old, new in zip(old_values, new_values, strict=True)]
        metric_summary[metric] = {
            "baseline": _distribution(old_values),
            "candidate": _distribution(new_values),
            "paired_delta": _distribution(deltas),
            "images_improved": sum(delta > 0 for delta in deltas),
            "images_worsened": sum(delta < 0 for delta in deltas),
            "images_unchanged": sum(delta == 0 for delta in deltas),
        }
    negative = negatives[0]
    true_negative_summary: dict[str, object] = {
        "filename": "nuc_005.tif",
        "baseline_false_positive_pixels": negative["baseline_false_positive_pixels"],
        "candidate_false_positive_pixels": negative["candidate_false_positive_pixels"],
        "paired_false_positive_pixels_delta": (
            int(negative["candidate_false_positive_pixels"])
            - int(negative["baseline_false_positive_pixels"])
        ),
    }
    return rows, {"image_count": 33, **metric_summary}, true_negative_summary


def run_stress(
    project_root: Path,
    split_manifest: Path,
    baseline_model: Path,
    candidate_model: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Verify frozen inputs, predict both models, then write a new small report."""

    project_root = project_root.resolve()
    split_manifest = split_manifest.resolve()
    baseline_model = baseline_model.resolve()
    candidate_model = candidate_model.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite stress results: {output_dir}")
    for protected in (project_root / "label_ref", project_root / "organoid_data"):
        protected = protected.resolve()
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError("Stress output must stay outside source data")

    target = get_target_spec("nuc")
    pairs = pair_tiff_directories(
        project_root / target.input_relative_dir,
        project_root / target.label_relative_dir,
    )
    _, _, validation, _ = validate_frozen_split(split_manifest, pairs)
    if len(validation) != 34:
        raise ValueError("Expected exactly 34 frozen validation pairs")
    provenance = _validate_provenance(
        split_manifest, baseline_model, candidate_model
    )

    # TensorFlow loads only after every source hash and run provenance check.
    import tensorflow as tf

    model = tf.keras.models.load_model(baseline_model, compile=False)
    baseline_rows = _predict_model(model, validation)
    del model
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(candidate_model, compile=False)
    candidate_rows = _predict_model(model, validation)
    del model
    tf.keras.backend.clear_session()
    rows, nonempty, true_negative = _combine_results(
        baseline_rows, candidate_rows
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "stress_validation_per_image.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    provenance["per_image_csv_sha256"] = _sha256(csv_path)
    summary: dict[str, object] = {
        "target": "nuc",
        "purpose": PURPOSE,
        "interpretation": (
            "Same-batch synthetic stress diagnostic on the frozen validation "
            "split. The trained-on perturbation may favor the candidate; this "
            "is not independent target-domain confirmation and selects no model."
        ),
        "transform": {
            "normalization_percentiles": list(NORMALIZATION_PERCENTILES),
            "sigma_pixels": SIGMA_PIXELS,
            "alpha": ALPHA,
            "boundary": "reflect",
            "threshold": THRESHOLD,
            "tile_size": TILE_SIZE,
            "overlap": OVERLAP,
            "prediction_batch_size": PREDICTION_BATCH_SIZE,
        },
        "provenance": provenance,
        "nonempty_validation": nonempty,
        "true_negative_validation": true_negative,
    }
    with (output_dir / "stress_validation_summary.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_stress(
        args.project_root, args.split_manifest, args.baseline_model,
        args.candidate_model, args.output_dir,
    )
    dice = summary["nonempty_validation"]["dice"]
    print(
        "NUC_STRESS_COMPLETE "
        f"baseline_mean_dice={dice['baseline']['mean']:.6f} "
        f"candidate_mean_dice={dice['candidate']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
