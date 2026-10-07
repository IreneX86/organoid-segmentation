"""Export held-out Mito masks at predeclared thresholds after model training.

The training run has already selected its checkpoint and validation threshold.
This script never selects a threshold from outer-test labels or retrains a model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import tensorflow as tf

from evaluation import predict_pairs, write_metric_report
from pipeline_config import PipelineConfig, get_target_spec
from segmentation_metrics import calculate_binary_metrics
from tiff_io import pair_tiff_directories, write_mask_tiff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--write-test-report",
        action="store_true",
        help="For a deferred outer test, write metrics after the validation-only decision.",
    )
    parser.add_argument(
        "--candidate-decisions",
        type=Path,
        help="Required for a validation-only candidate's outer-test export.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    experiment_path = run_dir / "experiment_config.json"
    if experiment_path.is_file():
        with experiment_path.open(encoding="utf-8") as handle:
            experiment = json.load(handle)
        if experiment.get("outer_test_deferred") is True:
            if args.candidate_decisions is None or not args.write_test_report:
                raise ValueError("Deferred candidate requires locked decisions and test report")
            with args.candidate_decisions.open(encoding="utf-8") as handle:
                decisions = json.load(handle)
            fold = experiment["fold"]
            matching = [item for item in decisions.get("decisions", []) if item["fold"] == fold]
            expected_variant = (
                "hard_negative"
                if experiment.get("hard_negative_fraction") == 0.2
                and experiment.get("architecture") == "baseline"
                and experiment.get("positive_patch_probability") == 0.6
                and experiment.get("loss") == "bce_dice"
                else
                "focal_dice"
                if experiment.get("hard_negative_fraction", 0.0) == 0.0
                and experiment.get("architecture") == "baseline"
                and experiment.get("positive_patch_probability") == 0.6
                and experiment.get("loss") == "focal_dice"
                else
                "three_pool"
                if experiment.get("architecture") == "three_pool"
                and experiment.get("positive_patch_probability") == 0.6
                and experiment.get("loss") == "bce_dice"
                else
                "sampling_035"
                if experiment.get("positive_patch_probability") == 0.35
                and experiment.get("loss") == "bce_dice"
                else "tversky_06"
                if experiment.get("positive_patch_probability") == 0.6
                and experiment.get("loss") == "bce_tversky_06"
                else None
            )
            expected_protocol = (
                "mito_hard_negative_validation_gate_v1"
                if expected_variant == "hard_negative"
                else "mito_focal_dice_validation_gate_v1"
                if expected_variant == "focal_dice"
                else "mito_three_pool_validation_gate_v1"
                if expected_variant == "three_pool"
                else "mito_one_factor_validation_gate_v1"
            )
            if (decisions.get("protocol") != expected_protocol
                    or decisions.get("variant") != expected_variant
                    or len(matching) != 1
                    or matching[0]["selected_candidate"] is not True):
                raise ValueError("This fold was not selected by the validation-only gate")
    if args.write_test_report and any(
        (run_dir / name).exists() for name in ("test_per_image.csv", "test_summary.json")
    ):
        raise FileExistsError("Refusing to overwrite an existing outer-test report")
    with (run_dir / "split.json").open(encoding="utf-8") as handle:
        split = json.load(handle)
    if split.get("target") != "mito":
        raise ValueError("This export is restricted to Mito")
    with (run_dir / "threshold.json").open(encoding="utf-8") as handle:
        chosen_threshold = float(json.load(handle)["threshold"])

    spec = get_target_spec("mito")
    root = args.project_root.resolve()
    pairs = pair_tiff_directories(
        root / spec.input_relative_dir, root / spec.label_relative_dir
    )
    test_names = set(split["test"])
    test_pairs = [pair for pair in pairs if pair.filename in test_names]
    if len(test_pairs) != len(test_names):
        raise ValueError("Outer-test files do not match the frozen split")

    config = PipelineConfig()
    model = tf.keras.models.load_model(run_dir / "baseline_unet.keras", compile=False)
    predictions = predict_pairs(
        model, test_pairs, config.patch_size, config.overlap, 4,
        config.lower_percentile, config.upper_percentile,
    )
    rows = []
    report_rows = []
    for pair, probability, truth in predictions:
        row: dict[str, float | str] = {"filename": pair.filename}
        # Both policies are fixed before reading any test metric. The trained
        # checkpoint and threshold.json came only from validation data.
        for policy, threshold in (("validation", chosen_threshold), ("fixed_050", 0.50)):
            mask = (probability >= threshold).astype("uint8")
            write_mask_tiff(run_dir / "predictions" / policy / pair.filename, mask)
            metrics = calculate_binary_metrics(truth, mask)
            if policy == "validation":
                report_rows.append({"filename": pair.filename, **metrics.to_dict()})
            for name in ("dice", "iou", "precision", "recall"):
                row[f"{policy}_{name}"] = getattr(metrics, name)
        rows.append(row)
    with (run_dir / "threshold_policy_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.write_test_report:
        # This is deliberately separate from train.py's validation-only phase.
        # The chosen training policy must be fixed before invoking this option.
        write_metric_report(run_dir, "test", report_rows)
    print(f"Exported {len(rows)} outer-test masks for each threshold policy")


if __name__ == "__main__":
    main()
