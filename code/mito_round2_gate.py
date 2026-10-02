"""Select a Mito candidate using only each fold's validation set.

This deliberately never opens an outer-test TIFF or mask. Decisions are
written once, before any candidate outer-test export is permitted.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def decide(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    """Apply the predeclared mean/tail/sparse-foreground validation rule."""

    baseline = {row["filename"]: row for row in baseline_rows}
    candidate = {row["filename"]: row for row in candidate_rows}
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("Validation image sets differ")
    names = sorted(baseline)
    for name in names:
        for row in (baseline[name], candidate[name]):
            value = float(row["dice"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Non-finite or invalid validation Dice: {name}")
        if abs(baseline[name]["foreground_fraction_true"]
               - candidate[name]["foreground_fraction_true"]) > 1e-10:
            raise ValueError(f"Validation label fraction mismatch: {name}")
    paired = [candidate[name]["dice"] - baseline[name]["dice"] for name in names]
    low_names = [name for name in names
                 if baseline[name]["foreground_fraction_true"] <= 0.05]
    mean_delta = statistics.mean(paired)
    low_delta = (statistics.mean(
        candidate[name]["dice"] - baseline[name]["dice"] for name in low_names
    ) if low_names else None)
    baseline_sd = statistics.stdev(baseline[name]["dice"] for name in names)
    candidate_sd = statistics.stdev(candidate[name]["dice"] for name in names)

    # An overall gain may be small if sparse images improve substantially.
    # Guard against large per-image losses and an unstable validation spread.
    promising = mean_delta >= 0.003 or (
        low_delta is not None and low_delta >= 0.010 and mean_delta >= 0
    )
    low_ok = low_delta is None or low_delta >= -0.005
    tail_ok = min(paired) >= -0.020
    spread_ok = candidate_sd - baseline_sd <= 0.015
    return {
        "baseline_validation_dice_mean": statistics.mean(
            baseline[name]["dice"] for name in names),
        "candidate_validation_dice_mean": statistics.mean(
            candidate[name]["dice"] for name in names),
        "mean_delta": mean_delta,
        "low_foreground_image_count": len(low_names),
        "low_foreground_mean_delta": low_delta,
        "worst_paired_delta": min(paired),
        "validation_std_delta": candidate_sd - baseline_sd,
        "selected_candidate": bool(promising and low_ok and tail_ok and spread_ok),
        "per_image": [
            {"filename": name,
             "foreground_fraction_true": baseline[name]["foreground_fraction_true"],
             "baseline_dice": baseline[name]["dice"],
             "candidate_dice": candidate[name]["dice"],
             "delta_dice": candidate[name]["dice"] - baseline[name]["dice"]}
            for name in names
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-model-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("three_pool", "hard_negative"),
        default="three_pool",
    )
    args = parser.parse_args()

    import tensorflow as tf
    from evaluation import metric_rows_from_predictions, predict_pairs
    from pipeline_config import PipelineConfig, get_target_spec
    from tiff_io import pair_tiff_directories

    root = args.project_root.resolve()
    spec = get_target_spec("mito")
    pairs = pair_tiff_directories(
        root / spec.input_relative_dir, root / spec.label_relative_dir
    )
    by_name = {pair.filename: pair for pair in pairs}
    config = PipelineConfig()
    decisions = []
    for fold in range(5):
        baseline = args.baseline_root / f"fold_{fold}"
        candidate = args.candidate_root / "mito" / f"fold_{fold}"
        baseline_split = json.loads((baseline / "split.json").read_text())
        candidate_split = json.loads((candidate / "split.json").read_text())
        if baseline_split != candidate_split:
            raise ValueError(f"Fold {fold} differs from frozen split")
        exp = json.loads((candidate / "experiment_config.json").read_text())
        expected_architecture = "three_pool" if args.variant == "three_pool" else "baseline"
        expected_hard_fraction = 0.2 if args.variant == "hard_negative" else 0.0
        if (exp.get("target") != "mito" or exp.get("fold") != fold
                or exp.get("architecture") != expected_architecture
                or exp.get("loss") != "bce_dice"
                or exp.get("positive_patch_probability") != 0.6
                or exp.get("hard_negative_fraction", 0.0) != expected_hard_fraction
                or exp.get("patch_size") != 512
                or exp.get("overlap") != 128
                or exp.get("batch_size") != 4
                or exp.get("steps_per_epoch") != 100
                or exp.get("epochs") != 100
                or exp.get("learning_rate") != 1e-4
                or exp.get("outer_test_deferred") is not True):
            raise ValueError(f"Fold {fold} is not the isolated {args.variant} candidate")
        if (candidate / "test_per_image.csv").exists() or (candidate / "predictions").exists():
            raise ValueError(f"Fold {fold} had premature outer-test evaluation")
        val_pairs = [by_name[name] for name in baseline_split["validation"]]
        fold_rows = []
        for role, model_path, threshold_path in (
            ("baseline", args.baseline_model_root / f"fold_{fold}" / "baseline_unet.keras",
             baseline / "threshold.json"),
            ("candidate", candidate / "baseline_unet.keras", candidate / "threshold.json"),
        ):
            model = tf.keras.models.load_model(model_path, compile=False)
            predictions = predict_pairs(
                model, val_pairs, config.patch_size, config.overlap, 4,
                config.lower_percentile, config.upper_percentile,
            )
            threshold = float(json.loads(threshold_path.read_text())["threshold"])
            fold_rows.append(metric_rows_from_predictions(predictions, threshold))
            del model
            tf.keras.backend.clear_session()
        result = decide(fold_rows[0], fold_rows[1])
        result["fold"] = fold
        decisions.append(result)
        print(f"fold={fold} mean_delta={result['mean_delta']:.6f} "
              f"low_delta={result['low_foreground_mean_delta']} "
              f"selected={result['selected_candidate']}", flush=True)

    report = {
        "protocol": ("mito_hard_negative_validation_gate_v1"
                     if args.variant == "hard_negative"
                     else "mito_three_pool_validation_gate_v1"),
        "variant": args.variant,
        "selection_rule": {
            "overall_mean_delta_at_least": 0.003,
            "or_low_foreground_delta_at_least": 0.010,
            "or_branch_requires_nonnegative_overall_mean": True,
            "low_foreground_delta_at_least": -0.005,
            "worst_paired_delta_at_least": -0.020,
            "validation_std_increase_at_most": 0.015,
        },
        "decisions": decisions,
    }
    path = args.candidate_root / "candidate_decisions.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"LOCKED_DECISIONS={path}", flush=True)


if __name__ == "__main__":
    main()
