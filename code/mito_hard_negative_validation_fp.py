"""Post-decision FP/recall diagnostics on inner validation images only.

This script requires locked decisions and never opens an outer-test mask. Its
descriptive results cannot revise the predeclared fold selections.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt

from evaluation import predict_pairs
from pipeline_config import PipelineConfig, get_target_spec
from segmentation_metrics import calculate_binary_metrics
from tiff_io import pair_tiff_directories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-model-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    decisions = json.loads((args.candidate_root / "candidate_decisions.json").read_text())
    if decisions.get("protocol") != "mito_hard_negative_validation_gate_v1":
        raise ValueError("Hard-negative fold decisions must already be locked")
    if len(decisions["decisions"]) != 5:
        raise ValueError("Expected five locked decisions")
    spec = get_target_spec("mito")
    pairs = pair_tiff_directories(
        args.project_root / spec.input_relative_dir,
        args.project_root / spec.label_relative_dir,
    )
    by_name = {pair.filename: pair for pair in pairs}
    config = PipelineConfig()
    rows = []
    for fold in range(5):
        split = json.loads((args.candidate_root / "mito" / f"fold_{fold}" / "split.json").read_text())
        frozen = json.loads((args.baseline_root / f"fold_{fold}" / "split.json").read_text())
        if split != frozen or set(split["validation"]) & set(split["test"]):
            raise ValueError(f"Fold {fold} changed or leaked the frozen split")
        validation_pairs = [by_name[name] for name in split["validation"]]
        for role, model_path, threshold_path in (
            ("baseline", args.baseline_model_root / f"fold_{fold}" / "baseline_unet.keras",
             args.baseline_root / f"fold_{fold}" / "threshold.json"),
            ("candidate", args.candidate_root / "mito" / f"fold_{fold}" / "baseline_unet.keras",
             args.candidate_root / "mito" / f"fold_{fold}" / "threshold.json"),
        ):
            threshold = float(json.loads(threshold_path.read_text())["threshold"])
            model = tf.keras.models.load_model(model_path, compile=False)
            predictions = predict_pairs(
                model, validation_pairs, config.patch_size, config.overlap, 4,
                config.lower_percentile, config.upper_percentile,
            )
            for pair, probability, truth_array in predictions:
                truth = truth_array.astype(bool)
                pred = probability >= threshold
                metric = calculate_binary_metrics(truth, pred)
                fp = ~truth & pred
                rows.append({
                    "fold": fold,
                    "filename": pair.filename,
                    "role": role,
                    "foreground_fraction": float(truth.mean()),
                    "threshold": threshold,
                    "dice": metric.dice,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "fp_pixels": int(fp.sum()),
                    "distant_fp_pixels": int(np.count_nonzero(
                        fp & (distance_transform_edt(~truth) > 10)
                    )),
                })
            del model
            tf.keras.backend.clear_session()
    keyed = {(row["fold"], row["filename"], row["role"]): row for row in rows}
    if len(keyed) != len(rows) or len(rows) != 72:
        raise ValueError("Expected baseline and candidate validation rows for 36 images")
    paired = []
    for fold, filename in sorted({(row["fold"], row["filename"]) for row in rows}):
        baseline = keyed[fold, filename, "baseline"]
        candidate = keyed[fold, filename, "candidate"]
        paired.append({
            "fold": fold,
            "filename": filename,
            "foreground_fraction": baseline["foreground_fraction"],
            **{f"delta_{name}": candidate[name] - baseline[name]
               for name in ("dice", "precision", "recall", "fp_pixels", "distant_fp_pixels")},
        })
    report = {}
    for name, subset in (
        ("all", paired),
        ("low_foreground", [r for r in paired if r["foreground_fraction"] <= 0.05]),
        ("higher_foreground", [r for r in paired if r["foreground_fraction"] > 0.05]),
    ):
        report[name] = {
            "image_count": len(subset),
            **{f"mean_delta_{metric}": statistics.mean(r[f"delta_{metric}"] for r in subset)
               for metric in ("dice", "precision", "recall", "fp_pixels", "distant_fp_pixels")},
            "images_dice_improved": sum(r["delta_dice"] > 0 for r in subset),
            "images_dice_worsened": sum(r["delta_dice"] < 0 for r in subset),
        }
    for filename, data in (("validation_fp_rows.csv", rows), ("validation_fp_paired.csv", paired)):
        with (args.candidate_root / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (args.candidate_root / "validation_fp_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
