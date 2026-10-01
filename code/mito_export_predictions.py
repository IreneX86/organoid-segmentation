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

from evaluation import predict_pairs
from pipeline_config import PipelineConfig, get_target_spec
from segmentation_metrics import calculate_binary_metrics
from tiff_io import pair_tiff_directories, write_mask_tiff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
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
    for pair, probability, truth in predictions:
        row: dict[str, float | str] = {"filename": pair.filename}
        # Both policies are fixed before reading any test metric. The trained
        # checkpoint and threshold.json came only from validation data.
        for policy, threshold in (("validation", chosen_threshold), ("fixed_050", 0.50)):
            mask = (probability >= threshold).astype("uint8")
            write_mask_tiff(run_dir / "predictions" / policy / pair.filename, mask)
            metrics = calculate_binary_metrics(truth, mask)
            for name in ("dice", "iou", "precision", "recall"):
                row[f"{policy}_{name}"] = getattr(metrics, name)
        rows.append(row)
    with (run_dir / "threshold_policy_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} outer-test masks for each threshold policy")


if __name__ == "__main__":
    main()
