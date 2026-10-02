"""Describe FP changes in locked, validation-selected Mito OOF masks.

This is post-decision evaluation only. It never changes a fold decision or
selects a threshold, and rejected folds use their frozen baseline masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from segmentation_metrics import calculate_binary_metrics
from tiff_io import read_binary_mask


def _counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    if truth.shape != prediction.shape:
        raise ValueError("Prediction and manual mask coordinates do not match")
    fp = ~truth & prediction
    return {
        "fp_pixels": int(fp.sum()),
        "distant_fp_pixels": int(np.count_nonzero(
            fp & (distance_transform_edt(~truth) > 10)
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--baseline-prediction-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads((args.candidate_root / "candidate_decisions.json").read_text())
    if gate.get("protocol") != "mito_hard_negative_validation_gate_v1":
        raise ValueError("Expected locked hard-negative validation decisions")
    oof = json.loads((args.candidate_root / "candidate_oof_summary.json").read_text())
    expected = {row["filename"]: row for row in oof["per_image"]}
    if len(expected) != 36:
        raise ValueError("Expected exactly 36 OOF images")
    output_rows = []
    for row in sorted(expected.values(), key=lambda item: item["filename"]):
        fold = int(row["fold"])
        filename = str(row["filename"])
        selected = bool(row["selected_candidate"])
        truth = read_binary_mask(args.label_dir / filename).astype(bool)
        baseline = read_binary_mask(
            args.baseline_prediction_root / f"fold_{fold}" /
            "predictions" / "validation" / filename
        ).astype(bool)
        chosen_path = (
            args.candidate_root / "mito" / f"fold_{fold}" /
            "predictions" / "validation" / filename
        ) if selected else (
            args.baseline_prediction_root / f"fold_{fold}" /
            "predictions" / "validation" / filename
        )
        chosen = read_binary_mask(chosen_path).astype(bool)
        base_metric = calculate_binary_metrics(truth, baseline)
        chosen_metric = calculate_binary_metrics(truth, chosen)
        if (abs(base_metric.dice - float(row["baseline_dice"])) > 1e-8
                or abs(chosen_metric.dice - float(row["dice"])) > 1e-8):
            raise ValueError(f"Mask and locked OOF metric mismatch: {filename}")
        base_counts = _counts(truth, baseline)
        chosen_counts = _counts(truth, chosen)
        output_rows.append({
            "filename": filename,
            "fold": fold,
            "selected_candidate": selected,
            "foreground_fraction": float(row["foreground_fraction_true"]),
            "dice_delta": chosen_metric.dice - base_metric.dice,
            "precision_delta": chosen_metric.precision - base_metric.precision,
            "recall_delta": chosen_metric.recall - base_metric.recall,
            "baseline_fp_pixels": base_counts["fp_pixels"],
            "selected_fp_pixels": chosen_counts["fp_pixels"],
            "delta_fp_pixels": chosen_counts["fp_pixels"] - base_counts["fp_pixels"],
            "baseline_distant_fp_pixels": base_counts["distant_fp_pixels"],
            "selected_distant_fp_pixels": chosen_counts["distant_fp_pixels"],
            "delta_distant_fp_pixels": (
                chosen_counts["distant_fp_pixels"] - base_counts["distant_fp_pixels"]
            ),
        })
    if len(output_rows) != 36 or len({r["filename"] for r in output_rows}) != 36:
        raise ValueError("Incomplete selected-policy OOF analysis")
    summary = {}
    for name, subset in (
        ("all", output_rows),
        ("low_foreground", [r for r in output_rows if r["foreground_fraction"] <= 0.05]),
        ("higher_foreground", [r for r in output_rows if r["foreground_fraction"] > 0.05]),
    ):
        summary[name] = {
            "image_count": len(subset),
            "mean_delta_dice": statistics.mean(r["dice_delta"] for r in subset),
            "mean_delta_precision": statistics.mean(r["precision_delta"] for r in subset),
            "mean_delta_recall": statistics.mean(r["recall_delta"] for r in subset),
            "baseline_fp_pixels": sum(r["baseline_fp_pixels"] for r in subset),
            "selected_fp_pixels": sum(r["selected_fp_pixels"] for r in subset),
            "baseline_distant_fp_pixels": sum(r["baseline_distant_fp_pixels"] for r in subset),
            "selected_distant_fp_pixels": sum(r["selected_distant_fp_pixels"] for r in subset),
            "images_dice_improved": sum(r["dice_delta"] > 0 for r in subset),
            "images_dice_worsened": sum(r["dice_delta"] < 0 for r in subset),
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_image_fp.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with (args.output_dir / "fp_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
