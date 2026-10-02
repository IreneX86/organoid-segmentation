"""Read-only, per-image audit of frozen Mito out-of-fold prediction errors.

This script never trains, selects a threshold, or changes masks. It requires
SciPy for spatial measurements and writes only derived tables to --output-dir.
The 36-image OOF map is checked against the frozen baseline metrics before
any error mechanism is summarized.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from segmentation_metrics import calculate_binary_metrics
from tiff_io import normalize_image, read_binary_mask, read_tiff_2d


MEASUREMENTS = (
    "dice", "precision", "recall", "foreground_fraction", "false_positive_rate",
    "fp_within_2px", "fp_within_5px", "fp_farther_10px",
    "fp_in_gt_touching_prediction", "fn_within_2px", "fp_above_tn_p90",
    "fp_local_contrast_above_tn_p90", "gt_edge_fraction", "gt_components_ge_10",
    "pred_components_ge_10", "gt_component_median_area_ge_10",
    "gt_area_share_components_10_to_49", "gt_split_component_fraction",
    "pred_merge_component_fraction", "background_grid_median_range",
    "hard_negative_candidate_count", "hard_negative_candidate_fp_enrichment",
)


def _fraction(mask: np.ndarray, condition: np.ndarray) -> float:
    count = int(mask.sum())
    return float(condition[mask].mean()) if count else 0.0


def _component_relationships(
    truth: np.ndarray, prediction: np.ndarray, predicted_labels: np.ndarray
) -> dict[str, float | int]:
    """Pixel-component descriptors, not counts of biological mitochondria."""

    gt_labels, gt_number = ndi.label(
        truth, structure=np.ones((3, 3), dtype=np.uint8)
    )
    pred_number = int(predicted_labels.max())
    gt_sizes = np.bincount(gt_labels.ravel(), minlength=gt_number + 1)
    pred_sizes = np.bincount(predicted_labels.ravel(), minlength=pred_number + 1)
    gt_valid = np.flatnonzero(gt_sizes >= 10)
    pred_valid = np.flatnonzero(pred_sizes >= 10)
    gt_valid = gt_valid[gt_valid != 0]
    pred_valid = pred_valid[pred_valid != 0]

    # A component pair must share at least three TP pixels. Pixel connectivity
    # is sensitive to annotation thickness, so these are only diagnostics.
    tp = truth & prediction
    encoded = gt_labels[tp].astype(np.int64) * (pred_number + 1) + predicted_labels[tp]
    pairs, overlaps = np.unique(encoded, return_counts=True)
    gt_partner = pairs // (pred_number + 1)
    pred_partner = pairs % (pred_number + 1)
    valid_pair = (
        (overlaps >= 3)
        & (gt_sizes[gt_partner] >= 10)
        & (pred_sizes[pred_partner] >= 10)
    )
    gt_partner_counts = np.bincount(gt_partner[valid_pair], minlength=gt_number + 1)
    pred_partner_counts = np.bincount(pred_partner[valid_pair], minlength=pred_number + 1)
    gt_small = (gt_sizes >= 10) & (gt_sizes < 50)
    gt_small[0] = False
    return {
        "gt_components_ge_10": int(len(gt_valid)),
        "pred_components_ge_10": int(len(pred_valid)),
        "gt_component_median_area_ge_10": (
            float(np.median(gt_sizes[gt_valid])) if len(gt_valid) else 0.0
        ),
        "gt_area_share_components_10_to_49": float(gt_sizes[gt_small].sum() / truth.sum()),
        "gt_split_component_fraction": (
            float(np.mean(gt_partner_counts[gt_valid] >= 2)) if len(gt_valid) else 0.0
        ),
        "pred_merge_component_fraction": (
            float(np.mean(pred_partner_counts[pred_valid] >= 2)) if len(pred_valid) else 0.0
        ),
    }


def _bootstrap_mean_difference(
    low: np.ndarray, high: np.ndarray, rng: np.random.Generator
) -> list[float]:
    """Bootstrap images, not pixels; the interval remains descriptive."""

    low_draws = rng.choice(low, size=(5000, len(low)), replace=True).mean(axis=1)
    high_draws = rng.choice(high, size=(5000, len(high)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(low_draws - high_draws, [2.5, 97.5])]


def audit_one(
    filename: str,
    fold: int,
    image_path: Path,
    mask_path: Path,
    prediction_path: Path,
    expected_dice: float,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    image = read_tiff_2d(image_path)
    truth = read_binary_mask(mask_path).astype(bool)
    prediction = read_binary_mask(prediction_path).astype(bool)
    if image.shape != truth.shape or truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch for {filename}")

    measured = calculate_binary_metrics(truth, prediction)
    if abs(measured.dice - expected_dice) > 1e-9:
        raise ValueError(f"Prediction is not the frozen OOF mask for {filename}")
    tp = truth & prediction
    fp = ~truth & prediction
    fn = truth & ~prediction
    tn = ~truth & ~prediction
    fp_count = int(fp.sum())
    fn_count = int(fn.sum())
    normalized = normalize_image(image)

    # Distance to GT gives a scale-free description in native image pixels.
    # It is not itself proof that an FP was caused by boundary expansion.
    distance_to_gt = ndi.distance_transform_edt(~truth)
    distance_inside_gt = ndi.distance_transform_edt(truth)
    predicted_components, count = ndi.label(
        prediction, structure=np.ones((3, 3), dtype=np.uint8)
    )
    gt_touching = np.zeros(count + 1, dtype=bool)
    gt_touching[np.unique(predicted_components[truth])] = True
    gt_touching[0] = False

    # Compare FP intensities with a deterministic TN pixel sample. A local
    # residual controls for slow field illumination; sigma=12 px is diagnostic,
    # not a tuned preprocessing parameter or proposed training hyperparameter.
    tn_values = normalized[tn]
    sample_size = min(100_000, len(tn_values))
    sample_indices = rng.choice(len(tn_values), size=sample_size, replace=False)
    tn_sample = tn_values[sample_indices]
    local_residual = normalized - ndi.gaussian_filter(normalized, sigma=12)
    tn_local_sample = local_residual[tn][sample_indices]
    bright_cutoff = float(np.percentile(tn_sample, 90))
    local_cutoff = float(np.percentile(tn_local_sample, 90))
    # Feasibility of one *future training-fold-only* sampling rule. Looking at
    # archived OOF predictions here is explicitly exploratory; it cannot be
    # used to tune this rule and then claim an unbiased OOF improvement.
    negative = ~truth
    hard_candidates = (
        negative
        & (distance_to_gt > 10)
        & (normalized >= np.percentile(normalized[negative], 90))
        & (local_residual >= np.percentile(local_residual[negative], 90))
    )
    background_fp_rate = fp_count / int(negative.sum())

    h, w = truth.shape
    grid_background_medians = []
    for y in range(0, h, h // 4):
        for x in range(0, w, w // 4):
            block = normalized[y : min(y + h // 4, h), x : min(x + w // 4, w)]
            block_gt = truth[y : min(y + h // 4, h), x : min(x + w // 4, w)]
            if (~block_gt).any():
                grid_background_medians.append(float(np.median(block[~block_gt])))

    raw_percentiles = np.percentile(image, [1, 50, 90, 99, 99.8])
    gt_border = truth & ~ndi.binary_erosion(truth)
    row: dict[str, float | int | str] = {
        "filename": filename,
        "fold": fold,
        "height": h,
        "width": w,
        "foreground_fraction": float(truth.mean()),
        "dice": measured.dice,
        "iou": measured.iou,
        "precision": measured.precision,
        "recall": measured.recall,
        "true_positive_pixels": int(tp.sum()),
        "false_positive_pixels": fp_count,
        "false_negative_pixels": fn_count,
        "false_positive_rate": background_fp_rate,
        "fp_within_2px": _fraction(fp, distance_to_gt <= 2),
        "fp_within_5px": _fraction(fp, distance_to_gt <= 5),
        "fp_farther_10px": _fraction(fp, distance_to_gt > 10),
        "fp_in_gt_touching_prediction": _fraction(fp, gt_touching[predicted_components]),
        "fn_within_2px": _fraction(fn, distance_inside_gt <= 2),
        "fp_normalized_intensity_median": float(np.median(normalized[fp])) if fp_count else 0.0,
        "tn_normalized_intensity_median": float(np.median(tn_sample)),
        "fp_above_tn_p90": _fraction(fp, normalized > bright_cutoff),
        "fp_local_residual_median": float(np.median(local_residual[fp])) if fp_count else 0.0,
        "tn_local_residual_median": float(np.median(tn_local_sample)),
        "fp_local_contrast_above_tn_p90": _fraction(fp, local_residual > local_cutoff),
        "gt_edge_fraction": float(gt_border.sum() / truth.sum()),
        **_component_relationships(truth, prediction, predicted_components),
        "background_grid_median_range": max(grid_background_medians) - min(grid_background_medians),
        "hard_negative_candidate_count": int(hard_candidates.sum()),
        "hard_negative_candidate_fp_enrichment": (
            float(prediction[hard_candidates].mean() / background_fp_rate)
            if hard_candidates.any() and background_fp_rate > 0 else 0.0
        ),
        "raw_p1": float(raw_percentiles[0]),
        "raw_p50": float(raw_percentiles[1]),
        "raw_p90": float(raw_percentiles[2]),
        "raw_p99": float(raw_percentiles[3]),
        "raw_p99_8": float(raw_percentiles[4]),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--oof-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.oof_csv.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    if len(baseline_rows) != 36 or len({r["filename"] for r in baseline_rows}) != 36:
        raise ValueError("Expected exactly 36 unique frozen OOF rows")
    rng = np.random.default_rng(20261002)
    rows = []
    for baseline in baseline_rows:
        filename = baseline["filename"]
        fold = int(baseline["fold"])
        prediction = (
            args.baseline_root / f"fold_{fold}" / "predictions" / "validation" / filename
        )
        rows.append(audit_one(
            filename, fold, args.image_dir / filename, args.mask_dir / filename,
            prediction, float(baseline["dice"]), rng,
        ))
    rows.sort(key=lambda row: str(row["filename"]))
    low = [row for row in rows if float(row["foreground_fraction"]) <= 0.05]
    high = [row for row in rows if float(row["foreground_fraction"]) > 0.05]
    if len(low) != 7 or len(high) != 29:
        raise ValueError("Frozen foreground groups differ from the baseline report")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {
        "source": "frozen Mito five-fold OOF predictions, validation-selected threshold",
        "image_count": len(rows),
        "low_foreground_rule": "manual foreground fraction <= 0.05; diagnostic grouping only",
        "groups": {},
    }
    for name, group in (("low_foreground", low), ("higher_foreground", high)):
        summary["groups"][name] = {
            "image_count": len(group),
            **{
                metric: {
                    "mean": float(np.mean([float(r[metric]) for r in group])),
                    "median": float(np.median([float(r[metric]) for r in group])),
                }
                for metric in MEASUREMENTS
            },
        }
    summary["low_minus_high_95pct_image_bootstrap_interval"] = {
        metric: _bootstrap_mean_difference(
            np.array([float(r[metric]) for r in low]),
            np.array([float(r[metric]) for r in high]), rng,
        )
        for metric in ("dice", "precision", "false_positive_rate", "fp_within_2px")
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for group_name, group in (("low_fg", low), ("high_fg", high)):
        order = sorted(group, key=lambda row: float(row["dice"]))
        print(group_name, "low-dice examples", [row["filename"] for row in order[:2]])
        print(group_name, "high-dice examples", [row["filename"] for row in order[-2:]])
    print(f"Wrote read-only 36-image audit to {output}")


if __name__ == "__main__":
    main()
