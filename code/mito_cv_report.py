"""Audit and summarize a completed, locked five-fold Mito evaluation.

This command reads existing reports only. It never trains a model or changes
TIFFs, masks, predictions, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = ("dice", "iou", "precision", "recall")


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_cv(run_root: Path, num_folds: int = 5) -> dict[str, object]:
    """Require disjoint outer tests, one shared fold map, and 36 OOF rows."""

    if num_folds != 5:
        raise ValueError("The locked Mito protocol uses exactly five folds")
    seen: set[str] = set()
    expected_fold_ids: dict[str, int] | None = None
    fold_reports: list[dict[str, object]] = []
    all_rows: list[dict[str, float | str]] = []

    for fold in range(num_folds):
        directory = run_root / f"fold_{fold}"
        with (directory / "split.json").open(encoding="utf-8") as handle:
            split = json.load(handle)
        if split.get("target") != "mito" or split.get("fold") != fold:
            raise ValueError(f"Wrong target or fold in {directory / 'split.json'}")
        if split.get("num_folds") != num_folds:
            raise ValueError(f"Wrong num_folds in {directory / 'split.json'}")
        fold_ids = split["fold_ids"]
        if expected_fold_ids is None:
            expected_fold_ids = fold_ids
        elif fold_ids != expected_fold_ids:
            raise ValueError("Fold assignments changed between runs")

        groups = {part: set(split[part]) for part in ("train", "validation", "test")}
        if any(len(groups[part]) != len(split[part]) for part in groups):
            raise ValueError(f"Duplicate filename within split in fold {fold}")
        if any(groups[a] & groups[b] for a, b in (
            ("train", "validation"), ("train", "test"), ("validation", "test")
        )):
            raise ValueError(f"Train/validation/test overlap in fold {fold}")
        if set.union(*groups.values()) != set(fold_ids):
            raise ValueError(f"Incomplete split in fold {fold}")
        if groups["test"] & seen:
            raise ValueError(f"Outer-test images repeated in fold {fold}")
        if any(fold_ids[name] != fold for name in groups["test"]):
            raise ValueError(f"Test membership contradicts fold_ids in fold {fold}")
        validation_fold = (fold + 1) % num_folds
        if any(fold_ids[name] != validation_fold for name in groups["validation"]):
            raise ValueError(f"Validation membership contradicts fold_ids in fold {fold}")
        if any(fold_ids[name] in {fold, validation_fold} for name in groups["train"]):
            raise ValueError(f"Training membership contradicts fold_ids in fold {fold}")
        seen.update(groups["test"])

        with (directory / "test_per_image.csv").open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        if {row["filename"] for row in records} != groups["test"] or len(records) != len(groups["test"]):
            raise ValueError(f"Test metric rows differ from split in fold {fold}")
        with (directory / "threshold_policy_per_image.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            policy_records = list(csv.DictReader(handle))
        policies = {row["filename"]: row for row in policy_records}
        if set(policies) != groups["test"] or len(policy_records) != len(groups["test"]):
            raise ValueError(f"Threshold-policy rows differ from split in fold {fold}")

        rows: list[dict[str, float | str]] = []
        for record in records:
            row: dict[str, float | str] = {"filename": record["filename"]}
            for metric in METRICS:
                value = float(record[metric])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"Invalid {metric} for {record['filename']}")
                row[metric] = value
                validation_value = float(policies[record["filename"]][f"validation_{metric}"])
                # GPU inference can flip a handful of pixels at a threshold
                # boundary across runs. A larger mismatch indicates that the
                # exported masks do not represent the saved evaluation.
                if not math.isclose(value, validation_value, abs_tol=1e-4):
                    raise ValueError(f"Saved test metric and exported mask disagree for {record['filename']}")
                fixed_value = float(policies[record["filename"]][f"fixed_050_{metric}"])
                if not math.isfinite(fixed_value) or not 0.0 <= fixed_value <= 1.0:
                    raise ValueError(f"Invalid fixed-threshold {metric} for {record['filename']}")
                row[f"fixed_050_{metric}"] = fixed_value
            fraction = float(record["foreground_fraction_true"])
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(f"Invalid foreground fraction for {record['filename']}")
            row["foreground_fraction_true"] = fraction
            row["fold"] = fold
            rows.append(row)
        all_rows.extend(rows)
        fold_reports.append({
            "fold": fold,
            "image_count": len(rows),
            "metrics": {
                metric: _distribution([float(row[metric]) for row in rows])
                for metric in METRICS
            },
        })

    if expected_fold_ids is None or seen != set(expected_fold_ids):
        raise ValueError("Not every labelled image has one outer-test prediction")
    if len(seen) != 36:
        raise ValueError(f"Expected 36 Mito images, found {len(seen)}")

    pooled: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in all_rows]
        pooled[metric] = _distribution(values)
    dice_values = sorted(float(row["dice"]) for row in all_rows)
    pooled["dice"]["lower_quartile"] = statistics.median(dice_values[:18])
    fractions = [float(row["foreground_fraction_true"]) for row in all_rows]
    dice = [float(row["dice"]) for row in all_rows]
    correlation = (
        statistics.correlation(fractions, dice)
        if len(set(fractions)) > 1 and len(set(dice)) > 1 else None
    )
    fixed: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        fixed[metric] = _distribution([float(row[f"fixed_050_{metric}"]) for row in all_rows])
    all_rows.sort(key=lambda row: str(row["filename"]))
    by_fraction = sorted(all_rows, key=lambda row: float(row["foreground_fraction_true"]))
    foreground_quartiles = []
    for quartile in range(4):
        group = by_fraction[quartile * 9:(quartile + 1) * 9]
        foreground_quartiles.append({
            "quartile": quartile + 1,
            "foreground_min": float(group[0]["foreground_fraction_true"]),
            "foreground_max": float(group[-1]["foreground_fraction_true"]),
            "dice": _distribution([float(row["dice"]) for row in group]),
        })
    return {
        "protocol": "mito_locked_five_fold_outer_test",
        "image_count": len(all_rows),
        "folds": fold_reports,
        "pooled_per_image": pooled,
        "fixed_050_per_image": fixed,
        "fixed_minus_validation_dice_per_image": _distribution([
            float(row["fixed_050_dice"]) - float(row["dice"]) for row in all_rows
        ]),
        "foreground_fraction_vs_dice_pearson": correlation,
        "foreground_fraction_quartiles": foreground_quartiles,
        "per_image": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Directory containing fold_0 ... fold_4")
    args = parser.parse_args()
    print(json.dumps(summarize_cv(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
