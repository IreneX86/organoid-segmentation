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
        rows: list[dict[str, float | str]] = []
        for record in records:
            row: dict[str, float | str] = {"filename": record["filename"]}
            for metric in METRICS:
                value = float(record[metric])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"Invalid {metric} for {record['filename']}")
                row[metric] = value
            rows.append(row)
        all_rows.extend(rows)
        fold_reports.append({
            "fold": fold,
            "image_count": len(rows),
            "mean_dice": statistics.mean(float(row["dice"]) for row in rows),
        })

    if expected_fold_ids is None or seen != set(expected_fold_ids):
        raise ValueError("Not every labelled image has one outer-test prediction")
    if len(seen) != 36:
        raise ValueError(f"Expected 36 Mito images, found {len(seen)}")

    pooled: dict[str, float] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in all_rows]
        pooled[f"{metric}_mean"] = statistics.mean(values)
        pooled[f"{metric}_std"] = statistics.stdev(values)
    dice_values = sorted(float(row["dice"]) for row in all_rows)
    pooled["dice_min"] = dice_values[0]
    pooled["dice_median"] = statistics.median(dice_values)
    pooled["dice_lower_quartile"] = statistics.median(dice_values[:18])
    return {
        "protocol": "mito_locked_five_fold_outer_test",
        "image_count": len(all_rows),
        "folds": fold_reports,
        "pooled_per_image": pooled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Directory containing fold_0 ... fold_4")
    args = parser.parse_args()
    print(json.dumps(summarize_cv(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
