"""Lock each Mito fold's sampling choice using validation histories only.

Run this after all five candidate folds have finished validation-only training
and before exporting any candidate outer-test prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _best_validation_dice(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        values = [float(row["val_full_dice"]) for row in csv.DictReader(handle)]
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError(f"Invalid validation history: {path}")
    return max(values)


def select_folds(baseline_root: Path, candidate_root: Path) -> dict[str, object]:
    """Apply the preregistered +0.01 per-fold validation gate."""

    decisions = []
    for fold in range(5):
        baseline = baseline_root / f"fold_{fold}"
        candidate = candidate_root / "mito" / f"fold_{fold}"
        with (baseline / "split.json").open(encoding="utf-8") as handle:
            baseline_split = json.load(handle)
        with (candidate / "split.json").open(encoding="utf-8") as handle:
            candidate_split = json.load(handle)
        if baseline_split != candidate_split:
            raise ValueError(f"Candidate fold {fold} differs from frozen baseline split")
        with (candidate / "experiment_config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        if (config.get("target") != "mito" or config.get("fold") != fold
                or config.get("positive_patch_probability") != 0.35
                or config.get("outer_test_deferred") is not True):
            raise ValueError(f"Candidate fold {fold} has wrong experiment configuration")
        if any((candidate / name).exists() for name in (
            "test_per_image.csv", "test_summary.json", "threshold_policy_per_image.csv",
        )) or (candidate / "predictions").exists():
            raise ValueError(f"Candidate fold {fold} was evaluated before validation-only selection")
        if not (candidate / "baseline_unet.keras").is_file() or not (
            candidate / "threshold.json"
        ).is_file():
            raise FileNotFoundError(f"Candidate fold {fold} lacks model or threshold")
        baseline_dice = _best_validation_dice(baseline / "training_history.csv")
        candidate_dice = _best_validation_dice(candidate / "training_history.csv")
        decisions.append({
            "fold": fold,
            "baseline_best_validation_dice": baseline_dice,
            "candidate_best_validation_dice": candidate_dice,
            "selected_positive_patch_probability": (
                0.35 if candidate_dice - baseline_dice >= 0.01 else 0.6
            ),
        })
    return {
        "protocol": "mito_sampling_035_validation_gate_v1",
        "minimum_validation_dice_gain": 0.01,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    report = select_folds(args.baseline_root, args.candidate_root)
    output = args.candidate_root / "sampling_decisions.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
