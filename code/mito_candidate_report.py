"""Summarize a validation-gated Mito candidate against the frozen OOF baseline.

This reads locked decisions and existing reports. It never chooses a model or
threshold from outer-test labels and never changes an experiment result.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = ("dice", "iou", "precision", "recall")


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    result = {row["filename"]: row for row in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate image metric row in {path}")
    return result


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_candidate(baseline_root: Path, candidate_root: Path) -> dict[str, object]:
    with (candidate_root / "candidate_decisions.json").open(encoding="utf-8") as handle:
        gate = json.load(handle)
    if gate.get("protocol") not in {
        "mito_one_factor_validation_gate_v1",
        "mito_three_pool_validation_gate_v1",
        "mito_hard_negative_validation_gate_v1",
    }:
        raise ValueError("Candidate decisions are not from the locked validation gate")
    decisions = gate["decisions"]
    if len(decisions) != 5 or [item["fold"] for item in decisions] != list(range(5)):
        raise ValueError("Expected one ordered decision for each of five folds")

    per_image = []
    fold_reports = []
    for fold, decision in enumerate(decisions):
        baseline_dir = baseline_root / f"fold_{fold}"
        candidate_dir = candidate_root / "mito" / f"fold_{fold}"
        with (baseline_dir / "split.json").open(encoding="utf-8") as handle:
            baseline_split = json.load(handle)
        with (candidate_dir / "split.json").open(encoding="utf-8") as handle:
            candidate_split = json.load(handle)
        if baseline_split != candidate_split:
            raise ValueError(f"Fold {fold} changed the frozen split")
        baseline_rows = _rows(baseline_dir / "test_per_image.csv")
        if set(baseline_rows) != set(baseline_split["test"]):
            raise ValueError(f"Fold {fold} baseline test rows mismatch")
        selected = bool(decision["selected_candidate"])
        selected_rows = (
            _rows(candidate_dir / "test_per_image.csv") if selected else baseline_rows
        )
        if set(selected_rows) != set(baseline_rows):
            raise ValueError(f"Fold {fold} selected test rows mismatch")
        fold_rows = []
        for filename in sorted(baseline_rows):
            base = baseline_rows[filename]
            chosen = selected_rows[filename]
            row: dict[str, str | float | int | bool] = {
                "filename": filename,
                "fold": fold,
                "selected_candidate": selected,
                "foreground_fraction_true": float(base["foreground_fraction_true"]),
            }
            for metric in METRICS:
                row[metric] = float(chosen[metric])
                row[f"baseline_{metric}"] = float(base[metric])
                row[f"delta_{metric}"] = float(chosen[metric]) - float(base[metric])
            fold_rows.append(row)
        per_image.extend(fold_rows)
        fold_reports.append({
            "fold": fold,
            "selected_candidate": selected,
            "image_count": len(fold_rows),
            "dice": _distribution([float(row["dice"]) for row in fold_rows]),
        })
    if len(per_image) != 36 or len({row["filename"] for row in per_image}) != 36:
        raise ValueError("Expected exactly 36 distinct outer-test images")
    return {
        "protocol": "mito_validation_gated_candidate_oof_v1",
        "variant": gate["variant"],
        "image_count": 36,
        "selected_candidate_folds": sum(item["selected_candidate"] for item in decisions),
        "metrics": {
            metric: _distribution([float(row[metric]) for row in per_image])
            for metric in METRICS
        },
        "paired_delta": {
            metric: _distribution([float(row[f"delta_{metric}"]) for row in per_image])
            for metric in METRICS
        },
        "folds": fold_reports,
        "per_image": sorted(per_image, key=lambda row: str(row["filename"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_candidate(args.baseline_root, args.candidate_root)
    output = args.candidate_root / "candidate_oof_summary.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
