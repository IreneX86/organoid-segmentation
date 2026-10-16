"""Compare two Nuc internal-validation CSVs by the same image filenames.

This reads only the saved per-image metrics. In particular, it does not read
source TIFFs, infer labels from screenshots, or select a new threshold. Both
models' ``selected`` columns may use different validation-selected thresholds;
the fixed 0.50 comparison is the direct same-threshold comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


POLICIES = ("fixed_050", "selected")
METRICS = ("dice", "iou", "precision", "recall")
GROUPS = ("nonempty", "true_negative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> dict[str, dict[str, str]]:
    required = {"filename", "truth_group"}
    for policy in POLICIES:
        required.add(f"{policy}_false_positive_pixels")
        required.update(f"{policy}_{metric}" for metric in METRICS)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if columns is None or len(columns) != len(set(columns)):
            raise ValueError(f"Missing or duplicate CSV columns: {path}")
        if not required.issubset(columns):
            raise ValueError(f"Missing validation metric columns: {path}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            name = row["filename"]
            if not name or name in rows:
                raise ValueError(f"Blank or duplicate filename in {path}: {name!r}")
            if row["truth_group"] not in GROUPS:
                raise ValueError(f"Invalid truth_group for {name} in {path}")
            for policy in POLICIES:
                _nonnegative_integer(row[f"{policy}_false_positive_pixels"], name)
                for metric in METRICS:
                    value = row[f"{policy}_{metric}"]
                    if row["truth_group"] == "true_negative":
                        if value not in ("", None):
                            raise ValueError(
                                f"True-negative {name} must have blank {metric}"
                            )
                    else:
                        _unit_float(value, name, metric)
            rows[name] = row
    if not rows:
        raise ValueError(f"Validation CSV is empty: {path}")
    return rows


def _nonnegative_integer(value: str | None, name: str) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid false-positive count for {name}: {value!r}") from error
    if number < 0:
        raise ValueError(f"Negative false-positive count for {name}")
    return number


def _unit_float(value: str | None, name: str, metric: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {metric} for {name}: {value!r}") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"Out-of-range {metric} for {name}: {value!r}")
    return number


def _summarize_metric(baseline: list[float], candidate: list[float]) -> dict[str, float | int]:
    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        index = fraction * (len(ordered) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])

    deltas = [new - old for old, new in zip(baseline, candidate, strict=True)]
    return {
        "baseline_mean": statistics.mean(baseline),
        "baseline_median": statistics.median(baseline),
        "baseline_std": statistics.stdev(baseline) if len(baseline) > 1 else 0.0,
        "baseline_min": min(baseline),
        "baseline_p10": percentile(baseline, 0.10),
        "candidate_mean": statistics.mean(candidate),
        "candidate_median": statistics.median(candidate),
        "candidate_std": statistics.stdev(candidate) if len(candidate) > 1 else 0.0,
        "candidate_min": min(candidate),
        "candidate_p10": percentile(candidate, 0.10),
        "paired_delta_mean": statistics.mean(deltas),
        "paired_delta_median": statistics.median(deltas),
        "paired_delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "paired_delta_min": min(deltas),
        "paired_delta_max": max(deltas),
        "images_improved": sum(delta > 0 for delta in deltas),
        "images_worsened": sum(delta < 0 for delta in deltas),
        "images_unchanged": sum(delta == 0 for delta in deltas),
    }


def compare_validation_csv(
    baseline_path: Path,
    candidate_path: Path,
    *,
    expected_nonempty: int = 33,
    expected_true_negative: int = 1,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return a diagnostic summary and paired rows; never modify either input."""

    if expected_nonempty < 1 or expected_true_negative < 0:
        raise ValueError("Expected group counts must be nonnegative, with nonempty >= 1")
    baseline_rows = _load_rows(baseline_path)
    candidate_rows = _load_rows(candidate_path)
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("Baseline and candidate validation filenames differ")

    names = sorted(baseline_rows)
    for name in names:
        if baseline_rows[name]["truth_group"] != candidate_rows[name]["truth_group"]:
            raise ValueError(f"Baseline and candidate truth_group differ for {name}")
    nonempty_names = [
        name for name in names if baseline_rows[name]["truth_group"] == "nonempty"
    ]
    negative_names = [
        name for name in names if baseline_rows[name]["truth_group"] == "true_negative"
    ]
    if (len(nonempty_names) != expected_nonempty
            or len(negative_names) != expected_true_negative):
        raise ValueError(
            "Unexpected validation group counts: "
            f"nonempty={len(nonempty_names)}, true_negative={len(negative_names)}"
        )

    paired_rows: list[dict[str, object]] = []
    policy_summary: dict[str, object] = {}
    for policy in POLICIES:
        metric_summary: dict[str, object] = {}
        for metric in METRICS:
            baseline_values = [
                _unit_float(baseline_rows[name][f"{policy}_{metric}"], name, metric)
                for name in nonempty_names
            ]
            candidate_values = [
                _unit_float(candidate_rows[name][f"{policy}_{metric}"], name, metric)
                for name in nonempty_names
            ]
            metric_summary[metric] = _summarize_metric(
                baseline_values, candidate_values
            )
        baseline_negative_fp = sum(
            _nonnegative_integer(
                baseline_rows[name][f"{policy}_false_positive_pixels"], name
            )
            for name in negative_names
        )
        candidate_negative_fp = sum(
            _nonnegative_integer(
                candidate_rows[name][f"{policy}_false_positive_pixels"], name
            )
            for name in negative_names
        )
        policy_summary[policy] = {
            "nonempty": metric_summary,
            "true_negative": {
                "image_count": len(negative_names),
                "baseline_false_positive_pixels_total": baseline_negative_fp,
                "candidate_false_positive_pixels_total": candidate_negative_fp,
                "paired_false_positive_pixels_delta_total": (
                    candidate_negative_fp - baseline_negative_fp
                ),
            },
        }

    for name in names:
        old = baseline_rows[name]
        new = candidate_rows[name]
        paired: dict[str, object] = {
            "filename": name,
            "truth_group": old["truth_group"],
        }
        for policy in POLICIES:
            old_fp = _nonnegative_integer(old[f"{policy}_false_positive_pixels"], name)
            new_fp = _nonnegative_integer(new[f"{policy}_false_positive_pixels"], name)
            paired[f"{policy}_baseline_false_positive_pixels"] = old_fp
            paired[f"{policy}_candidate_false_positive_pixels"] = new_fp
            paired[f"{policy}_paired_false_positive_pixels_delta"] = new_fp - old_fp
            for metric in METRICS:
                if old["truth_group"] == "true_negative":
                    old_metric = new_metric = delta = ""
                else:
                    old_metric = _unit_float(old[f"{policy}_{metric}"], name, metric)
                    new_metric = _unit_float(new[f"{policy}_{metric}"], name, metric)
                    delta = new_metric - old_metric
                paired[f"{policy}_baseline_{metric}"] = old_metric
                paired[f"{policy}_candidate_{metric}"] = new_metric
                paired[f"{policy}_paired_{metric}_delta"] = delta
        paired_rows.append(paired)

    summary: dict[str, object] = {
        "interpretation": (
            "Same-batch paired internal diagnostic, not an independent "
            "generalization estimate. Selected policies may have different "
            "model-specific validation-selected thresholds."
        ),
        "baseline_csv": str(baseline_path.resolve()),
        "baseline_csv_sha256": _sha256(baseline_path),
        "candidate_csv": str(candidate_path.resolve()),
        "candidate_csv_sha256": _sha256(candidate_path),
        "nonempty_image_count": len(nonempty_names),
        "true_negative_image_count": len(negative_names),
        "policies": policy_summary,
    }
    return summary, paired_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-nonempty", type=int, default=33)
    parser.add_argument("--expected-true-negative", type=int, default=1)
    args = parser.parse_args()

    input_paths = {args.baseline_csv.resolve(), args.candidate_csv.resolve()}
    output_paths = {args.output_json.resolve(), args.output_csv.resolve()}
    if len(output_paths) != 2 or input_paths & output_paths:
        raise ValueError("Output paths must be distinct from each other and inputs")
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Refusing to overwrite an existing comparison output")
    summary, paired_rows = compare_validation_csv(
        args.baseline_csv, args.candidate_csv,
        expected_nonempty=args.expected_nonempty,
        expected_true_negative=args.expected_true_negative,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with args.output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)


if __name__ == "__main__":
    main()
