"""Report the predeclared LD validation-gated OOF procedure.

This is descriptive reporting after all per-fold decisions are locked. It
must not be used to retune the gate or choose another candidate on these 36
images. Baseline-selected folds reuse the frozen baseline outer-test results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ld_component_gate import compute_gate
from ld_cv_report import METRICS, _statistics, aggregate_cv
from ld_evaluate import _load_json, _sha256
from tiff_io import read_tiff_2d


def _fold_dir(root: Path, fold: int) -> Path:
    folder = root / "ld" if (root / "ld").is_dir() else root
    return folder / f"fold_{fold}"


def _candidate_test_rows(
    run: Path, fold: int, expected_names: set[str]
) -> list[dict[str, float | str]]:
    final = run / "outer_test"
    manifest = _load_json(final / "evaluation_manifest.json")
    if (manifest.get("candidate") != "component_uniform_positive_center_v1"
            or manifest.get("fold") != fold
            or set(manifest.get("test_filenames", [])) != expected_names
            or manifest.get("selection_lock_sha256") != _sha256(run / "selection.lock.json")
            or manifest.get("decision_lock_sha256") != _sha256(run / "decision.lock.json")):
        raise ValueError("Candidate outer-test result differs from its frozen gate")
    threshold = float(manifest.get("validation_threshold", -1))
    if (threshold != float(_load_json(run / "threshold.json")["threshold"])
            or not 0 < threshold < 1):
        raise ValueError("Candidate outer-test threshold differs from validation lock")
    artifact_files = manifest.get("files")
    if (not isinstance(artifact_files, list)
            or len(artifact_files) != len(expected_names)
            or {item["filename"] for item in artifact_files} != expected_names):
        raise ValueError("Candidate outer-test TIFF manifest is incomplete")
    for item in artifact_files:
        name = str(item["filename"])
        shape = item.get("shape")
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(length) is not int or length < 1 for length in shape)):
            raise ValueError(f"Invalid candidate pixel grid: {name}")
        probability_path = final / "probabilities" / name
        if item.get("probability_sha256") != _sha256(probability_path):
            raise ValueError(f"Candidate probability TIFF changed: {name}")
        probability = read_tiff_2d(probability_path)
        if (list(probability.shape) != shape
                or not np.issubdtype(probability.dtype, np.floating)
                or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)):
            raise ValueError(f"Candidate probability TIFF is invalid: {name}")
        for policy, cutoff in (("validation", threshold), ("fixed_050", 0.5)):
            mask_path = final / "predictions" / policy / name
            if item.get(f"{policy}_mask_sha256") != _sha256(mask_path):
                raise ValueError(f"Candidate {policy} mask TIFF changed: {name}")
            saved_mask = read_tiff_2d(mask_path)
            expected_mask = (probability >= cutoff).astype(np.uint8) * 255
            if (saved_mask.shape != probability.shape
                    or saved_mask.dtype != np.uint8
                    or not np.array_equal(saved_mask, expected_mask)):
                raise ValueError(f"Candidate {policy} mask grid/values differ: {name}")
    for filename, field in (
        ("test_per_image.csv", "test_per_image_sha256"),
        ("test_summary.json", "test_summary_sha256"),
        ("threshold_policy_per_image.csv", "threshold_policy_per_image_sha256"),
    ):
        if manifest.get(field) != _sha256(final / filename):
            raise ValueError(f"Candidate outer-test metrics changed: {filename}")
    with (final / "test_per_image.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (final / "threshold_policy_per_image.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        policy_rows = list(csv.DictReader(handle))
    if (len(rows) != len(expected_names)
            or {row["filename"] for row in rows} != expected_names
            or len(policy_rows) != len(expected_names)
            or {row["filename"] for row in policy_rows} != expected_names):
        raise ValueError("Candidate outer-test rows omit or duplicate test images")
    saved_summary = _load_json(final / "test_summary.json")
    if int(saved_summary.get("image_count", -1)) != len(expected_names):
        raise ValueError("Candidate outer-test summary is incomplete")
    policies = {row["filename"]: row for row in policy_rows}
    output: list[dict[str, float | str]] = []
    for row in rows:
        name = row["filename"]
        converted: dict[str, float | str] = {"filename": name}
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid candidate {metric}: {name}")
            converted[metric] = value
        for metric in ("dice", "iou", "precision", "recall"):
            selected = float(policies[name][f"validation_{metric}"])
            fixed = float(policies[name][f"fixed_050_{metric}"])
            if (not np.isclose(selected, float(converted[metric]), atol=1e-9)
                    or not np.isfinite(fixed) or not 0 <= fixed <= 1):
                raise ValueError(f"Candidate threshold policies disagree: {name}")
            converted[f"fixed_050_{metric}"] = fixed
        output.append(converted)
    return output


def aggregate_gated_candidate(
    baseline_root: Path,
    candidate_root: Path,
    fold_map: Path,
    label_manifest: Path,
    label_dir: Path,
) -> tuple[list[dict[str, float | str | int]], dict[str, object]]:
    """Verify each fold-local gate, then compare selected OOF with baseline."""

    decisions: list[dict[str, object]] = []
    for fold in range(5):
        baseline = _fold_dir(baseline_root, fold)
        candidate = _fold_dir(candidate_root, fold)
        recorded = _load_json(candidate / "decision.lock.json")
        recomputed = compute_gate(
            baseline, candidate, fold_map, label_manifest, label_dir,
        )
        if recorded != recomputed or recorded.get("fold") != fold:
            raise ValueError(f"Candidate fold {fold} decision changed after validation")
        decisions.append(recorded)

    baseline_rows, baseline_report = aggregate_cv(baseline_root)
    baseline_by_name = {str(row["filename"]): row for row in baseline_rows}
    if len(baseline_by_name) != 36:
        raise ValueError("Frozen LD baseline lacks complete 36-image OOF coverage")
    selected_by_name: dict[str, dict[str, float | str | int]] = {}
    fold_decisions: list[dict[str, object]] = []
    for fold, decision in enumerate(decisions):
        run = _fold_dir(candidate_root, fold)
        names = {
            str(row["filename"]) for row in baseline_rows if int(row["fold"]) == fold
        }
        if decision["selected"] == "candidate":
            candidates = _candidate_test_rows(run, fold, names)
            for row in candidates:
                selected_by_name[str(row["filename"])] = {"fold": fold, **row}
        elif decision["selected"] == "baseline":
            if (run / "outer_test").exists():
                raise ValueError("Baseline-selected fold unexpectedly opened candidate outer test")
            for name in names:
                selected_by_name[name] = dict(baseline_by_name[name])
        else:
            raise ValueError(f"Candidate fold {fold} has no valid decision")
        fold_decisions.append({
            "fold": fold,
            "selected": decision["selected"],
            "mean_paired_validation_dice_delta_050": decision["mean_paired_dice_delta_050"],
            "decision_lock_sha256": _sha256(run / "decision.lock.json"),
        })
    if set(selected_by_name) != set(baseline_by_name):
        raise ValueError("Gated candidate lacks exactly one selected OOF row per image")

    paired: list[dict[str, float | str | int]] = []
    for name in sorted(selected_by_name):
        selected = selected_by_name[name]
        baseline = baseline_by_name[name]
        if int(selected["fold"]) != int(baseline["fold"]):
            raise ValueError("Candidate image was scored in a different outer fold")
        if not np.isclose(
            float(selected["foreground_fraction_true"]),
            float(baseline["foreground_fraction_true"]), atol=1e-12,
        ):
            raise ValueError("Candidate and baseline used different outer-test labels")
        row: dict[str, float | str | int] = {
            "fold": int(selected["fold"]),
            "filename": name,
            "selected": decisions[int(selected["fold"])]["selected"],
        }
        for metric in ("dice", "iou", "precision", "recall"):
            row[f"baseline_{metric}"] = float(baseline[metric])
            row[f"selected_{metric}"] = float(selected[metric])
            row[f"paired_{metric}_delta"] = float(selected[metric]) - float(baseline[metric])
            row[f"selected_fixed_050_{metric}"] = float(selected[f"fixed_050_{metric}"])
        row["foreground_fraction_true"] = float(baseline["foreground_fraction_true"])
        paired.append(row)
    summary: dict[str, object] = {
        "target": "ld",
        "procedure": "predeclared_fold_local_validation_gated_component_sampling",
        "interpretation": "exploratory internal OOF; not independent external confirmation",
        "fold_count": 5,
        "image_count": len(paired),
        "candidate_fold_count": sum(item["selected"] == "candidate" for item in decisions),
        "fold_decisions": fold_decisions,
        "baseline_oof": baseline_report["oof"],
        "selected_oof": {
            metric: _statistics([float(row[f"selected_{metric}"]) for row in paired])
            for metric in ("dice", "iou", "precision", "recall")
        },
        "paired_delta": {
            metric: _statistics([float(row[f"paired_{metric}_delta"]) for row in paired])
            for metric in ("dice", "iou", "precision", "recall")
        },
    }
    return paired, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = aggregate_gated_candidate(
        args.baseline_run_root, args.candidate_run_root, args.fold_map,
        args.label_manifest, args.label_dir,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired_oof_per_image.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "gated_oof_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary["selected_oof"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
