"""Lock the LD component-sampling decision using only inner validation labels.

Run this separately for each fold after candidate training, before any
candidate outer-test evaluation. Never aggregate validation folds to choose a
single global method: each image is outer test in another rotated fold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ld_component_train import CODE_RELATIVE_PATHS, _code_sha256_manifest, _sha256
from ld_evaluate import _load_json, _validate_locked_run as validate_baseline
from segmentation_metrics import calculate_binary_metrics
from tiff_io import read_binary_mask


PROTOCOL_RELATIVE = "experiments/ld_component_uniform_sampling_protocol.md"
COMPONENT_MODEL = "component_unet.keras"
UNCHANGED_SCIENCE_KEYS = (
    "architecture", "loss", "seed_base", "seed", "patch_size", "overlap",
    "batch_size", "prediction_batch_size", "steps_per_epoch", "epochs",
    "learning_rate", "positive_patch_probability", "lower_percentile",
    "upper_percentile", "checkpoint_selection", "threshold_selection",
)


def _expected_partitions(fold_ids: dict[str, int], fold: int) -> dict[str, set[str]]:
    validation_fold = (fold + 1) % 5
    return {
        "test": {name for name, index in fold_ids.items() if index == fold},
        "validation": {
            name for name, index in fold_ids.items() if index == validation_fold
        },
        "train": {
            name for name, index in fold_ids.items()
            if index not in (fold, validation_fold)
        },
    }


def validate_candidate(
    run_dir: Path, fold_map: Path, label_manifest: Path
) -> tuple[dict[str, object], dict[str, object], float]:
    """Verify candidate science, source snapshot, and pre-test selection lock."""

    run_dir = run_dir.resolve()
    config = _load_json(run_dir / "experiment_config.json")
    split = _load_json(run_dir / "split.json")
    locked = _load_json(run_dir / "selection.lock.json")
    threshold = float(_load_json(run_dir / "threshold.json")["threshold"])
    fold = split.get("fold")
    if (config.get("target") != "ld" or split.get("target") != "ld"
            or locked.get("target") != "ld" or type(fold) is not int
            or fold not in range(5) or config.get("fold") != fold
            or locked.get("fold") != fold or config.get("num_folds") != 5
            or split.get("num_folds") != 5):
        raise ValueError("Candidate run does not identify one frozen LD fold")
    if (config.get("candidate") != "component_uniform_positive_center_v1"
            or config.get("positive_center_policy")
            != "uniform_8_connected_training_component_then_pixel"
            or config.get("positive_patch_probability") != 0.6
            or not config.get("outer_test_deferred")):
        raise ValueError("Run does not follow the component-uniform protocol")
    expected_hashes = {
        "fold_map_sha256": _sha256(fold_map),
        "label_manifest_sha256": _sha256(label_manifest),
        "checkpoint_sha256": _sha256(run_dir / COMPONENT_MODEL),
        "threshold_sha256": _sha256(run_dir / "threshold.json"),
        "experiment_config_sha256": _sha256(run_dir / "experiment_config.json"),
        "split_sha256": _sha256(run_dir / "split.json"),
        "validation_artifacts_manifest_sha256": _sha256(
            run_dir / "validation_artifacts" / "manifest.json"
        ),
        "training_component_counts_sha256": _sha256(
            run_dir / "training_component_counts.json"
        ),
    }
    for name, digest in expected_hashes.items():
        if locked.get(name) != digest:
            raise ValueError(f"Candidate {name} differs from its selection lock")
        if name in ("fold_map_sha256", "label_manifest_sha256") and config.get(name) != digest:
            raise ValueError(f"Candidate {name} differs from its experiment configuration")
    if locked.get("threshold") != threshold or not 0 < threshold < 1:
        raise ValueError("Candidate threshold differs from its selection lock")
    actual_code = _code_sha256_manifest()
    if (set(actual_code) != set(CODE_RELATIVE_PATHS)
            or config.get("code_sha256") != actual_code
            or locked.get("code_sha256") != actual_code
            or config.get("protocol_sha256") != actual_code[PROTOCOL_RELATIVE]):
        raise ValueError("Candidate source or protocol changed since fitting")
    for name, digest in actual_code.items():
        if _sha256(run_dir / "source_snapshot" / name) != digest:
            raise ValueError(f"Candidate source snapshot changed: {name}")

    frozen = _load_json(fold_map)
    fold_ids = frozen.get("fold_ids")
    if (frozen.get("target") != "ld" or frozen.get("num_folds") != 5
            or not isinstance(fold_ids, dict) or len(fold_ids) != 36
            or split.get("fold_ids") != fold_ids):
        raise ValueError("Candidate split differs from frozen 36-image LD map")
    expected = _expected_partitions(fold_ids, fold)
    if any(set(split.get(part, [])) != names for part, names in expected.items()):
        raise ValueError("Candidate train/validation/test partition changed")
    components = _load_json(run_dir / "training_component_counts.json")
    counts = components.get("eight_connected_components_per_image")
    if (components.get("scope") != "training_masks_only"
            or not isinstance(counts, dict) or set(counts) != expected["train"]
            or any(type(value) is not int or value < 0 for value in counts.values())):
        raise ValueError("Candidate training component-count record is invalid")
    return config, split, threshold


def _validation_metrics(
    run_dir: Path, names: set[str], label_dir: Path, label_records: dict[str, dict]
) -> dict[str, dict[str, float]]:
    """Recompute fixed-0.50 metrics from locked validation artifacts only."""

    artifacts = run_dir / "validation_artifacts"
    manifest = _load_json(artifacts / "manifest.json")
    records = manifest.get("files")
    if (not isinstance(records, list) or manifest.get("image_count") != len(names)
            or {item["filename"] for item in records} != names):
        raise ValueError("Validation artifact manifest has incorrect filenames")
    expected_csv = artifacts / "threshold_policy_per_image.csv"
    with expected_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(names) or {row["filename"] for row in csv_rows} != names:
        raise ValueError("Validation policy CSV differs from frozen validation set")
    csv_by_name = {row["filename"]: row for row in csv_rows}
    output: dict[str, dict[str, float]] = {}
    for item in records:
        name = str(item["filename"])
        prediction = artifacts / "predictions" / "fixed_050" / name
        selected = artifacts / "predictions" / "validation" / name
        probability = artifacts / "probabilities" / name
        if (item.get("fixed_050_mask_sha256") != _sha256(prediction)
                or item.get("validation_mask_sha256") != _sha256(selected)
                or item.get("probability_sha256") != _sha256(probability)):
            raise ValueError(f"Validation artifacts changed: {name}")
        label = label_dir / name
        if _sha256(label) != label_records[name]["derived_mask_sha256"]:
            raise ValueError(f"Derived validation mask changed: {name}")
        truth = read_binary_mask(label)
        predicted = read_binary_mask(prediction)
        metrics = calculate_binary_metrics(truth, predicted)
        saved = csv_by_name[name]
        for metric in ("dice", "iou", "precision", "recall"):
            if not np.isclose(getattr(metrics, metric), float(saved[f"fixed_050_{metric}"]),
                              atol=1e-6, rtol=0):
                raise ValueError(f"Stored validation {metric} differs for {name}")
        output[name] = {
            "dice": metrics.dice,
            "precision": metrics.precision,
            "recall": metrics.recall,
        }
    return output


def compute_gate(
    baseline_run: Path,
    candidate_run: Path,
    fold_map: Path,
    label_manifest: Path,
    label_dir: Path,
) -> dict[str, object]:
    """Return the deterministic per-fold decision; never read outer masks."""

    baseline_run = baseline_run.resolve()
    candidate_run = candidate_run.resolve()
    fold_map = fold_map.resolve()
    label_manifest = label_manifest.resolve()
    label_dir = label_dir.resolve()
    baseline_config, baseline_split, _ = validate_baseline(
        baseline_run, fold_map, label_manifest
    )
    candidate_config, candidate_split, _ = validate_candidate(
        candidate_run, fold_map, label_manifest
    )
    if (baseline_split != candidate_split
            or baseline_config.get("input_dir") != candidate_config.get("input_dir")
            or Path(str(baseline_config.get("label_dir"))) != label_dir
            or Path(str(candidate_config.get("label_dir"))) != label_dir):
        raise ValueError("Baseline and candidate did not use the same LD split and paths")
    if any(
        baseline_config.get(key) != candidate_config.get(key)
        for key in UNCHANGED_SCIENCE_KEYS
    ):
        raise ValueError("Candidate altered a scientific factor beyond positive-center sampling")
    fold = int(candidate_split["fold"])
    validation_names = set(candidate_split["validation"])
    derived = _load_json(label_manifest)
    records = {item["filename"]: item for item in derived.get("files", [])}
    if (derived.get("rule_id") != "ld_antialias_midpoint_v1"
            or set(records) != set(candidate_split["fold_ids"])):
        raise ValueError("Derived LD label manifest differs from the frozen definition")
    base = _validation_metrics(baseline_run, validation_names, label_dir, records)
    candidate = _validation_metrics(candidate_run, validation_names, label_dir, records)
    rows: list[dict[str, float | str]] = []
    for name in sorted(validation_names):
        rows.append({
            "filename": name,
            "baseline_dice_050": base[name]["dice"],
            "candidate_dice_050": candidate[name]["dice"],
            "paired_dice_delta_050": candidate[name]["dice"] - base[name]["dice"],
            "baseline_precision_050": base[name]["precision"],
            "candidate_precision_050": candidate[name]["precision"],
            "paired_precision_delta_050": (
                candidate[name]["precision"] - base[name]["precision"]
            ),
            "baseline_recall_050": base[name]["recall"],
            "candidate_recall_050": candidate[name]["recall"],
        })
    dice = np.asarray([float(row["paired_dice_delta_050"]) for row in rows])
    precision = np.asarray([float(row["paired_precision_delta_050"]) for row in rows])
    checks = {
        "mean_dice_delta_at_least_0_01": bool(dice.mean() >= 0.01),
        "strict_majority_dice_improved": bool(np.count_nonzero(dice > 0) > len(dice) / 2),
        "no_single_dice_drop_more_than_0_10": bool(np.all(dice >= -0.10)),
        "mean_precision_drop_no_more_than_0_03": bool(precision.mean() >= -0.03),
    }
    return {
        "target": "ld",
        "candidate": "component_uniform_positive_center_v1",
        "fold": fold,
        "protocol_sha256": candidate_config["protocol_sha256"],
        "fold_map_sha256": _sha256(fold_map),
        "label_manifest_sha256": _sha256(label_manifest),
        "baseline_selection_lock_sha256": _sha256(baseline_run / "selection.lock.json"),
        "candidate_selection_lock_sha256": _sha256(candidate_run / "selection.lock.json"),
        "baseline_validation_manifest_sha256": _sha256(
            baseline_run / "validation_artifacts" / "manifest.json"
        ),
        "candidate_validation_manifest_sha256": _sha256(
            candidate_run / "validation_artifacts" / "manifest.json"
        ),
        "validation_filenames": sorted(validation_names),
        "paired_validation_rows": rows,
        "mean_paired_dice_delta_050": float(dice.mean()),
        "mean_paired_precision_delta_050": float(precision.mean()),
        "gate_checks": checks,
        "selected": "candidate" if all(checks.values()) else "baseline",
        "outer_test_labels_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    args = parser.parse_args()
    decision_path = args.candidate_run_dir / "decision.lock.json"
    if decision_path.exists():
        raise FileExistsError(f"Candidate decision is already locked: {decision_path}")
    decision = compute_gate(
        args.baseline_run_dir, args.candidate_run_dir, args.fold_map,
        args.label_manifest, args.label_dir,
    )
    with decision_path.open("x", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    print(f"LD_COMPONENT_GATE_LOCKED fold={decision['fold']} selected={decision['selected']}")


if __name__ == "__main__":
    main()
