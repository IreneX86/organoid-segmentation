"""Lock the global LD final method using predeclared inner-validation evidence.

This module reads each fold's locked validation predictions and derived masks,
selection lock, training history, and component-sampling gate. It never opens
an ``outer_test`` artifact or uses outer-test metrics.
The decision is exploratory because the rotated validation sets cover the same
36 images used in the LD cross-validation study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from ld_component_gate import _validation_metrics
from ld_final_policy import SCIENTIFIC_KEYS, _validation_best_epoch


BASELINE_ID = "ld_baseline_locked"
CANDIDATE_ID = "component_uniform_positive_center_v1"
POLICY_PATH = Path(__file__).resolve().parents[1] / "experiments" / "ld_production_selection_policy.md"
PAIRED_METRICS = ("dice", "precision", "recall")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _number(value: object, description: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {description}: boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {description}: {value!r}") from exc
    if not math.isfinite(number) or (unit_interval and not 0 <= number <= 1):
        raise ValueError(f"Invalid {description}: {value!r}")
    return number


def _close(actual: float, expected: float, description: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Validation evidence differs: {description}")


def _fold_dir(root: Path, fold: int) -> Path:
    fold_root = root / "ld" if (root / "ld").is_dir() else root
    return fold_root / f"fold_{fold}"


def _expected_partitions(fold_ids: dict[str, int], fold: int) -> dict[str, set[str]]:
    return {
        "test": {name for name, assigned in fold_ids.items() if assigned == fold},
        "validation": {
            name for name, assigned in fold_ids.items()
            if assigned == (fold + 1) % 5
        },
        "train": {
            name for name, assigned in fold_ids.items()
            if assigned not in (fold, (fold + 1) % 5)
        },
    }


def _validation_csv(path: Path, expected_names: set[str]) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected_names) or {row.get("filename") for row in rows} != expected_names:
        raise ValueError(f"Validation CSV does not cover the frozen validation fold: {path}")
    by_name = {}
    for row in rows:
        name = str(row["filename"])
        by_name[name] = {
            metric: _number(row[f"fixed_050_{metric}"], f"{name} {metric}",
                            unit_interval=True)
            for metric in PAIRED_METRICS
        }
    return by_name


def _load_run(
    directory: Path, fold: int, fold_ids: dict[str, int],
    map_sha: str, label_sha: str, label_dir: Path,
    label_records: dict[str, dict], *, candidate: bool,
) -> dict:
    """Validate one pre-test run without resolving any outer-test path."""
    config_path = directory / "experiment_config.json"
    split_path = directory / "split.json"
    lock_path = directory / "selection.lock.json"
    threshold_path = directory / "threshold.json"
    history_path = directory / "training_history.csv"
    manifest_path = directory / "validation_artifacts" / "manifest.json"
    csv_path = directory / "validation_artifacts" / "threshold_policy_per_image.csv"
    checkpoint = directory / ("component_unet.keras" if candidate else "baseline_unet.keras")
    config, split, lock = _json(config_path), _json(split_path), _json(lock_path)
    threshold_record, manifest = _json(threshold_path), _json(manifest_path)
    if (config.get("target") != "ld" or split.get("target") != "ld"
            or lock.get("target") != "ld" or config.get("fold") != fold
            or split.get("fold") != fold or lock.get("fold") != fold
            or config.get("num_folds") != 5 or split.get("num_folds") != 5
            or config.get("outer_test_deferred") is not True):
        raise ValueError(f"Invalid locked LD fold record: {directory}")
    if candidate:
        if (config.get("candidate") != CANDIDATE_ID
                or config.get("positive_center_policy")
                != "uniform_8_connected_training_component_then_pixel"):
            raise ValueError(f"Candidate method differs from the fixed protocol: {directory}")
    elif config.get("candidate") is not None:
        raise ValueError(f"Baseline run is marked as a candidate: {directory}")
    if split.get("fold_ids") != fold_ids:
        raise ValueError(f"Fold assignments differ from frozen LD map: {directory}")
    expected = _expected_partitions(fold_ids, fold)
    if any(set(split.get(part, [])) != names for part, names in expected.items()):
        raise ValueError(f"Train/validation/test partition differs: {directory}")
    if (config.get("fold_map_sha256") != map_sha
            or config.get("label_manifest_sha256") != label_sha):
        raise ValueError(f"Run uses another fold map or label version: {directory}")
    if not isinstance(config.get("label_dir"), str) or not config["label_dir"]:
        raise ValueError(f"Run lacks its recorded derived-label directory: {directory}")
    hashes = {
        "fold_map_sha256": map_sha,
        "label_manifest_sha256": label_sha,
        "experiment_config_sha256": _sha256(config_path),
        "split_sha256": _sha256(split_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "threshold_sha256": _sha256(threshold_path),
        "validation_artifacts_manifest_sha256": _sha256(manifest_path),
    }
    if candidate:
        component_path = directory / "training_component_counts.json"
        counts = _json(component_path)
        by_image = counts.get("eight_connected_components_per_image")
        if (counts.get("scope") != "training_masks_only" or not isinstance(by_image, dict)
                or set(by_image) != expected["train"]
                or any(type(value) is not int or value < 0 for value in by_image.values())):
            raise ValueError(f"Component counts are not training-only: {directory}")
        hashes["training_component_counts_sha256"] = _sha256(component_path)
    if any(lock.get(key) != value for key, value in hashes.items()):
        raise ValueError(f"Validation selection lock differs from saved run: {directory}")
    code_hashes = config.get("code_sha256")
    if not isinstance(code_hashes, dict) or not code_hashes or lock.get("code_sha256") != code_hashes:
        raise ValueError(f"Missing locked source-code hashes: {directory}")
    for name, digest in code_hashes.items():
        # Run snapshots avoid depending on a mutable current checkout.
        if (not isinstance(name, str) or name.startswith("/")
                or ".." in Path(name).parts or _sha256(directory / "source_snapshot" / name) != digest):
            raise ValueError(f"Source snapshot differs: {name}")
    if candidate:
        protocol_name = "experiments/ld_component_uniform_sampling_protocol.md"
        if config.get("protocol_sha256") != code_hashes.get(protocol_name):
            raise ValueError(f"Candidate protocol hash differs: {directory}")
    threshold = _number(threshold_record.get("threshold"), "selected threshold")
    if not 0 < threshold < 1 or lock.get("threshold") != threshold:
        raise ValueError(f"Validation-selected threshold differs from lock: {directory}")
    files = manifest.get("files")
    if (manifest.get("target") != "ld" or manifest.get("image_count") != len(expected["validation"])
            or not isinstance(files, list) or len(files) != len(expected["validation"])
            or {entry.get("filename") for entry in files} != expected["validation"]
            or manifest.get("threshold") != threshold):
        raise ValueError(f"Validation manifest differs from frozen fold: {directory}")
    metrics = _validation_csv(csv_path, expected["validation"])
    # The existing fold-local gate verifier checks all three saved TIFF hashes,
    # the derived label hash, and the fixed-0.50 CSV against recomputed metrics.
    # Restrict names to this fold's validation partition before calling it.
    recomputed = _validation_metrics(
        directory, expected["validation"], label_dir, label_records
    )
    for name in sorted(expected["validation"]):
        for metric in PAIRED_METRICS:
            if not math.isclose(metrics[name][metric], recomputed[name][metric],
                                rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Saved validation metric differs from TIFF: {name} {metric}")
    epoch_count, best_dice = _validation_best_epoch(history_path)
    if epoch_count < 1 or epoch_count > int(config["epochs"]):
        raise ValueError(f"Invalid validation-selected epoch: {directory}")
    # FullImageValidationCallback computes per-image fixed-0.50 Dice then its
    # arithmetic mean; the selected checkpoint uses this same val_full_dice.
    validation_mean = sum(recomputed[name]["dice"] for name in sorted(recomputed)) / len(recomputed)
    if not math.isclose(best_dice, validation_mean, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"Best-epoch Dice differs from locked validation predictions: {directory}")
    return {
        "config": config, "split": split, "lock": lock, "metrics": metrics,
        "fold_record": {
            "fold": fold,
            "validation_count": len(expected["validation"]),
            "validation_best_epoch_count": epoch_count,
            "validation_best_dice_at_0_5": best_dice,
            "validation_selected_threshold": threshold,
            "selection_lock_sha256": _sha256(lock_path),
            "training_history_sha256": _sha256(history_path),
            "validation_metrics_csv_sha256": _sha256(csv_path),
            "split_sha256": hashes["split_sha256"],
            "checkpoint_sha256": hashes["checkpoint_sha256"],
        },
        "validation_manifest_sha256": hashes["validation_artifacts_manifest_sha256"],
    }


def _check_gate(
    gate_path: Path, fold: int, baseline: dict, candidate: dict,
    map_sha: str, label_sha: str,
) -> tuple[dict, list[dict[str, float | str]]]:
    gate = _json(gate_path)
    expected_hashes = {
        "fold_map_sha256": map_sha,
        "label_manifest_sha256": label_sha,
        "baseline_selection_lock_sha256": baseline["fold_record"]["selection_lock_sha256"],
        "candidate_selection_lock_sha256": candidate["fold_record"]["selection_lock_sha256"],
        "baseline_validation_manifest_sha256": baseline["validation_manifest_sha256"],
        "candidate_validation_manifest_sha256": candidate["validation_manifest_sha256"],
        "protocol_sha256": candidate["config"]["protocol_sha256"],
    }
    if (gate.get("target") != "ld" or gate.get("candidate") != CANDIDATE_ID
            or gate.get("fold") != fold or gate.get("outer_test_labels_opened") is not False
            or any(gate.get(key) != value for key, value in expected_hashes.items())):
        raise ValueError(f"Fold-local gate provenance differs: {gate_path}")
    expected_names = set(baseline["metrics"])
    rows = gate.get("paired_validation_rows")
    if (not isinstance(rows, list) or len(rows) != len(expected_names)
            or gate.get("validation_filenames") != sorted(expected_names)
            or {row.get("filename") for row in rows} != expected_names):
        raise ValueError(f"Fold-local gate rows differ from validation split: {gate_path}")
    normalized = []
    for row in rows:
        name = row["filename"]
        base, cand = baseline["metrics"][name], candidate["metrics"][name]
        for metric in PAIRED_METRICS:
            _close(_number(row[f"baseline_{metric}_050"], f"baseline {metric}"),
                   base[metric], f"{name} baseline {metric}")
            _close(_number(row[f"candidate_{metric}_050"], f"candidate {metric}"),
                   cand[metric], f"{name} candidate {metric}")
        dice_delta = cand["dice"] - base["dice"]
        precision_delta = cand["precision"] - base["precision"]
        _close(_number(row["paired_dice_delta_050"], "paired Dice delta"),
               dice_delta, f"{name} Dice delta")
        _close(_number(row["paired_precision_delta_050"], "paired precision delta"),
               precision_delta, f"{name} precision delta")
        normalized.append({"filename": name, "dice_delta": dice_delta,
                           "precision_delta": precision_delta})
    dice = [float(row["dice_delta"]) for row in normalized]
    precision = [float(row["precision_delta"]) for row in normalized]
    # Each frozen validation fold has only seven or eight rows. Keep the
    # original gate's row order and float64-style sequential reduction.
    mean_dice = sum(dice) / len(dice)
    mean_precision = sum(precision) / len(precision)
    expected_checks = {
        "mean_dice_delta_at_least_0_01": mean_dice >= 0.01,
        "strict_majority_dice_improved": sum(delta > 0 for delta in dice) > len(dice) / 2,
        "no_single_dice_drop_more_than_0_10": all(delta >= -0.10 for delta in dice),
        "mean_precision_drop_no_more_than_0_03": mean_precision >= -0.03,
    }
    if gate.get("gate_checks") != expected_checks or gate.get("selected") != (
        "candidate" if all(expected_checks.values()) else "baseline"
    ):
        raise ValueError(f"Fold-local gate decision disagrees with validation rows: {gate_path}")
    _close(_number(gate.get("mean_paired_dice_delta_050"), "fold Dice delta"),
           mean_dice, f"fold {fold} mean Dice delta")
    _close(_number(gate.get("mean_paired_precision_delta_050"), "fold precision delta"),
           mean_precision, f"fold {fold} mean precision delta")
    return {
        "fold": fold,
        "gate_selected": gate["selected"],
        "gate_sha256": _sha256(gate_path),
        "validation_count": len(normalized),
        "mean_paired_dice_delta_050": mean_dice,
        "mean_paired_precision_delta_050": mean_precision,
        "minimum_paired_dice_delta_050": min(dice),
    }, normalized


def build_method_policy(
    baseline_cv_root: Path, candidate_cv_root: Path,
    fold_map_path: Path, label_manifest_path: Path, label_dir: Path,
) -> dict:
    """Apply all four locked global criteria to 36 paired validation images."""
    frozen = _json(fold_map_path)
    fold_ids = frozen.get("fold_ids")
    if (frozen.get("target") != "ld" or frozen.get("num_folds") != 5
            or not isinstance(fold_ids, dict) or len(fold_ids) != 36
            or any(type(value) is not int or value not in range(5)
                   for value in fold_ids.values())
            or set(fold_ids.values()) != set(range(5))):
        raise ValueError("Expected the frozen 36-image, five-fold LD map")
    map_sha, label_sha = _sha256(fold_map_path), _sha256(label_manifest_path)
    manifest = _json(label_manifest_path)
    records = manifest.get("files")
    if (manifest.get("rule_id") != "ld_antialias_midpoint_v1"
            or manifest.get("label_semantics") != {"background": 0, "foreground": 1}
            or not isinstance(records, list) or len(records) != 36
            or any(not isinstance(item, dict) or not isinstance(item.get("filename"), str)
                   or not isinstance(item.get("derived_mask_sha256"), str)
                   or len(item["derived_mask_sha256"]) != 64 for item in records)
            or {item["filename"] for item in records} != set(fold_ids)):
        raise ValueError("Derived LD label manifest has wrong semantics or nonunique records")
    label_dir = label_dir.resolve()
    # Archived runs may have been trained under a remote absolute path while
    # this policy is built locally. Relocation is safe only after byte-for-byte
    # verification of all 36 derived masks against the frozen manifest. The
    # baseline and candidate configs must still agree on their original path.
    for item in records:
        name = item["filename"]
        if Path(name).name != name or Path(name).suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Invalid derived-label filename: {name}")
        if _sha256(label_dir / name) != item["derived_mask_sha256"]:
            raise ValueError(f"Relocated derived label differs from manifest: {name}")
    label_records = {item["filename"]: item for item in records}
    baseline_records, candidate_records, gates, pairs = [], [], [], []
    comparison_provenance = []
    baseline_settings = candidate_settings = None
    for fold in range(5):
        baseline_dir = _fold_dir(baseline_cv_root, fold)
        candidate_dir = _fold_dir(candidate_cv_root, fold)
        baseline = _load_run(baseline_dir, fold, fold_ids, map_sha, label_sha,
                             label_dir, label_records, candidate=False)
        candidate = _load_run(candidate_dir, fold, fold_ids, map_sha, label_sha,
                              label_dir, label_records, candidate=True)
        if baseline["split"] != candidate["split"]:
            raise ValueError(f"Fold {fold} baseline and candidate use different splits")
        shared_keys = tuple(key for key in SCIENTIFIC_KEYS if key != "code_sha256") + (
            "seed", "input_dir", "label_dir",
        )
        if any(baseline["config"].get(key) != candidate["config"].get(key)
               for key in shared_keys):
            raise ValueError(f"Fold {fold} altered another scientific factor")
        bs = {key: baseline["config"].get(key) for key in SCIENTIFIC_KEYS}
        cs = {key: candidate["config"].get(key) for key in SCIENTIFIC_KEYS}
        cs["positive_center_policy"] = candidate["config"]["positive_center_policy"]
        if baseline_settings is None:
            baseline_settings, candidate_settings = bs, cs
        elif bs != baseline_settings or cs != candidate_settings:
            raise ValueError("Five folds have inconsistent scientific settings")
        gate, fold_pairs = _check_gate(candidate_dir / "decision.lock.json", fold,
                                       baseline, candidate, map_sha, label_sha)
        gates.append(gate)
        pairs.extend(fold_pairs)
        baseline_records.append(baseline["fold_record"])
        candidate_records.append(candidate["fold_record"])
        comparison_provenance.append({
            "fold": fold,
            "split_sha256": baseline["fold_record"]["split_sha256"],
            "baseline_selection_lock_sha256": baseline["fold_record"]["selection_lock_sha256"],
            "candidate_selection_lock_sha256": candidate["fold_record"]["selection_lock_sha256"],
            "baseline_validation_metrics_csv_sha256": baseline["fold_record"][
                "validation_metrics_csv_sha256"
            ],
            "candidate_validation_metrics_csv_sha256": candidate["fold_record"][
                "validation_metrics_csv_sha256"
            ],
            "gate_sha256": gate["gate_sha256"],
        })
    names = [row["filename"] for row in pairs]
    if len(names) != 36 or set(names) != set(fold_ids):
        raise ValueError("Paired inner-validation must cover all 36 images exactly once")
    dice = [float(row["dice_delta"]) for row in pairs]
    gate_wins = sum(row["gate_selected"] == "candidate" for row in gates)
    mean_dice = math.fsum(dice) / 36
    checks = {
        "candidate_gate_wins_at_least_4_of_5": gate_wins >= 4,
        "all_36_mean_paired_dice_delta_at_least_0_01": mean_dice >= 0.01,
        "each_fold_mean_precision_delta_at_least_minus_0_03": all(
            row["mean_paired_precision_delta_050"] >= -0.03 for row in gates
        ),
        "no_individual_dice_delta_below_minus_0_10": all(
            delta >= -0.10 for delta in dice
        ),
    }
    selected = "candidate" if all(checks.values()) else "baseline"
    records = candidate_records if selected == "candidate" else baseline_records
    settings = candidate_settings if selected == "candidate" else baseline_settings
    assert settings is not None
    return {
        "policy_version": "ld_global_validation_method_v1",
        "target": "ld",
        "candidate_id": CANDIDATE_ID if selected == "candidate" else BASELINE_ID,
        "selected_method": selected,
        "selection_basis": "predeclared_five_fold_inner_validation_only",
        "outer_test_used_for_selection": False,
        "policy_protocol_sha256": _sha256(POLICY_PATH),
        "fold_map_sha256": map_sha,
        "label_manifest_sha256": label_sha,
        "derived_label_verification": "all_36_local_TIFF_SHA256_match_frozen_manifest",
        "image_count": 36,
        "global_checks": checks,
        "gate_wins": gate_wins,
        "mean_paired_dice_delta_050_all_36": mean_dice,
        "minimum_paired_dice_delta_050_all_36": min(dice),
        "fold_gate_records": gates,
        "comparison_provenance": comparison_provenance,
        "final_epoch_count": int(statistics.median(
            [row["validation_best_epoch_count"] for row in records]
        )),
        "deployment_threshold": float(statistics.median(
            [row["validation_selected_threshold"] for row in records]
        )),
        "epoch_rule": "median of five earliest best val_full_dice epoch counts",
        "threshold_rule": "median of all five validation-selected thresholds",
        "scientific_settings": settings,
        "fold_validation_records": records,
        "evidence_scope": "36 paired inner-validation appearances; exploratory same-dataset selection",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cv-root", type=Path, required=True)
    parser.add_argument("--candidate-cv-root", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Policy already exists: {args.output}")
    policy = build_method_policy(
        args.baseline_cv_root, args.candidate_cv_root,
        args.fold_map, args.label_manifest, args.label_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"LD_METHOD_POLICY_LOCKED selected={policy['selected_method']} "
          f"epochs={policy['final_epoch_count']} "
          f"threshold={policy['deployment_threshold']:.2f}")


if __name__ == "__main__":
    main()
