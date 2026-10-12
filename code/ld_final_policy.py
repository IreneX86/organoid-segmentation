"""Lock an LD production policy using five folds' validation records only.

This command never reads ``outer_test`` results. Candidate choice must already
have been made from the inner-validation evidence; its identifier is recorded
here for provenance, not selected by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


SCIENTIFIC_KEYS = (
    "architecture", "loss", "seed_base", "patch_size", "overlap",
    "batch_size", "prediction_batch_size", "steps_per_epoch", "epochs",
    "learning_rate", "positive_patch_probability", "lower_percentile",
    "upper_percentile", "checkpoint_selection", "threshold_selection",
    "label_manifest_sha256", "fold_map_sha256", "code_sha256",
)

# This entry point currently supports the frozen baseline only. A later LD
# candidate needs its own validated policy builder, not a renamed baseline.
BASELINE_CANDIDATE_ID = "ld_baseline_locked"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validation_best_epoch(path: Path) -> tuple[int, float]:
    """Match earliest-maximum validation Dice used by ModelCheckpoint."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty training history: {path}")
    best_epoch = -1
    best_dice = -math.inf
    for row in rows:
        epoch = int(row["epoch"])
        score = float(row["val_full_dice"])
        if epoch < 0 or not math.isfinite(score):
            raise ValueError(f"Invalid epoch or validation Dice in {path}")
        if score > best_dice:
            best_epoch, best_dice = epoch, score
    return best_epoch + 1, best_dice


def build_production_policy(
    cv_root: Path,
    fold_map_path: Path,
    label_manifest_path: Path,
    candidate_id: str,
) -> dict[str, object]:
    """Derive fixed epochs and threshold without touching any outer-test file."""
    if candidate_id != BASELINE_CANDIDATE_ID:
        raise ValueError(
            f"This policy supports only candidate_id={BASELINE_CANDIDATE_ID!r}"
        )
    cv_root = cv_root.resolve()
    fold_root = cv_root / "ld" if (cv_root / "ld").is_dir() else cv_root
    frozen = _read_json(fold_map_path)
    if frozen.get("target") != "ld" or frozen.get("num_folds") != 5:
        raise ValueError("Expected a frozen five-fold LD map")
    fold_ids = frozen.get("fold_ids")
    if not isinstance(fold_ids, dict) or len(fold_ids) != 36:
        raise ValueError("Expected exactly 36 LD fold assignments")
    fold_map_sha = _sha256(fold_map_path)
    label_manifest_sha = _sha256(label_manifest_path)
    settings: dict[str, object] | None = None
    fold_records: list[dict[str, object]] = []
    best_epoch_counts: list[int] = []
    selected_thresholds: list[float] = []

    for fold in range(5):
        directory = fold_root / f"fold_{fold}"
        config = _read_json(directory / "experiment_config.json")
        split = _read_json(directory / "split.json")
        lock = _read_json(directory / "selection.lock.json")
        threshold_record = _read_json(directory / "threshold.json")
        if (config.get("target"), config.get("fold"), split.get("target"),
                split.get("fold")) != ("ld", fold, "ld", fold):
            raise ValueError(f"Fold {fold} has mismatched LD records")
        if config.get("outer_test_deferred") is not True:
            raise ValueError(f"Fold {fold} did not defer the outer test")
        if split.get("fold_ids") != fold_ids:
            raise ValueError(f"Fold {fold} used another assignment")
        validation_fold = (fold + 1) % 5
        expected = {
            "test": {name for name, assigned in fold_ids.items() if assigned == fold},
            "validation": {
                name for name, assigned in fold_ids.items()
                if assigned == validation_fold
            },
            "train": {
                name for name, assigned in fold_ids.items()
                if assigned not in (fold, validation_fold)
            },
        }
        if any(set(split.get(key, [])) != names for key, names in expected.items()):
            raise ValueError(f"Fold {fold} partitions differ from frozen rotation")
        provenance_hashes = {
            "fold_map_sha256": fold_map_sha,
            "label_manifest_sha256": label_manifest_sha,
            "experiment_config_sha256": _sha256(directory / "experiment_config.json"),
            "split_sha256": _sha256(directory / "split.json"),
            "checkpoint_sha256": _sha256(directory / "baseline_unet.keras"),
            "threshold_sha256": _sha256(directory / "threshold.json"),
            "validation_artifacts_manifest_sha256": _sha256(
                directory / "validation_artifacts" / "manifest.json"
            ),
        }
        if any(lock.get(key) != value for key, value in provenance_hashes.items()):
            raise ValueError(f"Fold {fold} validation selection lock has changed")
        if lock.get("code_sha256") != config.get("code_sha256"):
            raise ValueError(f"Fold {fold} code provenance has changed")
        fold_settings = {key: config.get(key) for key in SCIENTIFIC_KEYS}
        if settings is None:
            settings = fold_settings
        elif fold_settings != settings:
            raise ValueError("Five folds must use the same scientific configuration")
        threshold = float(threshold_record["threshold"])
        if not 0 < threshold < 1 or lock.get("threshold") != threshold:
            raise ValueError(f"Fold {fold} has an invalid validation-selected threshold")
        epoch_count, val_dice = _validation_best_epoch(directory / "training_history.csv")
        if epoch_count > int(config["epochs"]):
            raise ValueError(f"Fold {fold} best epoch exceeds the training budget")
        best_epoch_counts.append(epoch_count)
        selected_thresholds.append(threshold)
        fold_records.append({
            "fold": fold, "validation_best_epoch_count": epoch_count,
            "validation_best_dice_at_0_5": val_dice,
            "validation_selected_threshold": threshold,
            "selection_lock_sha256": _sha256(directory / "selection.lock.json"),
            "training_history_sha256": _sha256(directory / "training_history.csv"),
        })

    assert settings is not None
    if settings["architecture"] != "unet_2d_baseline" or settings["loss"] != "bce_dice":
        raise ValueError("The current production entry point supports only the LD U-Net baseline")
    policy = {
        "policy_version": "ld_validation_median_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "ld", "candidate_id": candidate_id,
        "selection_basis": "five_fold_inner_validation_only",
        "outer_test_used_for_selection": False,
        "label_manifest_sha256": label_manifest_sha,
        "fold_map_sha256": fold_map_sha,
        "image_count": 36,
        "final_epoch_count": int(statistics.median(best_epoch_counts)),
        "deployment_threshold": float(statistics.median(selected_thresholds)),
        "epoch_rule": "median of five earliest best val_full_dice epoch counts",
        "threshold_rule": "median of five validation-selected thresholds",
        "scientific_settings": settings,
        "fold_validation_records": fold_records,
    }
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = build_production_policy(
        args.cv_root, args.fold_map, args.label_manifest, args.candidate_id
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True)
    print(f"LD_PRODUCTION_POLICY_LOCKED epochs={policy['final_epoch_count']} "
          f"threshold={policy['deployment_threshold']:.2f}")


if __name__ == "__main__":
    main()
