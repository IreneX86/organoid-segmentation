"""Aggregate five locked LD focal-loss outer-test folds descriptively.

This is a reporting step only. It checks that each of the 36 labelled images
occurs in exactly one outer-test fold and that all folds used one scientific
configuration and one version of the derived LD labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ld_focal_train import CODE_RELATIVE_PATHS


METRICS = (
    "dice", "iou", "precision", "recall", "foreground_fraction_true",
    "foreground_fraction_predicted",
)
SCIENTIFIC_KEYS = (
    "architecture", "loss", "seed_base", "patch_size", "overlap", "batch_size",
    "prediction_batch_size", "steps_per_epoch", "epochs", "learning_rate",
    "positive_patch_probability", "lower_percentile", "upper_percentile",
    "checkpoint_selection", "threshold_selection", "label_manifest_sha256",
    "fold_map_sha256", "code_sha256",
)


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Metric values must be non-empty and finite")
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()), "max": float(array.max()),
    }


def aggregate_cv(run_root: Path) -> tuple[list[dict[str, str | float | int]], dict[str, object]]:
    """Read completed folds and fail on missing, duplicate, or inconsistent OOF data."""
    run_root = run_root.resolve()
    fold_root = run_root / "ld" if (run_root / "ld").is_dir() else run_root
    all_rows: list[dict[str, str | float | int]] = []
    seen: set[str] = set()
    canonical_fold_ids: dict[str, int] | None = None
    canonical_config: dict[str, object] | None = None
    fold_summaries: list[dict[str, object]] = []

    for fold in range(5):
        directory = fold_root / f"fold_{fold}"
        split = _load_json(directory / "split.json")
        config = _load_json(directory / "experiment_config.json")
        locked = _load_json(directory / "selection.lock.json")
        evaluated = _load_json(directory / "outer_test" / "evaluation_manifest.json")
        threshold = _load_json(directory / "threshold.json")
        if (split.get("target"), split.get("fold"), split.get("num_folds")) != (
            "ld", fold, 5
        ):
            raise ValueError(f"Fold {fold} has an invalid split manifest")
        if (config.get("target") != "ld" or config.get("fold") != fold
                or config.get("candidate") != "ld_focal_dice_gamma2_v1"
                or config.get("loss") != "focal_bce_gamma2_plus_soft_dice"):
            raise ValueError(f"Fold {fold} has an invalid LD experiment configuration")
        if (type(config.get("seed_base")) is not int
                or config.get("seed") != config["seed_base"] + fold):
            raise ValueError(f"Fold {fold} used a seed outside the frozen fold schedule")
        if evaluated.get("fold") != fold or locked.get("fold") != fold:
            raise ValueError(f"Fold {fold} has mismatched selection/evaluation records")
        if evaluated.get("selection_lock_sha256") != _sha256(
            directory / "selection.lock.json"
        ):
            raise ValueError(f"Fold {fold} was evaluated with a different selection lock")
        if locked.get("split_sha256") != _sha256(directory / "split.json"):
            raise ValueError(f"Fold {fold} split differs from its selection lock")
        if locked.get("experiment_config_sha256") != _sha256(
            directory / "experiment_config.json"
        ):
            raise ValueError(f"Fold {fold} configuration differs from its selection lock")
        if locked.get("target") != "ld":
            raise ValueError(f"Fold {fold} has a non-LD selection lock")
        for name in ("label_manifest_sha256", "fold_map_sha256"):
            if locked.get(name) != config.get(name):
                raise ValueError(f"Fold {fold} provenance differs from its selection lock")
        if locked.get("checkpoint_sha256") != _sha256(directory / "focal_unet.keras"):
            raise ValueError(f"Fold {fold} checkpoint differs from its selection lock")
        if locked.get("threshold_sha256") != _sha256(directory / "threshold.json"):
            raise ValueError(f"Fold {fold} threshold file differs from its selection lock")
        validation_dir = directory / "validation_artifacts"
        validation_manifest = _load_json(validation_dir / "manifest.json")
        if locked.get("validation_artifacts_manifest_sha256") != _sha256(
            validation_dir / "manifest.json"
        ):
            raise ValueError(f"Fold {fold} validation artifacts differ from the selection lock")
        validation_files = validation_manifest.get("files")
        if (not isinstance(validation_files, list)
                or validation_manifest.get("image_count") != len(split.get("validation", []))
                or {item["filename"] for item in validation_files}
                != set(split.get("validation", []))):
            raise ValueError(f"Fold {fold} validation artifact names are incomplete")
        for item in validation_files:
            name = item["filename"]
            artifact_paths = {
                "probability_sha256": validation_dir / "probabilities" / name,
                "validation_mask_sha256": (
                    validation_dir / "predictions" / "validation" / name
                ),
                "fixed_050_mask_sha256": (
                    validation_dir / "predictions" / "fixed_050" / name
                ),
            }
            for field, path in artifact_paths.items():
                if item.get(field) != _sha256(path):
                    raise ValueError(f"Fold {fold} validation artifact changed: {path}")
        if locked.get("threshold") != threshold.get("threshold"):
            raise ValueError(f"Fold {fold} threshold differs from its selection lock")
        code_hashes = config.get("code_sha256")
        if (not isinstance(code_hashes, dict)
                or set(code_hashes) != set(CODE_RELATIVE_PATHS)
                or locked.get("code_sha256") != code_hashes):
            raise ValueError(f"Fold {fold} has inconsistent source-code provenance")
        for name, expected in code_hashes.items():
            if _sha256(directory / "source_snapshot" / name) != expected:
                raise ValueError(f"Fold {fold} frozen source differs: {name}")

        fold_ids = split.get("fold_ids")
        if not isinstance(fold_ids, dict):
            raise ValueError(f"Fold {fold} has no frozen fold IDs")
        if canonical_fold_ids is None:
            canonical_fold_ids = fold_ids
            if len(canonical_fold_ids) != 36:
                raise ValueError("The LD study requires exactly 36 labelled images")
        elif fold_ids != canonical_fold_ids:
            raise ValueError("Different LD folds used different frozen assignments")
        settings = {key: config.get(key) for key in SCIENTIFIC_KEYS}
        if canonical_config is None:
            canonical_config = settings
        elif settings != canonical_config:
            raise ValueError(f"Fold {fold} used different scientific settings or label version")
        validation_fold = (fold + 1) % 5
        expected_partition = {
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
        if any(set(split.get(key, [])) != names for key, names in expected_partition.items()):
            raise ValueError(f"Fold {fold} partitions differ from the rotated fold assignment")
        partition = [set(split.get(key, [])) for key in ("train", "validation", "test")]
        if any(partition[i] & partition[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"Fold {fold} has overlapping partitions")
        if set.union(*partition) != set(fold_ids):
            raise ValueError(f"Fold {fold} does not cover the 36-image dataset")
        test_names = set(split["test"])
        if not test_names or test_names != {
            name for name, assigned in fold_ids.items() if assigned == fold
        }:
            raise ValueError(f"Fold {fold} test names disagree with frozen assignments")
        if test_names & seen:
            raise ValueError("An image appears in more than one outer-test fold")
        seen.update(test_names)
        if set(evaluated.get("test_filenames", [])) != test_names:
            raise ValueError(f"Fold {fold} evaluation manifest omits test images")

        with (directory / "outer_test" / "test_per_image.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            raw_rows = list(csv.DictReader(handle))
        if {row["filename"] for row in raw_rows} != test_names or len(raw_rows) != len(test_names):
            raise ValueError(f"Fold {fold} per-image metrics do not match its test set")
        with (directory / "outer_test" / "threshold_policy_per_image.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            policy_rows = list(csv.DictReader(handle))
        if len(policy_rows) != len(test_names) or {
            row["filename"] for row in policy_rows
        } != test_names:
            raise ValueError(f"Fold {fold} threshold policies do not cover its test set")
        policies = {row["filename"]: row for row in policy_rows}
        saved_summary = _load_json(directory / "outer_test" / "test_summary.json")
        if int(saved_summary.get("image_count", -1)) != len(test_names):
            raise ValueError(f"Fold {fold} has an incomplete test summary")
        fold_rows: list[dict[str, str | float | int]] = []
        for row in raw_rows:
            converted: dict[str, str | float | int] = {
                "fold": fold, "filename": row["filename"],
            }
            for metric in METRICS:
                number = float(row[metric])
                if not np.isfinite(number) or not 0 <= number <= 1:
                    raise ValueError(f"Invalid {metric} for {row['filename']}")
                converted[metric] = number
            policy = policies[row["filename"]]
            for metric in ("dice", "iou", "precision", "recall"):
                selected = float(policy[f"validation_{metric}"])
                fixed = float(policy[f"fixed_050_{metric}"])
                if not np.isfinite(fixed) or not 0 <= fixed <= 1:
                    raise ValueError(f"Invalid fixed threshold {metric} for {row['filename']}")
                if not np.isclose(selected, float(converted[metric]), atol=1e-9):
                    raise ValueError(f"Selected policy {metric} differs for {row['filename']}")
                converted[f"fixed_050_{metric}"] = fixed
            fold_rows.append(converted)
        all_rows.extend(fold_rows)
        for metric in ("dice", "iou", "precision", "recall"):
            mean = _statistics([float(row[metric]) for row in fold_rows])["mean"]
            if not np.isclose(float(saved_summary[f"{metric}_mean"]), mean, atol=1e-9):
                raise ValueError(f"Fold {fold} summary and per-image {metric} disagree")
        fold_summaries.append({
            "fold": fold, "n": len(fold_rows),
            "threshold": float(threshold["threshold"]),
            "dice": _statistics([float(row["dice"]) for row in fold_rows]),
            "iou": _statistics([float(row["iou"]) for row in fold_rows]),
            "precision": _statistics([float(row["precision"]) for row in fold_rows]),
            "recall": _statistics([float(row["recall"]) for row in fold_rows]),
        })

    if canonical_fold_ids is None or seen != set(canonical_fold_ids):
        raise ValueError("Five folds do not provide exactly one OOF score per image")
    all_rows.sort(key=lambda row: str(row["filename"]).lower())
    report: dict[str, object] = {
        "target": "ld", "fold_count": 5, "image_count": len(all_rows),
        "settings": canonical_config, "folds": fold_summaries,
        "oof": {
            metric: _statistics([float(row[metric]) for row in all_rows])
            for metric in METRICS
        },
        "fixed_050_oof": {
            metric: _statistics([float(row[f"fixed_050_{metric}"]) for row in all_rows])
            for metric in ("dice", "iou", "precision", "recall")
        },
    }
    return all_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, report = aggregate_cv(args.run_root)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "oof_per_image.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "oof_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report["oof"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
