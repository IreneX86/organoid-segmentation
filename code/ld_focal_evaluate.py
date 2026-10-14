"""Evaluate one already-locked LD focal checkpoint on its outer-test fold once.

This command is deliberately separate from training. The selection lock binds
the checkpoint, threshold, fold map, and derived-label provenance before any
outer-test label is opened. Results are not used to choose this fold's model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from ld_focal_train import CODE_RELATIVE_PATHS, _code_sha256_manifest
from segmentation_metrics import calculate_binary_metrics
from tiff_io import pair_tiff_directories, write_mask_tiff


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _verify_all_gate_decisions(
    all_runs_root: Path, fold_map: Path, label_manifest: Path,
    current_run_dir: Path,
) -> None:
    """Require five fold-local decisions before any outer-test mask is read.

    This safeguard sits in the evaluator as well as the RunPod runner so a
    standalone invocation cannot accidentally score one candidate early.
    It never reads outer-test labels or candidate metrics.
    """
    protocol = (Path(__file__).resolve().parents[1]
                / "experiments" / "ld_focal_dice_protocol.md")
    fold_map_hash = _sha256(fold_map)
    label_hash = _sha256(label_manifest)
    expected_root = all_runs_root / "ld"
    if current_run_dir.parent != expected_root:
        raise ValueError("Current focal run is outside the complete five-fold root")
    for fold in range(5):
        fold_dir = expected_root / f"fold_{fold}"
        decision = _load_json(fold_dir / "decision.lock.json")
        if (decision.get("format") != "ld_focal_fold_local_decision_v1"
                or decision.get("target") != "ld"
                or decision.get("candidate") != "ld_focal_dice_gamma2_v1"
                or decision.get("fold") != fold
                or decision.get("selected") not in {"candidate", "baseline"}
                or decision.get("outer_test_labels_opened") is not False
                or decision.get("protocol_sha256") != _sha256(protocol)
                or decision.get("fold_map_sha256") != fold_map_hash
                or decision.get("label_manifest_sha256") != label_hash
                or decision.get("candidate_selection_lock_sha256")
                != _sha256(fold_dir / "selection.lock.json")
                or decision.get("candidate_validation_manifest_sha256")
                != _sha256(fold_dir / "validation_artifacts" / "manifest.json")):
            raise ValueError(f"Missing or mismatched validation-only gate for fold {fold}")


def _validate_locked_run(run_dir: Path, fold_map: Path, label_manifest: Path) -> tuple[
    dict[str, object], dict[str, object], float
]:
    config = _load_json(run_dir / "experiment_config.json")
    split = _load_json(run_dir / "split.json")
    locked = _load_json(run_dir / "selection.lock.json")
    checkpoint = run_dir / "focal_unet.keras"
    threshold_path = run_dir / "threshold.json"
    if config.get("target") != split.get("target") or split.get("target") != "ld":
        raise ValueError("This command only evaluates LD runs")
    if locked.get("target") != "ld":
        raise ValueError("Selection lock does not identify LD")
    if (config.get("candidate") != "ld_focal_dice_gamma2_v1"
            or config.get("loss") != "focal_bce_gamma2_plus_soft_dice"
            or config.get("protocol_sha256")
            != _sha256(Path(__file__).resolve().parents[1]
                       / "experiments" / "ld_focal_dice_protocol.md")):
        raise ValueError("Expected the preregistered LD focal-loss candidate")
    if not config.get("outer_test_deferred"):
        raise ValueError("The run was not trained with deferred outer-test evaluation")
    if config.get("fold") != split.get("fold") or locked.get("fold") != split.get("fold"):
        raise ValueError("Run fold identifiers disagree")
    if config.get("num_folds") != 5 or split.get("num_folds") != 5:
        raise ValueError("Expected the frozen five-fold LD design")
    expected_hashes = {
        "fold_map_sha256": _sha256(fold_map),
        "label_manifest_sha256": _sha256(label_manifest),
        "checkpoint_sha256": _sha256(checkpoint),
        "threshold_sha256": _sha256(threshold_path),
        "experiment_config_sha256": _sha256(run_dir / "experiment_config.json"),
        "split_sha256": _sha256(run_dir / "split.json"),
        "validation_artifacts_manifest_sha256": _sha256(
            run_dir / "validation_artifacts" / "manifest.json"
        ),
    }
    for name, actual in expected_hashes.items():
        if name in ("fold_map_sha256", "label_manifest_sha256") and config.get(name) != actual:
            raise ValueError(f"Configuration {name} does not match the current file")
        if locked.get(name) != actual:
            raise ValueError(f"Locked {name} does not match the current file")
    code_hashes = config.get("code_sha256")
    if (not isinstance(code_hashes, dict)
            or set(code_hashes) != set(CODE_RELATIVE_PATHS)
            or locked.get("code_sha256") != code_hashes
            or _code_sha256_manifest() != code_hashes):
        raise ValueError("Current LD source code differs from the locked training code")
    for name, expected in code_hashes.items():
        if _sha256(run_dir / "source_snapshot" / name) != expected:
            raise ValueError(f"Frozen LD source snapshot differs: {name}")
    threshold = float(_load_json(threshold_path)["threshold"])
    if not 0 < threshold < 1 or locked.get("threshold") != threshold:
        raise ValueError("Validation-selected threshold differs from the selection lock")

    frozen = _load_json(fold_map)
    fold_ids = frozen.get("fold_ids")
    if frozen.get("target") != "ld" or frozen.get("num_folds") != 5:
        raise ValueError("Fold map is not the frozen five-fold LD map")
    if fold_ids != split.get("fold_ids"):
        raise ValueError("Run split differs from the frozen LD fold map")
    if not isinstance(fold_ids, dict):
        raise ValueError("Frozen fold map contains no fold assignments")
    fold = int(split["fold"])
    validation_fold = (fold + 1) % 5
    expected_partition = {
        "test": {name for name, assigned in fold_ids.items() if assigned == fold},
        "validation": {
            name for name, assigned in fold_ids.items() if assigned == validation_fold
        },
        "train": {
            name for name, assigned in fold_ids.items()
            if assigned not in (fold, validation_fold)
        },
    }
    if not expected_partition["test"] or any(
        set(split.get(key, [])) != names for key, names in expected_partition.items()
    ):
        raise ValueError("Train/validation/test names differ from the rotated fold assignment")
    partition = [set(split.get(key, [])) for key in ("train", "validation", "test")]
    if any(partition[i] & partition[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Train, validation and outer-test filenames overlap")
    if set.union(*partition) != set(fold_ids):
        raise ValueError("Run split does not cover the frozen LD dataset")
    return config, split, threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--all-runs-root", type=Path, required=True,
                        help="Root containing all five locked LD focal folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    from evaluation import predict_pair_probability, write_metric_report

    run_dir = args.run_dir.resolve()
    final_dir = run_dir / "outer_test"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to rescore an existing outer test: {final_dir}")
    input_dir = args.input_dir.resolve()
    label_dir = args.label_dir.resolve()
    config, split, threshold = _validate_locked_run(
        run_dir, args.fold_map.resolve(), args.label_manifest.resolve()
    )
    _verify_all_gate_decisions(
        args.all_runs_root.resolve(), args.fold_map.resolve(),
        args.label_manifest.resolve(), run_dir,
    )
    if input_dir != Path(str(config["input_dir"])) or label_dir != Path(str(config["label_dir"])):
        raise ValueError("Evaluation data paths differ from the locked training paths")

    # All selection and path checks precede this point. Only now may the outer
    # labels be loaded, and only to report final held-out performance.
    pairs = pair_tiff_directories(input_dir, label_dir)
    test_names = set(split["test"])
    if {pair.filename for pair in pairs} != set(split["fold_ids"]):
        raise ValueError("Evaluation data filenames differ from the frozen dataset")
    test_pairs = [pair for pair in pairs if pair.filename in test_names]
    if len(test_pairs) != len(test_names):
        raise ValueError("Some frozen outer-test images are missing")
    frozen = _load_json(args.fold_map.resolve())
    derived = _load_json(args.label_manifest.resolve())
    records = {item["filename"]: item for item in derived.get("files", [])}
    if set(records) != set(split["fold_ids"]):
        raise ValueError("Derived-label manifest does not cover the frozen LD dataset")
    input_hashes = frozen.get("input_sha256", {})
    for pair in test_pairs:
        if input_hashes and _sha256(pair.image_path) != input_hashes[pair.filename]:
            raise ValueError(f"Outer-test input TIFF changed: {pair.filename}")
        if _sha256(pair.label_path) != records[pair.filename]["derived_mask_sha256"]:
            raise ValueError(f"Outer-test derived label changed: {pair.filename}")
    model = tf.keras.models.load_model(run_dir / "focal_unet.keras", compile=False)

    # Atomic directory publication prevents partial test outputs from appearing
    # complete after an interrupted GPU run. A failed scratch directory is kept
    # for diagnosis; a successful result is never overwritten.
    scratch = Path(tempfile.mkdtemp(prefix=".outer_test_in_progress_", dir=run_dir))
    rows: list[dict[str, float | str]] = []
    threshold_rows: list[dict[str, float | str]] = []
    for pair in test_pairs:
        probability, truth = predict_pair_probability(
            model, pair, tile_size=int(config["patch_size"]),
            overlap=int(config["overlap"]),
            prediction_batch_size=int(config["prediction_batch_size"]),
            lower_percentile=float(config["lower_percentile"]),
            upper_percentile=float(config["upper_percentile"]),
        )
        if probability.shape != truth.shape:
            raise RuntimeError(f"Prediction grid differs from truth for {pair.filename}")
        probability_path = scratch / "probabilities" / pair.filename
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(probability_path, probability.astype(np.float32), metadata=None)
        comparison: dict[str, float | str] = {"filename": pair.filename}
        for policy, cutoff in (("validation", threshold), ("fixed_050", 0.5)):
            predicted = (probability >= cutoff).astype(np.uint8)
            write_mask_tiff(scratch / "predictions" / policy / pair.filename, predicted)
            metrics = calculate_binary_metrics(truth, predicted)
            if policy == "validation":
                rows.append({"filename": pair.filename, **metrics.to_dict()})
            for metric in ("dice", "iou", "precision", "recall"):
                comparison[f"{policy}_{metric}"] = getattr(metrics, metric)
        threshold_rows.append(comparison)

    write_metric_report(scratch, "test", rows)
    with (scratch / "threshold_policy_per_image.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    with (scratch / "evaluation_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "target": "ld", "fold": split["fold"],
            "selection_lock_sha256": _sha256(run_dir / "selection.lock.json"),
            "validation_threshold": threshold,
            "test_filenames": sorted(test_names),
            "probabilities_saved": True,
        }, handle, indent=2, sort_keys=True)
    os.replace(scratch, final_dir)
    print(f"LD_FOCAL_OUTER_TEST_COMPLETE fold={split['fold']} images={len(rows)}")


if __name__ == "__main__":
    main()
