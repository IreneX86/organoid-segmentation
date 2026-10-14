"""Lock a fold-local LD highlight-normalization decision before outer testing.

The baseline exporter is run locally against an archived *same-fold inner
validation* run. It writes a small JSON reference; no model or TIFF is copied.
The comparator can run on a separate host after candidate training. It reads
only that fold's inner-validation derived labels and prediction artifacts,
then writes ``decision.lock.json`` using exclusive, atomic publication.

Neither mode opens an ``outer_test`` directory or an outer-test label TIFF.
The rotated five-fold validation design is exploratory and does not provide
an independent global model-selection set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


CANDIDATE_ID = "highlight_upper_p99_95_v1"
PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "experiments" / "ld_highlight_normalization_protocol.md"
SMALL_COMPONENT_MAX_PIXELS = 25
UNCHANGED_SCIENCE_KEYS = (
    "architecture", "batch_size", "checkpoint_selection", "epochs", "learning_rate",
    "loss", "lower_percentile", "num_folds", "outer_test_deferred", "overlap",
    "patch_size", "positive_patch_probability", "prediction_batch_size", "seed",
    "seed_base", "steps_per_epoch", "target", "threshold_selection",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _safe_filename(value: Any) -> str:
    if (not isinstance(value, str) or Path(value).name != value
            or not value.lower().endswith(".tif")):
        raise ValueError(f"Unsafe validation TIFF filename: {value!r}")
    return value


def _frozen_inputs(fold_map: Path, label_manifest: Path) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    mapping = json_object(fold_map)
    fold_ids = mapping.get("fold_ids")
    if (mapping.get("target") != "ld" or mapping.get("num_folds") != 5
            or not isinstance(fold_ids, dict) or len(fold_ids) != 36
            or any(type(index) is not int or index not in range(5)
                   for index in fold_ids.values())):
        raise ValueError("Expected the frozen 36-image LD five-fold map")
    if any(_safe_filename(name) != name for name in fold_ids):
        raise ValueError("Invalid frozen filename")
    labels = json_object(label_manifest)
    files = labels.get("files")
    if labels.get("rule_id") != "ld_antialias_midpoint_v1" or not isinstance(files, list):
        raise ValueError("Expected the frozen LD midpoint derived-label manifest")
    records = {_safe_filename(item["filename"]): item for item in files}
    if len(records) != len(files) or set(records) != set(fold_ids):
        raise ValueError("Derived-label filenames differ from frozen fold map")
    return fold_ids, records


def _expected_split(fold_ids: dict[str, int], fold: int) -> dict[str, set[str]]:
    return {
        "test": {name for name, index in fold_ids.items() if index == fold},
        "validation": {name for name, index in fold_ids.items() if index == (fold + 1) % 5},
        "train": {name for name, index in fold_ids.items()
                  if index not in (fold, (fold + 1) % 5)},
    }


def _validated_run(
    run_dir: Path, fold_map: Path, label_manifest: Path, *, candidate: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    """Verify a run using its archived source, never the mutable checkout.

    The frozen baseline was trained before later LD source edits. Comparing
    its code hashes to today's ``ld_train.py`` would reject a valid archive;
    comparing every archived snapshot to its locked hash preserves provenance.
    """
    if candidate and (run_dir / "outer_test").exists():
        raise FileExistsError("Refusing to lock a gate after candidate outer-test evaluation")
    config_path = run_dir / "experiment_config.json"
    split_path = run_dir / "split.json"
    lock_path = run_dir / "selection.lock.json"
    threshold_path = run_dir / "threshold.json"
    manifest_path = run_dir / "validation_artifacts" / "manifest.json"
    config, split, lock = map(json_object, (config_path, split_path, lock_path))
    fold = split.get("fold")
    if (type(fold) is not int or fold not in range(5)
            or any(record.get("target") != "ld" for record in (config, split, lock))
            or any(record.get("fold") != fold for record in (config, lock))
            or config.get("num_folds") != 5 or split.get("num_folds") != 5
            or config.get("outer_test_deferred") is not True):
        raise ValueError("Run is not one locked, deferred LD five-fold experiment")
    checkpoint = run_dir / ("baseline_unet.keras")
    expected_hashes = {
        "fold_map_sha256": sha256(fold_map),
        "label_manifest_sha256": sha256(label_manifest),
        "checkpoint_sha256": sha256(checkpoint),
        "threshold_sha256": sha256(threshold_path),
        "experiment_config_sha256": sha256(config_path),
        "split_sha256": sha256(split_path),
        "validation_artifacts_manifest_sha256": sha256(manifest_path),
    }
    for key, digest in expected_hashes.items():
        if lock.get(key) != digest:
            raise ValueError(f"Locked {key} differs from run artifact")
        if key in ("fold_map_sha256", "label_manifest_sha256") and config.get(key) != digest:
            raise ValueError(f"Configured {key} differs from frozen input")
    source_hashes = config.get("code_sha256")
    if (not isinstance(source_hashes, dict) or not source_hashes
            or lock.get("code_sha256") != source_hashes):
        raise ValueError("Run source manifest differs from selection lock")
    for name, digest in source_hashes.items():
        if (not isinstance(name, str) or not name.startswith("code/")
                or ".." in Path(name).parts or Path(name).is_absolute()
                or sha256(run_dir / "source_snapshot" / name) != digest):
            raise ValueError(f"Run source snapshot differs from locked hash: {name}")
    threshold = float(json_object(threshold_path)["threshold"])
    if not math.isfinite(threshold) or not 0 < threshold < 1 or lock.get("threshold") != threshold:
        raise ValueError("Validation-selected threshold differs from selection lock")
    fold_ids, _ = _frozen_inputs(fold_map, label_manifest)
    expected = _expected_split(fold_ids, fold)
    if (split.get("fold_ids") != fold_ids
            or any(set(split.get(part, [])) != names for part, names in expected.items())
            or not all(expected.values())):
        raise ValueError("Run train/validation/test split differs from frozen map")
    if candidate:
        if (config.get("candidate") != CANDIDATE_ID
                or config.get("protocol_sha256") != sha256(PROTOCOL_PATH)
                or config.get("lower_percentile") != 1.0
                or config.get("upper_percentile") != 99.95):
            raise ValueError("Candidate violates fixed highlight-normalization protocol")
    elif config.get("upper_percentile") != 99.8:
        raise ValueError("Frozen baseline does not use p99.8")
    return config, split, lock, threshold


def _read_binary(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"Expected one integer 2D binary mask: {path}")
    levels = np.unique(image)
    if levels.size > 2 or not set(levels.tolist()).issubset({0, 1, 255}):
        raise ValueError(f"Expected 0/1 or 0/255 mask levels: {path}")
    return image != 0


def _metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if truth.shape != pred.shape:
        raise ValueError("Validation truth and prediction grids differ")
    tp = int(np.count_nonzero(truth & pred))
    fp = int(np.count_nonzero(~truth & pred))
    fn = int(np.count_nonzero(truth & ~pred))
    n_true, n_pred = tp + fn, tp + fp
    return {
        "dice": 1.0 if n_true + n_pred == 0 else 2.0 * tp / (n_true + n_pred),
        "precision": (1.0 if n_true == 0 else 0.0) if n_pred == 0 else tp / n_pred,
        "recall": (1.0 if n_pred == 0 else 0.0) if n_true == 0 else tp / n_true,
    }


def small_component_counts(truth: np.ndarray, pred: np.ndarray) -> tuple[int, int]:
    """Count ≤25-pixel, eight-connected semantic regions and any-overlap hits.

    This sparse flood fill avoids adding SciPy to the gate's CPU-only runtime.
    The quantity is a proxy: touching droplets may form one semantic region.
    """
    if truth.ndim != 2 or truth.shape != pred.shape:
        raise ValueError("Component truth and prediction grids differ")
    height, width = truth.shape
    visited = np.zeros(truth.shape, dtype=bool)
    components = hits = 0
    ys, xs = np.nonzero(truth)
    for first_y, first_x in zip(ys, xs):
        y, x = int(first_y), int(first_x)
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        area = overlap = 0
        while stack:
            row, col = stack.pop()
            area += 1
            overlap += bool(pred[row, col])
            for ny in range(max(0, row - 1), min(height, row + 2)):
                for nx in range(max(0, col - 1), min(width, col + 2)):
                    if truth[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if area <= SMALL_COMPONENT_MAX_PIXELS:
            components += 1
            hits += overlap > 0
    return components, hits


def _validation_rows(
    run_dir: Path, names: set[str], label_dir: Path, records: dict[str, dict[str, Any]],
    selected_threshold: float,
) -> list[dict[str, Any]]:
    """Recompute fixed-0.50 validation metrics; never construct an outer path."""
    artifacts = run_dir / "validation_artifacts"
    manifest = json_object(artifacts / "manifest.json")
    files = manifest.get("files")
    if (manifest.get("target") != "ld" or manifest.get("image_count") != len(names)
            or not isinstance(files, list) or len(files) != len(names)
            or {_safe_filename(item["filename"]) for item in files} != names
            or manifest.get("threshold") != selected_threshold):
        raise ValueError("Validation artifact manifest differs from locked fold")
    with (artifacts / "threshold_policy_per_image.csv").open(newline="", encoding="utf-8") as stream:
        policy = list(csv.DictReader(stream))
    if len(policy) != len(names) or {row["filename"] for row in policy} != names:
        raise ValueError("Validation metric CSV differs from locked filenames")
    policy_by_name = {row["filename"]: row for row in policy}
    rows: list[dict[str, Any]] = []
    for item in sorted(files, key=lambda record: record["filename"]):
        name = _safe_filename(item["filename"])
        label_path = label_dir / name
        mask_path = artifacts / "predictions" / "fixed_050" / name
        selected_path = artifacts / "predictions" / "validation" / name
        probability_path = artifacts / "probabilities" / name
        if (sha256(label_path) != records[name].get("derived_mask_sha256")
                or sha256(mask_path) != item.get("fixed_050_mask_sha256")
                or sha256(selected_path) != item.get("validation_mask_sha256")
                or sha256(probability_path) != item.get("probability_sha256")):
            raise ValueError(f"Changed derived label or validation artifact: {name}")
        truth = _read_binary(label_path)
        prediction = _read_binary(mask_path)
        probability = np.asarray(tifffile.imread(probability_path))
        if (probability.ndim != 2 or not np.issubdtype(probability.dtype, np.floating)
                or probability.shape != truth.shape or prediction.shape != truth.shape
                or list(truth.shape) != item.get("shape")
                or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)
                or not np.array_equal(prediction, probability >= 0.5)):
            raise ValueError(f"Invalid fixed-0.50 probability/mask geometry: {name}")
        scores = _metrics(truth, prediction)
        saved = policy_by_name[name]
        for key, value in scores.items():
            if not math.isclose(value, float(saved[f"fixed_050_{key}"]), rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Recomputed validation {key} differs from CSV: {name}")
        small_count, small_hits = small_component_counts(truth, prediction)
        rows.append({
            "filename": name,
            "label_sha256": records[name]["derived_mask_sha256"],
            "fixed_050_mask_sha256": item["fixed_050_mask_sha256"],
            "probability_sha256": item["probability_sha256"],
            "small_component_count_proxy": small_count,
            "small_component_detected_any_proxy": small_hits,
            **scores,
        })
    return rows


def _atomic_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace locked result: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def export_baseline(
    baseline_run_dir: Path, fold_map: Path, label_manifest: Path, label_dir: Path,
) -> dict[str, Any]:
    config, split, lock, threshold = _validated_run(
        baseline_run_dir, fold_map, label_manifest, candidate=False
    )
    _, records = _frozen_inputs(fold_map, label_manifest)
    names = set(split["validation"])
    rows = _validation_rows(baseline_run_dir, names, label_dir, records, threshold)
    return {
        "format": "ld_highlight_fold_local_baseline_reference_v1",
        "target": "ld",
        "fold": split["fold"],
        "fold_map_sha256": sha256(fold_map),
        "label_manifest_sha256": sha256(label_manifest),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "baseline_config_sha256": sha256(baseline_run_dir / "experiment_config.json"),
        "baseline_split_sha256": sha256(baseline_run_dir / "split.json"),
        "baseline_selection_lock_sha256": sha256(baseline_run_dir / "selection.lock.json"),
        "baseline_validation_manifest_sha256": sha256(
            baseline_run_dir / "validation_artifacts" / "manifest.json"),
        "baseline_source_sha256": lock["code_sha256"],
        "baseline_science": {key: config[key] for key in UNCHANGED_SCIENCE_KEYS},
        "baseline_upper_percentile": config["upper_percentile"],
        "validation_filenames": sorted(names),
        "fixed_050_rows": rows,
        "outer_test_labels_opened": False,
    }


def _gate_checks(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, float | int | None]]:
    if not baseline_rows or len(baseline_rows) != len(candidate_rows):
        raise ValueError("Paired fold-local validation rows are missing")
    base_by_name = {row["filename"]: row for row in baseline_rows}
    cand_by_name = {row["filename"]: row for row in candidate_rows}
    if set(base_by_name) != set(cand_by_name):
        raise ValueError("Candidate and baseline validation filenames differ")
    dice_delta = [cand_by_name[name]["dice"] - base_by_name[name]["dice"]
                  for name in sorted(base_by_name)]
    precision_delta = [cand_by_name[name]["precision"] - base_by_name[name]["precision"]
                       for name in sorted(base_by_name)]
    baseline_count = sum(row["small_component_count_proxy"] for row in baseline_rows)
    candidate_count = sum(row["small_component_count_proxy"] for row in candidate_rows)
    if baseline_count != candidate_count:
        raise ValueError("Small-component GT counts changed between paired runs")
    base_hits = sum(row["small_component_detected_any_proxy"] for row in baseline_rows)
    candidate_hits = sum(row["small_component_detected_any_proxy"] for row in candidate_rows)
    component_delta = ((candidate_hits - base_hits) / baseline_count
                       if baseline_count else None)
    baseline_component_fraction = base_hits / baseline_count if baseline_count else None
    candidate_component_fraction = candidate_hits / baseline_count if baseline_count else None
    ceiling = baseline_component_fraction is not None and baseline_component_fraction > 0.90
    observed: dict[str, float | int | None] = {
        "mean_paired_dice_delta_050": float(np.mean(dice_delta)),
        "mean_paired_precision_delta_050": float(np.mean(precision_delta)),
        "worst_paired_dice_delta_050": float(min(dice_delta)),
        "small_component_count_proxy": baseline_count,
        "baseline_small_component_detected_any_proxy": base_hits,
        "candidate_small_component_detected_any_proxy": candidate_hits,
        "baseline_small_component_overlap_fraction": baseline_component_fraction,
        "candidate_small_component_overlap_fraction": candidate_component_fraction,
        "small_component_overlap_fraction_delta": component_delta,
        "baseline_component_ceiling_rule_applies": int(ceiling),
    }
    checks = {
        "mean_dice_drop_no_more_than_0_01": observed["mean_paired_dice_delta_050"] >= -0.01 - 1e-12,
        "small_component_ceiling_aware_condition": (
            (candidate_hits >= base_hits and observed["mean_paired_dice_delta_050"] >= 0.01 - 1e-12)
            if ceiling else
            (component_delta is not None and component_delta >= 0.10 - 1e-12)),
        "mean_precision_drop_no_more_than_0_03": (
            observed["mean_paired_precision_delta_050"] >= -0.03 - 1e-12),
        "no_single_image_dice_drop_more_than_0_10": (
            observed["worst_paired_dice_delta_050"] >= -0.10 - 1e-12),
    }
    return checks, observed


def compare_fold(
    baseline_reference: Path, candidate_run_dir: Path, fold_map: Path,
    label_manifest: Path, label_dir: Path,
) -> dict[str, Any]:
    reference = json_object(baseline_reference)
    config, split, lock, threshold = _validated_run(
        candidate_run_dir, fold_map, label_manifest, candidate=True
    )
    fold = split["fold"]
    if (reference.get("format") != "ld_highlight_fold_local_baseline_reference_v1"
            or reference.get("target") != "ld" or reference.get("fold") != fold
            or reference.get("fold_map_sha256") != sha256(fold_map)
            or reference.get("label_manifest_sha256") != sha256(label_manifest)
            or reference.get("protocol_sha256") != sha256(PROTOCOL_PATH)
            or reference.get("baseline_upper_percentile") != 99.8
            or reference.get("validation_filenames") != sorted(split["validation"])
            or reference.get("outer_test_labels_opened") is not False):
        raise ValueError("Baseline reference is not the matching frozen fold")
    baseline_science = reference.get("baseline_science")
    if (not isinstance(baseline_science, dict)
            or any(config.get(key) != baseline_science.get(key)
                   for key in UNCHANGED_SCIENCE_KEYS)):
        raise ValueError("Candidate changed a scientific factor besides upper percentile")
    # The training entry point must change to expose the p99.95 option. All
    # other archived baseline source modules that define the model, loss,
    # patches, normalization implementation and inference must stay bytewise
    # identical, preventing a hidden second scientific factor.
    baseline_source = reference.get("baseline_source_sha256")
    candidate_source = lock["code_sha256"]
    if (not isinstance(baseline_source, dict)
            or set(baseline_source) - set(candidate_source)
            or any(candidate_source[name] != digest
                   for name, digest in baseline_source.items()
                   if name != "code/ld_train.py")):
        raise ValueError("Candidate changed archived source outside the LD training entry point")
    _, records = _frozen_inputs(fold_map, label_manifest)
    rows = _validation_rows(
        candidate_run_dir, set(split["validation"]), label_dir, records, threshold
    )
    baseline_rows = reference.get("fixed_050_rows")
    if not isinstance(baseline_rows, list) or len(baseline_rows) != len(rows):
        raise ValueError("Missing baseline validation rows")
    for baseline_row, candidate_row in zip(baseline_rows, rows):
        if (baseline_row.get("filename") != candidate_row["filename"]
                or baseline_row.get("label_sha256") != candidate_row["label_sha256"]):
            raise ValueError("Baseline and candidate used different validation labels")
    checks, observed = _gate_checks(baseline_rows, rows)
    paired = []
    for baseline_row, candidate_row in zip(baseline_rows, rows):
        paired.append({
            "filename": candidate_row["filename"],
            "baseline_dice_050": baseline_row["dice"],
            "candidate_dice_050": candidate_row["dice"],
            "paired_dice_delta_050": candidate_row["dice"] - baseline_row["dice"],
            "baseline_precision_050": baseline_row["precision"],
            "candidate_precision_050": candidate_row["precision"],
            "baseline_recall_050": baseline_row["recall"],
            "candidate_recall_050": candidate_row["recall"],
            "small_component_count_proxy": candidate_row["small_component_count_proxy"],
            "baseline_small_component_detected_any_proxy": (
                baseline_row["small_component_detected_any_proxy"]),
            "candidate_small_component_detected_any_proxy": (
                candidate_row["small_component_detected_any_proxy"]),
        })
    return {
        "format": "ld_highlight_fold_local_decision_v1",
        "target": "ld", "candidate": CANDIDATE_ID, "fold": fold,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "fold_map_sha256": sha256(fold_map),
        "label_manifest_sha256": sha256(label_manifest),
        "baseline_reference_sha256": sha256(baseline_reference),
        "candidate_selection_lock_sha256": sha256(candidate_run_dir / "selection.lock.json"),
        "candidate_validation_manifest_sha256": sha256(
            candidate_run_dir / "validation_artifacts" / "manifest.json"),
        "candidate_code_sha256": lock["code_sha256"],
        "validation_filenames": sorted(split["validation"]),
        "paired_validation_rows": paired,
        "observed": observed,
        "gate_checks": checks,
        "selected": "candidate" if all(checks.values()) else "baseline",
        "outer_test_labels_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    exporter = subparsers.add_parser("export-baseline")
    exporter.add_argument("--baseline-run-dir", type=Path, required=True)
    comparator = subparsers.add_parser("compare")
    comparator.add_argument("--baseline-reference", type=Path, required=True)
    comparator.add_argument("--candidate-run-dir", type=Path, required=True)
    for command in (exporter, comparator):
        command.add_argument("--fold-map", type=Path, required=True)
        command.add_argument("--label-manifest", type=Path, required=True)
        command.add_argument("--label-dir", type=Path, required=True)
        command.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-baseline":
        payload = export_baseline(
            args.baseline_run_dir, args.fold_map, args.label_manifest, args.label_dir
        )
        _atomic_exclusive_json(args.output_file, payload)
        print(f"LD_HIGHLIGHT_BASELINE_REFERENCE_EXPORTED fold={payload['fold']}")
    else:
        payload = compare_fold(
            args.baseline_reference, args.candidate_run_dir, args.fold_map,
            args.label_manifest, args.label_dir,
        )
        _atomic_exclusive_json(args.output_file, payload)
        print(f"LD_HIGHLIGHT_FOLD_GATE_LOCKED fold={payload['fold']} selected={payload['selected']}")


if __name__ == "__main__":
    main()
