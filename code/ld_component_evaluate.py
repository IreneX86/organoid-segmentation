"""Evaluate an LD component-sampling fold only after its validation gate locks.

If the per-fold gate chooses baseline, no candidate outer-test label is read;
the frozen baseline evaluation is reused in the eventual comparison report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from ld_component_gate import compute_gate
from ld_evaluate import _load_json, _sha256
from segmentation_metrics import calculate_binary_metrics
from tiff_io import pair_tiff_directories, read_tiff_2d, write_mask_tiff


def _checked_probability(
    probability: np.ndarray, truth: np.ndarray, filename: str
) -> np.ndarray:
    """Reject a changed grid or invalid probabilities before any thresholding."""

    values = np.asarray(probability)
    if values.ndim != 2 or values.shape != truth.shape:
        raise RuntimeError(f"Prediction grid differs from truth for {filename}")
    if (not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError(f"Prediction is not finite probability in [0,1]: {filename}")
    return values.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.candidate_run_dir.resolve()
    final_dir = run_dir / "outer_test"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to rescore an existing outer test: {final_dir}")
    decision_path = run_dir / "decision.lock.json"
    if not decision_path.is_file():
        raise FileNotFoundError("Per-fold validation-only decision must be locked first")

    # Compute and compare the gate again before any outer-test mask is opened.
    # This verifies both candidate and baseline selection locks and the exact
    # validation artifacts, labels, split, source code, and prospective rule.
    expected = compute_gate(
        args.baseline_run_dir, run_dir, args.fold_map,
        args.label_manifest, args.label_dir,
    )
    decision = _load_json(decision_path)
    if decision != expected:
        raise ValueError("Decision lock differs from the validation-only gate")
    if decision["selected"] == "baseline":
        print(f"LD_COMPONENT_GATE_BASELINE_REUSE fold={decision['fold']}")
        return
    if decision["selected"] != "candidate":
        raise ValueError("Decision lock has an invalid selected method")

    config = _load_json(run_dir / "experiment_config.json")
    split = _load_json(run_dir / "split.json")
    input_dir = args.input_dir.resolve()
    label_dir = args.label_dir.resolve()
    if (input_dir != Path(str(config["input_dir"]))
            or label_dir != Path(str(config["label_dir"]))):
        raise ValueError("Evaluation data paths differ from locked candidate training")

    # After this point the test masks may be loaded, strictly for reporting.
    import tensorflow as tf

    from evaluation import predict_pair_probability, write_metric_report

    pairs = pair_tiff_directories(input_dir, label_dir)
    test_names = set(split["test"])
    if {pair.filename for pair in pairs} != set(split["fold_ids"]):
        raise ValueError("Evaluation TIFF names differ from the frozen LD set")
    test_pairs = [pair for pair in pairs if pair.filename in test_names]
    if len(test_pairs) != len(test_names):
        raise ValueError("Some frozen outer-test images are missing")
    frozen = _load_json(args.fold_map.resolve())
    derived = _load_json(args.label_manifest.resolve())
    records = {item["filename"]: item for item in derived.get("files", [])}
    if set(records) != set(split["fold_ids"]):
        raise ValueError("Derived-label manifest differs from the frozen LD set")
    input_hashes = frozen.get("input_sha256", {})
    for pair in test_pairs:
        if input_hashes and _sha256(pair.image_path) != input_hashes[pair.filename]:
            raise ValueError(f"Outer-test input TIFF changed: {pair.filename}")
        if _sha256(pair.label_path) != records[pair.filename]["derived_mask_sha256"]:
            raise ValueError(f"Outer-test derived mask changed: {pair.filename}")
    model = tf.keras.models.load_model(run_dir / "component_unet.keras", compile=False)
    threshold = float(_load_json(run_dir / "threshold.json")["threshold"])

    # Publish complete test outputs atomically and never overwrite a run.
    scratch = Path(tempfile.mkdtemp(prefix=".outer_test_in_progress_", dir=run_dir))
    rows: list[dict[str, float | str]] = []
    threshold_rows: list[dict[str, float | str]] = []
    file_records: list[dict[str, object]] = []
    for pair in test_pairs:
        probability, truth = predict_pair_probability(
            model, pair, tile_size=int(config["patch_size"]),
            overlap=int(config["overlap"]),
            prediction_batch_size=int(config["prediction_batch_size"]),
            lower_percentile=float(config["lower_percentile"]),
            upper_percentile=float(config["upper_percentile"]),
        )
        probability = _checked_probability(probability, truth, pair.filename)
        probability_path = scratch / "probabilities" / pair.filename
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(probability_path, probability, metadata=None)
        if not np.array_equal(read_tiff_2d(probability_path), probability):
            raise RuntimeError(f"Probability TIFF read-back failed: {pair.filename}")
        file_record: dict[str, object] = {
            "filename": pair.filename,
            "shape": list(truth.shape),
            "probability_sha256": _sha256(probability_path),
        }
        comparison: dict[str, float | str] = {"filename": pair.filename}
        for policy, cutoff in (("validation", threshold), ("fixed_050", 0.5)):
            predicted = (probability >= cutoff).astype(np.uint8)
            mask_path = scratch / "predictions" / policy / pair.filename
            write_mask_tiff(mask_path, predicted)
            if not np.array_equal(read_tiff_2d(mask_path), predicted * 255):
                raise RuntimeError(f"Prediction mask TIFF read-back failed: {pair.filename}")
            file_record[f"{policy}_mask_sha256"] = _sha256(mask_path)
            metrics = calculate_binary_metrics(truth, predicted)
            if policy == "validation":
                rows.append({"filename": pair.filename, **metrics.to_dict()})
            for metric in ("dice", "iou", "precision", "recall"):
                comparison[f"{policy}_{metric}"] = getattr(metrics, metric)
        threshold_rows.append(comparison)
        file_records.append(file_record)

    write_metric_report(scratch, "test", rows)
    with (scratch / "threshold_policy_per_image.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    with (scratch / "evaluation_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "target": "ld", "candidate": "component_uniform_positive_center_v1",
            "fold": split["fold"],
            "selection_lock_sha256": _sha256(run_dir / "selection.lock.json"),
            "decision_lock_sha256": _sha256(decision_path),
            "validation_threshold": threshold,
            "test_filenames": sorted(test_names),
            "probabilities_saved": True,
            "files": file_records,
            "test_per_image_sha256": _sha256(scratch / "test_per_image.csv"),
            "test_summary_sha256": _sha256(scratch / "test_summary.json"),
            "threshold_policy_per_image_sha256": _sha256(
                scratch / "threshold_policy_per_image.csv"
            ),
        }, handle, indent=2, sort_keys=True)
    os.replace(scratch, final_dir)
    print(f"LD_COMPONENT_OUTER_TEST_COMPLETE fold={split['fold']} images={len(rows)}")


if __name__ == "__main__":
    main()
