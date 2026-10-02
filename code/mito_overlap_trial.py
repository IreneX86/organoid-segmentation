"""Validation-gated overlap-256 trial using frozen Mito baseline checkpoints.

Run `select` first. Only `evaluate` may open candidate outer-test masks, and
only after all five validation decisions have been saved. No retraining occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mito_round2_gate import decide


def _context(args):
    from pipeline_config import get_target_spec
    from tiff_io import pair_tiff_directories

    root = args.project_root.resolve()
    spec = get_target_spec("mito")
    pairs = pair_tiff_directories(
        root / spec.input_relative_dir, root / spec.label_relative_dir
    )
    return {pair.filename: pair for pair in pairs}


def _predict_rows(model, pairs, overlap, threshold):
    from evaluation import metric_rows_from_predictions, predict_pairs

    predictions = predict_pairs(model, pairs, 512, overlap, 4, 1.0, 99.8)
    return predictions, metric_rows_from_predictions(predictions, threshold)


def select(args) -> None:
    import tensorflow as tf

    by_name = _context(args)
    decisions = []
    for fold in range(5):
        baseline = args.baseline_root / f"fold_{fold}"
        split = json.loads((baseline / "split.json").read_text())
        threshold = float(json.loads((baseline / "threshold.json").read_text())["threshold"])
        model_path = args.baseline_model_root / f"fold_{fold}" / "baseline_unet.keras"
        model = tf.keras.models.load_model(model_path, compile=False)
        val_pairs = [by_name[name] for name in split["validation"]]
        _, baseline_rows = _predict_rows(model, val_pairs, 128, threshold)
        _, candidate_rows = _predict_rows(model, val_pairs, 256, threshold)
        result = decide(baseline_rows, candidate_rows)
        result["fold"] = fold
        result["threshold"] = threshold
        decisions.append(result)
        del model
        tf.keras.backend.clear_session()
        print(f"OVERLAP_VALIDATION_FOLD={fold} "
              f"delta={result['mean_delta']:.6f} "
              f"selected={result['selected_candidate']}", flush=True)
    report = {
        "protocol": "mito_overlap256_validation_gate_v1",
        "variant": "overlap256",
        "baseline_overlap": 128,
        "candidate_overlap": 256,
        "threshold_policy": "frozen baseline validation-selected threshold",
        "decisions": decisions,
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    with (args.output_root / "candidate_decisions.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("OVERLAP_DECISIONS_LOCKED", flush=True)


def evaluate(args) -> None:
    import tensorflow as tf
    from evaluation import write_metric_report
    from segmentation_metrics import calculate_binary_metrics
    from tiff_io import write_mask_tiff

    decisions_path = args.output_root / "candidate_decisions.json"
    decisions = json.loads(decisions_path.read_text())
    if (decisions.get("protocol") != "mito_overlap256_validation_gate_v1"
            or len(decisions["decisions"]) != 5
            or [item["fold"] for item in decisions["decisions"]] != list(range(5))):
        raise ValueError("Missing locked five-fold overlap decisions")
    by_name = _context(args)
    per_image = []
    for item in decisions["decisions"]:
        fold = item["fold"]
        baseline = args.baseline_root / f"fold_{fold}"
        split = json.loads((baseline / "split.json").read_text())
        with (baseline / "test_per_image.csv").open(newline="") as handle:
            baseline_rows = {row["filename"]: row for row in csv.DictReader(handle)}
        if set(baseline_rows) != set(split["test"]):
            raise ValueError(f"Baseline outer-test rows mismatch fold {fold}")
        selected_rows = baseline_rows
        if item["selected_candidate"]:
            output = args.output_root / f"fold_{fold}"
            if output.exists():
                raise FileExistsError(f"Refusing existing output: {output}")
            output.mkdir()
            model = tf.keras.models.load_model(
                args.baseline_model_root / f"fold_{fold}" / "baseline_unet.keras",
                compile=False,
            )
            test_pairs = [by_name[name] for name in split["test"]]
            predictions, rows = _predict_rows(
                model, test_pairs, 256, float(item["threshold"])
            )
            for pair, probability, truth in predictions:
                binary = (probability >= float(item["threshold"])).astype("uint8")
                write_mask_tiff(output / "predictions" / pair.filename, binary)
                # Recheck the exported mask's intended metric on the original grid.
                expected = calculate_binary_metrics(truth, binary).dice
                measured = next(row["dice"] for row in rows
                                if row["filename"] == pair.filename)
                if abs(expected - measured) > 1e-10:
                    raise RuntimeError("Exported mask metric mismatch")
            write_metric_report(output, "test", rows)
            selected_rows = {row["filename"]: row for row in rows}
            del model
            tf.keras.backend.clear_session()
        for name in sorted(baseline_rows):
            base, chosen = baseline_rows[name], selected_rows[name]
            row = {
                "filename": name,
                "fold": fold,
                "selected_candidate": bool(item["selected_candidate"]),
                "foreground_fraction_true": float(base["foreground_fraction_true"]),
            }
            for metric in ("dice", "iou", "precision", "recall"):
                row[metric] = float(chosen[metric])
                row[f"baseline_{metric}"] = float(base[metric])
                row[f"delta_{metric}"] = float(chosen[metric]) - float(base[metric])
            per_image.append(row)
        print(f"OVERLAP_EVALUATED_FOLD={fold} "
              f"selected={item['selected_candidate']}", flush=True)
    if len(per_image) != 36 or len({r["filename"] for r in per_image}) != 36:
        raise ValueError("Expected exactly 36 distinct OOF images")
    from mito_candidate_report import _distribution
    metrics = ("dice", "iou", "precision", "recall")
    result = {
        "protocol": "mito_overlap256_selected_oof_v1",
        "variant": "overlap256",
        "image_count": 36,
        "selected_candidate_folds": sum(item["selected_candidate"]
                                        for item in decisions["decisions"]),
        "metrics": {metric: _distribution([row[metric] for row in per_image])
                    for metric in metrics},
        "paired_delta": {metric: _distribution([row[f"delta_{metric}"]
                                                for row in per_image])
                         for metric in metrics},
        "per_image": sorted(per_image, key=lambda row: row["filename"]),
    }
    with (args.output_root / "candidate_oof_summary.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("OVERLAP_EXPERIMENT_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "evaluate"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    (select if args.stage == "select" else evaluate)(args)


if __name__ == "__main__":
    main()
