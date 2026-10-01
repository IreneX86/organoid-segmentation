"""Full-resolution validation, threshold selection, and metric reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from segmentation_metrics import calculate_binary_metrics, summarize_metric_rows
from sliding_window import predict_sliding_window
from tiff_io import ImageLabelPair, load_validated_pair, normalize_image


def predict_pair_probability(
    model: object,
    pair: ImageLabelPair,
    tile_size: int,
    overlap: int,
    prediction_batch_size: int,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one full-resolution probability map and its binary ground truth."""

    image, mask = load_validated_pair(pair)
    normalized = normalize_image(image, lower_percentile, upper_percentile)
    probability = predict_sliding_window(
        model,
        normalized,
        tile_size=tile_size,
        overlap=overlap,
        prediction_batch_size=prediction_batch_size,
    )
    return probability, mask


def predict_pairs(
    model: object,
    pairs: list[ImageLabelPair],
    tile_size: int,
    overlap: int,
    prediction_batch_size: int,
    lower_percentile: float,
    upper_percentile: float,
) -> list[tuple[ImageLabelPair, np.ndarray, np.ndarray]]:
    predictions = []
    for pair in pairs:
        probability, mask = predict_pair_probability(
            model,
            pair,
            tile_size,
            overlap,
            prediction_batch_size,
            lower_percentile,
            upper_percentile,
        )
        predictions.append((pair, probability, mask))
    return predictions


def metric_rows_from_predictions(
    predictions: list[tuple[ImageLabelPair, np.ndarray, np.ndarray]],
    threshold: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for pair, probability, mask in predictions:
        metrics = calculate_binary_metrics(mask, probability >= threshold)
        rows.append({"filename": pair.filename, **metrics.to_dict()})
    return rows


def choose_threshold(
    predictions: list[tuple[ImageLabelPair, np.ndarray, np.ndarray]],
    thresholds: np.ndarray | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Select the threshold with the best mean per-image Dice on validation data."""

    if thresholds is None:
        thresholds = np.arange(0.10, 0.901, 0.05)
    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        rows = metric_rows_from_predictions(predictions, float(threshold))
        summary = summarize_metric_rows(rows)
        candidates.append(
            {
                "threshold": float(threshold),
                "dice_mean": summary["dice_mean"],
                "iou_mean": summary["iou_mean"],
            }
        )
    best = max(candidates, key=lambda item: (item["dice_mean"], item["iou_mean"]))
    return best["threshold"], candidates


def write_metric_report(
    output_dir: Path,
    prefix: str,
    rows: list[dict[str, float | str]],
) -> dict[str, float]:
    """Write per-image CSV and summary JSON files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_metric_rows(rows)
    csv_path = output_dir / f"{prefix}_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


class FullImageValidationCallback(tf.keras.callbacks.Callback):
    """Evaluate validation TIFFs on their full H×W grid after each epoch."""

    def __init__(
        self,
        pairs: list[ImageLabelPair],
        tile_size: int,
        overlap: int,
        prediction_batch_size: int,
        lower_percentile: float,
        upper_percentile: float,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not pairs:
            raise ValueError("FullImageValidationCallback requires validation pairs")
        self.pairs = pairs
        self.tile_size = tile_size
        self.overlap = overlap
        self.prediction_batch_size = prediction_batch_size
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.threshold = threshold

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs if logs is not None else {}
        predictions = predict_pairs(
            self.model,
            self.pairs,
            self.tile_size,
            self.overlap,
            self.prediction_batch_size,
            self.lower_percentile,
            self.upper_percentile,
        )
        rows = metric_rows_from_predictions(predictions, self.threshold)
        summary = summarize_metric_rows(rows)
        for metric_name in ("dice", "iou", "precision", "recall"):
            logs[f"val_full_{metric_name}"] = summary[f"{metric_name}_mean"]
        print(
            "\nFull-image validation: "
            f"Dice={summary['dice_mean']:.4f}, IoU={summary['iou_mean']:.4f}, "
            f"Precision={summary['precision_mean']:.4f}, "
            f"Recall={summary['recall_mean']:.4f}"
        )

