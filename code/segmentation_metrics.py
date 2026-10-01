"""Per-image binary segmentation metrics used for validation and testing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    foreground_fraction_true: float
    foreground_fraction_predicted: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> BinaryMetrics:
    """Calculate metrics for one 2D image with explicit empty-mask behavior."""

    true_mask = np.asarray(y_true, dtype=bool)
    predicted_mask = np.asarray(y_predicted, dtype=bool)
    if true_mask.shape != predicted_mask.shape:
        raise ValueError(
            f"Metric shape mismatch: truth={true_mask.shape}, prediction={predicted_mask.shape}"
        )

    true_positive = int(np.logical_and(true_mask, predicted_mask).sum())
    false_positive = int(np.logical_and(~true_mask, predicted_mask).sum())
    false_negative = int(np.logical_and(true_mask, ~predicted_mask).sum())

    true_count = true_positive + false_negative
    predicted_count = true_positive + false_positive
    dice_denominator = true_count + predicted_count
    union = true_positive + false_positive + false_negative

    dice = 1.0 if dice_denominator == 0 else 2.0 * true_positive / dice_denominator
    iou = 1.0 if union == 0 else true_positive / union
    precision = 1.0 if predicted_count == 0 and true_count == 0 else (
        0.0 if predicted_count == 0 else true_positive / predicted_count
    )
    recall = 1.0 if true_count == 0 and predicted_count == 0 else (
        0.0 if true_count == 0 else true_positive / true_count
    )
    return BinaryMetrics(
        dice=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        foreground_fraction_true=float(true_mask.mean()),
        foreground_fraction_predicted=float(predicted_mask.mean()),
    )


def summarize_metric_rows(rows: list[dict[str, float | str]]) -> dict[str, float]:
    """Return mean and sample standard deviation for all numeric metric fields."""

    if not rows:
        raise ValueError("Cannot summarize an empty metric list")
    numeric_keys = [key for key in rows[0] if key != "filename"]
    summary: dict[str, float] = {"image_count": float(len(rows))}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary

