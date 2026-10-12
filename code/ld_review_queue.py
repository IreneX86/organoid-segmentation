"""Rank LD batch predictions for human review without using ground truth.

Input is a completed ``ld_predict.py --save-probabilities`` run, its original
TIFF folder, and the frozen 36-mask midpoint reference manifest. This tool
only reads them and writes a new review directory. It never changes TIFFs,
thresholds, predictions, masks, or human annotations. Priority is an
uncalibrated triage heuristic, not a probability of error or a Dice estimate.

Example::

    python code/ld_review_queue.py \
      --prediction-dir /path/to/new_ld_predictions \
      --input-dir /path/to/original_ld_tiffs \
      --reference-manifest experiments/results/ld_binary_midpoint_v1_manifest.json \
      --output-dir outputs/ld_review_20261012 --contact-sheet

The optional contact sheet contains raw biological image crops. Keep its
output outside Git and verify flagged cases with a domain expert.
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

import numpy as np
import tifffile
from scipy import ndimage as ndi


BOUNDARY_WIDTH_PIXELS = 2
REFERENCE_LOWER_PERCENTILE = 5.0
REFERENCE_UPPER_PERCENTILE = 95.0
DEFAULT_CONFIDENCE_MARGIN = 0.10
CSV_FIELDS = (
    "priority_rank", "filename", "height", "width", "triage_score",
    "boundary_uncertainty_fraction", "uncertain_boundary_pixels",
    "predicted_boundary_pixels", "predicted_foreground_fraction",
    "reference_p05_foreground_fraction", "reference_p95_foreground_fraction",
    "foreground_outlier_direction", "foreground_outlier_log2_distance",
    "foreground_outlier_score", "review_reasons",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _real_tiffs(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"TIFF directory does not exist: {directory}")
    return {
        path.name: path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        and not path.name.startswith("._")
    }


def _reference_fractions(reference_manifest: Path) -> np.ndarray:
    reference = _read_json(reference_manifest)
    if (reference.get("rule_id") != "ld_antialias_midpoint_v1"
            or reference.get("label_semantics") != {"background": 0, "foreground": 1}
            or reference.get("pair_count") != 36):
        raise ValueError("Expected the frozen 36-mask LD midpoint reference")
    files = reference.get("files")
    if not isinstance(files, list) or len(files) != 36:
        raise ValueError("LD training reference must have 36 labelled masks")
    values = []
    names = set()
    for record in files:
        name = record.get("filename")
        shape = record.get("shape")
        count = record.get("foreground_pixels_50pct")
        if (not isinstance(name, str) or name in names
                or not isinstance(shape, list) or len(shape) != 2
                or any(type(length) is not int or length <= 0 for length in shape)
                or type(count) is not int or not 0 <= count <= shape[0] * shape[1]):
            raise ValueError("Invalid LD training reference record")
        names.add(name)
        values.append(count / (shape[0] * shape[1]))
    return np.asarray(values, dtype=np.float64)


def _boundary_band(mask: np.ndarray) -> np.ndarray:
    """Two native pixels on either side of a predicted 8-connected edge."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    outer = ndi.binary_dilation(mask, structure=structure,
                                iterations=BOUNDARY_WIDTH_PIXELS)
    inner = ndi.binary_erosion(mask, structure=structure,
                               iterations=BOUNDARY_WIDTH_PIXELS)
    return outer & ~inner


def _foreground_outlier(
    fraction: float, lower: float, upper: float, pixel_count: int,
) -> tuple[str, float, float]:
    """Return direction, log2 distance outside reference, capped priority term."""
    resolution = 1 / pixel_count
    if fraction < lower:
        direction = "below_p05"
        distance = math.log2(max(lower, resolution) / max(fraction, resolution))
    elif fraction > upper:
        direction = "above_p95"
        distance = math.log2(max(fraction, resolution) / max(upper, resolution))
    else:
        return "within_p05_p95", 0.0, 0.0
    distance = max(0.0, distance)
    return direction, distance, min(distance, 1.0)


def _measure(
    name: str, probability: np.ndarray, predicted: np.ndarray, threshold: float,
    margin: float, lower: float, upper: float,
) -> dict[str, object]:
    boundary = _boundary_band(predicted)
    uncertain = boundary & (np.abs(probability - threshold) <= margin)
    boundary_pixels = int(boundary.sum())
    uncertain_pixels = int(uncertain.sum())
    uncertainty = uncertain_pixels / boundary_pixels if boundary_pixels else None
    fraction = float(predicted.mean())
    direction, distance, outlier = _foreground_outlier(
        fraction, lower, upper, predicted.size,
    )
    reasons = []
    if not predicted.any():
        reasons.append("no predicted LD foreground")
    if direction == "below_p05":
        reasons.append("predicted foreground below training reference p05")
    elif direction == "above_p95":
        reasons.append("predicted foreground above training reference p95")
    if uncertainty is not None and uncertainty >= 0.25:
        reasons.append("at least 25% of predicted edge band near threshold")
    if not reasons:
        reasons.append("routine visual review")
    return {
        "filename": name, "height": predicted.shape[0], "width": predicted.shape[1],
        "triage_score": (uncertainty or 0.0) + outlier,
        "boundary_uncertainty_fraction": uncertainty,
        "uncertain_boundary_pixels": uncertain_pixels,
        "predicted_boundary_pixels": boundary_pixels,
        "predicted_foreground_fraction": fraction,
        "reference_p05_foreground_fraction": lower,
        "reference_p95_foreground_fraction": upper,
        "foreground_outlier_direction": direction,
        "foreground_outlier_log2_distance": distance,
        "foreground_outlier_score": outlier,
        "review_reasons": "; ".join(reasons),
    }


def _read_checked_prediction(
    input_path: Path, mask_path: Path, probability_path: Path,
    record: dict, threshold: float, foreground_value: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for field, path in (("input_sha256", input_path), ("mask_sha256", mask_path),
                        ("probability_sha256", probability_path)):
        if record.get(field) != _sha256(path):
            raise ValueError(f"{field} differs from prediction manifest: {path.name}")
    image = tifffile.imread(input_path)
    mask = tifffile.imread(mask_path)
    probability = tifffile.imread(probability_path)
    shape = (record.get("height"), record.get("width"))
    if (image.ndim != 2 or mask.shape != image.shape or probability.shape != image.shape
            or image.shape != shape or not np.issubdtype(image.dtype, np.number)
            or not np.isfinite(image).all()
            or mask.dtype != np.uint8 or not np.issubdtype(probability.dtype, np.floating)
            or not np.isfinite(probability).all()
            or np.any(probability < 0) or np.any(probability > 1)):
        raise ValueError(f"Input/mask/probability pixel grid or dtype is invalid: {input_path.name}")
    expected_mask = (probability >= threshold).astype(np.uint8) * foreground_value
    if not np.array_equal(mask, expected_mask):
        raise ValueError(f"Saved LD mask does not match locked probability threshold: {input_path.name}")
    binary = mask != 0
    if not math.isclose(float(binary.mean()), float(record.get("foreground_fraction", -1)),
                        rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Predicted foreground fraction changed: {input_path.name}")
    return image, binary, probability


def _crop_origin(center: int, limit: int, crop: int) -> int:
    return min(max(center - crop // 2, 0), max(limit - crop, 0))


def _render_contact_sheet(
    rows: list[dict[str, object]], input_dir: Path, prediction_dir: Path,
    threshold: float, margin: float, crop_size: int, top_k: int, output_path: Path,
) -> list[dict[str, int | str]]:
    """Show source-coordinate crops, never resize or write TIFF pixels back."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Optional contact sheet requires Pillow") from error

    chosen = rows[:min(top_k, len(rows))]
    # TIFFs may have different dimensions; preserve native pixels and pad rows.
    panel_width = crop_size
    row_height = crop_size + 46
    sheet = Image.new("RGB", (panel_width * 3, row_height * len(chosen)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    locations: list[dict[str, int | str]] = []
    for index, row in enumerate(chosen):
        name = str(row["filename"])
        image = tifffile.imread(input_dir / name)
        probability = tifffile.imread(prediction_dir / "probabilities" / name)
        predicted = probability >= threshold
        boundary = _boundary_band(predicted)
        uncertain = boundary & (np.abs(probability - threshold) <= margin)
        if uncertain.any():
            candidates = np.flatnonzero(uncertain)
            chosen_flat = candidates[np.argmin(np.abs(probability.flat[candidates] - threshold))]
            cy, cx = np.unravel_index(chosen_flat, probability.shape)
        elif predicted.any():
            cy, cx = np.unravel_index(np.argmax(probability), probability.shape)
        else:
            cy, cx = image.shape[0] // 2, image.shape[1] // 2
        height = min(crop_size, image.shape[0])
        width = min(crop_size, image.shape[1])
        y0 = _crop_origin(int(cy), image.shape[0], height)
        x0 = _crop_origin(int(cx), image.shape[1], width)
        locations.append({"filename": name, "x0": x0, "y0": y0,
                          "width": width, "height": height})
        source = image[y0:y0 + height, x0:x0 + width].astype(np.float32)
        low, high = np.percentile(image, [1.0, 99.8])
        grey = np.clip((source - low) / max(float(high - low), 1.0), 0, 1)
        grey8 = (grey * 255).astype(np.uint8)
        p = probability[y0:y0 + height, x0:x0 + width]
        heat = np.stack(((p * 255).astype(np.uint8),
                         np.zeros_like(grey8), ((1 - p) * 255).astype(np.uint8)), axis=-1)
        overlay = np.repeat(grey8[..., None], 3, axis=2)
        edge_crop = boundary[y0:y0 + height, x0:x0 + width]
        uncertain_crop = uncertain[y0:y0 + height, x0:x0 + width]
        overlay[edge_crop] = (0, 210, 210)
        overlay[uncertain_crop] = (255, 215, 0)
        panel_y = index * row_height + 42
        for column, pixels in enumerate((np.repeat(grey8[..., None], 3, axis=2),
                                         heat, overlay)):
            sheet.paste(Image.fromarray(pixels),
                        (column * panel_width, panel_y))
        draw.text((8, index * row_height + 5),
                  f"rank {index + 1}  {name}  crop (x={x0}, y={y0}, w={width}, h={height})",
                  fill="black", font=font)
        draw.text((8, index * row_height + 23),
                  "input                         probability (red=high)                         "
                  "edge=cyan; uncertain edge=yellow", fill="black", font=font)
    sheet.save(output_path)
    return locations


def build_review_queue(
    prediction_dir: Path, input_dir: Path, reference_manifest: Path,
    output_dir: Path, *, confidence_margin: float = DEFAULT_CONFIDENCE_MARGIN,
    contact_sheet: bool = False, top_k: int = 8, crop_size: int = 256,
) -> dict[str, object]:
    """Verify and rank a completed same-grid batch into a fresh local directory."""
    prediction_dir, input_dir, reference_manifest, output_dir = (
        path.resolve() for path in (prediction_dir, input_dir, reference_manifest, output_dir)
    )
    if (output_dir.exists() or output_dir == prediction_dir or output_dir == input_dir
            or prediction_dir in output_dir.parents or input_dir in output_dir.parents):
        raise FileExistsError("Review output must be new and separate from TIFF folders")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"Review output parent does not exist: {output_dir.parent}")
    if not 0 < confidence_margin < 0.5 or top_k < 1 or crop_size < 16:
        raise ValueError("Invalid uncertainty margin, top_k or crop_size")
    reference = _reference_fractions(reference_manifest)
    lower, upper = (float(value) for value in np.percentile(
        reference, [REFERENCE_LOWER_PERCENTILE, REFERENCE_UPPER_PERCENTILE]
    ))
    manifest_path = prediction_dir / "prediction_manifest.json"
    manifest = _read_json(manifest_path)
    threshold = float(manifest.get("threshold", -1))
    foreground_value = manifest.get("foreground_value")
    records = manifest.get("files")
    if (manifest.get("target") != "ld" or manifest.get("status") != "complete"
            or manifest.get("semantic_labels_internal") != {"background": 0, "foreground": 1}
            or manifest.get("probabilities_saved") is not True
            or not 0 < threshold < 1 or foreground_value not in (1, 255)
            or not isinstance(records, list) or not records
            or manifest.get("image_count") != len(records)):
        raise ValueError("Expected completed LD predictions saved with --save-probabilities")
    by_name = {record.get("filename"): record for record in records}
    if (len(by_name) != len(records)
            or any(not isinstance(name, str) or Path(name).name != name
                   or Path(name).suffix.lower() not in {".tif", ".tiff"}
                   or name.startswith("._") for name in by_name)):
        raise ValueError("Prediction manifest has duplicate or unsafe TIFF filenames")
    names = set(by_name)
    if (set(_real_tiffs(input_dir)) != names
            or set(_real_tiffs(prediction_dir)) != names
            or set(_real_tiffs(prediction_dir / "probabilities")) != names):
        raise ValueError("Input, mask and probability TIFF filenames are not one-to-one")

    rows: list[dict[str, object]] = []
    for name in sorted(names, key=str.lower):
        _, predicted, probability = _read_checked_prediction(
            input_dir / name, prediction_dir / name,
            prediction_dir / "probabilities" / name, by_name[name],
            threshold, int(foreground_value),
        )
        rows.append(_measure(name, probability, predicted, threshold,
                             confidence_margin, lower, upper))
    rows.sort(key=lambda row: (-float(row["triage_score"]), str(row["filename"]).lower()))
    for rank, row in enumerate(rows, start=1):
        row["priority_rank"] = rank

    with tempfile.TemporaryDirectory(prefix=".ld_review_in_progress_",
                                     dir=output_dir.parent) as temporary:
        scratch = Path(temporary)
        with (scratch / "review_queue.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        locations = (_render_contact_sheet(
            rows, input_dir, prediction_dir, threshold, confidence_margin,
            crop_size, top_k, scratch / "contact_sheet.png",
        ) if contact_sheet else [])
        summary: dict[str, object] = {
            "target": "ld", "scope": "heuristic human-review triage; no ground truth read",
            "warning": "Priority is not calibrated error probability or model accuracy; "
                       "foreground-ratio comparison assumes similar field of view. "
                       "Review flagged cases manually; contact-sheet colors are not labels.",
            "prediction_manifest_sha256": _sha256(manifest_path),
            "training_reference_manifest_sha256": _sha256(reference_manifest),
            "reference_rule_id": "ld_antialias_midpoint_v1",
            "reference_images": len(reference), "prediction_images": len(rows),
            "reference_foreground_p05": lower, "reference_foreground_p95": upper,
            "threshold": threshold, "confidence_margin_absolute": confidence_margin,
            "boundary_definition": "two native pixels inside/outside 8-connected predicted edge",
            "boundary_uncertainty_definition": "fraction of boundary-band pixels with "
                                               "absolute probability distance <= margin",
            "outlier_definition": "log2 distance outside training-reference p05–p95 "
                                  "foreground range, capped at 1 for priority",
            "priority_definition": "boundary_uncertainty_fraction (0 when no edge) + "
                                   "capped foreground_outlier_score; descending",
            "contact_sheet": "contact_sheet.png" if contact_sheet else None,
            "contact_sheet_crops_native_coordinates": locations,
        }
        with (scratch / "review_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        os.replace(scratch, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-margin", type=float,
                        default=DEFAULT_CONFIDENCE_MARGIN)
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    args = parser.parse_args()
    result = build_review_queue(
        args.prediction_dir, args.input_dir, args.reference_manifest, args.output_dir,
        confidence_margin=args.confidence_margin, contact_sheet=args.contact_sheet,
        top_k=args.top_k, crop_size=args.crop_size,
    )
    print(f"LD_REVIEW_QUEUE_COMPLETE images={result['prediction_images']} "
          f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
