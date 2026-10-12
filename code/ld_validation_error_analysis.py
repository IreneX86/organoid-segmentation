"""Describe LD errors using saved *validation* probabilities only.

This tool never opens ``outer_test`` artifacts or original Photoshop masks.
Its five-fold aggregate is exploratory: in the rotated CV design, an image
validated in one fold is an outer-test image in another. Consequently these
diagnostics must not be used to choose a global candidate and then present
the same five-fold OOF result as untouched confirmation.

Example::

    python code/ld_validation_error_analysis.py \
      --run-root runpod_results/ld_baseline_20261012/extracted \
      --input-dir ~/Desktop/Organoid/label_ref/LD/LD_input \
      --label-dir data/ld_binary_midpoint_v1/LD_output \
      --label-manifest data/ld_binary_midpoint_v1/LD_output/manifest.json \
      --output-dir outputs/ld_validation_error_analysis_20261012

The output directory must not already exist. It contains only small CSV/JSON
summaries and a local montage; raw TIFFs, probabilities, and masks stay put.
SciPy and Pillow are required locally for distance transforms and the montage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from segmentation_metrics import calculate_binary_metrics


LOW_FOREGROUND_FRACTION = 0.001  # At most 0.10% of image pixels.
SMALL_COMPONENT_MAX_PIXELS = 25  # A proxy, not an instance-label definition.
BOUNDARY_DISTANCE_PIXELS = 2.0
DISTANT_FP_PIXELS = 10.0
MONTAGE_TILE_PIXELS = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _safe_tiff_name(name: object) -> str:
    if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".tif"):
        raise ValueError(f"Unsafe or non-TIFF validation filename: {name!r}")
    return name


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def _normalised_image(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1.0, 99.8])
    span = max(float(high - low), 1.0)
    return np.clip((image.astype(np.float32) - float(low)) / span, 0.0, 1.0)


def analyse_case(
    image: np.ndarray,
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    """Quantify one 2D validation case without changing its model or threshold.

    False-positive distances are measured to the nearest true foreground
    pixel. False-negative boundary depth is measured inside the true mask;
    missed-object distance is measured to the nearest predicted foreground.
    Eight-connected semantic regions of at most 25 pixels are used only as a
    *small-object recall proxy* because touching droplets need not be separate
    instances in a binary annotation.
    """
    if image.ndim != 2 or truth.shape != image.shape or probability.shape != image.shape:
        raise ValueError("Input, derived label, and probability must share one 2D grid")
    if image.dtype.kind not in "ui" or truth.dtype != np.uint8:
        raise ValueError("Expected integer image and uint8 derived 0/1 label")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("Derived LD label must contain only 0 and 1")
    if probability.dtype.kind != "f" or not np.isfinite(probability).all():
        raise ValueError("Validation probabilities must be finite floating-point values")
    if np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("Validation probabilities must lie within [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Validation-selected threshold must lie within [0, 1]")

    gt = truth.astype(bool, copy=False)
    prediction = probability >= threshold
    tp = gt & prediction
    fp = ~gt & prediction
    fn = gt & ~prediction
    tn = ~gt & ~prediction
    metrics = calculate_binary_metrics(gt, prediction)

    if gt.any():
        fp_distance = ndi.distance_transform_edt(~gt)
        fn_depth = ndi.distance_transform_edt(gt) if not gt.all() else None
        fp_near = int(np.count_nonzero(fp & (fp_distance <= BOUNDARY_DISTANCE_PIXELS)))
        fp_distant = int(np.count_nonzero(fp & (fp_distance > DISTANT_FP_PIXELS)))
    else:
        fp_near, fp_distant = 0, int(fp.sum())
        fn_depth = None
    fn_near_boundary = (
        int(np.count_nonzero(fn & (fn_depth <= BOUNDARY_DISTANCE_PIXELS)))
        if fn_depth is not None else 0
    )
    if prediction.any():
        missed_distance = ndi.distance_transform_edt(~prediction)
        fn_far_from_prediction = int(np.count_nonzero(fn & (missed_distance > DISTANT_FP_PIXELS)))
    else:
        fn_far_from_prediction = int(fn.sum())

    components, component_count = ndi.label(gt, structure=np.ones((3, 3), dtype=bool))
    component_areas = np.bincount(components.ravel(), minlength=component_count + 1)[1:]
    component_hits = np.bincount(components[tp].ravel(), minlength=component_count + 1)[1:]
    small = component_areas <= SMALL_COMPONENT_MAX_PIXELS
    small_count = int(small.sum())
    small_detected_any = int(np.count_nonzero(component_hits[small] > 0))
    small_detected_half = int(np.count_nonzero(
        component_hits[small] >= 0.5 * component_areas[small]
    ))

    intensity = _normalised_image(image)
    result: dict[str, float | int | None] = {
        **metrics.to_dict(),
        "threshold": float(threshold),
        "foreground_pixels": int(gt.sum()),
        "predicted_pixels": int(prediction.sum()),
        "tp_pixels": int(tp.sum()),
        "fp_pixels": int(fp.sum()),
        "fn_pixels": int(fn.sum()),
        "fp_within_2px_of_gt": fp_near,
        "fp_farther_than_10px_from_gt": fp_distant,
        "fn_within_2px_of_gt_boundary": fn_near_boundary,
        "fn_farther_than_2px_inside_gt": int(fn.sum()) - fn_near_boundary,
        "fn_farther_than_10px_from_prediction": fn_far_from_prediction,
        "small_component_count_proxy": small_count,
        "small_component_detected_any_proxy": small_detected_any,
        "small_component_detected_half_proxy": small_detected_half,
        "small_component_recall_any_proxy": (
            small_detected_any / small_count if small_count else None
        ),
        "small_component_recall_half_proxy": (
            small_detected_half / small_count if small_count else None
        ),
        "fp_intensity_median_normalised": _median_or_none(intensity[fp]),
        "gt_intensity_median_normalised": _median_or_none(intensity[gt]),
        "tn_intensity_median_normalised": _median_or_none(intensity[tn]),
    }
    return result


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarise zero validation images")

    def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"image_count": 0}
        answer: dict[str, Any] = {"image_count": len(group)}
        for name in ("dice", "iou", "precision", "recall", "foreground_fraction_true"):
            values = np.asarray([row[name] for row in group], dtype=np.float64)
            answer[name] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()), "max": float(values.max()),
            }
        for name in ("tp_pixels", "fp_pixels", "fn_pixels", "fp_within_2px_of_gt",
                     "fp_farther_than_10px_from_gt", "fn_within_2px_of_gt_boundary",
                     "fn_farther_than_10px_from_prediction", "small_component_count_proxy",
                     "small_component_detected_any_proxy", "small_component_detected_half_proxy"):
            answer[f"{name}_total"] = int(sum(row[name] for row in group))
        small_total = answer["small_component_count_proxy_total"]
        answer["small_component_recall_any_proxy_micro"] = (
            answer["small_component_detected_any_proxy_total"] / small_total
            if small_total else None
        )
        answer["small_component_recall_half_proxy_micro"] = (
            answer["small_component_detected_half_proxy_total"] / small_total
            if small_total else None
        )
        return answer

    low = [row for row in rows if row["foreground_fraction_true"] <= LOW_FOREGROUND_FRACTION]
    higher = [row for row in rows if row["foreground_fraction_true"] > LOW_FOREGROUND_FRACTION]
    return {
        "scope": "saved LD validation predictions and derived validation labels only",
        "selection_warning": (
            "Exploratory descriptive summary only. Each fold's threshold was "
            "selected on its validation images, so these metrics are in-sample "
            "for threshold choice. Rotated folds expose each image as validation "
            "in one run and outer test in another; do not use this all-fold "
            "aggregate to choose a candidate and claim the same OOF evaluation "
            "is untouched. Never inspect outer_test artifacts here."
        ),
        "low_foreground_definition": f"foreground fraction <= {LOW_FOREGROUND_FRACTION}",
        "small_object_proxy_definition": (
            "8-connected semantic GT component area <=25 px; any-overlap and "
            ">=50%-overlap proxies are not instance-segmentation metrics"
        ),
        "distance_units": "native input pixels",
        "montage_legend": {
            "true_positive": "green", "false_positive": "red",
            "false_negative": "blue", "probability": "red high, blue low",
        },
        "all_validation_images": describe(rows),
        "low_foreground": describe(low),
        "higher_foreground": describe(higher),
    }


def _select_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for label, predicate in (
        ("low foreground", lambda row: row["foreground_fraction_true"] <= LOW_FOREGROUND_FRACTION),
        ("higher foreground", lambda row: row["foreground_fraction_true"] > LOW_FOREGROUND_FRACTION),
    ):
        group = sorted((row for row in rows if predicate(row)), key=lambda row: (row["dice"], row["filename"]))
        for qualifier, item in (("lowest Dice", group[0] if group else None),
                                ("highest Dice", group[-1] if group else None)):
            if item is not None and item not in chosen:
                chosen.append({**item, "selection_reason": f"{label}; {qualifier}"})
    return chosen


def _overlay(image_byte: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    display = np.repeat(image_byte[..., None], 3, axis=2).astype(np.float32)
    for mask, color in ((gt & pred, (0, 255, 0)),
                        (~gt & pred, (255, 35, 35)),
                        (gt & ~pred, (35, 145, 255))):
        display[mask] = 0.35 * display[mask] + 0.65 * np.asarray(color)
    return display.astype(np.uint8)


def _montage(
    selected: list[dict[str, Any]], input_dir: Path, label_dir: Path,
    output_path: Path,
) -> None:
    """Make a local review sheet; never export source TIFFs or labels."""
    tile = MONTAGE_TILE_PIXELS
    margin = 6
    title_height = 34
    columns = ("Input", "GT", "Pred", "Prob: red=high", "G:TP R:FP B:FN", "Error crop")
    sheet = Image.new("RGB", (6 * (tile + margin), len(selected) * (tile + title_height + margin)), "#151515")
    draw = ImageDraw.Draw(sheet)
    for row_number, record in enumerate(selected):
        name = record["filename"]
        image = tifffile.imread(input_dir / name)
        truth = tifffile.imread(label_dir / name).astype(bool)
        probability = tifffile.imread(Path(record["probability_path"]))
        pred = probability >= record["threshold"]
        image_byte = (_normalised_image(image) * 255).astype(np.uint8)
        grey = np.repeat(image_byte[..., None], 3, axis=2)
        gt_rgb = grey.copy(); gt_rgb[truth] = (30, 230, 30)
        pred_rgb = grey.copy(); pred_rgb[pred] = (220, 40, 220)
        p_byte = np.uint8(np.clip(probability * 255, 0, 255))
        heat = np.stack((p_byte, p_byte // 3, 255 - p_byte), axis=2)
        overlay = _overlay(image_byte, truth, pred)
        errors = truth ^ pred
        components, count = ndi.label(errors, structure=np.ones((3, 3), dtype=bool))
        if count:
            areas = np.bincount(components.ravel()); areas[0] = 0
            centre = np.argwhere(components == int(np.argmax(areas))).mean(axis=0)
        else:
            centre = np.asarray(image.shape) / 2
        height, width = image.shape
        y0 = int(np.clip(round(centre[0]) - tile // 2, 0, max(0, height - tile)))
        x0 = int(np.clip(round(centre[1]) - tile // 2, 0, max(0, width - tile)))
        crop = overlay[y0:y0 + tile, x0:x0 + tile]
        panels = (grey, gt_rgb, pred_rgb, heat, overlay, crop)
        row_y = row_number * (tile + title_height + margin)
        heading = f"{name} | {record['selection_reason']} | Dice {record['dice']:.3f}"
        draw.text((8, row_y + 2), heading, fill="white")
        for column, (title, panel) in enumerate(zip(columns, panels)):
            x = column * (tile + margin)
            draw.text((x + 3, row_y + 18), title, fill="#dddddd")
            bitmap = Image.fromarray(panel).resize((tile, tile), Image.Resampling.NEAREST if title in ("GT", "Pred", "TP/FP/FN") else Image.Resampling.BILINEAR)
            sheet.paste(bitmap, (x, row_y + title_height))
    sheet.save(output_path)


def analyse_validation_runs(
    run_root: Path,
    input_dir: Path,
    label_dir: Path,
    label_manifest: Path,
    output_dir: Path,
    folds: tuple[int, ...] = tuple(range(5)),
    expected_images: int = 36,
) -> dict[str, Any]:
    """Analyse declared validation members; never traverse an outer-test tree."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {output_dir}")
    label_data = _json_object(label_manifest)
    if (label_data.get("rule_id") != "ld_antialias_midpoint_v1"
            or label_data.get("label_semantics") != {"background": 0, "foreground": 1}):
        raise ValueError("Expected the frozen midpoint 0/1 LD label manifest")
    label_records = {entry["filename"]: entry for entry in label_data["files"]}
    fold_root = run_root / "ld" if (run_root / "ld").is_dir() else run_root
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in folds:
        if not 0 <= fold <= 4:
            raise ValueError("Fold must be within 0–4")
        directory = fold_root / f"fold_{fold}"
        split = _json_object(directory / "split.json")
        validation = directory / "validation_artifacts"
        manifest = _json_object(validation / "manifest.json")
        declared_names = {_safe_tiff_name(name) for name in split.get("validation", [])}
        entries = manifest.get("files")
        if (split.get("target") != "ld" or split.get("fold") != fold
                or manifest.get("target") != "ld" or not isinstance(entries, list)
                or manifest.get("image_count") != len(entries)):
            raise ValueError(f"Invalid LD validation metadata for fold {fold}")
        artifact_names = [_safe_tiff_name(entry.get("filename")) for entry in entries]
        if len(set(artifact_names)) != len(entries) or set(artifact_names) != declared_names:
            raise ValueError(f"Validation artifacts do not match fold {fold} validation split")
        threshold = float(manifest["threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Invalid validation threshold in fold {fold}")
        for entry in entries:
            name = _safe_tiff_name(entry["filename"])
            if name in seen:
                raise ValueError(f"Validation image occurs in two folds: {name}")
            if name not in label_records:
                raise ValueError(f"Missing derived-label provenance for {name}")
            image_path = input_dir / name
            label_path = label_dir / name
            probability_path = validation / "probabilities" / name
            if _sha256(image_path) != label_records[name]["source_image_sha256"]:
                raise ValueError(f"LD input differs from label preparation: {name}")
            if _sha256(label_path) != label_records[name]["derived_mask_sha256"]:
                raise ValueError(f"LD derived label changed: {name}")
            if _sha256(probability_path) != entry["probability_sha256"]:
                raise ValueError(f"Saved validation probability changed: {name}")
            image = tifffile.imread(image_path)
            truth = tifffile.imread(label_path)
            probability = tifffile.imread(probability_path)
            measurements = analyse_case(image, truth, probability, threshold)
            rows.append({"fold": fold, "filename": name, **measurements,
                         "probability_path": str(probability_path.resolve())})
            seen.add(name)
    if len(rows) != expected_images:
        raise ValueError(f"Expected {expected_images} unique validation images, found {len(rows)}")

    rows.sort(key=lambda row: row["filename"])
    report = _summarise(rows)
    selected = _select_representatives(rows)
    report["representative_cases"] = [
        {"filename": row["filename"], "fold": row["fold"],
         "selection_reason": row["selection_reason"], "dice": row["dice"]}
        for row in selected
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = [key for key in rows[0] if key != "probability_path"]
    with (output_dir / "validation_errors_per_image.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    with (output_dir / "validation_error_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    _montage(selected, input_dir, label_dir, output_dir / "validation_error_montage.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyse_validation_runs(
        args.run_root, args.input_dir, args.label_dir, args.label_manifest, args.output_dir
    )
    print(json.dumps({"image_count": report["all_validation_images"]["image_count"],
                      "output_dir": str(args.output_dir),
                      "selection_warning": report["selection_warning"]}))


if __name__ == "__main__":
    main()
