"""Build a blinded, local-only Mito annotation review packet.

This implements experiments/mito_annotation_semantics_review_protocol.md.
Frozen outer-fold predictions and labels are used only to choose diagnostic
locations. The first-round reviewer directory contains anonymized input
pixels, a neutral crosshair, and blank forms; all categories, source names,
coordinates, and second-round GT contours stay in Git-ignored outputs/internal.
This is an exploratory label audit, not a new independent test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from segmentation_metrics import calculate_binary_metrics
from tiff_io import normalize_image, read_binary_mask, read_tiff_2d


SEED = "20261008"
CROP_SIZE = 320
CROP_CENTER = CROP_SIZE // 2
MIN_SEPARATION = 64
LOW_FOREGROUND_LIMIT = 0.05
SOURCE_CATEGORY = (
    "distant_fp_bright", "distant_fp_other", "bright_tn",
    "boundary_fp", "tp",
)
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SourceRow:
    filename: str
    fold: int
    foreground_fraction: float
    dice: float
    foreground_group: str = ""
    dice_group: str = ""


@dataclass(frozen=True)
class ReviewPoint:
    source: SourceRow
    category: str
    y: int
    x: int
    normalized_intensity: float
    local_residual: float


@dataclass(frozen=True)
class Case:
    normalized: np.ndarray
    truth: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    distance_to_gt: np.ndarray
    intensity_p90: float
    residual_p90: float


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_source_images(
    oof_rows: list[dict[str, str]], audit_rows: list[dict[str, str]]
) -> list[SourceRow]:
    """Use only the frozen OOF table for the protocol's 2×2 image draw."""

    if len(oof_rows) != 36 or len(audit_rows) != 36:
        raise ValueError("Expected 36 frozen OOF and 36 audit rows")
    audit = {row["filename"]: row for row in audit_rows}
    names = [row["filename"] for row in oof_rows]
    if len(set(names)) != 36 or set(names) != set(audit):
        raise ValueError("OOF and audit filenames must match uniquely")
    sources = []
    for row in oof_rows:
        name = row["filename"]
        if name.startswith("._") or Path(name).name != name:
            raise ValueError(f"Invalid source filename: {name}")
        fold = int(row["fold"])
        foreground = float(row["foreground_fraction_true"])
        dice = float(row["dice"])
        checked = audit[name]
        if (not 0 <= fold < 5 or not np.isfinite([foreground, dice]).all()
                or not 0 <= foreground <= 1 or not 0 <= dice <= 1
                or int(checked["fold"]) != fold
                or abs(float(checked["foreground_fraction"]) - foreground) > 1e-10
                or abs(float(checked["dice"]) - dice) > 1e-9):
            raise ValueError(f"Frozen OOF/audit disagreement: {name}")
        sources.append(SourceRow(name, fold, foreground, dice))

    selected = []
    for foreground_group, members in (
        ("low_foreground", [r for r in sources if r.foreground_fraction <= LOW_FOREGROUND_LIMIT]),
        ("higher_foreground", [r for r in sources if r.foreground_fraction > LOW_FOREGROUND_LIMIT]),
    ):
        if not members:
            raise ValueError(f"Empty foreground group: {foreground_group}")
        median_dice = statistics.median(r.dice for r in members)
        for dice_group, cell in (
            ("lower_dice", [r for r in members if r.dice < median_dice]),
            ("higher_dice", [r for r in members if r.dice >= median_dice]),
        ):
            # No borrowing across cells if fewer than two images exist.
            chosen = sorted(cell, key=lambda r: _sha(f"{SEED}|{r.filename}"))[:2]
            selected.extend(SourceRow(
                r.filename, r.fold, r.foreground_fraction, r.dice,
                foreground_group, dice_group,
            ) for r in chosen)
    return selected


def _load_case(
    row: SourceRow, image_dir: Path, mask_dir: Path, baseline_root: Path
) -> Case:
    image = read_tiff_2d(image_dir / row.filename)
    truth = read_binary_mask(mask_dir / row.filename).astype(bool)
    prediction_path = (
        baseline_root / f"fold_{row.fold}" / "predictions" / "validation" / row.filename
    )
    predicted = read_binary_mask(prediction_path).astype(bool)
    if image.shape != truth.shape or truth.shape != predicted.shape:
        raise ValueError(f"Image/mask/OOF geometry mismatch: {row.filename}")
    metrics = calculate_binary_metrics(truth, predicted)
    if (abs(metrics.dice - row.dice) > 1e-9
            or abs(metrics.foreground_fraction_true - row.foreground_fraction) > 1e-10):
        raise ValueError(f"OOF prediction does not reproduce frozen metrics: {row.filename}")
    normalized = normalize_image(image)
    residual = normalized - ndi.gaussian_filter(normalized, sigma=12)
    background = ~truth
    return Case(
        normalized=normalized,
        truth=truth,
        predicted=predicted,
        residual=residual,
        distance_to_gt=ndi.distance_transform_edt(background),
        intensity_p90=float(np.percentile(normalized[background], 90)),
        residual_p90=float(np.percentile(residual[background], 90)),
    )


def _candidate_masks(case: Case, bright_fp_intensity: float | None) -> dict[str, np.ndarray]:
    background = ~case.truth
    fp = case.predicted & background
    distant_fp = fp & (case.distance_to_gt > 10)
    bright = ((case.normalized >= case.intensity_p90)
              & (case.residual >= case.residual_p90))
    tn = background & ~case.predicted & (case.distance_to_gt > 10)
    matched_tn = (
        tn & (case.residual >= case.residual_p90)
        & (np.abs(case.normalized - bright_fp_intensity) <= 0.05)
        if bright_fp_intensity is not None else np.zeros_like(tn)
    )
    return {
        "distant_fp_bright": distant_fp & bright,
        "distant_fp_other": distant_fp & ~bright,
        "bright_tn": matched_tn,
        "boundary_fp": fp & (case.distance_to_gt <= 2),
        "tp": case.predicted & case.truth,
    }


def _pick_hash_order(
    eligible: np.ndarray, filename: str, category: str,
    previous: list[tuple[int, int]], limit: int,
) -> list[tuple[int, int]]:
    """Take the first SHA-ordered pixels respecting 64-pixel spacing."""

    ranked = [
        (_sha(f"{SEED}|{filename}|{category}|{int(y)}|{int(x)}"), int(y), int(x))
        for y, x in np.argwhere(eligible)
    ]
    ranked.sort()
    chosen: list[tuple[int, int]] = []
    for _, y, x in ranked:
        if all((y - py) ** 2 + (x - px) ** 2 >= MIN_SEPARATION ** 2
               for py, px in (*previous, *chosen)):
            chosen.append((y, x))
            if len(chosen) == limit:
                break
    return chosen


def select_points(row: SourceRow, case: Case) -> tuple[list[ReviewPoint], dict[str, dict[str, int]]]:
    """Draw at most 1+1 distant FP, 2 bright TN, 2 boundary FP, and 2 TP."""

    selected: list[ReviewPoint] = []
    previous: list[tuple[int, int]] = []
    counts: dict[str, dict[str, int]] = {}
    bright_intensity = None
    for category, limit in (
        ("distant_fp_bright", 1), ("distant_fp_other", 1),
        ("bright_tn", 2), ("boundary_fp", 2), ("tp", 2),
    ):
        eligible = _candidate_masks(case, bright_intensity)[category]
        points = _pick_hash_order(eligible, row.filename, category, previous, limit)
        counts[category] = {
            "eligible_pixels": int(eligible.sum()),
            "selected_points": len(points),
            "shortfall": limit - len(points),
        }
        for y, x in points:
            selected.append(ReviewPoint(
                row, category, y, x,
                float(case.normalized[y, x]), float(case.residual[y, x]),
            ))
            previous.append((y, x))
            if category == "distant_fp_bright":
                bright_intensity = float(case.normalized[y, x])
    return selected, counts


def _crop(array: np.ndarray, y: int, x: int, fill: int) -> np.ndarray:
    """Keep the target at pixel (160,160), padding rather than shifting."""

    y0, x0 = y - CROP_CENTER, x - CROP_CENTER
    source_y0, source_x0 = max(y0, 0), max(x0, 0)
    source_y1, source_x1 = min(y0 + CROP_SIZE, array.shape[0]), min(x0 + CROP_SIZE, array.shape[1])
    destination_y0, destination_x0 = source_y0 - y0, source_x0 - x0
    crop = np.full((CROP_SIZE, CROP_SIZE), fill, dtype=array.dtype)
    crop[
        destination_y0:destination_y0 + source_y1 - source_y0,
        destination_x0:destination_x0 + source_x1 - source_x0,
    ] = array[source_y0:source_y1, source_x0:source_x1]
    return crop


def _input_png(normalized: np.ndarray) -> Image.Image:
    # A single mapping is applied to the entire source image, never per crop.
    return Image.fromarray(np.rint(normalized * 255).astype(np.uint8))


def _marked_crop(normalized: np.ndarray, y: int, x: int) -> Image.Image:
    whole = np.rint(normalized * 255).astype(np.uint8)
    image = Image.fromarray(_crop(whole, y, x, 128)).convert("RGB")
    draw = ImageDraw.Draw(image)
    c = CROP_CENTER
    # Identical neutral reticle on every image; the target pixel stays visible.
    for endpoints in (((c - 14, c), (c - 5, c)), ((c + 5, c), (c + 14, c)),
                      ((c, c - 14), (c, c - 5)), ((c, c + 5), (c, c + 14))):
        draw.line((*endpoints[0], *endpoints[1]), fill=(255, 255, 255), width=3)
        draw.line((*endpoints[0], *endpoints[1]), fill=(0, 0, 0), width=1)
    return image


def _round_two_crop(normalized: np.ndarray, truth: np.ndarray, y: int, x: int) -> Image.Image:
    """Prepare GT contour privately; release only after round one is locked."""

    image = _marked_crop(normalized, y, x)
    contour = truth & ~ndi.binary_erosion(truth)
    cropped = _crop(contour.astype(np.uint8), y, x, 0).astype(bool)
    rgb = np.asarray(image).copy()
    rgb[cropped] = (255, 190, 0)
    return Image.fromarray(rgb)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_packet(
    image_dir: Path, mask_dir: Path, baseline_root: Path, oof_csv: Path,
    audit_csv: Path, output_dir: Path,
) -> dict[str, object]:
    """Create an immutable first-round packet and held-back second-round key."""

    output_dir = output_dir.resolve()
    allowed_root = (REPO_ROOT / "outputs").resolve()
    if output_dir == allowed_root or not output_dir.is_relative_to(allowed_root):
        raise ValueError(f"Review packet must be a new directory under {allowed_root}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite review packet: {output_dir}")
    selected_images = select_source_images(_read_csv(oof_csv), _read_csv(audit_csv))
    selected_points = []
    candidate_counts = {}
    for source in selected_images:
        case = _load_case(source, image_dir, mask_dir, baseline_root)
        points, counts = select_points(source, case)
        selected_points.extend(points)
        candidate_counts[source.filename] = counts
    # The final anonymous order is independent of category or sampling cell.
    selected_points.sort(key=lambda point: _sha(
        f"{SEED}|{point.source.filename}|{point.y}|{point.x}"
    ))
    image_names = sorted(
        (source.filename for source in selected_images),
        key=lambda name: _sha(f"{SEED}|{name}"),
    )
    image_ids = {name: f"I{index:04d}" for index, name in enumerate(image_names, 1)}

    allowed_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mito_review_build_", dir=allowed_root) as temp:
        packet = Path(temp)
        crops_dir = packet / "reviewer" / "crops"
        full_dir = packet / "reviewer" / "full_images"
        held_dir = packet / "internal" / "round2_hold"
        for directory in (crops_dir, full_dir, held_dir):
            directory.mkdir(parents=True)
        (packet / "internal").chmod(0o700)
        cases = {}
        for source in selected_images:
            cases[source.filename] = _load_case(source, image_dir, mask_dir, baseline_root)
            _input_png(cases[source.filename].normalized).save(
                full_dir / f"{image_ids[source.filename]}.png"
            )
        key_rows = []
        review_rows = []
        for index, point in enumerate(selected_points, 1):
            review_id = f"R{index:04d}"
            case = cases[point.source.filename]
            _marked_crop(case.normalized, point.y, point.x).save(
                crops_dir / f"{review_id}.png"
            )
            _round_two_crop(case.normalized, case.truth, point.y, point.x).save(
                held_dir / f"{review_id}.png"
            )
            review_rows.append({
                "anonymous_id": review_id,
                "full_image_id": image_ids[point.source.filename],
                "mito_at_center_yes_no_uncertain": "",
                "confidence_high_medium_low": "",
                "if_no_other_structure_artifact_bright_background_unknown": "",
                "boundary_clear_or_ambiguous": "",
                "brief_reason": "",
            })
            key_rows.append({
                "anonymous_id": review_id,
                "full_image_id": image_ids[point.source.filename],
                "filename": point.source.filename,
                "fold": point.source.fold,
                "foreground_group": point.source.foreground_group,
                "dice_group": point.source.dice_group,
                "category": point.category,
                "y": point.y,
                "x": point.x,
                "foreground_fraction": point.source.foreground_fraction,
                "oof_dice": point.source.dice,
                "normalized_intensity": point.normalized_intensity,
                "local_residual": point.local_residual,
            })
        _write_csv(packet / "reviewer" / "round1_blank.csv", review_rows, list(review_rows[0]))
        _write_csv(packet / "internal" / "key.csv", key_rows, list(key_rows[0]))
        (packet / "reviewer" / "INSTRUCTIONS.txt").write_text(
            "Independent first-round review only. Every crop is a full-image-normalized "
            "320x320 input view. Judge whether the structure at the center crosshair "
            "should be labelled Mito under the frozen SOP. The crosshair does not "
            "encode any class. Ixxxx is the same unmarked full input for context. "
            "Record yes/no/uncertain and confidence before discussing answers. "
            "Do not open any internal files.\n",
            encoding="utf-8",
        )
        (packet / "internal" / "RELEASE_AFTER_ROUND1.txt").write_text(
            "Do not release round2_hold or key.csv until both independent "
            "first-round answer sheets are locked. Round-two images show only "
            "the original manual-mask contour (amber); they never show model "
            "predictions or scores. Preserve original reviewer answers.\n",
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "protocol": "experiments/mito_annotation_semantics_review_protocol.md",
            "seed": SEED,
            "selected_image_count": len(selected_images),
            "selected_point_count": len(selected_points),
            "crop_size": CROP_SIZE,
            "target_pixel_in_crop": [CROP_CENTER, CROP_CENTER],
            "min_native_pixel_separation": MIN_SEPARATION,
            "selected_images": [
                {"filename": s.filename, "fold": s.fold,
                 "foreground_group": s.foreground_group, "dice_group": s.dice_group}
                for s in selected_images
            ],
            "candidate_counts_and_shortfalls": candidate_counts,
            "categories": list(SOURCE_CATEGORY),
            "round1_locked": False,
            "round2_released": False,
        }
        (packet / "internal" / "selection_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (packet / "PACKET_COMPLETE").write_text(
            "First-round packet complete; release reviewer/ only.\n", encoding="utf-8"
        )
        packet.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_packet(
        args.image_dir, args.mask_dir, args.baseline_root, args.oof_csv,
        args.audit_csv, args.output_dir,
    )
    print(json.dumps({
        "selected_image_count": report["selected_image_count"],
        "selected_point_count": report["selected_point_count"],
        "candidate_shortfalls": {
            name: {kind: values["shortfall"] for kind, values in counts.items()}
            for name, counts in report["candidate_counts_and_shortfalls"].items()
        },
        "reviewer_directory": str(args.output_dir.resolve() / "reviewer"),
    }, indent=2))


if __name__ == "__main__":
    main()
