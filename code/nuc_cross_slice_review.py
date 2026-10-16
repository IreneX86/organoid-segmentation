#!/usr/bin/env python3
"""Create a read-only, CPU-only cross-slice review packet for frozen Nuc masks.

This is an exploratory viewer, not segmentation or validation. Candidate locations
come only from image intensities; the archived 2D mask is drawn for comparison
after candidates have been ranked. No labels, reviewer circles, model inference,
training, or changes to source TIFFs are involved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage


FILE_RE = re.compile(r"^fluo_(\d+)\.tif$", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Fixed viewing heuristics, deliberately not fitted to labels or target hints.
SMALL_BLUR_PX = 10.0
LARGE_BLUR_PX = 32.0
PEAK_SEPARATION_PX = 60
EDGE_MARGIN_PX = 48
MAX_MARKERS_PER_SLICE = 40
NEIGHBOR_WEIGHT = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tiffs_by_number(directory: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in directory.iterdir():
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        number = int(match.group(1))
        if number in paths:
            raise ValueError(f"Duplicate slice number {number} in {directory}")
        if not path.is_file():
            raise ValueError(f"Expected a regular TIFF file: {path}")
        paths[number] = path
    if not paths:
        raise ValueError(f"No fluo_###.tif files in {directory}")
    return paths


def expected_hashes(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ValueError("Input manifest must contain a files list")
    hashes: dict[str, str] = {}
    for entry in payload["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Input manifest contains a non-object files entry")
        name, expected = entry.get("name"), entry.get("input_sha256")
        if not isinstance(name, str) or FILE_RE.fullmatch(name) is None:
            raise ValueError(f"Invalid manifest filename: {name!r}")
        if not isinstance(expected, str) or SHA256_RE.fullmatch(expected.lower()) is None:
            raise ValueError(f"Invalid manifest SHA256 for {name!r}")
        if name in hashes:
            raise ValueError(f"Duplicate manifest filename: {name}")
        hashes[name] = expected.lower()
    if not hashes:
        raise ValueError("Input manifest files list is empty")
    return hashes


def read_grayscale(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(f"Expected a single 2D grayscale TIFF: {path}, got {image.shape}")
    if not np.issubdtype(image.dtype, np.number) or not np.all(np.isfinite(image)):
        raise ValueError(f"Expected finite numeric pixels: {path}")
    return image


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = read_grayscale(path)
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} differs from image shape {shape}: {path}")
    values = np.unique(mask)
    if not set(values.tolist()).issubset({0, 1, 255}):
        raise ValueError(f"Mask must be frozen binary 0/1 or 0/255: {path}")
    return mask != 0


def contrast_bounds(center: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(center, [1.0, 99.8])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Center slice has no usable display contrast")
    return float(lo), float(hi)


def normalized(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def dark_core_response(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Positive where a dark local core lies inside a brighter broad region.

    This multiscale difference has no learned parameters, object classifier, or
    calibrated threshold. It can also respond to non-nuclear texture/artifacts.
    """
    scaled = normalized(image, lo, hi)
    broad = ndimage.gaussian_filter(scaled, LARGE_BLUR_PX, mode="reflect")
    core = ndimage.gaussian_filter(scaled, SMALL_BLUR_PX, mode="reflect")
    return broad - core


def candidate_rows(
    center_response: np.ndarray,
    previous_response: np.ndarray | None,
    next_response: np.ndarray | None,
    mask: np.ndarray,
    slice_name: str,
) -> list[dict[str, object]]:
    height, width = center_response.shape
    maxima = ndimage.maximum_filter(center_response, size=PEAK_SEPARATION_PX + 1)
    local_peaks = (center_response == maxima) & (center_response > 0)
    margin = min(EDGE_MARGIN_PX, (height - 1) // 2, (width - 1) // 2)
    local_peaks[:margin, :] = False
    local_peaks[height - margin :, :] = False
    local_peaks[:, :margin] = False
    local_peaks[:, width - margin :] = False
    ys, xs = np.nonzero(local_peaks)
    if len(xs) == 0:
        return []

    center_values = center_response[ys, xs]
    previous_values = previous_response[ys, xs] if previous_response is not None else None
    next_values = next_response[ys, xs] if next_response is not None else None
    neighbor_values = [values for values in (previous_values, next_values) if values is not None]
    support = np.mean([np.maximum(values, 0) for values in neighbor_values], axis=0) if neighbor_values else np.zeros_like(center_values)
    rank_score = center_values + NEIGHBOR_WEIGHT * support
    # The mask does not enter selection or ranking. Tie breakers are deterministic.
    order = np.lexsort((xs, ys, -center_values, -rank_score))[:MAX_MARKERS_PER_SLICE]
    rows: list[dict[str, object]] = []
    for rank, index in enumerate(order, start=1):
        y, x = int(ys[index]), int(xs[index])
        rows.append({
            "slice": slice_name,
            "rank": rank,
            "x": x,
            "y": y,
            "center_response": round(float(center_values[index]), 6),
            "previous_response": round(float(previous_values[index]), 6) if previous_values is not None else "",
            "next_response": round(float(next_values[index]), 6) if next_values is not None else "",
            "neighbor_support_mean_positive": round(float(support[index]), 6),
            "rank_score": round(float(rank_score[index]), 6),
            "available_neighbors": len(neighbor_values),
            "inside_frozen_mask_at_point": bool(mask[y, x]),
        })
    return rows


def triptych(
    previous: np.ndarray | None,
    center: np.ndarray,
    next_image: np.ndarray | None,
    mask: np.ndarray,
    candidates: list[dict[str, object]],
    lo: float,
    hi: float,
) -> tuple[Image.Image, Image.Image]:
    height, width = center.shape
    panels: list[np.ndarray] = []
    for image in (previous, center, next_image):
        gray = np.zeros((height, width), dtype=np.uint8) if image is None else np.rint(normalized(image, lo, hi) * 255).astype(np.uint8)
        panels.append(gray)
    gray_triptych = Image.fromarray(np.concatenate(panels, axis=1))

    rgb = np.stack([np.concatenate(panels, axis=1)] * 3, axis=2)
    center_panel = rgb[:, width : 2 * width]
    if np.any(mask):
        tint = np.array([255, 80, 220], dtype=np.float32)
        center_panel[mask] = np.rint(0.62 * center_panel[mask] + 0.38 * tint).astype(np.uint8)
        boundary = mask & ~ndimage.binary_erosion(mask)
        center_panel[boundary] = [255, 40, 210]
    overlay = Image.fromarray(rgb)
    draw = ImageDraw.Draw(overlay)
    labels = ("PREVIOUS", "CENTER · FROZEN 2D MASK", "NEXT")
    for panel_index, label in enumerate(labels):
        if (panel_index == 0 and previous is None) or (panel_index == 2 and next_image is None):
            label = "NO ADJACENT SLICE"
        origin_x = panel_index * width + 8
        draw.rectangle((origin_x - 4, 4, origin_x + 260, 25), fill=(0, 0, 0))
        draw.text((origin_x, 8), label, fill=(255, 255, 255))

    # The same image-derived XY locations appear on all available slices.
    # Numbered markers are only on the center panel to keep neighbors readable.
    for row in candidates:
        x, y, rank = int(row["x"]), int(row["y"]), int(row["rank"])
        for panel_index, image in enumerate((previous, center, next_image)):
            if image is None:
                continue
            px = x + panel_index * width
            color = (255, 200, 20)
            draw.ellipse((px - 13, y - 13, px + 13, y + 13), outline=color, width=2)
            draw.line((px - 4, y, px + 4, y), fill=color, width=2)
            draw.line((px, y - 4, px, y + 4), fill=color, width=2)
            if panel_index == 1:
                draw.text((px + 16, y - 10), str(rank), fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
    return gray_triptych, overlay


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_index(summary: dict[str, object], slices: list[dict[str, object]]) -> str:
    lines = [
        "# Nuc 跨切片人工复核包",
        "",
        "每行是一张已有冻结 2D 预测对应的中心切片。三联图依次为前一层、中心层、后一层；三幅图共用**中心图 p1–p99.8** 算出的显示范围。灰度三联图和叠加三联图均保留原始像素网格，没有缩小采样。",
        "",
        "洋红色仅表示中心层已有的冻结预测 mask；黄色编号是**仅从图像亮暗结构生成的探索性待复核位置**，同一坐标在相邻层重复标出。候选按中心层暗芯/宽尺度亮周围响应，加上相邻层的正响应均值排序。已有 mask 不参与候选发现或排序；`掩膜外叠加图`只隐藏中心点落在 mask 内的黄色点，保留原始编号和排名。黄色点不等于漏核、真值或新标签。",
        "",
        "**没有目标完整人工真值，因此没有 Dice、召回率或准确率。不会自动修改任何标签、输入图、mask 或模型。** 中心层的第一/末层可能只有一侧邻图；缺失面板不复制边缘图。相邻层的同一 XY 也未逐张验证亚像素配准或物理 Z 间距，边缘层和内部层的排序不应直接互比。即使邻层支持较高，纹理、背景和其他结构也可能产生响应。",
        "",
        "## 文件",
        "",
        "- [候选位置 CSV](candidates.csv)：每层至多 40 个点；数值只是浏览优先级，不是概率。",
        "- [逐层汇总 CSV](slices.csv) 和 [完整运行记录 JSON](summary.json)：含输入及冻结 mask SHA256。",
        "",
        "| 中心图 | 前层 | 后层 | 全部点 | 点中心在掩膜外 | 灰度三联图 | 全部点叠加图 | 掩膜外叠加图 |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in slices:
        name = str(item["slice"])
        base = Path(name).stem
        lines.append(
            f"| `{name}` | {item['previous'] or '缺失'} | {item['next'] or '缺失'} | {item['candidate_count']} | "
            f"{item['candidate_outside_frozen_mask_at_point_count']} | [查看](grayscale/{base}.png) | "
            f"[查看](overlay/{base}.png) | [查看](overlay_outside_mask/{base}.png) |"
        )
    lines += [
        "",
        "运行规则固定在 `code/nuc_cross_slice_review.py`；这些尺度只是可重复的查看启发式，未用目标红圈或 `Nuc_label` 调节，也未按候选数评价模型。输入 manifest 只核验其中列出的 TIFF；额外邻层会记录实际 SHA256，并标为未列入 manifest。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    image_dir = args.image_dir.resolve()
    mask_dir = args.mask_dir.resolve()
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError("--image-dir and --mask-dir must be existing directories")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if (
        output_dir == image_dir or output_dir == mask_dir
        or output_dir in image_dir.parents or output_dir in mask_dir.parents
        or image_dir in output_dir.parents or mask_dir in output_dir.parents
    ):
        raise ValueError("Output directory cannot equal, contain, or be inside an input directory")
    if not output_dir.parent.is_dir():
        raise ValueError(f"Output parent directory does not exist: {output_dir.parent}")

    images = tiffs_by_number(image_dir)
    masks = tiffs_by_number(mask_dir)
    unknown = sorted(set(masks) - set(images))
    if unknown:
        raise ValueError(f"Frozen masks have no matching image slice numbers: {unknown}")
    manifest_path = args.input_manifest.resolve() if args.input_manifest else None
    expected = expected_hashes(manifest_path)
    if expected:
        missing = sorted(set(expected) - {path.name for path in images.values()})
        if missing:
            raise ValueError(f"Manifest TIFFs missing from --image-dir: {missing}")
        unlisted_masks = sorted(path.name for path in masks.values() if path.name not in expected)
        if unlisted_masks:
            raise ValueError(f"Mask centers absent from input manifest: {unlisted_masks}")

    needed = set(masks)
    for number in masks:
        needed.update(n for n in (number - 1, number + 1) if n in images)
    image_hashes = {images[number].name: sha256(images[number]) for number in sorted(needed)}
    for name, digest in expected.items():
        actual = image_hashes.get(name) or sha256(image_dir / name)
        if actual != digest:
            raise ValueError(f"Input manifest SHA256 mismatch for {name}: expected {digest}, got {actual}")
    mask_hashes = {masks[number].name: sha256(masks[number]) for number in sorted(masks)}

    # Complete hash verification and path checks before creating any output.
    output_dir.mkdir()
    (output_dir / "grayscale").mkdir()
    (output_dir / "overlay").mkdir()
    (output_dir / "overlay_outside_mask").mkdir()
    all_candidates: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    image_cache: dict[int, np.ndarray] = {}

    def image(number: int) -> np.ndarray:
        if number not in image_cache:
            image_cache[number] = read_grayscale(images[number])
        return image_cache[number]

    for number in sorted(masks):
        center = image(number)
        previous = image(number - 1) if number - 1 in images else None
        next_image = image(number + 1) if number + 1 in images else None
        if previous is not None and previous.shape != center.shape:
            raise ValueError(f"Adjacent image shape differs at {images[number - 1]}")
        if next_image is not None and next_image.shape != center.shape:
            raise ValueError(f"Adjacent image shape differs at {images[number + 1]}")
        mask = read_mask(masks[number], center.shape)
        lo, hi = contrast_bounds(center)
        center_response = dark_core_response(center, lo, hi)
        previous_response = dark_core_response(previous, lo, hi) if previous is not None else None
        next_response = dark_core_response(next_image, lo, hi) if next_image is not None else None
        candidates = candidate_rows(center_response, previous_response, next_response, mask, images[number].name)
        outside_candidates = [row for row in candidates if not row["inside_frozen_mask_at_point"]]
        all_candidates.extend(candidates)
        gray, overlay = triptych(previous, center, next_image, mask, candidates, lo, hi)
        _, outside_overlay = triptych(previous, center, next_image, mask, outside_candidates, lo, hi)
        basename = images[number].stem + ".png"
        gray.save(output_dir / "grayscale" / basename)
        overlay.save(output_dir / "overlay" / basename)
        outside_overlay.save(output_dir / "overlay_outside_mask" / basename)
        slice_rows.append({
            "slice": images[number].name,
            "previous": images[number - 1].name if previous is not None else "",
            "next": images[number + 1].name if next_image is not None else "",
            "width": int(center.shape[1]),
            "height": int(center.shape[0]),
            "display_p1": round(lo, 4),
            "display_p99_8": round(hi, 4),
            "frozen_mask_foreground_pixels": int(mask.sum()),
            "candidate_count": len(candidates),
            "candidate_outside_frozen_mask_at_point_count": len(outside_candidates),
            "image_sha256": image_hashes[images[number].name],
            "mask_sha256": mask_hashes[masks[number].name],
        })
        for stale in list(image_cache):
            if stale < number - 1:
                del image_cache[stale]

    candidate_fields = [
        "slice", "rank", "x", "y", "center_response", "previous_response",
        "next_response", "neighbor_support_mean_positive", "rank_score",
        "available_neighbors", "inside_frozen_mask_at_point",
    ]
    slice_fields = [
        "slice", "previous", "next", "width", "height", "display_p1", "display_p99_8",
        "frozen_mask_foreground_pixels", "candidate_count", "candidate_outside_frozen_mask_at_point_count",
        "image_sha256", "mask_sha256",
    ]
    write_csv(output_dir / "candidates.csv", all_candidates, candidate_fields)
    write_csv(output_dir / "slices.csv", slice_rows, slice_fields)
    summary: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exploratory cross-slice human review; no ground truth or accuracy metric",
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "input_manifest": str(manifest_path) if manifest_path else None,
        "input_manifest_sha256": sha256(manifest_path) if manifest_path else None,
        "image_sha256": image_hashes,
        "images_not_listed_in_manifest": sorted(set(image_hashes) - set(expected)) if expected else [],
        "mask_sha256": mask_hashes,
        "slice_count": len(slice_rows),
        "candidate_count": len(all_candidates),
        "candidate_outside_frozen_mask_at_point_count": sum(
            int(item["candidate_outside_frozen_mask_at_point_count"]) for item in slice_rows
        ),
        "heuristic": {
            "center_display_percentiles": [1.0, 99.8],
            "small_gaussian_sigma_px": SMALL_BLUR_PX,
            "large_gaussian_sigma_px": LARGE_BLUR_PX,
            "peak_separation_px": PEAK_SEPARATION_PX,
            "edge_margin_px": EDGE_MARGIN_PX,
            "max_markers_per_slice": MAX_MARKERS_PER_SLICE,
            "neighbor_weight": NEIGHBOR_WEIGHT,
            "candidate_ranking": "center broad-minus-core response + 0.5 × mean positive available-neighbor response at identical XY",
        },
        "caveats": [
            "Image-only markers are not predicted masks, ground truth, confirmed omissions, or calibrated probabilities.",
            "No complete target ground truth exists here; Dice, recall, and accuracy cannot be calculated.",
            "Missing first/last neighboring slices are shown as blank panels, never replicated.",
            "Identical XY across neighboring images assumes only file ordering; subpixel alignment and physical Z spacing were not verified here.",
            "Frozen masks are visual references only and do not affect marker selection or ranking.",
            "The outside-mask overlay is only a display filter at each marker's center point; it does not identify true omissions.",
            "No existing TIFF, mask, label, or model was changed.",
        ],
        "slices": slice_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "index.md").write_text(markdown_index(summary, slice_rows))
    print(f"Wrote {len(slice_rows)} slice triptychs and {len(all_candidates)} exploratory markers to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="One real organoid's sequential fluo TIFF directory")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Read-only archived 2D binary mask TIFF directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; fails if it exists")
    parser.add_argument("--input-manifest", type=Path, help="Optional JSON with files[].name and files[].input_sha256")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
