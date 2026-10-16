"""Read-only audit of candidate Nuc inference stacks.

This script inspects TIFF headers and samples image pairs. It does not infer
acquisition order from filenames: numeric adjacency is only a hypothesis to
test against the image content. No image, mask, or prediction is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile


FILE_RE = re.compile(r"^fluo_(\d+)\.tif$", re.IGNORECASE)
GAPS = (1, 5, 10)
PAIRS_PER_GAP = 12


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    left = first.ravel().astype(np.float64)
    right = second.ravel().astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    return float((left @ right) / denominator) if denominator else float("nan")


def _thumbnail(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    image = tifffile.imread(path)
    if image.shape != (1200, 1200) or image.ndim != 2:
        raise ValueError(f"Unsupported shape for sampled image: {path}: {image.shape}")
    small = image.reshape(300, 4, 300, 4).mean(axis=(1, 3), dtype=np.float32)
    low, high = np.percentile(small, (1, 99.8))
    scaled = np.clip((small - low) / max(high - low, 1), 0, 1).astype(np.float32)
    # A per-tile residual reduces the influence of static low-frequency haze.
    tile_means = scaled.reshape(15, 20, 15, 20).mean(axis=(1, 3))
    local = scaled - np.repeat(np.repeat(tile_means, 20, 0), 20, 1)
    return scaled, local, (float(low), float(high))


def _phase_shift(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return signed (dy, dx) at the input array's pixel resolution."""
    window = np.hanning(first.shape[0])[:, None] * np.hanning(first.shape[1])[None, :]
    first_fft = np.fft.rfft2(first * window)
    second_fft = np.fft.rfft2(second * window)
    cross = first_fft * np.conj(second_fft)
    cross /= np.maximum(np.abs(cross), 1e-12)
    peak = np.unravel_index(np.argmax(np.fft.irfft2(cross, s=first.shape)), first.shape)
    displacement = np.asarray(peak, dtype=float)
    displacement = np.where(displacement > np.asarray(first.shape) / 2,
                            displacement - np.asarray(first.shape), displacement)
    return float(displacement[0]), float(displacement[1])


def _translation_px(first: np.ndarray, second: np.ndarray) -> float:
    dy, dx = _phase_shift(first, second)
    return float(4 * math.hypot(dy, dx))


def _native_registration(first_path: Path, second_path: Path) -> dict[str, float]:
    """Estimate native-pixel XY shift on image gradients, no image mutation."""
    inputs = []
    for path in (first_path, second_path):
        raw = tifffile.imread(path).astype(np.float32)
        low, high = np.percentile(raw, (1, 99.8))
        scaled = np.clip((raw - low) / max(high - low, 1), 0, 1)
        gy, gx = np.gradient(scaled)
        inputs.append(np.hypot(gy, gx).astype(np.float32))
    dy, dx = _phase_shift(*inputs)
    return {"dy_px": dy, "dx_px": dx,
            "translation_px": float(math.hypot(dy, dx))}


def _summary(values: list[float]) -> dict[str, float | int | None]:
    valid = np.asarray([number for number in values if np.isfinite(number)], dtype=float)
    if not len(valid):
        return {"n": 0, "median": None, "min": None, "max": None,
                "q10": None, "q90": None}
    return {
        "n": int(len(valid)), "median": float(np.median(valid)),
        "min": float(valid.min()), "max": float(valid.max()),
        "q10": float(np.percentile(valid, 10)),
        "q90": float(np.percentile(valid, 90)),
    }


def _points(first: int, last: int, count: int) -> list[int]:
    return sorted({int(round(value)) for value in np.linspace(first, last, count)})


def _boxmean(image: np.ndarray, width: int) -> np.ndarray:
    """Exact reflect-padded box mean used by the previous fixed diagnostic."""
    radius = width // 2
    padded = np.pad(image, radius, mode="reflect")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0, dtype=np.float64),
                                 axis=1, dtype=np.float64), ((1, 0), (1, 0)))
    return ((integral[width:, width:] - integral[:-width, width:]
             - integral[width:, :-width] + integral[:-width, :-width])
            / (width * width)).astype(np.float32)


def _input_only_detail(path: Path) -> dict[str, float]:
    """Same p1–99.8 + 5x5 high-pass q90 statistic, with no mask/probability."""
    image = tifffile.imread(path).astype(np.float32)
    low, high = np.percentile(image, [1, 99.8])
    image = (np.clip((image - low) / (high - low), 0, 1).astype(np.float32)
             if high > low else np.zeros_like(image))
    middle = (image >= .15) & (image <= .85)
    detail = np.abs(image - _boxmean(image, 5))
    return {
        "detail_q90_box5": (float(np.quantile(detail[middle], .9))
                            if np.any(middle) else float("nan")),
        "raw_p1": float(low), "raw_p99_8": float(high),
        "normalized_q90": float(np.quantile(image, .9)),
    }


def _domain_summary(paths: list[Path], cache: dict[Path, dict]) -> dict:
    records = []
    for path in paths:
        if path not in cache:
            cache[path] = _input_only_detail(path)
        records.append(cache[path])
    return {
        "n_images": len(records),
        "detail_q90_box5": _summary([r["detail_q90_box5"] for r in records]),
        "raw_p1": _summary([r["raw_p1"] for r in records]),
        "raw_p99_8": _summary([r["raw_p99_8"] for r in records]),
        "normalized_q90": _summary([r["normalized_q90"] for r in records]),
    }


def _cross_folder_similarity(paths: list[Path], thumbnail_cache: dict) -> dict:
    """Screen sampled target folders for obvious reused image content."""
    vectors = []
    for path in paths:
        if path not in thumbnail_cache:
            thumbnail_cache[path] = _thumbnail(path)
        local = thumbnail_cache[path][1]
        small = local.reshape(100, 3, 100, 3).mean(axis=(1, 3)).ravel()
        small = small - small.mean()
        norm = float(np.linalg.norm(small))
        vectors.append(small / norm if norm else small)
    matrix = np.stack(vectors).astype(np.float32)
    similarity = matrix @ matrix.T
    same_folder = np.asarray([path.parent.parent.name for path in paths])
    similarity[same_folder[:, None] == same_folder[None, :]] = -np.inf
    best_indices = similarity.argmax(axis=1)
    best_values = similarity[np.arange(len(paths)), best_indices]
    highest = int(np.argmax(best_values))
    return {
        "method": "100x100 local-detail thumbnails; each of 12 fixed samples per folder compared only with other folders",
        "best_other_folder_local_r": _summary(best_values.tolist()),
        "highest_pair": {
            "first": f"{paths[highest].parent.parent.name}/{paths[highest].name}",
            "second": f"{paths[int(best_indices[highest])].parent.parent.name}/{paths[int(best_indices[highest])].name}",
            "local_r": float(best_values[highest]),
        },
        "sample_pairs_r_at_least_0_99": int(np.sum(best_values >= .99)),
    }


def inspect(root: Path, training_input_dir: Path | None = None,
            training_split: Path | None = None) -> dict:
    stacks = sorted(path / "fluo" for path in root.iterdir()
                    if path.is_dir() and (path / "fluo").is_dir())
    output: dict = {
        "method": {
            "type": "read-only systematic sample of numeric file neighbors",
            "gaps": list(GAPS), "pairs_per_gap_per_stack": PAIRS_PER_GAP,
            "thumbnail": "1200x1200 to 300x300 exact 4x4 means; per-image p1-p99.8 clip",
            "local_detail": "thumbnail minus non-overlapping 20x20 tile mean",
            "translation": "FFT phase correlation on detail thumbnails, multiplied by 4",
            "limitation": "Similarity and numeric order do not establish Z, acquisition time, physical spacing, or XY registration at full resolution.",
        },
        "stacks": [],
    }
    cache: dict[Path, tuple[np.ndarray, np.ndarray, tuple[float, float]]] = {}
    domain_cache: dict[Path, dict] = {}
    for folder in stacks:
        indexed: dict[int, Path] = {}
        unmatched = []
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            match = FILE_RE.fullmatch(path.name)
            if match:
                number = int(match.group(1))
                if number in indexed:
                    raise ValueError(f"Duplicate numeric index in {folder}: {number}")
                indexed[number] = path
            else:
                unmatched.append(path.name)
        numbers = sorted(indexed)
        if not numbers:
            continue
        shapes, dtypes, page_counts, descriptions, software, tag_sets, ome_count = (
            Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), 0)
        for path in indexed.values():
            with tifffile.TiffFile(path) as tiff:
                page = tiff.pages[0]
                shapes[str(page.shape)] += 1
                dtypes[str(page.dtype)] += 1
                page_counts[len(tiff.pages)] += 1
                descriptions[str(page.description)[:120]] += 1
                software[str(page.tags.get("Software").value if page.tags.get("Software") else "none")[:120]] += 1
                tag_sets[", ".join(sorted(tag.name for tag in page.tags.values()))] += 1
                ome_count += int(bool(tiff.ome_metadata))
        sampled_pairs: list[dict] = []
        for gap in GAPS:
            eligible = [number for number in numbers if number + gap in indexed]
            if not eligible:
                continue
            for list_position in _points(0, len(eligible) - 1, PAIRS_PER_GAP):
                number = eligible[list_position]
                a, b = indexed[number], indexed[number + gap]
                if a not in cache:
                    cache[a] = _thumbnail(a)
                if b not in cache:
                    cache[b] = _thumbnail(b)
                global_a, detail_a, intensity_a = cache[a]
                global_b, detail_b, intensity_b = cache[b]
                sampled_pairs.append({
                    "index_a": number, "index_b": number + gap, "gap": gap,
                    "global_r": _pearson(global_a, global_b),
                    "local_detail_r": _pearson(detail_a, detail_b),
                    "phase_translation_px": _translation_px(detail_a, detail_b),
                    "p1_ratio_b_over_a": intensity_b[0] / max(intensity_a[0], 1e-9),
                    "p99_8_ratio_b_over_a": intensity_b[1] / max(intensity_a[1], 1e-9),
                })
        stack_data = {
            "folder": folder.parent.name,
            "count": len(numbers), "first_index": numbers[0],
            "last_index": numbers[-1],
            "missing_indices": sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers)),
            "unmatched_files": unmatched,
            "shapes": dict(shapes), "dtypes": dict(dtypes),
            "page_counts": dict(page_counts),
            "descriptions": dict(descriptions), "software": dict(software),
            "tag_name_sets": dict(tag_sets),
            "ome_tiff_count": ome_count,
            "by_gap": {},
        }
        adjacent = [number for number in numbers if number + 1 in indexed]
        registration_indices = [adjacent[index] for index in
                                _points(0, len(adjacent) - 1, 3)]
        stack_data["native_xy_registration_samples"] = [
            {"index_a": number, "index_b": number + 1,
             **_native_registration(indexed[number], indexed[number + 1])}
            for number in registration_indices
        ]
        if training_input_dir is not None and training_split is not None:
            # Twelve evenly spaced samples represent each full target folder.
            sample_indices = _points(0, len(numbers) - 1, 12)
            stack_data["input_only_domain_sample"] = {
                "sampled_numeric_indices": [numbers[index] for index in sample_indices],
                **_domain_summary([indexed[numbers[index]] for index in sample_indices],
                                  domain_cache),
            }
        for gap in GAPS:
            subset = [pair for pair in sampled_pairs if pair["gap"] == gap]
            stack_data["by_gap"][str(gap)] = {
                "global_r": _summary([pair["global_r"] for pair in subset]),
                "local_detail_r": _summary([pair["local_detail_r"] for pair in subset]),
                "phase_translation_px": _summary([pair["phase_translation_px"] for pair in subset]),
                "p99_8_ratio_b_over_a": _summary([pair["p99_8_ratio_b_over_a"] for pair in subset]),
                "sampled_pairs": subset,
            }
        output["stacks"].append(stack_data)
    output["total_tiffs"] = sum(item["count"] for item in output["stacks"])
    output["total_stacks"] = len(output["stacks"])
    if training_input_dir is not None and training_split is not None:
        split = json.loads(training_split.read_text())
        train_paths = [training_input_dir / name for name in split["train"]]
        target_paths = [
            root / item["folder"] / "fluo" / f"fluo_{number:03}.tif"
            for item in output["stacks"]
            for number in item["input_only_domain_sample"]["sampled_numeric_indices"]
        ]
        training_domain = _domain_summary(train_paths, domain_cache)
        target_domain = _domain_summary(target_paths, domain_cache)
        training_p10 = training_domain["detail_q90_box5"]["q10"]
        target_domain["count_below_training_detail_p10"] = sum(
            domain_cache[path]["detail_q90_box5"] < training_p10
            for path in target_paths
        )
        for item in output["stacks"]:
            sampled_paths = [root / item["folder"] / "fluo" / f"fluo_{number:03}.tif"
                             for number in item["input_only_domain_sample"]["sampled_numeric_indices"]]
            item["input_only_domain_sample"]["count_below_training_detail_p10"] = sum(
                domain_cache[path]["detail_q90_box5"] < training_p10
                for path in sampled_paths
            )
        output["input_only_domain_audit"] = {
            "method": "Existing p1-p99.8 normalization; 5x5 reflect box high-pass absolute residual q90, restricted to separately selected normalized pixels in [0.15,0.85]",
            "sampling": "All frozen training inputs; 12 fixed equally spaced inputs per target folder",
            "training_only": training_domain,
            "target_all_folders": target_domain,
        }
        output["cross_folder_sample_screen"] = _cross_folder_similarity(
            target_paths, cache)
    output["aggregate_by_gap"] = {}
    for gap in GAPS:
        pairs = [pair for stack in output["stacks"]
                 for pair in stack["by_gap"][str(gap)]["sampled_pairs"]]
        output["aggregate_by_gap"][str(gap)] = {
            "global_r": _summary([pair["global_r"] for pair in pairs]),
            "local_detail_r": _summary([pair["local_detail_r"] for pair in pairs]),
            "phase_translation_px": _summary([pair["phase_translation_px"] for pair in pairs]),
        }
    return output


def make_svg(result: dict) -> str:
    width, height = 1120, 540
    margins = (100, 50, 80, 90)
    left, top, bottom, right = margins
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#0891b2", "#0d9488", "#65a30d", "#ca8a04",
              "#ea580c", "#dc2626", "#db2777", "#9333ea", "#475569"]
    def x_for(gap: int) -> float:
        return left + plot_w * {1: .17, 5: .50, 10: .83}[gap]
    def y_for(correlation: float) -> float:
        return top + plot_h * (1 - min(max(correlation, .50), 1.0) - 0.0) / .50
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            '<text x="100" y="30" font-size="20" font-family="Arial" fill="#111827">Target TIFF numeric-neighbor continuity audit</text>',
            '<text x="100" y="49" font-size="12" font-family="Arial" fill="#475569">Local-detail Pearson r; 12 systematic pairs per gap per folder; numeric gap is not verified physical Z</text>']
    for tick in (.5, .6, .7, .8, .9, 1.0):
        y = y_for(tick)
        body.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="Arial" fill="#6b7280">{tick:.1f}</text>')
    for gap in GAPS:
        x = x_for(gap)
        body.append(f'<text x="{x:.1f}" y="{height-53}" text-anchor="middle" font-size="12" font-family="Arial" fill="#374151">{gap}</text>')
    for stack_index, stack in enumerate(result["stacks"]):
        color = colors[stack_index % len(colors)]
        for gap in GAPS:
            for pair in stack["by_gap"][str(gap)]["sampled_pairs"]:
                value = pair["local_detail_r"]
                if math.isfinite(value):
                    jitter = (stack_index - (len(result["stacks"]) - 1) / 2) * 8
                    body.append(f'<circle cx="{x_for(gap)+jitter:.1f}" cy="{y_for(value):.1f}" r="3.1" fill="{color}" opacity=".57"/>')
    body.extend([
        f'<text x="{left+plot_w/2:.1f}" y="{height-23}" text-anchor="middle" font-size="13" font-family="Arial" fill="#374151">Numeric TIFF index gap</text>',
        f'<text transform="translate(25 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-size="13" font-family="Arial" fill="#374151">Local-detail Pearson correlation (display limited to 0.5–1)</text>',
        f'<text x="{width-right-2}" y="{height-18}" text-anchor="end" font-size="11" font-family="Arial" fill="#6b7280">Colored points = separate folders; N={len(result["stacks"])} folders</text>',
        '</svg>'])
    return "\n".join(body) + "\n"


def make_domain_svg(result: dict) -> str:
    """Plot the fixed input-only detail statistic without any target labels."""
    domain = result["input_only_domain_audit"]
    width, height = 1050, 590
    left, right, top, row = 235, 70, 95, 42
    axis_width = width - left - right
    maximum = .025
    x_for = lambda value: left + axis_width * min(max(value, 0), maximum) / maximum
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="235" y="32" font-size="20" font-family="Arial" fill="#111827">Input-only local-detail distribution</text>',
        '<text x="235" y="53" font-size="12" font-family="Arial" fill="#475569">Frozen training inputs (n=133) versus 12 equally spaced images per target folder; no target labels used</text>',
    ]
    baseline = domain["training_only"]["detail_q90_box5"]
    body.append(f'<rect x="{x_for(baseline["q10"]):.1f}" y="80" width="{x_for(baseline["q90"])-x_for(baseline["q10"]):.1f}" height="{11*row+10}" fill="#dbeafe" opacity=".55"/>')
    for value in (0, .005, .010, .015, .020, .025):
        x = x_for(value)
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="80" y2="{top+10*row+20}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{x:.1f}" y="{height-42}" text-anchor="middle" font-size="11" font-family="Arial" fill="#6b7280">{value:.3f}</text>')
    rows = [("Frozen train (133)", baseline, "#1d4ed8")]
    rows.extend((stack["folder"], stack["input_only_domain_sample"]["detail_q90_box5"], "#ea580c")
                for stack in result["stacks"])
    for index, (name, stats, color) in enumerate(rows):
        y = top + row * index
        body.append(f'<text x="{left-12}" y="{y+4}" text-anchor="end" font-size="11" font-family="Arial" fill="#374151">{name}</text>')
        body.append(f'<line x1="{x_for(stats["q10"]):.1f}" x2="{x_for(stats["q90"]):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        body.append(f'<circle cx="{x_for(stats["median"]):.1f}" cy="{y}" r="6" fill="{color}" stroke="white" stroke-width="2"/>')
    body.extend([
        f'<text x="{left+axis_width/2}" y="{height-16}" text-anchor="middle" font-size="12" font-family="Arial" fill="#374151">p1–99.8 normalized 5×5 high-pass absolute residual q90 (unitless)</text>',
        '<text x="945" y="69" text-anchor="end" font-size="11" font-family="Arial" fill="#64748b">line = p10–p90; dot = median</text>',
        '</svg>',
    ])
    return "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--domain-svg", type=Path)
    parser.add_argument("--training-input-dir", type=Path)
    parser.add_argument("--training-split", type=Path)
    args = parser.parse_args()
    if bool(args.training_input_dir) != bool(args.training_split):
        parser.error("--training-input-dir and --training-split must be given together")
    result = inspect(args.root, args.training_input_dir, args.training_split)
    # Keep sampled pair details in memory for the figure, while recording only
    # compact per-folder/pooled statistics in the repository JSON artifact.
    compact = json.loads(json.dumps(result))
    for stack in compact["stacks"]:
        for gap in GAPS:
            stack["by_gap"][str(gap)].pop("sampled_pairs")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")
    args.svg.write_text(make_svg(result))
    if args.domain_svg:
        args.domain_svg.write_text(make_domain_svg(result))
    print(f"Read-only audit: {result['total_tiffs']} TIFFs across {result['total_stacks']} folders")
    for gap, values in result["aggregate_by_gap"].items():
        print(f"index gap {gap}: local r median {values['local_detail_r']['median']:.4f}; phase shift median {values['phase_translation_px']['median']:.1f}px")


if __name__ == "__main__":
    main()
