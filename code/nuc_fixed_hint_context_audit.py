"""Read-only contrast trajectories at previously reviewed, fixed Nuc hints.

The input hint JSON is a local, Git-ignored diagnostic containing approximate
bounding boxes from a hand-drawn screenshot. This script never fits a model,
selects a threshold, or changes a TIFF. Its committed outputs contain group
aggregates only: neither image pixels nor hint coordinates are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from tiff_io import normalize_image


N_IMAGES = 10
MISS_COVERAGE = 0.05
COVERED_COVERAGE = 0.50
STRONGER_MARGIN = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _core_minus_ring(image: np.ndarray, bbox: list[int]) -> float:
    """Replicate the fixed ellipse proxy in the earlier probability audit."""

    if len(bbox) != 4 or any(not isinstance(value, int) for value in bbox):
        raise ValueError("Each local hint needs a four-integer bounding box")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("A local hint falls outside the image grid")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = max((x1 - x0) / 2, 3), max((y1 - y0) / 2, 3)
    ax0 = max(0, int(cx - 1.45 * rx) - 1)
    ax1 = min(image.shape[1], int(cx + 1.45 * rx) + 2)
    ay0 = max(0, int(cy - 1.45 * ry) - 1)
    ay1 = min(image.shape[0], int(cy + 1.45 * ry) + 2)
    yy, xx = np.mgrid[ay0:ay1, ax0:ax1]
    radius = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    values = image[ay0:ay1, ax0:ax1]
    core = values[radius < 0.45]
    ring = values[(radius >= 0.85) & (radius < 1.25)]
    if not core.size or not ring.size:
        raise ValueError("A local hint has an empty proxy region")
    return float(np.mean(core) - np.mean(ring))


def _quantiles(rows: np.ndarray) -> dict[str, list[float]]:
    return {
        "q25": [float(v) for v in np.quantile(rows, 0.25, axis=0)],
        "median": [float(v) for v in np.median(rows, axis=0)],
        "q75": [float(v) for v in np.quantile(rows, 0.75, axis=0)],
    }


def inspect(hint_json: Path, input_dir: Path) -> dict[str, object]:
    source = json.loads(hint_json.read_text(encoding="utf-8"))
    hints = source.get("hints")
    if not isinstance(hints, list) or len(hints) != 26:
        raise ValueError("Expected the archived 26 non-edge rough hints")
    groups: dict[str, list[int]] = {"missed": [], "covered": [], "intermediate": []}
    for index, item in enumerate(hints):
        coverage = float(item["frozen_threshold_coverage"])
        if not np.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError("An archived coverage value is invalid")
        if coverage < MISS_COVERAGE:
            groups["missed"].append(index)
        elif coverage >= COVERED_COVERAGE:
            groups["covered"].append(index)
        else:
            groups["intermediate"].append(index)

    trajectories = np.empty((len(hints), N_IMAGES), dtype=np.float64)
    image_hashes: dict[str, str] = {}
    for image_index in range(1, N_IMAGES + 1):
        name = f"fluo_{image_index:03d}.tif"
        path = input_dir / name
        raw = tifffile.imread(path)
        if raw.shape != (1200, 1200) or raw.ndim != 2:
            raise ValueError(f"Unexpected 2D shape for {name}")
        image_hashes[name] = _sha256(path)
        image = normalize_image(raw, 1.0, 99.8)
        for hint_index, item in enumerate(hints):
            trajectories[hint_index, image_index - 1] = _core_minus_ring(
                image, item["bbox_xyxy"]
            )
    if not np.isfinite(trajectories).all():
        raise ValueError("A contrast proxy was not finite")

    missed = trajectories[groups["missed"]]
    covered = trajectories[groups["covered"]]
    if not len(missed) or not len(covered):
        raise ValueError("The archived hints did not contain both comparison groups")
    stronger_early = np.min(missed[:, 1:3], axis=1) < missed[:, 0] - STRONGER_MARGIN
    stronger_any = np.min(missed[:, 1:], axis=1) < missed[:, 0] - STRONGER_MARGIN
    output: dict[str, object] = {
        "method": {
            "purpose": "descriptive fixed-location image-only context check; no fitting or target GT",
            "input_images": "fluo_001.tif through fluo_010.tif",
            "normalization": "independent p1-p99.8 per complete image, unchanged from 2D baseline",
            "proxy": "fixed approximate ellipse mean(core radius<0.45) minus mean(ring 0.85<=radius<1.25)",
            "grouping": "archived fluo_001 frozen-baseline rough-core coverage <0.05 missed; >=0.50 covered; middle excluded",
            "stronger_definition": "a more negative proxy than fluo_001 by >0.02 normalized intensity units",
            "caveats": [
                "Rough screenshot hints are not complete or pixel-accurate ground truth.",
                "The boxes are fixed in XY with no registration; tissue can drift through the apparent sequence.",
                "The first frame is a stack boundary for a symmetric 3/5-slice model unless earlier images exist.",
                "Groups were selected after seeing baseline predictions; this is selection-exposed descriptive analysis.",
                "Numeric filename adjacency is not verified physical Z order; these results do not prove 2.5D benefit.",
            ],
        },
        "input_integrity": {
            "archived_hint_json_sha256": _sha256(hint_json),
            "image_sha256": image_hashes,
        },
        "counts": {key: len(indices) for key, indices in groups.items()},
        "fluo_indices": list(range(1, N_IMAGES + 1)),
        "missed_proxy": _quantiles(missed),
        "covered_proxy": _quantiles(covered),
        "missed_early_context_stronger_by_0_02": int(np.sum(stronger_early)),
        "missed_any_later_stronger_by_0_02": int(np.sum(stronger_any)),
    }
    return output


def plot(summary: dict[str, object], path: Path) -> None:
    """Render aggregate trajectories as a self-contained SVG, without plots libs."""

    width, height = 900, 480
    left, top, right, bottom = 94, 63, 870, 380
    tracks = [summary["missed_proxy"], summary["covered_proxy"]]
    all_values = [float(v) for track in tracks for part in ("q25", "q75")
                  for v in track[part]]
    y_min = min(all_values + [0.0]) - 0.01
    y_max = max(all_values + [0.0]) + 0.01

    def xcoord(index: int) -> float:
        return left + (index - 1) * (right - left) / (N_IMAGES - 1)

    def ycoord(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="94" y="30" font-family="Arial,sans-serif" font-size="20" fill="#1f2937">Rough Nuc hints: fixed-location contrast</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#3b4450"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#3b4450"/>',
    ]
    for index in range(1, N_IMAGES + 1):
        x = xcoord(index)
        lines.extend([
            f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom+5}" stroke="#3b4450"/>',
            f'<text x="{x:.1f}" y="{bottom+23}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#3b4450">{index}</text>',
        ])
    for value in np.linspace(y_min, y_max, 6):
        y = ycoord(float(value))
        lines.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e7eb"/>',
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#3b4450">{value:.2f}</text>',
        ])
    y_zero = ycoord(0.0)
    lines.append(f'<line x1="{left}" y1="{y_zero:.1f}" x2="{right}" y2="{y_zero:.1f}" stroke="#858d98" stroke-dasharray="5 5"/>')
    for key, color, label in (
        ("missed_proxy", "#bb4450", "Low baseline coverage (&lt;5%)"),
        ("covered_proxy", "#28658c", "High baseline coverage (≥50%)"),
    ):
        track = summary[key]
        upper = [(xcoord(i), ycoord(v)) for i, v in enumerate(track["q75"], 1)]
        lower = [(xcoord(i), ycoord(v)) for i, v in enumerate(track["q25"], 1)]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower[::-1])
        median = [(xcoord(i), ycoord(v)) for i, v in enumerate(track["median"], 1)]
        path_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in median)
        lines.append(f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.16"/>')
        lines.append(f'<polyline points="{path_points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lines.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.3" fill="{color}"/>'
                     for x, y in median)
        legend_y = 81 if key == "missed_proxy" else 105
        lines.extend([
            f'<line x1="570" y1="{legend_y-4}" x2="595" y2="{legend_y-4}" stroke="{color}" stroke-width="3"/>',
            f'<text x="603" y="{legend_y}" font-family="Arial,sans-serif" font-size="12" fill="#27303d">{label}</text>',
        ])
    lines.extend([
        '<text x="482" y="438" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#27303d">Numeric fluo index (physical Z order unverified)</text>',
        '<text x="23" y="225" transform="rotate(-90 23 225)" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#27303d">Core − ring, normalized intensity</text>',
        '<text x="94" y="465" font-family="Arial,sans-serif" font-size="11" fill="#5b6471">Median ± interquartile range; rough selection-exposed hints, no XY registration or target GT.</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hint-json", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.resolve() == args.hint_json.resolve():
        raise ValueError("Aggregate output cannot overwrite archived hints")
    result = inspect(args.hint_json, args.input_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(result, args.output_svg)
    print(json.dumps({"counts": result["counts"],
                      "missed_early_context_stronger_by_0_02": result["missed_early_context_stronger_by_0_02"],
                      "output_json": str(args.output_json), "output_svg": str(args.output_svg)}))


if __name__ == "__main__":
    main()
