"""Reproduce review-guided ER masks on the last N slices of selected cells.

Each cell may have a different stack depth. Review ellipses are source-pixel
prompts for that specific cell and Z range, not transferable training labels.
The existing ER bright/line algorithm and its thresholds remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tifffile

from er_bright_line_pipeline import run_batch, sha256
from er_ridge_batch_predict import pairs


def _check_ellipses(review: dict, height: int, width: int) -> None:
    unknown = set(review) - {"exclude_ellipses", "rescue_ellipses"}
    if unknown:
        raise ValueError(f"Unknown review action: {sorted(unknown)}")
    for action in ("exclude_ellipses", "rescue_ellipses"):
        for ellipse in review.get(action, []):
            if set(ellipse) != {"center_xy", "radius_xy"}:
                raise ValueError(f"Malformed {action} ellipse: {ellipse}")
            center, radius = ellipse["center_xy"], ellipse["radius_xy"]
            if len(center) != 2 or len(radius) != 2:
                raise ValueError(f"Malformed {action} coordinates: {ellipse}")
            x, y = center
            rx, ry = radius
            if not (0 <= x < width and 0 <= y < height and rx > 0 and ry > 0):
                raise ValueError(f"Review ellipse outside source image: {ellipse}")


def prepare_plan(
    input_root: Path, base_mask_root: Path, config: dict, tail_slices: int
) -> list[dict]:
    """Validate all selected pairs and source-Z review keys before writing."""
    if tail_slices < 1:
        raise ValueError("tail_slices must be positive")
    selected = config.get("cells")
    if not isinstance(selected, dict) or not selected:
        raise ValueError("Review config must contain a nonempty 'cells' object")
    discovered = {}
    for er_path, fluo_path, _ in pairs(input_root):
        cell = er_path.parent.name
        if cell in discovered:
            raise ValueError(f"Duplicate cell folder name: {cell}")
        discovered[cell] = (er_path, fluo_path)
    if set(selected) - set(discovered):
        raise FileNotFoundError(f"Review cells not found: {sorted(set(selected)-set(discovered))}")
    plan = []
    for cell, entry in selected.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Review entry must be an object: {cell}")
        er_path, fluo_path = discovered[cell]
        base_path = base_mask_root / cell / "ER_mask.tif"
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        with (
            tifffile.TiffFile(er_path) as er,
            tifffile.TiffFile(fluo_path) as fluo,
            tifffile.TiffFile(base_path) as base,
        ):
            shape = er.series[0].shape
            if len(shape) != 3 or shape != fluo.series[0].shape or shape != base.series[0].shape:
                raise ValueError(f"ER, Fluo and base mask must share a ZYX shape: {cell}")
        depth, height, width = shape
        if depth < tail_slices:
            raise ValueError(f"{cell} has only {depth} slices")
        z_start = depth - tail_slices
        expected = list(range(z_start, depth))
        if entry.get("z_zero_based") != expected:
            raise ValueError(
                f"Review Z indices for {cell} must equal last {tail_slices}: {expected}"
            )
        review = entry.get("review")
        if not isinstance(review, dict) or not review:
            raise ValueError(f"Missing review actions for {cell}")
        _check_ellipses(review, height, width)
        plan.append({"cell": cell, "source_er": str(er_path), "source_fluo": str(fluo_path),
                     "base_mask": str(base_path), "z_start": z_start, "z_stop": depth})
    return plan


def run_reviewed_tail(
    input_root: Path, base_mask_root: Path, review_config: Path,
    output_root: Path, tail_slices: int = 5,
) -> list[Path]:
    """Apply only predeclared, cell-specific reviews to new output folders."""
    input_root = input_root.expanduser().resolve()
    base_mask_root = base_mask_root.expanduser().resolve()
    review_config = review_config.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    config = json.loads(review_config.read_text())
    plan = prepare_plan(input_root, base_mask_root, config, tail_slices)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "run_plan.json").write_text(json.dumps({
        "review_config": str(review_config), "review_config_sha256": sha256(review_config),
        "tail_slices": tail_slices, "cells": plan,
    }, indent=2) + "\n")
    results = []
    for item in plan:
        cell = item["cell"]
        reviews = {cell: {str(z): config["cells"][cell]["review"]
                          for z in range(item["z_start"], item["z_stop"])}}
        # run_batch accepts a direct cell folder and records original global Z.
        result = run_batch(
            Path(item["source_er"]).parent, output_root / cell,
            base_mask_root=Path(item["base_mask"]).parent, reviews=reviews,
            z_start=item["z_start"], z_stop=item["z_stop"],
        )
        results.extend(result)
    (output_root / "batch_complete.json").write_text(json.dumps({
        "status": "COMPLETE", "cells": len(plan), "slices": tail_slices * len(plan),
        "line_masks": [str(path) for path in results],
    }, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--base-mask-root", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tail-slices", type=int, default=5)
    args = parser.parse_args()
    run_reviewed_tail(args.input_root, args.base_mask_root, args.review_config,
                      args.output_root, args.tail_slices)


if __name__ == "__main__":
    main()
