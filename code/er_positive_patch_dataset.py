"""Aligned Fluo / approved ER-positive crops; unknown pixels are NOT background.

This prepares partial supervision, not a ready-to-train positive-only model.
No external model predictions or ER-channel intensities enter the input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from er_fluo_student import TEST_CELL, VALIDATION_CELL, sha256
from er_positive_paired_crops import best_tile


def encode(seed):
    if seed.dtype != np.uint8 or not np.isin(seed, (0, 255)).all():
        raise ValueError("Expected binary uint8 source mask")
    return np.where(seed > 0, 1, 255).astype(np.uint8)


def supervision(labels):
    """Return binary target and loss weight; use weights in BOTH BCE and Dice."""
    if not np.isin(labels, (0, 1, 255)).all():
        raise ValueError("Invalid partial-label codes")
    return (labels == 1).astype(np.float32), (labels != 255).astype(np.float32)


def require_two_classes(labels):
    """Prevent silently training a degenerate all-positive classifier."""
    supervision(labels)
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Training requires documented negative AND positive supervision")


def prepare(mask_root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    manifests = sorted(mask_root.glob("*/*/manifest.json"))
    if len(manifests) != 8:
        raise ValueError("Expected eight stack manifests")
    selected = [p for p in manifests if p.parent.name not in {VALIDATION_CELL, TEST_CELL}]
    if len(selected) != 6:
        raise ValueError("Expected six training organoids")
    output.mkdir(parents=True)
    records, sources, empty = [], [], []
    for path in selected:
        source = json.loads(path.read_text())
        fluo_path = Path(source["source_fluo"])
        mask_path = path.parent / "ER_mask_line.tif"
        expected_shape = tuple(source["source_shape"])
        if source["source_z_start_zero_based"] != 0 or source["source_z_stop_exclusive"] != expected_shape[0]:
            raise ValueError("Incomplete stack")
        for file, expected in ((fluo_path, source["source_fluo_sha256"]),
                               (mask_path, source["outputs"]["ER_mask_line"]["sha256"])):
            if sha256(file) != expected:
                raise ValueError(f"Changed source {file}")
        sources.append({"cell": path.parent.name, "manifest": str(path.resolve()),
                        "manifest_sha256": sha256(path), "shape": expected_shape,
                        "fluo_sha256": source["source_fluo_sha256"],
                        "mask_sha256": source["outputs"]["ER_mask_line"]["sha256"]})
        with tifffile.TiffFile(fluo_path) as f, tifffile.TiffFile(mask_path) as m:
            if any(t.series[0].shape != expected_shape or len(t.pages) != expected_shape[0] for t in (f, m)):
                raise ValueError("Stack alignment mismatch")
            for z in range(expected_shape[0]):
                mask = m.pages[z].asarray()
                encode(mask)  # Validate before selecting or skipping.
                if not np.any(mask):
                    empty.append({"cell": path.parent.name, "z": z})
                    continue  # Empty teacher mask does not prove negative tissue.
                y, x, count = best_tile(mask)
                image = f.pages[z].asarray()[y:y+384, x:x+384]
                seed = mask[y:y+384, x:x+384]
                name = f"{path.parent.name}_z{z:03d}_y{y}_x{x}.npz"
                target = output / name
                np.savez_compressed(target, fluo=image, er_positive_mask=seed,
                                    partial_label=encode(seed))
                records.append({"file": name, "sha256": sha256(target), "cell": path.parent.name,
                                "split": "train", "z_zero_based": z, "y0": y, "x0": x,
                                "shape": list(image.shape), "positive_pixels": count})
        print(f"Prepared {path.parent.name}", flush=True)
    result = {"status": "PAIRED_POSITIVES_READY; negative supervision unresolved",
              "training_ready": False, "input": "unaltered Fluo crop only",
              "selection": "one densest fixed-grid 384px crop per positive training slice",
              "codes": {"1": "old ER pseudo-positive", "255": "ignore", "0": "reserved for documented negatives; none generated"},
              "excluded_cells": [VALIDATION_CELL, TEST_CELL], "sources": sources,
              "empty_slices_skipped": empty, "records": records}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify(output: Path):
    manifest = json.loads((output / "manifest.json").read_text())
    count = positives = 0
    for row in manifest["records"]:
        file = output / row["file"]
        if row["cell"] in {VALIDATION_CELL, TEST_CELL} or sha256(file) != row["sha256"]:
            raise ValueError(f"Split/hash failed {file}")
        with np.load(file, allow_pickle=False) as data:
            if set(data.files) != {"fluo", "er_positive_mask", "partial_label"}:
                raise ValueError("Unexpected channels")
            if any(data[k].shape != (384, 384) for k in data.files):
                raise ValueError("Shape mismatch")
            if not np.array_equal(encode(data["er_positive_mask"]), data["partial_label"]):
                raise ValueError("Label mapping mismatch")
            target, weight = supervision(data["partial_label"])
            if int(target.sum()) != row["positive_pixels"] or not np.array_equal(target, weight):
                raise ValueError("Positive/ignore semantics mismatch")
            positives += int(target.sum())
        count += 1
    report = {"status": "PASS", "patches": count, "positive_pixels": positives,
              "train_cells": len(manifest["sources"]), "training_ready": False}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "verify"])
    p.add_argument("--mask-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "prepare":
        if a.mask_root is None:
            p.error("prepare requires --mask-root")
        prepare(a.mask_root, a.output)
    print(json.dumps(verify(a.output)))
