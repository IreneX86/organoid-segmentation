"""Prepare aligned ER-positive review crops from TRAIN organoids only.

These crops are annotation candidates, not complete binary ground truth: an
unmarked pixel may still contain ER. No model should treat zeros as negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from er_fluo_student import TEST_CELL, VALIDATION_CELL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_tile(mask: np.ndarray, size: int = 384) -> tuple[int, int, int]:
    """Choose the most ER-positive tile on a fixed, reproducible XY grid."""
    height, width = mask.shape
    if min(height, width) < size:
        raise ValueError("Image smaller than crop size")
    ys = sorted(set(range(0, height - size + 1, size - 112)) | {height - size})
    xs = sorted(set(range(0, width - size + 1, size - 112)) | {width - size})
    scored = [(int(np.count_nonzero(mask[y:y + size, x:x + size])), y, x)
              for y in ys for x in xs]
    count, y, x = max(scored, key=lambda row: (row[0], -row[1], -row[2]))
    if count == 0:
        raise ValueError("No ER-positive seed in this plane")
    return y, x, count


def display(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99.8))
    return np.uint8(np.clip((image.astype(np.float32) - low) /
                            max(1.0, float(high - low)), 0, 1) * 255)


def prepare(mask_root: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(output_root)
    manifests = sorted(mask_root.rglob("manifest.json"))
    if len(manifests) != 8:
        raise ValueError(f"Expected eight completed stacks, found {len(manifests)}")
    train = [p for p in manifests if p.parent.name not in {VALIDATION_CELL, TEST_CELL}]
    if len(train) != 6:
        raise ValueError("Expected six training organoids")
    output_root.mkdir(parents=True)
    records: list[dict] = []
    for manifest_path in train:
        source = json.loads(manifest_path.read_text())
        if source["source_z_start_zero_based"] != 0 or \
                source["source_z_stop_exclusive"] != source["source_shape"][0]:
            raise ValueError(f"Incomplete source stack: {manifest_path}")
        er_path = Path(source["source_er"])
        fluo_path = Path(source["source_fluo"])
        mask_path = manifest_path.parent / "ER_mask_line.tif"
        for path, expected in ((er_path, source["source_er_sha256"]),
                               (fluo_path, source["source_fluo_sha256"]),
                               (mask_path, source["outputs"]["ER_mask_line"]["sha256"])):
            if sha256(path) != expected:
                raise ValueError(f"Source changed: {path}")
        depth, height, width = source["source_shape"]
        z_samples = sorted({depth // 4, depth // 2, 3 * depth // 4})
        with tifffile.TiffFile(er_path) as er_file, \
                tifffile.TiffFile(fluo_path) as fluo_file, \
                tifffile.TiffFile(mask_path) as mask_file:
            if any(f.series[0].shape != (depth, height, width) or len(f.pages) != depth
                   for f in (er_file, fluo_file, mask_file)):
                raise ValueError(f"Z/XY stack shape mismatch: {manifest_path}")
            for z in z_samples:
                old = mask_file.pages[z].asarray()
                if old.dtype != np.uint8 or not np.isin(old, (0, 255)).all():
                    raise ValueError(f"Invalid seed mask {mask_path} z={z}")
                y, x, count = best_tile(old)
                region = np.s_[y:y + 384, x:x + 384]
                er = er_file.pages[z].asarray()[region]
                fluo = fluo_file.pages[z].asarray()[region]
                seed = old[region]
                if er.shape != fluo.shape or fluo.shape != seed.shape or seed.shape != (384, 384):
                    raise ValueError(f"Crop alignment failed {manifest_path} z={z}")
                name = f"{manifest_path.parent.name}_z{z:03d}_y{y}_x{x}"
                target = output_root / name
                target.mkdir()
                paths = {"ER_reference": er, "Fluo_input": fluo,
                         "observed_ER_positive_seed_NOT_COMPLETE_LABEL": seed}
                hashes = {}
                for stem, image in paths.items():
                    path = target / f"{stem}.tif"
                    tifffile.imwrite(path, image)
                    hashes[stem] = sha256(path)
                fluo_rgb = np.repeat(display(fluo)[..., None], 3, axis=2)
                fluo_rgb[seed > 0] = (255, 220, 0)
                overview = np.concatenate((np.repeat(display(er)[..., None], 3, axis=2),
                                           np.repeat(display(fluo)[..., None], 3, axis=2),
                                           fluo_rgb), axis=1)
                Image.fromarray(overview).save(target / "ER_Fluo_seed_overlay.png")
                records.append({"cell": manifest_path.parent.name,
                                "domain": manifest_path.parent.parent.name.strip(),
                                "split": "train", "z_zero_based": z, "y0": y, "x0": x,
                                "shape_yx": [384, 384], "observed_seed_pixels": count,
                                "source_sha256": {"ER": source["source_er_sha256"],
                                                  "Fluo": source["source_fluo_sha256"],
                                                  "old_mask": source["outputs"]["ER_mask_line"]["sha256"]},
                                "crop_sha256": hashes, "directory": name,
                                "status": "review candidate; zero in seed means unknown, not negative"})
    manifest = {"purpose": "aligned ER-positive / Fluo training review crops",
                "scope": "six training organoids only; validation and test unopened",
                "selection": "Z quartiles; fixed XY grid tile with most old-mask positives",
                "label_semantics": "255 in observed seed is a candidate positive; 0 is unknown",
                "training_ready": False, "records": records}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_root / "README.md").write_text(
        "# ER-positive paired crops\n\n"
        "Each crop maps the same Z and XY window in ER and Fluo. The yellow "
        "overlay shows existing ER-derived pseudo-mask positives. Existing "
        "masks omit visible ER, so every unmarked pixel is **unknown**, not a "
        "negative training label. The seeds themselves remain candidates "
        "because the teacher is not manual ground truth. Review/extend "
        "positives and mark definite negatives before training a masked-loss "
        "Fluo-only student. ER is reference only and must never enter the "
        "student input. Whole-organoid validation/test stacks were unopened.\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.mask_root, args.output_root)
    print(json.dumps({"output": str(args.output_root), "crops": len(result["records"]),
                      "train_cells": len({r["cell"] for r in result["records"]}),
                      "training_ready": result["training_ready"]}))
