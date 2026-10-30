"""Prepare Fluo-visible ER-like annotation review crops without changing source data.

The proposed positive mask is a *suggestion*, not a training label. Uncertain
pixels must remain unlabelled until a reviewer checks the Fluo image first and
uses the ER image only to confirm identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.filters import meijering
import tifffile


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1, 99.8))
    return np.clip((image.astype(np.float32) - lo) / max(float(hi - lo), 1.0), 0, 1)


def suggested_positive(fluo: np.ndarray, old_er_mask: np.ndarray) -> np.ndarray:
    """Conservative seed candidates; short proximity helps retain line continuity.

    This cannot identify an ER-like network from Fluo by itself. It is always
    intersected with the old ER-derived mask and is *not* an approved label.
    """
    image = normalized(fluo)
    contrast = image - gaussian_filter(image, 5)
    ridge = meijering(image, sigmas=(1, 2, 3), black_ridges=False)
    supported = (contrast > 0.005) & (ridge > 0.025)
    yy, xx = np.ogrid[-2:3, -2:3]
    near = binary_dilation(supported, structure=(xx * xx + yy * yy <= 4))
    return old_er_mask.astype(bool) & near


def select_rows(survey: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(survey.open(newline="")))
    cells = sorted({row["cell"] for row in rows if row["split"] == "train"})
    if len(cells) != 6:
        raise ValueError(f"Expected six training organoids, found {len(cells)}")
    selected = []
    for cell in cells:
        eligible = [row for row in rows if row["split"] == "train" and row["cell"] == cell
                    and 0.01 <= float(row["foreground_fraction"]) <= 0.12]
        if len(eligible) < 2:
            raise ValueError(f"Insufficient eligible crops for {cell}")
        eligible.sort(key=lambda row: (float(row["fluo_auc"]), int(row["z"]),
                                       int(row["y0"]), int(row["x0"])))
        selected.extend((eligible[0], eligible[len(eligible) // 2]))
    return selected


def one_dir(root: Path, cell: str) -> Path:
    matches = [p for p in root.rglob(cell) if p.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"Expected one source directory for {cell}: {matches}")
    return matches[0]


def prepare(source: Path, masks: Path, survey: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    chosen = select_rows(survey)
    output.mkdir(parents=True)
    records = []
    for index, row in enumerate(chosen, 1):
        cell, z = row["cell"], int(row["z"])
        y0, x0 = int(row["y0"]), int(row["x0"])
        sl = np.s_[y0:y0 + 384, x0:x0 + 384]
        src_dir, mask_dir = one_dir(source, cell), one_dir(masks, cell)
        er_path = next(p for p in src_dir.iterdir() if p.name.lower() == "er.tif")
        fluo_path = src_dir / "fluo.tif"
        old_path = mask_dir / "ER_mask_line.tif"
        fluo = tifffile.imread(fluo_path, key=z)[sl]
        er = tifffile.imread(er_path, key=z)[sl]
        old = tifffile.imread(old_path, key=z)[sl]
        if fluo.shape != (384, 384) or er.shape != fluo.shape or old.shape != fluo.shape:
            raise ValueError(f"Unaligned crop for {cell} z{z}")
        if old.dtype != np.uint8 or not np.isin(old, (0, 255)).all():
            raise ValueError(f"Old mask is not binary uint8: {old_path}")
        suggestion = suggested_positive(fluo, old > 0).astype(np.uint8) * 255
        tag = f"{index:02d}_{cell}_z{z:03d}_y{y0}_x{x0}"
        dst = output / tag
        dst.mkdir()
        files = {"fluo": ("fluo.tif", fluo), "er_reference": ("er_reference.tif", er),
                 "old_er_pseudolabel": ("old_er_pseudolabel.tif", old),
                 "suggested_positive_NOT_LABEL": ("suggested_positive_NOT_LABEL.tif", suggestion)}
        hashes = {}
        for key, (name, array) in files.items():
            path = dst / name
            tifffile.imwrite(path, array)
            hashes[key] = digest(path)
        records.append({"id": tag, "cell": cell, "domain": row["domain"], "split": "train",
                        "z_zero_based": z, "y0": y0, "x0": x0,
                        "original_foreground_pixels": int(np.count_nonzero(old)),
                        "suggested_pixels": int(np.count_nonzero(suggestion)),
                        "fluo_intensity_auc_of_old_mask": float(row["fluo_auc"]),
                        "source_sha256": {"fluo": digest(fluo_path), "er": digest(er_path),
                                          "old_mask": digest(old_path)}, "crop_sha256": hashes})
    (output / "manifest.json").write_text(json.dumps({
        "purpose": "Fluo-visible ER-like annotation review; no training labels approved",
        "scope": "six training organoids only; no validation/test read",
        "selection": "per cell lowest and median Fluo-intensity AUC among eligible survey tiles",
        "suggestion": "old ER-derived mask intersected with local Fluo ridge/contrast support within 2 px",
        "records": records}, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Fluo-visible ER-like review crops\n\n"
        "Each folder contains aligned 384×384 Fluo, paired ER reference, old ER-derived mask, "
        "and an automatic positive suggestion. The suggestion is **not** a label and is not "
        "training-ready; it may omit real visible filaments or retain cell boundaries. Review "
        "Fluo first, then use ER only to confirm ER identity. Mark visible supported lines "
        "positive, definite non-ER structures negative, and unclear/ER-only areas ignore. "
        "Source TIFFs and old masks are unchanged. Whole-organoid train/validation/test "
        "separation must remain in force.\n")
    print(json.dumps({"output": str(output), "crops": len(records),
                      "cells": len({r['cell'] for r in records})}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.masks, args.survey, args.output)
