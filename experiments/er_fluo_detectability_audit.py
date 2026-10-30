"""Audit whether existing ER-positive seed lines carry aligned Fluo signal.

This is a read-only detectability diagnostic, not segmentation accuracy. The
old mask contains omissions, so its zeros are NEVER treated as ground-truth
background. We instead compare seed skeleton scores with local shifted copies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.stats import rankdata
from skimage.filters import meijering
from skimage.morphology import skeletonize
import tifffile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1, 99.8))
    return np.clip((image.astype(np.float32) - lo) / max(1.0, float(hi - lo)), 0, 1)


def score(image: np.ndarray) -> np.ndarray:
    scaled = normalize(image)
    thin = meijering(gaussian_filter(scaled, 0.7), sigmas=(1, 2, 3),
                     black_ridges=False)
    contrast = np.maximum(gaussian_filter(scaled, 0.7) - gaussian_filter(scaled, 7), 0)
    return (thin * contrast).astype(np.float32)


def shift_without_wrap(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def auc(pos: np.ndarray, control: np.ndarray) -> float:
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(control, dtype=np.float64)
    if not p.size or not n.size:
        raise ValueError("Empty score group")
    ranks = rankdata(np.concatenate((p, n)), method="average")
    return float((ranks[:p.size].sum() - p.size * (p.size + 1) / 2) /
                 (p.size * n.size))


def local_control(skeleton: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """A nearby displaced geometry control; NOT a true negative mask."""
    control = np.zeros_like(skeleton)
    for dy, dx in ((-16, 0), (16, 0), (0, -16), (0, 16)):
        control |= shift_without_wrap(skeleton, dy, dx)
    return control & ~binary_dilation(seed, iterations=5)


def write_figure(rows: list[dict], path: Path) -> None:
    width, height = 1100, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 18), "Aligned ER-positive line vs local displaced geometry", fill="black")
    draw.text((35, 42), "AUC = probability aligned seed scores above nearby shifted control; NOT segmentation accuracy", fill="black")
    for i, channel in enumerate(("ER", "Fluo")):
        y0 = 115 + 350 * i
        draw.text((35, y0 - 25), f"{channel} ridge x local contrast (AUC)", fill="black")
        draw.line((180, y0 + 300, 1035, y0 + 300), fill="black", width=2)
        for tick in (0.5, 0.75, 1.0):
            xx = 180 + (tick - 0.5) * 1710
            draw.line((xx, y0 + 300, xx, y0 + 308), fill="black")
            draw.text((xx - 12, y0 + 312), str(tick), fill="black")
        for j, row in enumerate(rows):
            value = row[f"{channel.lower()}_aligned_vs_shift_auc"]
            yy = y0 + 10 + j * 16
            xx = 180 + (value - 0.5) * 1710
            color = (33, 110, 190) if row["domain"].lower().startswith("mcf") else (156, 76, 24)
            draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=color)
            draw.text((35, yy - 7), f"{row['cell'].split('_')[2]} Z{row['z_zero_based']}", fill="black")
    draw.text((35, 848), "Blue: MCF-7 | Brown: other acquisition group | 18 training crops, six organoids", fill="black")
    image.save(path)


def run(crops: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest_path = crops / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["training_ready"] is not False or len(manifest["records"]) != 18:
        raise ValueError("Expected locked training-only review set")
    if any(r["split"] != "train" for r in manifest["records"]):
        raise ValueError("Non-training crop in review set")
    output.mkdir(parents=True)
    rows = []
    for record in manifest["records"]:
        directory = crops / record["directory"]
        paths = {"er": directory / "ER_reference.tif",
                 "fluo": directory / "Fluo_input.tif",
                 "seed": directory / "observed_ER_positive_seed_NOT_COMPLETE_LABEL.tif"}
        for key, stem in (("er", "ER_reference"), ("fluo", "Fluo_input"),
                          ("seed", "observed_ER_positive_seed_NOT_COMPLETE_LABEL")):
            if sha256(paths[key]) != record["crop_sha256"][stem]:
                raise ValueError(f"Source crop changed: {paths[key]}")
        seed = tifffile.imread(paths["seed"]) > 0
        skeleton = skeletonize(seed)
        control = local_control(skeleton, seed)
        if skeleton.sum() < 100 or control.sum() < 100:
            raise ValueError(f"Too little seed/control in {directory}")
        row = {"cell": record["cell"], "domain": record["domain"],
               "z_zero_based": record["z_zero_based"],
               "seed_skeleton_pixels": int(skeleton.sum()),
               "shifted_control_pixels": int(control.sum())}
        for key in ("er", "fluo"):
            signal = score(tifffile.imread(paths[key]))
            row[f"{key}_aligned_vs_shift_auc"] = auc(signal[skeleton], signal[control])
            row[f"{key}_aligned_median_score"] = float(np.median(signal[skeleton]))
            row[f"{key}_shifted_median_score"] = float(np.median(signal[control]))
        rows.append(row)
    with (output / "per_crop.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_figure(rows, output / "aligned_signal_auc.png")
    cells = sorted({r["cell"] for r in rows})
    per_cell = []
    for cell in cells:
        group = [r for r in rows if r["cell"] == cell]
        per_cell.append({"cell": cell, "domain": group[0]["domain"],
                         "ER_median_auc": float(np.median([r["er_aligned_vs_shift_auc"] for r in group])),
                         "Fluo_median_auc": float(np.median([r["fluo_aligned_vs_shift_auc"] for r in group]))})
    result = {"definition": "line score = Meijering ridge response x positive local contrast",
              "comparison": "existing positive skeleton versus copies shifted 16 px in XY, excluding 5-px neighborhood of old seed",
              "warning": "AUC is a detectability probe, not segmentation accuracy; displaced control may contain real ER and old seed has omissions",
              "crop_manifest_sha256": sha256(manifest_path), "n_crops": len(rows),
              "n_train_organoids": len(cells), "per_cell": per_cell,
              "median_fluo_auc": float(np.median([r["fluo_aligned_vs_shift_auc"] for r in rows])),
              "median_ER_auc": float(np.median([r["er_aligned_vs_shift_auc"] for r in rows]))}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.crops, args.output)
    print(json.dumps({"n_crops": report["n_crops"],
                      "median_fluo_auc": report["median_fluo_auc"],
                      "median_er_auc": report["median_ER_auc"]}))
