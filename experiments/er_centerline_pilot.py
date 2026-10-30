"""Exploratory ER ridge-centerline pilot using the 40-slice review export.

This produces independent candidates for visual review. It does not generate
ground truth, train a model, or replace any previously accepted ER mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.filters import sato
from skimage.morphology import disk, remove_small_objects, skeletonize

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from er_bright_line_pipeline import settings_for_profile, sha256  # noqa: E402

SOURCE = PROJECT / "outputs/er_random5_profiled_20261028/originals_masks_correlation"
DEST = PROJECT / "outputs/er_centerline_pilot_20261028"
sys.path.insert(0, str(PROJECT / "outputs/er_random5_profiled_20261028"))
from export_sources_and_correlation import tissue_support  # noqa: E402
SELECT = {
    "cell_20260814_23298_NZ_50": 39,
    "cell_20260814_23299_NZ_200": 56,
    "cell_20260814_23300_NZ_100": 49,
    "cell_20260904_23961_NZ_100": 44,
    "cell_20260904_23962_NZ_170": 105,
    "cell_20260904_23963_NZ_40": 24,
    "cell_20260904_23965_NZ_140": 109,
    "cell_20260904_23966_NZ_100": 55,
}


def candidate(er: np.ndarray, fluo: np.ndarray, profile: str) -> tuple[np.ndarray, dict]:
    """Seek signal-aligned narrow ridges instead of thinning a binary blob."""
    settings = settings_for_profile(profile)
    image = er.astype(np.float32)
    background = gaussian_filter(image, settings.er_background_sigma)
    highpass = gaussian_filter(image, 0.8) - background
    relative = np.clip(highpass / np.maximum(background, settings.relative_background_floor), 0, 2)
    ridge = sato(relative, sigmas=(0.8, 1.5, 2.5), black_ridges=False)
    support = tissue_support(fluo)
    p50, p75, p99 = np.percentile(image, [50, 75, 99])
    # This fixed exploratory rule requires both ridge shape and local ER
    # contrast. It is not tuned against manual ER labels (none exist here).
    contrast_floor = max(settings.minimum_highpass * 0.6, 0.018 * (p99-p50))
    ridge_floor = settings.weak_ridge_floor * 0.7
    raw = (ridge > ridge_floor) & (highpass > contrast_floor) & (image > p75) & support
    raw = remove_small_objects(raw, min_size=30)
    thin = binary_dilation(skeletonize(raw), structure=disk(1)) & raw
    return (thin.astype(np.uint8)*255), {
        "contrast_floor": float(contrast_floor), "ridge_floor": float(ridge_floor),
        "raw_pixels": int(raw.sum()), "line_pixels": int(thin.sum()),
        "support_fraction": float(support.mean()),
    }


def display(er: np.ndarray, mask: np.ndarray | None = None) -> Image.Image:
    lo, hi = np.percentile(er, [1, 99.8])
    gray = np.clip((er.astype(np.float32)-lo)/max(hi-lo,1)*255,0,255).astype(np.uint8)
    rgb = np.repeat(gray[...,None],3,axis=2)
    if mask is not None:
        selected = mask > 0
        rgb[selected] = (0.4*rgb[selected] + 0.6*np.array([255,210,0])).astype(np.uint8)
    return Image.fromarray(rgb).resize((600,600),Image.Resampling.LANCZOS)


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Refusing overwrite: {DEST}")
    # The review export has already been checked against the raw source planes.
    manifest = json.loads((SOURCE / "correlation_results.json").read_text())
    records = {(row["cell"], row["z_one_based"]): row for row in manifest["slices"]}
    DEST.mkdir()
    results = []
    for cell, z in SELECT.items():
        record = records[cell,z]
        er = tifffile.imread(record["er_original"])
        fluo = tifffile.imread(record["fluo_original"])
        old = tifffile.imread(record["line_mask"])
        new, audit = candidate(er, fluo, record["profile"])
        if new.shape != old.shape or new.dtype != np.uint8 or not np.isin(new, [0,255]).all():
            raise RuntimeError(f"Invalid output: {cell} Z{z}")
        cell_dir = DEST / cell
        cell_dir.mkdir()
        path = cell_dir / f"ER_centerline_candidate_z{z:03d}.tif"
        tifffile.imwrite(path, new, compression="zlib")
        sheet = Image.new("RGB", (600*4, 636), "#101010")
        draw = ImageDraw.Draw(sheet)
        parts = (display(er), display(er,old), display(er,new),
                 Image.fromarray(new).convert("RGB").resize((600,600),Image.Resampling.NEAREST))
        for col,(title,part) in enumerate(zip(("original ER", "current mask", "ridge centerline candidate", "pure centerline mask"),parts)):
            draw.text((col*600+8,8),f"{cell} Z{z} — {title}",fill="white")
            sheet.paste(part,(col*600,36))
        sheet_path=cell_dir/f"comparison_z{z:03d}.png"
        sheet.save(sheet_path)
        results.append({"cell":cell,"z_one_based":z,"profile":record["profile"],
                        "old_pixels":int(np.count_nonzero(old)),"new_pixels":int(np.count_nonzero(new)),
                        "new_pixels_outside_old":int(np.count_nonzero((new>0)&(old==0))),
                        "old_pixels_not_in_new":int(np.count_nonzero((old>0)&(new==0))),
                        "candidate_mask":str(path),"sha256":sha256(path),
                        "comparison":str(sheet_path),"audit":audit})
        print(f"{cell} z{z}: old={results[-1]['old_pixels']} new={results[-1]['new_pixels']}",flush=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "status":"exploratory image-derived line candidates, not manual ground truth",
        "source_export":str(SOURCE),"selection":SELECT,"results":results,
        "limitations":["No independent manual ER masks for accuracy evaluation",
                       "Thin-line extraction can miss blurred sheets and generate noise in texture"],
    },indent=2)+"\n")


if __name__ == "__main__":
    main()
