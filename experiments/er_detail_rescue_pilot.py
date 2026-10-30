"""Conservative ER detail rescue: preserve old mask, add supported interior ridges.

Exploratory image-derived candidates only. No source TIFF or accepted mask is
modified, and no claim of increased accuracy is made without ER ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage.filters import sato
from skimage.morphology import disk, remove_small_objects, skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from er_bright_line_pipeline import process_slice, settings_for_profile, sha256  # noqa: E402

PLAN = ROOT / "outputs/er_old_line_thin_random5_20261028/manifest.json"
DEST = ROOT / "outputs/er_detail_rescue_pilot_20261028"


def rescue(er: np.ndarray, bright: np.ndarray, old: np.ndarray, profile: str) -> tuple[np.ndarray, dict]:
    settings = settings_for_profile(profile)
    image = gaussian_filter(er.astype(np.float32), 0.7)
    background = gaussian_filter(image, settings.er_background_sigma)
    highpass = image - background
    relative = np.clip(highpass / np.maximum(background, settings.relative_background_floor), 0, 2)
    ridge = sato(relative, sigmas=(0.8, 1.4, 2.2), black_ridges=False)
    p50, p99 = np.percentile(er, (50, 99))
    contrast_floor = max(settings.minimum_highpass * 0.5, 0.018 * (p99 - p50))
    raw = (ridge > settings.weak_ridge_floor * 0.7) & (highpass > contrast_floor) & (bright > 0)
    raw = remove_small_objects(raw, min_size=25)
    fine = binary_dilation(skeletonize(raw), structure=disk(1)) & raw
    # Retain coherent filaments that approach the established line. This
    # excludes free-floating texture and never expands outside the old bright
    # candidate's footprint.
    groups, count = label(binary_dilation(fine, structure=disk(1)))
    size = np.bincount(groups.ravel(), weights=fine.ravel(), minlength=count + 1)
    contact = np.bincount(groups.ravel(), weights=(fine & binary_dilation(old > 0, structure=disk(5))).ravel(), minlength=count + 1)
    keep = (size >= 20) & (contact >= 5)
    keep[0] = False
    fine &= keep[groups]
    additions = fine & (old == 0)
    result = (old > 0) | additions
    return result.astype(np.uint8) * 255, {
        "contrast_floor": float(contrast_floor),
        "ridge_floor": float(settings.weak_ridge_floor * 0.7),
        "interior_added_pixels": int(additions.sum()),
        "result_pixels": int(result.sum()),
    }


def panel(er: np.ndarray, mask: np.ndarray | None = None, color=(255, 205, 0)) -> Image.Image:
    low, high = np.percentile(er, (1, 99.8))
    gray = np.clip((er.astype(np.float32) - low) / max(high - low, 1) * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if mask is not None:
        selected = mask > 0
        rgb[selected] = (0.45 * rgb[selected] + 0.55 * np.array(color)).astype(np.uint8)
    return Image.fromarray(rgb).resize((600, 600), Image.Resampling.LANCZOS)


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Refusing overwrite: {DEST}")
    plan = json.loads(PLAN.read_text())
    DEST.mkdir()
    records = []
    for row in plan["records"]:
        z = row["source_z_zero_based"]
        er = tifffile.imread(row["outputs"]["er"]["path"])
        fluo = tifffile.imread(row["outputs"]["fluo"]["path"])
        old = tifffile.imread(row["outputs"]["old"]["path"])
        with tifffile.TiffFile(row["base_mask"]) as tiff:
            base = tiff.pages[z].asarray()
        bright, reproduced, _ = process_slice(er, fluo, base,
            settings=settings_for_profile(row["profile"]),
            artifact_bbox_yxyx=row["artifact_bbox_yxyx"])
        if not np.array_equal(reproduced, old):
            raise RuntimeError("Failed to reproduce old mask")
        new, audit = rescue(er, bright, old, row["profile"])
        if not np.all(new >= old):
            raise RuntimeError("Rescue must preserve every old mask pixel")
        folder = DEST / row["cell"]
        folder.mkdir()
        mask_path = folder / "ER_mask_line_with_interior_detail.tif"
        tifffile.imwrite(mask_path, new, compression="zlib")
        additions = (new > 0) & (old == 0)
        parts = (panel(er), panel(er, old), panel(er, new), panel(er, additions, color=(40, 230, 90)))
        board = Image.new("RGB", (2400, 636), "#101010")
        draw = ImageDraw.Draw(board)
        titles = ("original ER", "accepted old line", "old line + interior ridges", "added pixels in green")
        for i, (part, title) in enumerate(zip(parts, titles)):
            draw.text((i * 600 + 8, 8), f"{row['cell']} Z{z + 1} {title}", fill="white")
            board.paste(part, (i * 600, 36))
        png = folder / "comparison.png"
        board.save(png)
        records.append({"cell": row["cell"], "z_one_based": z + 1,
                        "profile": row["profile"], "mask": str(mask_path),
                        "mask_sha256": sha256(mask_path), "comparison": str(png),
                        "old_pixels": int(np.count_nonzero(old)), "audit": audit})
        print(f"{row['cell']} Z{z + 1}: added {audit['interior_added_pixels']} pixels", flush=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "status": "exploratory image-derived candidate, not manually validated",
        "source_plan": str(PLAN), "results": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
