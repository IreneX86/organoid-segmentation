"""Exploratory ER mesh-line candidates, without changing reviewed masks.

The existing line mask is used only as a broad spatial prior. Actual new
filaments must have positive ER-channel ridge and local-contrast evidence.
These image-derived candidates are not manual ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, label
from skimage.filters import sato
from skimage.morphology import disk, remove_small_objects, skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "outputs/er_random5_profiled_20261028"))
from er_bright_line_pipeline import settings_for_profile, sha256  # noqa: E402
from export_sources_and_correlation import tissue_support  # noqa: E402

SOURCE = ROOT / "outputs/er_random5_profiled_20261028/originals_masks_correlation"
DEST = ROOT / "outputs/er_meshline_pilot_20261028"
SLICES = (
    ("cell_20260814_23298_NZ_50", 39),
    ("cell_20260814_23299_NZ_200", 56),
    ("cell_20260814_23300_NZ_100", 49),
    ("cell_20260904_23961_NZ_100", 44),
    ("cell_20260904_23962_NZ_170", 105),
    ("cell_20260904_23963_NZ_40", 24),
    ("cell_20260904_23965_NZ_140", 109),
    ("cell_20260904_23966_NZ_100", 55),
)


def extract(er: np.ndarray, fluo: np.ndarray, old: np.ndarray, profile: str) -> tuple[np.ndarray, dict]:
    image = er.astype(np.float32)
    settings = settings_for_profile(profile)
    background = gaussian_filter(image, settings.er_background_sigma)
    highpass = gaussian_filter(image, 0.7) - background
    relative = np.clip(highpass / np.maximum(background, settings.relative_background_floor), 0, 2)
    ridge = sato(relative, sigmas=(0.75, 1.25, 2.0), black_ridges=False)
    p50, p99 = np.percentile(image, [50, 99])
    # The lower contrast floor is limited to the vicinity of existing ER
    # structures. There is no attempt to infer a filament from a dark region.
    contrast = max(settings.minimum_highpass * 0.45, 0.012 * (p99 - p50))
    core = (ridge > settings.weak_ridge_floor * 0.58) & (highpass > contrast)
    context = binary_dilation(old > 0, structure=disk(14)) | binary_fill_holes(old > 0)
    context &= tissue_support(fluo)
    core &= context
    core = remove_small_objects(core, min_size=18)
    # Skeleton follows the measured narrow ridge, rather than the centroid of
    # a thick binary blob. A one-pixel halo makes the result reviewable at 1:1.
    line = binary_dilation(skeletonize(core), structure=disk(1)) & core
    groups, count = label(line)
    lengths = np.bincount(groups.ravel(), minlength=count + 1)
    keep = lengths >= 13
    keep[0] = False
    line &= keep[groups]
    return line.astype(np.uint8) * 255, {
        "contrast_floor": float(contrast),
        "ridge_floor": float(settings.weak_ridge_floor * 0.58),
        "context_pixels": int(context.sum()),
        "raw_pixels": int(core.sum()),
        "line_pixels": int(line.sum()),
    }


def gray(er: np.ndarray) -> np.ndarray:
    low, high = np.percentile(er, (1, 99.8))
    return np.clip((er.astype(np.float32) - low) / max(high - low, 1) * 255, 0, 255).astype(np.uint8)


def panel(er: np.ndarray, mask: np.ndarray | None) -> Image.Image:
    raw = gray(er)
    rgb = np.repeat(raw[..., None], 3, axis=2)
    if mask is not None:
        selected = mask > 0
        # Magenta reveals narrow lines without hiding grayscale ER signal.
        rgb[selected] = (0.35 * rgb[selected] + 0.65 * np.array([255, 45, 205])).astype(np.uint8)
    return Image.fromarray(rgb).resize((600, 600), Image.Resampling.LANCZOS)


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite {DEST}")
    manifest = json.loads((SOURCE / "correlation_results.json").read_text())
    records = {(item["cell"], item["z_one_based"]): item for item in manifest["slices"]}
    DEST.mkdir()
    results = []
    for cell, z in SLICES:
        item = records[cell, z]
        er = tifffile.imread(item["er_original"])
        fluo = tifffile.imread(item["fluo_original"])
        old = tifffile.imread(item["line_mask"])
        new, audit = extract(er, fluo, old, item["profile"])
        if new.shape != old.shape or new.dtype != np.uint8 or not np.isin(new, (0, 255)).all():
            raise RuntimeError(f"Bad candidate for {cell} z{z}")
        folder = DEST / cell
        folder.mkdir()
        mask_path = folder / f"ER_meshline_candidate_z{z:03d}.tif"
        tifffile.imwrite(mask_path, new, compression="zlib")
        board = Image.new("RGB", (2400, 636), "#101010")
        draw = ImageDraw.Draw(board)
        images = (panel(er, None), panel(er, old), panel(er, new),
                  Image.fromarray(new).convert("RGB").resize((600, 600), Image.Resampling.NEAREST))
        names = ("original ER", "existing line mask", "new ER mesh candidate", "pure binary mask")
        for i, (name, img) in enumerate(zip(names, images)):
            draw.text((i * 600 + 8, 8), f"{cell} Z{z} {name}", fill="white")
            board.paste(img, (i * 600, 36))
        png_path = folder / f"comparison_z{z:03d}.png"
        board.save(png_path)
        results.append({"cell": cell, "z_one_based": z, "profile": item["profile"],
                        "old_pixels": int(np.count_nonzero(old)), "new_pixels": int(np.count_nonzero(new)),
                        "new_outside_old": int(np.count_nonzero((new > 0) & (old == 0))),
                        "mask": str(mask_path), "sha256": sha256(mask_path),
                        "comparison": str(png_path), "audit": audit})
        print(f"{cell} z{z}: {results[-1]['new_pixels']} line pixels", flush=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "status": "exploratory image-derived ER mesh candidates, not human annotations",
        "results": results,
        "limitations": ["No manual ER ground truth to assess false positives or missed filaments",
                        "Existing mask used as broad spatial prior; missed cells far away remain possible"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
