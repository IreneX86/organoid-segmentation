"""Intermediate ER line-width candidate between historical and thin pilots.

Exploratory visualization only; neither source images nor existing masks change.
The fixed five-image comparison is not an accuracy evaluation without ER GT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, label, uniform_filter
from skimage.morphology import disk, skeletonize

from er_adaptive_width_detail_pilot import line_detail, panel


ROOT = Path(__file__).resolve().parents[1]
OLD_PLAN = ROOT / "outputs/er_old_line_thin_random5_20261028/manifest.json"
Z_PILOT = ROOT / "outputs/er_z_supported_neurite_pilot_20261028"
THIN_PILOT = ROOT / "outputs/er_adaptive_width_detail_pilot_20261028"
DEST = ROOT / "outputs/er_balanced_width_detail_pilot_20261028"


def balanced_mask(er: np.ndarray, old: np.ndarray, z_addition: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    center = skeletonize(old)
    image = gaussian_filter(er.astype(np.float32), 0.65)
    highpass = image - gaussian_filter(image, 18)
    threshold = float(np.percentile(highpass[old], 60)) if old.any() else float("inf")
    # Roughly 5 px around the original network centerline, and 7 px only
    # where the raw ER image supports a strong line. Never reach the old 9 px.
    body = old & binary_dilation(center, structure=disk(2))
    body |= old & binary_dilation(center, structure=disk(3)) & (highpass >= threshold)
    detail = line_detail(z_addition)
    p50, p99 = np.percentile(er, (50, 99))
    detail = binary_dilation(detail, structure=disk(1)) & (
        highpass > max(4, 0.025 * (p99 - p50))
    )
    result = body | detail
    assert np.all(result[center])
    return result, detail, {"highpass_p60_on_old": threshold,
                            "old_pixels": int(old.sum()),
                            "body_pixels": int(body.sum()),
                            "detail_pixels": int((detail & ~old).sum()),
                            "result_pixels": int(result.sum())}


def detail_crop(er: np.ndarray, old: np.ndarray, thin: np.ndarray,
                balanced: np.ndarray, cell: str, z: int) -> Image.Image:
    density = uniform_filter(old.astype(np.float32), size=301)
    y, x = np.unravel_index(np.argmax(density), density.shape)
    y = min(max(y - 150, 0), er.shape[0] - 300)
    x = min(max(x - 150, 0), er.shape[1] - 300)
    cut = (slice(y, y + 300), slice(x, x + 300))
    lower, upper = np.percentile(er, (1, 99.8))
    gray = np.clip((er[cut].astype(np.float32) - lower) * 255 /
                   max(upper - lower, 1), 0, 255).astype(np.uint8)
    board = Image.new("RGB", (2400, 632), "#101010")
    draw = ImageDraw.Draw(board)
    for i, (name, mask) in enumerate((("original ER", None), ("old mask", old),
                                       ("prior thin", thin), ("balanced", balanced))):
        rgb = np.repeat(gray[..., None], 3, axis=2)
        if mask is not None:
            selected = mask[cut]
            rgb[selected] = (rgb[selected] * 0.35 + np.array([255, 205, 0]) * 0.65).astype(np.uint8)
        part = Image.fromarray(rgb).resize((600, 600), Image.Resampling.NEAREST)
        board.paste(part, (600 * i, 32))
        draw.text((600 * i + 8, 7), f"{cell} Z{z} {name}, x{x}:{x+300} y{y}:{y+300}",
                  fill="white")
    return board


def main() -> None:
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir()
    results = []
    rows = json.loads(OLD_PLAN.read_text())["records"]
    for row in rows:
        cell = row["cell"]
        er = tifffile.imread(row["outputs"]["er"]["path"])
        old = tifffile.imread(row["outputs"]["old"]["path"]) > 0
        thin = tifffile.imread(THIN_PILOT / cell / "ER_mask_adaptive_width_candidate.tif") > 0
        z_addition = tifffile.imread(Z_PILOT / cell / "candidate_added_only.tif") > 0
        result, detail, audit = balanced_mask(er, old, z_addition)
        before_connectivity_restore = result.copy()
        original_ids, n = label(old, structure=np.ones((3, 3), dtype=bool))
        result_ids, _ = label(result, structure=np.ones((3, 3), dtype=bool))
        restored = 0
        for original in range(1, n + 1):
            ids = np.unique(result_ids[original_ids == original])
            ids = ids[ids > 0]
            if ids.size != 1:
                # Preserve the old outline for this ambiguous component only.
                # The user's priority is to keep validated old structures.
                result[original_ids == original] = True
                restored += 1
        result_ids, _ = label(result, structure=np.ones((3, 3), dtype=bool))
        for original in range(1, n + 1):
            ids = np.unique(result_ids[original_ids == original])
            if np.count_nonzero(ids) != 1:
                raise RuntimeError(f"Could not preserve old structure {original} in {cell}")
        audit["result_pixels"] = int(result.sum())
        audit["pixels_restored_for_connectivity"] = int((result & ~before_connectivity_restore).sum())
        folder = DEST / cell
        folder.mkdir()
        tifffile.imwrite(folder / "ER_mask_balanced_candidate.tif",
                         result.astype(np.uint8) * 255, compression="zlib")
        tifffile.imwrite(folder / "added_detail_only.tif",
                         (detail & ~old).astype(np.uint8) * 255, compression="zlib")
        board = Image.new("RGB", (2400, 636), "#101010")
        draw = ImageDraw.Draw(board)
        for i, (name, part) in enumerate(zip(
            ("original ER", "old mask", "prior thin", "balanced candidate"),
            (panel(er), panel(er, old), panel(er, thin), panel(er, result)))):
            board.paste(part, (600 * i, 36))
            draw.text((600 * i + 8, 8), f"{cell} Z{row['source_z_one_based']} {name}", fill="white")
        board.save(folder / "comparison.png")
        detail_crop(er, old, thin, result, cell, row["source_z_one_based"]).save(
            folder / "detail_crop_2x.png")
        audit.update({"cell": cell, "z": row["source_z_one_based"],
                      "old_structures_restored_for_connectivity": restored,
                      "comparison": str(folder / "comparison.png")})
        results.append(audit)
        print(audit, flush=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "status": "exploratory unvalidated intermediate-width ER mask",
        "source_old": str(OLD_PLAN), "source_z_candidate": str(Z_PILOT),
        "source_thin": str(THIN_PILOT), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
