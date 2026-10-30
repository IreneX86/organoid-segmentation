"""Exploratory ER line rescue using original-image neuriteness and adjacent Z.

This is a single, fixed comparison inspired by ER-network enhancement work,
not an implementation of nERdy and not a validated mask. It never changes the
historical ER mask; additional pixels are review candidates only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage.filters import meijering
from skimage.morphology import disk, remove_small_objects, skeletonize


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/er_old_line_thin_random5_20261028/manifest.json"
DEST = ROOT / "outputs/er_z_supported_neurite_pilot_20261028"


def response(er: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = gaussian_filter(er.astype(np.float32), 0.65)
    background = gaussian_filter(image, 18)
    highpass = image - background
    p50, p99 = np.percentile(er, [50, 99])
    # Image-derived local contrast avoids a threshold on display-adjusted RGB.
    relative = np.clip(highpass / np.maximum(background, max(60, 0.25 * p50)), 0, 1)
    ridge = meijering(relative, sigmas=(1.0, 1.8, 2.8), black_ridges=False)
    contrast_floor = max(4, 0.025 * (p99 - p50))
    candidate = (highpass > contrast_floor) & (ridge > 0.085)
    return candidate, ridge


def overlay(er: np.ndarray, mask: np.ndarray | None = None,
            color: tuple[int, int, int] = (255, 205, 0)) -> Image.Image:
    low, high = np.percentile(er, (1, 99.8))
    gray = np.clip((er.astype(np.float32) - low) * 255 / max(high - low, 1), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if mask is not None:
        rgb[mask] = (0.4 * rgb[mask] + 0.6 * np.array(color)).astype(np.uint8)
    return Image.fromarray(rgb).resize((600, 600), Image.Resampling.LANCZOS)


def main() -> None:
    if DEST.exists():
        raise FileExistsError(DEST)
    rows = json.loads(PLAN.read_text())["records"]
    DEST.mkdir()
    records = []
    for row in rows:
        z = row["source_z_zero_based"]
        with tifffile.TiffFile(row["source_er"]) as stack:
            frames = [stack.pages[min(max(k, 0), len(stack.pages)-1)].asarray()
                      for k in (z-1, z, z+1)]
        old = tifffile.imread(row["outputs"]["old"]["path"]) > 0
        candidates = [response(frame)[0] for frame in frames]
        current = candidates[1]
        # Independent Z evidence: neighboring frames must contain a nearby
        # locally bright, elongated response. A pixel need not overlap exactly.
        neighbor = binary_dilation(candidates[0] | candidates[2], structure=disk(3))
        supported = current & neighbor & binary_dilation(old, structure=disk(24))
        supported = remove_small_objects(supported, min_size=30)
        groups, count = label(binary_dilation(supported, structure=disk(1)))
        sizes = np.bincount(groups.ravel(), weights=supported.ravel(), minlength=count+1)
        touches = np.bincount(groups.ravel(),
                              weights=(supported & binary_dilation(old, structure=disk(7))).ravel(),
                              minlength=count+1)
        keep = (sizes >= 30) & (touches >= 3)
        keep[0] = False
        supported &= keep[groups]
        addition = binary_dilation(skeletonize(supported), structure=disk(1)) & supported & ~old
        combined = old | addition
        folder = DEST / row["cell"]
        folder.mkdir()
        tifffile.imwrite(folder / "candidate_added_only.tif", addition.astype(np.uint8)*255,
                         compression="zlib")
        tifffile.imwrite(folder / "old_plus_z_supported_candidate.tif",
                         combined.astype(np.uint8)*255, compression="zlib")
        board = Image.new("RGB", (2400, 636), "#111111")
        draw = ImageDraw.Draw(board)
        panels = [overlay(frames[1]), overlay(frames[1], old),
                  overlay(frames[1], addition, (30, 230, 90)),
                  overlay(frames[1], combined)]
        for i, (title, panel) in enumerate(zip(
            ("original ER", "old accepted mask", "candidate additions only", "old + candidate"), panels)):
            draw.text((i*600+8, 8), f"{row['cell']} z{z+1} {title}", fill="white")
            board.paste(panel, (i*600, 36))
        board.save(folder / "comparison.png")
        records.append({"cell": row["cell"], "z": z+1,
                        "old_pixels": int(old.sum()), "candidate_added_pixels": int(addition.sum()),
                        "candidate_components": int(label(addition)[1]),
                        "comparison": str(folder / "comparison.png")})
        print(records[-1], flush=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "status": "exploratory candidate, no ER ground-truth validation",
        "method": "original-image Meijering neuriteness with adjacent-Z support; old pixels retained",
        "parameters": {"sigmas": [1.0, 1.8, 2.8], "ridge_floor": 0.085,
                       "contrast_fraction": 0.025, "z_tolerance_px": 3,
                       "old_proximity_px": 24, "minimum_component_pixels": 30},
        "results": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
