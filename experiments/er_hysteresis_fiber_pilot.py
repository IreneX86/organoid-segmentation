"""ER ridge-following candidate: local contrast, hysteresis and adjacent Z.

This is a read-only exploratory mask comparison. It does not implement nERdy's
MATLAB Jerman filter, and does not replace any accepted ER masks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage.filters import apply_hysteresis_threshold, meijering
from skimage.morphology import disk, skeletonize


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/er_old_line_thin_random5_20261028/manifest.json"
DEST = ROOT / "outputs/er_hysteresis_fiber_topology_safe_20261028"


def ridge_network(er: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Smooth camera-scale noise while retaining 2-6 px membrane filaments;
    # subtract broad illumination instead of globally amplifying background.
    image = gaussian_filter(er.astype(np.float32), 0.65)
    background = gaussian_filter(image, 18)
    highpass = image - background
    p50, p99 = np.percentile(er, (50, 99))
    relative = np.clip(highpass / np.maximum(background, max(60, 0.25 * p50)), 0, 1)
    ridge = meijering(relative, sigmas=(1.0, 1.8, 2.8), black_ridges=False)
    contrast = highpass > max(4, 0.025 * (p99 - p50))
    # A low ridge response is accepted only if connected to a stronger seed.
    coherent = apply_hysteresis_threshold(np.where(contrast, ridge, 0),
                                          low=0.055, high=0.17)
    return coherent, ridge


def filter_supported_lines(coherent: np.ndarray, old: np.ndarray,
                           adjacent: np.ndarray) -> tuple[np.ndarray, dict]:
    zone = binary_dilation(old, structure=disk(30))
    z_support = binary_dilation(adjacent, structure=disk(3))
    raw = coherent & zone & z_support
    center = skeletonize(raw)
    ids, count = label(binary_dilation(center, structure=disk(1)),
                       structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(ids.ravel(), weights=center.ravel(), minlength=count + 1)
    contact = np.bincount(ids.ravel(), weights=(center & binary_dilation(old, structure=disk(6))).ravel(),
                          minlength=count + 1)
    keep = (sizes >= 24) & (contact >= 3)
    keep[0] = False
    lines = center & keep[ids]
    return lines, {"coherent_pixels": int(coherent.sum()),
                   "z_supported_pixels": int(raw.sum()),
                   "line_center_pixels": int(lines.sum()),
                   "line_groups_kept": int(keep.sum())}


def render(er: np.ndarray, mask: np.ndarray | None) -> Image.Image:
    lo, hi = np.percentile(er, (1, 99.8))
    gray = np.clip((er.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if mask is not None:
        rgb[mask] = (0.4 * rgb[mask] + 0.6 * np.array([255, 205, 0])).astype(np.uint8)
    return Image.fromarray(rgb).resize((600, 600), Image.Resampling.LANCZOS)


def main() -> None:
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir()
    records = []
    for row in json.loads(PLAN.read_text())["records"]:
        z = row["source_z_zero_based"]
        with tifffile.TiffFile(row["source_er"]) as stack:
            frames = [stack.pages[min(max(k, 0), len(stack.pages) - 1)].asarray()
                      for k in (z - 1, z, z + 1)]
        old = tifffile.imread(row["outputs"]["old"]["path"]) > 0
        current, _ = ridge_network(frames[1])
        neighbors = ridge_network(frames[0])[0] | ridge_network(frames[2])[0]
        lines, audit = filter_supported_lines(current, old, neighbors)
        # Keep an intermediate width around the established mask centerline.
        old_center = skeletonize(old)
        old_body = old & binary_dilation(old_center, structure=disk(2))
        detail = binary_dilation(lines, structure=disk(1))
        candidate = old_body | detail
        old_ids, old_count = label(old, structure=np.ones((3, 3), dtype=bool))
        new_ids, _ = label(candidate, structure=np.ones((3, 3), dtype=bool))
        restored = 0
        for index in range(1, old_count + 1):
            ids = np.unique(new_ids[old_ids == index])
            ids = ids[ids > 0]
            if ids.size != 1:
                candidate[old_ids == index] = True
                restored += 1
        new_ids, _ = label(candidate, structure=np.ones((3, 3), dtype=bool))
        for index in range(1, old_count + 1):
            ids = np.unique(new_ids[old_ids == index])
            if np.count_nonzero(ids) != 1:
                raise RuntimeError(f"Old ER component {index} split after restoration")
        folder = DEST / row["cell"]
        folder.mkdir()
        tifffile.imwrite(folder / "ER_mask_hysteresis_candidate.tif", candidate.astype(np.uint8) * 255,
                         compression="zlib")
        tifffile.imwrite(folder / "ER_ridge_additions_only.tif",
                         (detail & ~old).astype(np.uint8) * 255, compression="zlib")
        board = Image.new("RGB", (2400, 636), "#101010")
        draw = ImageDraw.Draw(board)
        for i, (name, part) in enumerate(zip(
            ("original ER", "old mask", "hysteresis fiber candidate", "ridge lines only"),
            (render(frames[1], None), render(frames[1], old),
             render(frames[1], candidate), render(frames[1], detail)))):
            board.paste(part, (600 * i, 36))
            draw.text((600 * i + 8, 8), f"{row['cell']} Z{z+1} {name}", fill="white")
        board.save(folder / "comparison.png")
        rec = {"cell": row["cell"], "z": z + 1, "old_pixels": int(old.sum()),
               "candidate_pixels": int(candidate.sum()),
               "ridge_additions_outside_old": int((detail & ~old).sum()),
               "old_centerline_retained": bool(np.all(candidate[old_center])),
               "old_components_restored_for_connectivity": restored,
               "comparison": str(folder / "comparison.png"), **audit}
        records.append(rec)
        print(rec, flush=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "status": "exploratory, unvalidated ER candidate",
        "fixed_parameters": {"ridge_sigmas": [1.0, 1.8, 2.8], "hysteresis_low": 0.055,
                             "hysteresis_high": 0.17, "z_tolerance_px": 3,
                             "old_support_px": 30, "old_body_radius_px": 2},
        "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
