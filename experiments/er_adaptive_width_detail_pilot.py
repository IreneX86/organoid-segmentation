"""Exploratory thin ER-line rendering with Z-supported detail, no source changes.

The old ER mask remains the reference. This experiment narrows its binary
centerline according to original-image contrast and adds only elongated,
adjacent-Z-supported candidate lines from a separately frozen pilot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage.morphology import disk, skeletonize


ROOT = Path(__file__).resolve().parents[1]
OLD_PLAN = ROOT / "outputs/er_old_line_thin_random5_20261028/manifest.json"
Z_PILOT = ROOT / "outputs/er_z_supported_neurite_pilot_20261028"
DEST = ROOT / "outputs/er_adaptive_width_detail_pilot_20261028"


def adaptive_width(er: np.ndarray, old: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return 3-px core, expanded to 5 px only where input has local contrast."""
    center = skeletonize(old)
    narrow = old & binary_dilation(center, structure=disk(1))
    broader = old & binary_dilation(center, structure=disk(2))
    image = gaussian_filter(er.astype(np.float32), 0.65)
    local = image - gaussian_filter(image, 18)
    signal = local[old]
    floor = float(np.percentile(signal, 60)) if signal.size else float("inf")
    thin = narrow | (broader & (local >= floor))
    # The centerline must survive. We do not restore thick source components.
    assert np.all(thin[center])
    return thin, {"local_contrast_p60_on_old": floor,
                  "old_pixels": int(old.sum()), "narrowed_pixels": int(thin.sum())}


def line_detail(addition: np.ndarray) -> np.ndarray:
    """Keep elongated Z-supported components, rejecting short bright flecks."""
    ids, _ = label(addition, structure=np.ones((3, 3), dtype=bool))
    keep = np.zeros(ids.max() + 1, dtype=bool)
    from scipy.ndimage import find_objects
    for index, box in enumerate(find_objects(ids), 1):
        if box is None:
            continue
        area = int(np.count_nonzero(ids[box] == index))
        y_span = box[0].stop - box[0].start
        x_span = box[1].stop - box[1].start
        if area >= 12 and max(x_span, y_span) >= 14:
            keep[index] = True
    return addition & keep[ids]


def panel(er: np.ndarray, mask: np.ndarray | None = None,
          color: tuple[int, int, int] = (255, 205, 0)) -> Image.Image:
    low, high = np.percentile(er, (1, 99.8))
    gray = np.clip((er.astype(np.float32) - low) * 255 / max(high-low, 1), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if mask is not None:
        rgb[mask] = (0.4*rgb[mask] + 0.6*np.array(color)).astype(np.uint8)
    return Image.fromarray(rgb).resize((600, 600), Image.Resampling.LANCZOS)


def main() -> None:
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir()
    rows = json.loads(OLD_PLAN.read_text())["records"]
    results = []
    for row in rows:
        cell = row["cell"]
        er = tifffile.imread(row["outputs"]["er"]["path"])
        old = tifffile.imread(row["outputs"]["old"]["path"]) > 0
        candidate = tifffile.imread(Z_PILOT / cell / "candidate_added_only.tif") > 0
        thin, audit = adaptive_width(er, old)
        detail = line_detail(candidate)
        new = thin | detail
        folder = DEST / cell
        folder.mkdir()
        tifffile.imwrite(folder / "ER_mask_adaptive_width_candidate.tif",
                         new.astype(np.uint8)*255, compression="zlib")
        tifffile.imwrite(folder / "added_detail_only.tif", detail.astype(np.uint8)*255,
                         compression="zlib")
        board = Image.new("RGB", (2400, 636), "#111111")
        draw = ImageDraw.Draw(board)
        parts = (panel(er), panel(er, old), panel(er, new),
                 panel(er, detail, (30, 230, 90)))
        for i, (title, part) in enumerate(zip(
            ("original ER", "old mask", "adaptive-width + Z detail", "Z detail only"), parts)):
            draw.text((i*600 + 8, 8), f"{cell} Z{row['source_z_one_based']} {title}", fill="white")
            board.paste(part, (i*600, 36))
        board.save(folder / "comparison.png")
        results.append({"cell": cell, "z": row["source_z_one_based"],
                        "old_pixels": audit["old_pixels"],
                        "new_pixels": int(new.sum()),
                        "removed_from_old": int((old & ~new).sum()),
                        "added_from_z_candidate": int(detail.sum()),
                        "local_contrast_p60_on_old": audit["local_contrast_p60_on_old"],
                        "comparison": str(folder / "comparison.png")})
        print(results[-1], flush=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "status": "exploratory unvalidated ER mask candidate",
        "source_old": str(OLD_PLAN), "source_z_candidate": str(Z_PILOT),
        "geometry": "old skeleton radius 1; radius 2 where original ER highpass >= old-mask p60; elongated adjacent-Z detail",
        "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
