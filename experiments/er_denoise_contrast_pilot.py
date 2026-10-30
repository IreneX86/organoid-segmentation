"""Small, read-only ER preprocessing comparison on five frozen random planes.

The established ridge/base mask, biological thresholds, and artifact exclusions
remain fixed. This does not replace the accepted line masks or raw TIFFs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from er_bright_line_pipeline import process_slice, settings_for_profile, sha256  # noqa: E402

SOURCE = ROOT / "outputs/er_old_line_thin_random5_20261028/manifest.json"
DEST = ROOT / "outputs/er_denoise_contrast_pilot_20261028"


def preprocess(er: np.ndarray, mode: str) -> np.ndarray:
    """Return a float32 image on the source intensity scale."""
    smooth = gaussian_filter(er.astype(np.float32), 0.7)
    if mode == "denoise_only":
        return smooth
    if mode != "denoise_local_contrast":
        raise ValueError(mode)
    low, high = np.percentile(smooth, (1, 99.8))
    scale = max(high - low, 1)
    normalized = np.clip((smooth - low) / scale, 0, 1)
    adapted = equalize_adapthist(normalized, kernel_size=128, clip_limit=0.008)
    # A 20% blend limits the change to image appearance/intensity; it does not
    # generate a mask or fabricate lines where the original signal is absent.
    return (0.8 * smooth + 0.2 * (adapted * scale + low)).astype(np.float32)


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
    records = json.loads(SOURCE.read_text())["records"]
    DEST.mkdir()
    output = []
    for row in records:
        z = row["source_z_zero_based"]
        er = tifffile.imread(row["outputs"]["er"]["path"])
        fluo = tifffile.imread(row["outputs"]["fluo"]["path"])
        old = tifffile.imread(row["outputs"]["old"]["path"])
        with tifffile.TiffFile(row["base_mask"]) as source:
            base = source.pages[z].asarray()
        settings = settings_for_profile(row["profile"])
        _, reproduced, _ = process_slice(er, fluo, base, settings=settings,
                                          artifact_bbox_yxyx=row["artifact_bbox_yxyx"])
        if not np.array_equal(old, reproduced):
            raise RuntimeError(f"Failed to reproduce old mask: {row['cell']} Z{z+1}")
        predictions = {}
        folder = DEST / row["cell"]
        folder.mkdir()
        for mode in ("denoise_only", "denoise_local_contrast"):
            processed = preprocess(er, mode)
            _, line, _ = process_slice(processed, fluo, base, settings=settings,
                                       artifact_bbox_yxyx=row["artifact_bbox_yxyx"])
            path = folder / f"ER_mask_{mode}.tif"
            tifffile.imwrite(path, line, compression="zlib")
            predictions[mode] = {"mask": str(path), "sha256": sha256(path),
                                 "pixels": int(np.count_nonzero(line)),
                                 "new_pixels": int(np.count_nonzero((line > 0) & (old == 0))),
                                 "lost_pixels": int(np.count_nonzero((line == 0) & (old > 0)))}
        # Five panes show original evidence, accepted output, controlled
        # preprocess variants and exactly which pixels contrast adds.
        changed = tifffile.imread(predictions["denoise_local_contrast"]["mask"])
        parts = (panel(er), panel(er, old),
                 panel(er, tifffile.imread(predictions["denoise_only"]["mask"])),
                 panel(er, changed), panel(er, (changed > 0) & (old == 0), (40, 220, 90)))
        sheet = Image.new("RGB", (3000, 636), "#101010")
        draw = ImageDraw.Draw(sheet)
        titles = ("original ER", "accepted old line", "denoise only", "denoise + mild local contrast", "new pixels in green")
        for i, (part, title) in enumerate(zip(parts, titles)):
            draw.text((i * 600 + 8, 8), f"{row['cell']} Z{z + 1} {title}", fill="white")
            sheet.paste(part, (i * 600, 36))
        comparison = folder / "comparison.png"
        sheet.save(comparison)
        output.append({"cell": row["cell"], "z_one_based": z + 1,
                       "profile": row["profile"], "old_pixels": int(np.count_nonzero(old)),
                       "predictions": predictions, "comparison": str(comparison)})
        print(f"{row['cell']} Z{z + 1}: old {output[-1]['old_pixels']}, "
              f"denoise {predictions['denoise_only']['pixels']}, "
              f"contrast {predictions['denoise_local_contrast']['pixels']}", flush=True)
    (DEST / "manifest.json").write_text(json.dumps({
        "status": "unvalidated preprocessing comparison; not manual ER ground truth",
        "source_selection": str(SOURCE),
        "conditions": ["accepted old line", "Gaussian denoise sigma 0.7",
                       "Gaussian denoise sigma 0.7 + 20% CLAHE blend (128 px tiles, clip 0.008)"],
        "results": output}, indent=2) + "\n")


if __name__ == "__main__":
    main()
