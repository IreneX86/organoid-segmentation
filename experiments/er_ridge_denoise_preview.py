"""Remove sparse, unstable specks from an exploratory ER mask stack.

This is a conservative post-processing preview, not a validated ER model.
The original mask TIFF is read-only and the output directory must be new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, gaussian_filter, label


def denoise_stack(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    if mask.ndim != 3 or mask.shape[0] < 2 or not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError("Expected at least two Z slices with uint8 0/255 mask")
    binary = mask > 0
    contextual = np.zeros_like(binary)
    # A tiny fragment far from other ER candidates is more likely to be noise.
    # The density threshold was selected on this ten-slice exploratory preview.
    for z in range(len(binary)):
        density = gaussian_filter(binary[z].astype(np.float32), sigma=25)
        contextual[z] = binary[z] & (density > 0.05)

    output = np.zeros_like(binary)
    audits = []
    neighbor_structure = np.ones((5, 5), dtype=bool)
    for z in range(len(binary)):
        adjacent = np.zeros(binary.shape[1:], dtype=bool)
        for neighbor in (z - 1, z + 1):
            if 0 <= neighbor < len(binary):
                adjacent |= binary_dilation(contextual[neighbor], structure=neighbor_structure)
        components, count = label(contextual[z])
        size = np.bincount(components.ravel(), minlength=count + 1)
        adjacent_pixels = np.bincount(
            components.ravel(), weights=adjacent.ravel(), minlength=count + 1
        )
        # Keep substantial 2D structures. For a small piece, require evidence
        # at almost the same XY location in a neighboring Z slice.
        retain = (size >= 25) | (adjacent_pixels >= 2)
        retain[0] = False
        output[z] = retain[components]
        audits.append({
            "slice_in_input_stack": z,
            "before_pixels": int(binary[z].sum()),
            "after_pixels": int(output[z].sum()),
            "removed_pixels": int((binary[z] & ~output[z]).sum()),
        })
    return np.where(output, 255, 0).astype(np.uint8), audits


def _display(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99.8])
    return np.uint8(np.clip((image - low) / max(high - low, 1) * 255, 0, 255))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--er", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-z-start", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    mask = tifffile.imread(args.mask)
    output, audits = denoise_stack(mask)
    with tifffile.TiffFile(args.er) as er_file:
        source_depth, height, width = er_file.series[0].shape
        if output.shape[1:] != (height, width) or args.source_z_start + len(output) > source_depth:
            raise ValueError("Mask size/Z range does not match ER source")
        args.out.mkdir(parents=True)
        output_path = args.out / "ER_mask_ridge_denoised.tif"
        tifffile.imwrite(output_path, output, imagej=True)
        tiles = []
        font = ImageFont.load_default()
        for local_z, result in enumerate(output):
            source_z = args.source_z_start + local_z
            gray = _display(er_file.pages[source_z].asarray())
            rgb = np.stack([gray, gray, gray], axis=-1)
            selected = result > 0
            rgb[selected] = np.uint8(rgb[selected].astype(np.float32) * 0.4 + np.array([255, 200, 0]) * 0.6)
            overlay = Image.fromarray(rgb)
            overlay.save(args.out / f"overlay_z{source_z + 1:03d}.png")
            tile = overlay.resize((420, 420))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 75, 20), fill="black")
            draw.text((5, 4), f"z{source_z + 1:03d}", fill="white", font=font)
            tiles.append(tile)
            audits[local_z]["source_z_zero_based"] = source_z
        cols = 5
        sheet = Image.new("RGB", (cols * 420, ((len(tiles) + cols - 1) // cols) * 420), (18, 18, 18))
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % cols) * 420, (index // cols) * 420))
        sheet.save(args.out / "contact_sheet.png")
    manifest = {
        "status": "exploratory; no ground truth available",
        "source_mask": str(args.mask.resolve()),
        "source_er": str(args.er.resolve()),
        "output_shape": list(output.shape),
        "background": 0,
        "foreground": 255,
        "method": "local candidate density sigma25 >0.05; small 2D components <25px require >=2 pixels near adjacent Z",
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "slices": audits,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(output)} slices: {output_path}")
    print(f"Removed {sum(item['removed_pixels'] for item in audits)} of {int((mask > 0).sum())} foreground pixels")


if __name__ == "__main__":
    main()
