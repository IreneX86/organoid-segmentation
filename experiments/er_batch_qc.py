"""Render and validate representative Z slices from an ER candidate stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--er", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--z-indices", type=int, nargs="+", metavar="Z",
        help="Optional 1-based Z slices to render instead of five representative slices",
    )
    args = parser.parse_args()
    mask = tifffile.imread(args.mask)
    if mask.ndim != 3 or mask.dtype != np.uint8 or not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError("Expected 3D uint8 0/255 mask")
    with tifffile.TiffFile(args.er) as er_file:
        if er_file.series[0].shape != mask.shape:
            raise ValueError(f"Source/mask mismatch: {er_file.series[0].shape} vs {mask.shape}")
        depth = len(mask)
        if args.z_indices:
            if any(z < 1 or z > depth for z in args.z_indices):
                raise ValueError(f"Requested Z slice outside 1..{depth}")
            indices = sorted({z - 1 for z in args.z_indices})
        else:
            indices = sorted({0, depth // 4, depth // 2, 3 * depth // 4, depth - 1})
        sheet = Image.new("RGB", (420 * len(indices), 420), (18, 18, 18))
        for col, z in enumerate(indices):
            image = er_file.pages[z].asarray().astype(np.float32)
            low, high = np.percentile(image, [1, 99.8])
            gray = np.uint8(np.clip((image - low) / max(high - low, 1) * 255, 0, 255))
            rgb = np.stack([gray] * 3, axis=-1)
            selected = mask[z] > 0
            rgb[selected] = np.uint8(
                0.4 * rgb[selected].astype(np.float32) + 0.6 * np.array([255, 200, 0])
            )
            tile = Image.fromarray(rgb).resize((420, 420))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 75, 20), fill="black")
            draw.text((5, 4), f"z{z + 1:03d}", fill="white", font=ImageFont.load_default())
            sheet.paste(tile, (col * 420, 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    stats = {
        "mask_shape": list(mask.shape),
        "foreground_pixels_by_z": np.count_nonzero(mask, axis=(1, 2)).astype(int).tolist(),
        "representative_z_zero_based": indices,
        "mask_is_binary_0_255": True,
    }
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
