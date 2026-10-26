"""Export a verified v4-base/v7-addition 3D TIFF and compact review montage.

The instance TIFF retains its numerical IDs. The PNG is display-only; coloured
outlines and yellow additions are not quantitative labels. This never writes
inside the source cell directory or modifies historical predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_gray(raw: np.ndarray) -> np.ndarray:
    low, high = np.percentile(raw[::4, ::4], (1, 99.5))
    return (np.clip((raw.astype(np.float32) - low) / max(high - low, 1), 0, 1) * 255).astype(np.uint8)


def boundaries(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edge & (labels > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cell", "v4", "v7", "fusion", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    for path in (args.cell / "Memb.tif", args.cell / "fluo.tif", args.v4, args.v7, args.fusion):
        if not path.is_file():
            raise FileNotFoundError(path)
    with tifffile.TiffFile(args.cell / "Memb.tif") as memb, \
            tifffile.TiffFile(args.cell / "fluo.tif") as fluo, \
            tifffile.TiffFile(args.v4) as base, \
            tifffile.TiffFile(args.v7) as candidate, \
            tifffile.TiffFile(args.fusion) as fusion:
        shape = memb.series[0].shape
        if any(tif.series[0].shape != shape for tif in (fluo, base, candidate, fusion)):
            raise ValueError("Input and all predictions must have identical ZYX geometry")
        if fusion.series[0].dtype != np.dtype("uint16"):
            raise ValueError("Expected numerical uint16 instance TIFF")
        z_count = shape[0]
        for z in range(z_count):
            b = base.pages[z].asarray()
            f = fusion.pages[z].asarray()
            if not np.array_equal(b[b > 0], f[b > 0]):
                raise AssertionError(f"v4 label changed at z={z}")
        zs = sorted(set(round(f * (z_count - 1)) for f in (.2, .5, .8)))
        width, height = 400, 400
        canvas = Image.new("RGB", (width * 4, height * len(zs)), "black")
        draw = ImageDraw.Draw(canvas)
        for row, z in enumerate(zs):
            raw_m = memb.pages[z].asarray()
            raw_f = fluo.pages[z].asarray()
            b = base.pages[z].asarray()
            v = candidate.pages[z].asarray()
            f = fusion.pages[z].asarray()
            gray = display_gray(raw_m)
            panels = [np.repeat(gray[..., None], 3, axis=-1) for _ in range(4)]
            panels[0] = np.repeat(display_gray(raw_f)[..., None], 3, axis=-1)
            panels[1][boundaries(b)] = (0, 220, 255)
            panels[2][boundaries(v)] = (50, 255, 100)
            panels[3][boundaries(f)] = (0, 220, 255)
            panels[3][(f > 0) & (b == 0)] = (255, 205, 0)
            for col, panel in enumerate(panels):
                im = Image.fromarray(panel).resize((width, height), Image.Resampling.BILINEAR)
                canvas.paste(im, (col * width, row * height))
                draw.text((col * width + 8, row * height + 8),
                          f"z={z}  {('fluo', 'v4', 'v7', 'fusion')[col]}", fill="white",
                          stroke_width=2, stroke_fill="black")
    args.out.mkdir(parents=True)
    target = args.out / "memb_instances_v4base_v7add_safe.tif"
    shutil.copy2(args.fusion, target)
    if sha256(target) != sha256(args.fusion):
        raise RuntimeError("Exported TIFF hash mismatch")
    canvas.save(args.out / "review.png", optimize=True)
    record = {"cell": args.cell.name, "shape_zyx": shape, "review_z": zs,
              "method": "frozen v4 base, conservative v7 additions, local11, global 3D stitch",
              "instance_tiff_sha256": sha256(target), "v4_sha256": sha256(args.v4),
              "v7_sha256": sha256(args.v7), "human_labels_used_for_selection": False}
    (args.out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
