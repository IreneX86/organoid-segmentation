"""Exploratory ER recall pass: add weak ridges near existing ER structure.

The input strong mask is preserved pixel-for-pixel. This script never writes
to the source TIFFs or the earlier predictions. No GT labels are available,
so this preview is not a measured accuracy improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label
from skimage.filters import sato
from skimage.measure import regionprops
from skimage.morphology import binary_closing, disk, remove_small_objects


def _candidate(
    er: np.ndarray,
    fluo: np.ndarray,
    artifact_xy: tuple[int, int],
    *,
    artifact_fail_closed: bool = True,
) -> tuple[np.ndarray, dict]:
    er = np.asarray(er, dtype=np.float32)
    fluo = np.asarray(fluo, dtype=np.float32)
    background = gaussian_filter(er, 25)
    high_pass = gaussian_filter(er, 1) - background
    relative = np.clip((er - background) / np.maximum(background, 250), 0, 2)
    ridge = sato(relative, sigmas=(1, 2, 4), black_ridges=False)
    fluo_detail = gaussian_filter(fluo, 3) - gaussian_filter(fluo, 60)
    support = remove_small_objects(
        binary_dilation(binary_closing(fluo_detail > 30, disk(10)), disk(25)),
        min_size=5000,
    )

    # Locate the nuisance object from the reliable, stricter response so that
    # lowering the weak-signal threshold cannot merge it into adjacent noise.
    reliable = remove_small_objects((ridge > 0.035) & (high_pass > 30) & support, min_size=8)
    grouped, _ = label(binary_closing(binary_dilation(reliable, disk(8)), disk(4)))
    groups = [region for region in regionprops(grouped) if region.area >= 10000]
    x, y = artifact_xy
    if not groups:
        raise RuntimeError("Cannot locate known artifact group")
    artifact = min(groups, key=lambda region: (region.centroid[0] - y) ** 2 + (region.centroid[1] - x) ** 2)
    distance = float(np.hypot(artifact.centroid[0] - y, artifact.centroid[1] - x))
    if distance > 50:
        if artifact_fail_closed:
            raise RuntimeError(f"Known artifact not near seed ({distance:.1f}px)")
        exclusion = np.zeros(er.shape, dtype=bool)
        artifact_status = "unresolved_keep_candidates"
    else:
        exclusion = binary_dilation(grouped == artifact.label, structure=np.ones((13, 13), dtype=bool))
        artifact_status = "matched"

    weak = remove_small_objects((ridge > 0.025) & (high_pass > 20) & support, min_size=8)
    weak[exclusion] = False
    return weak, {
        "artifact_bbox_yxyx": list(map(int, artifact.bbox)),
        "artifact_centroid_xy": [float(artifact.centroid[1]), float(artifact.centroid[0])],
        "artifact_seed_distance_px": distance,
        "artifact_status": artifact_status,
    }


def _render(er: np.ndarray, strong: np.ndarray, result: np.ndarray, z: int, out: Path) -> Image.Image:
    low, high = np.percentile(er, [1, 99.8])
    gray = np.uint8(np.clip((er - low) / max(high - low, 1) * 255, 0, 255))
    rgb = np.stack([gray] * 3, axis=-1)
    rgb[result] = np.uint8(0.4 * rgb[result].astype(np.float32) + 0.6 * np.array([255, 200, 0]))
    Image.fromarray(rgb).save(out / f"overlay_z{z + 1:03d}.png")
    added = result & ~strong
    comparison = rgb.copy()
    comparison[added] = np.uint8(0.3 * comparison[added].astype(np.float32) + 0.7 * np.array([0, 255, 255]))
    Image.fromarray(comparison).save(out / f"added_cyan_z{z + 1:03d}.png")
    tile = Image.fromarray(rgb).resize((420, 420))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 75, 20), fill="black")
    draw.text((5, 4), f"z{z + 1:03d}", fill="white", font=ImageFont.load_default())
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--strong-mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-z-start", type=int, default=0)
    parser.add_argument("--artifact-seed-xy", type=int, nargs=2, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    strong_uint8 = tifffile.imread(args.strong_mask)
    if strong_uint8.ndim != 3 or strong_uint8.dtype != np.uint8 or not set(np.unique(strong_uint8)).issubset({0, 255}):
        raise ValueError("Strong mask must be a 3D uint8 0/255 TIFF")
    strong = strong_uint8 > 0
    er_path = args.cell / "ER.tif"
    fluo_path = args.cell / "fluo.tif"
    weak_stack = np.zeros_like(strong)
    audit = []
    with tifffile.TiffFile(er_path) as er_file, tifffile.TiffFile(fluo_path) as fluo_file:
        if er_file.series[0].shape != fluo_file.series[0].shape:
            raise ValueError("ER/Fluo source stacks differ")
        if strong.shape[1:] != er_file.series[0].shape[1:] or args.source_z_start + len(strong) > er_file.series[0].shape[0]:
            raise ValueError("Strong mask does not align with source Z/Y/X")
        for local_z in range(len(strong)):
            source_z = args.source_z_start + local_z
            weak, details = _candidate(
                er_file.pages[source_z].asarray(),
                fluo_file.pages[source_z].asarray(),
                tuple(args.artifact_seed_xy),
            )
            # Keep a weak candidate only near an already detected ER structure.
            weak_stack[local_z] = weak & (distance_transform_edt(~strong[local_z]) <= 40) & ~strong[local_z]
            audit.append({"source_z_zero_based": source_z, **details, "nearby_weak_pixels": int(weak_stack[local_z].sum())})

        additions = np.zeros_like(strong)
        for z in range(len(strong)):
            nearby_z = np.zeros(strong.shape[1:], dtype=bool)
            for neighbor in (z - 1, z + 1):
                if 0 <= neighbor < len(strong):
                    nearby_z |= binary_dilation(weak_stack[neighbor], structure=np.ones((5, 5), dtype=bool))
            components, count = label(weak_stack[z])
            areas = np.bincount(components.ravel(), minlength=count + 1)
            overlap = np.bincount(components.ravel(), weights=nearby_z.ravel(), minlength=count + 1)
            keep = (areas >= 25) | (overlap >= 2)
            keep[0] = False
            additions[z] = keep[components]
        result = strong | additions
        if not np.all(result >= strong):
            raise AssertionError("The recall pass must retain the earlier mask")

        args.out.mkdir(parents=True)
        mask_path = args.out / "ER_mask_ridge_relaxed_context.tif"
        tifffile.imwrite(mask_path, result.astype(np.uint8) * 255, imagej=True)
        tiles = []
        for local_z in range(len(strong)):
            source_z = args.source_z_start + local_z
            tiles.append(_render(er_file.pages[source_z].asarray(), strong[local_z], result[local_z], source_z, args.out))
            audit[local_z]["added_pixels"] = int(additions[local_z].sum())
            audit[local_z]["final_pixels"] = int(result[local_z].sum())
            print(f"z{source_z + 1:03d}: +{audit[local_z]['added_pixels']} pixels", flush=True)
        cols = 5
        sheet = Image.new("RGB", (cols * 420, ((len(tiles) + cols - 1) // cols) * 420), (18, 18, 18))
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % cols) * 420, (index // cols) * 420))
        sheet.save(args.out / "contact_sheet.png")

    manifest = {
        "status": "exploratory; no ground-truth validation",
        "source_er": str(er_path.resolve()),
        "source_fluo": str(fluo_path.resolve()),
        "strong_mask": str(args.strong_mask.resolve()),
        "method": "Sato ridge >0.025 and high-pass >20; within 40px of strong mask; new small objects need adjacent-Z support",
        "artifact_seed_xy": args.artifact_seed_xy,
        "artifact_seed_scope": "This field of view only",
        "output_shape": list(result.shape),
        "background": 0,
        "foreground": 255,
        "output_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "slices": audit,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
