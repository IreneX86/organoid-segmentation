"""Independent ER ridge/Fluo preview for a short, reviewed Z range.

This exploratory script does not import the earlier ER threshold pipeline.
The optional artifact seed identifies one known nuisance cluster in one
field of view; it is not a trained or transferable artifact classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from skimage.filters import sato
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_objects


def segment_slice(
    er: np.ndarray,
    fluo: np.ndarray,
    artifact_xy: tuple[int, int] | None,
    *,
    artifact_fail_closed: bool = True,
) -> tuple[np.ndarray, dict]:
    if er.shape != fluo.shape or er.ndim != 2:
        raise ValueError("Aligned 2D ER and Fluo slices are required")
    er = er.astype(np.float32)
    fluo = fluo.astype(np.float32)

    # Remove broad illumination variation, then retain thin bright ridges at
    # several spatial scales. This differs from the previous two-cutoff mask.
    background = gaussian_filter(er, 25)
    high_pass = gaussian_filter(er, 1) - background
    normalized = np.clip((er - background) / np.maximum(background, 250), 0, 2)
    ridge = sato(normalized, sigmas=(1, 2, 4), black_ridges=False)
    candidate = remove_small_objects((ridge > 0.035) & (high_pass > 30), min_size=8)

    # Fluo is an auxiliary spatial support, not a pixel-wise ER label.
    fluo_detail = gaussian_filter(fluo, 3) - gaussian_filter(fluo, 60)
    support = remove_small_objects(
        binary_dilation(binary_closing(fluo_detail > 30, disk(10)), disk(25)),
        min_size=5000,
    )
    foreground = candidate & support
    before_artifact = int(foreground.sum())
    artifact_bbox = None
    artifact_centroid_xy = None
    artifact_status = "not_requested" if artifact_xy is None else "matched"
    artifact_pixels = 0
    if artifact_xy is not None:
        # A user-identified nuisance cluster is grouped from the *new* ridge
        # candidate and selected by proximity to the supplied seed. Abort if
        # no convincing matching cluster exists instead of deleting another.
        grouped = label(binary_closing(binary_dilation(foreground, disk(8)), disk(4)))
        regions = [region for region in regionprops(grouped) if region.area >= 10000]
        seed_x, seed_y = artifact_xy
        if not regions:
            raise RuntimeError("No component group available for artifact seed")
        selected = min(regions, key=lambda region: (region.centroid[0] - seed_y) ** 2 + (region.centroid[1] - seed_x) ** 2)
        distance = float(np.hypot(selected.centroid[0] - seed_y, selected.centroid[1] - seed_x))
        if distance > 50:
            if artifact_fail_closed:
                raise RuntimeError(f"Artifact seed did not match a nearby component group: {distance:.1f}px")
            artifact_status = "unresolved_keep_candidates"
        else:
            artifact_pixels = int((foreground & (grouped == selected.label)).sum())
            foreground[grouped == selected.label] = False
            artifact_bbox = list(map(int, selected.bbox))
            artifact_centroid_xy = [float(selected.centroid[1]), float(selected.centroid[0])]

    return np.where(foreground, 255, 0).astype(np.uint8), {
        "candidate_pixels": int(candidate.sum()),
        "after_fluo_support_pixels": before_artifact,
        "artifact_removed_pixels": artifact_pixels,
        "artifact_group_bbox_yxyx": artifact_bbox,
        "artifact_group_centroid_xy": artifact_centroid_xy,
        "artifact_status": artifact_status,
        "final_pixels": int(foreground.sum()),
    }


def display_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99.8])
    return np.uint8(np.clip((image - lo) / max(hi - lo, 1) * 255, 0, 255))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--z-start", type=int, default=0)
    parser.add_argument("--z-stop", type=int, required=True)
    parser.add_argument("--artifact-seed-xy", type=int, nargs=2)
    args = parser.parse_args()
    cell = args.cell.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    er_path, fluo_path = cell / "ER.tif", cell / "fluo.tif"
    masks = []
    audits = []
    thumbs = []
    font = ImageFont.load_default()
    with tifffile.TiffFile(er_path) as er_file, tifffile.TiffFile(fluo_path) as fluo_file:
        if er_file.series[0].shape != fluo_file.series[0].shape:
            raise ValueError("ER and Fluo stack shapes differ")
        depth, height, width = er_file.series[0].shape
        if not (0 <= args.z_start < args.z_stop <= depth):
            raise ValueError("Requested Z range is outside the source stack")
        for z in range(args.z_start, args.z_stop):
            er = er_file.pages[z].asarray()
            fluo = fluo_file.pages[z].asarray()
            mask, audit = segment_slice(er, fluo, tuple(args.artifact_seed_xy) if args.artifact_seed_xy else None)
            masks.append(mask)
            audits.append({"source_z_zero_based": z, **audit})
            er_gray = display_uint8(er)
            fluo_gray = display_uint8(fluo)
            overlay = np.stack([er_gray] * 3, axis=-1)
            selected = mask > 0
            overlay[selected] = np.uint8(0.4 * overlay[selected] + 0.6 * np.array([255, 200, 0]))
            page = Image.new("RGB", (1500, 535), (18, 18, 18))
            draw = ImageDraw.Draw(page)
            for column, (array, title) in enumerate(((er_gray, "ER"), (fluo_gray, "Fluo"), (overlay, "new mask overlay"))):
                page.paste(Image.fromarray(array).convert("RGB").resize((500, 500)), (column * 500, 35))
                draw.text((column * 500 + 10, 8), f"z{z + 1:03d} {title}", fill="white", font=font)
            page.save(out / f"comparison_z{z + 1:03d}.png")
            thumb = Image.fromarray(overlay).resize((420, 420))
            td = ImageDraw.Draw(thumb)
            td.rectangle((0, 0, 70, 20), fill="black")
            td.text((5, 4), f"z{z + 1:03d}", fill="white", font=font)
            thumbs.append(thumb)
            print(f"z{z + 1:03d}: {audit['final_pixels']} pixels; artifact removed {audit['artifact_removed_pixels']}", flush=True)
    mask_path = out / "ER_mask_ridge_trial.tif"
    tifffile.imwrite(mask_path, np.stack(masks), imagej=True)
    cols = 5
    sheet = Image.new("RGB", (cols * 420, ((len(thumbs) + cols - 1) // cols) * 420), (18, 18, 18))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 420, (index // cols) * 420))
    sheet.save(out / "contact_sheet.png")
    manifest = {
        "status": "exploratory preview; no ground-truth validation",
        "method": "ER multiscale Sato bright-ridge response + local high-pass + broad Fluo support",
        "artifact_seed_xy": args.artifact_seed_xy,
        "artifact_seed_scope": "This selected field of view and Z range only; do not generalize",
        "source_er": str(er_path),
        "source_fluo": str(fluo_path),
        "output_shape": [len(masks), height, width],
        "mask_dtype": "uint8",
        "background": 0,
        "foreground": 255,
        "output_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "slices": audits,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
