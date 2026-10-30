"""Exploratory ER segmentation from paired ER/Fluo TIFF stacks.

This is deterministic image processing, not a trained U-Net. Its defaults
reproduce the approved single-slice pilot's *automatic* rule. A reviewed ROI
can be supplied separately when reproducing a manually reviewed slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_fill_holes, gaussian_filter, label
from skimage.feature import blob_log
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_objects


@dataclass(frozen=True)
class ERParameters:
    fluo_small_sigma: float = 5
    fluo_large_sigma: float = 80
    fluo_structure_threshold: float = 100
    support_close_radius: int = 18
    support_min_area: int = 3500
    support_dilate_radius: int = 25
    er_small_sigma: float = 1
    er_background_sigma: float = 30
    er_background_floor: float = 200
    er_residual_threshold: float = 100
    er_relative_threshold: float = 0.25
    er_min_area: int = 8
    blob_min_sigma: float = 5
    blob_max_sigma: float = 24
    blob_num_sigma: int = 10
    blob_threshold: float = 0.16
    blob_overlap: float = 0.2
    blob_detected_sigma_min: float = 7
    blob_area_min: int = 200
    blob_solidity_min: float = 0.75
    blob_fluo_relative_max: float = 0.12
    fluo_background_sigma: float = 30


def segment_er_slice(
    er_image: np.ndarray,
    fluo_image: np.ndarray,
    params: ERParameters = ERParameters(),
    review_ellipses: tuple[dict, ...] = (),
) -> tuple[np.ndarray, dict]:
    """Return a same-size uint8 0/255 mask and per-slice audit information.

    Reviewed ellipses select component centroids. They are deliberately
    optional because a location marked on one image cannot generalize to a
    different Z slice or organoid.
    """
    if er_image.ndim != 2 or fluo_image.shape != er_image.shape:
        raise ValueError("ER and Fluo must be aligned 2D images")
    er = np.asarray(er_image, dtype=np.float32)
    fluo = np.asarray(fluo_image, dtype=np.float32)

    fluo_structure = gaussian_filter(fluo, params.fluo_small_sigma) - gaussian_filter(
        fluo, params.fluo_large_sigma
    )
    support = binary_dilation(
        remove_small_objects(
            binary_fill_holes(
                binary_closing(
                    fluo_structure > params.fluo_structure_threshold,
                    disk(params.support_close_radius),
                )
            ),
            min_size=params.support_min_area,
        ),
        disk(params.support_dilate_radius),
    )
    er_background = gaussian_filter(er, params.er_background_sigma)
    er_residual = gaussian_filter(er, params.er_small_sigma) - er_background
    er_relative = er_residual / np.maximum(er_background, params.er_background_floor)
    initial = remove_small_objects(
        (er_residual > params.er_residual_threshold)
        & (er_relative > params.er_relative_threshold)
        & support,
        min_size=params.er_min_area,
    )

    components, _ = label(initial)
    regions = {region.label: region for region in regionprops(components)}
    fluo_background = gaussian_filter(fluo, params.fluo_background_sigma)
    blobs = blob_log(
        np.clip(er_relative, 0, 4),
        min_sigma=params.blob_min_sigma,
        max_sigma=params.blob_max_sigma,
        num_sigma=params.blob_num_sigma,
        threshold=params.blob_threshold,
        overlap=params.blob_overlap,
    )
    automatic_ids: set[int] = set()
    for y, x, sigma in blobs:
        if sigma < params.blob_detected_sigma_min:
            continue
        component_id = int(components[round(y), round(x)])
        if component_id < 1 or component_id not in regions:
            continue
        region = regions[component_id]
        if region.area < params.blob_area_min or region.solidity < params.blob_solidity_min:
            continue
        pixels = components == component_id
        fluo_relative = float(
            np.mean((fluo[pixels] - fluo_background[pixels]) / np.maximum(fluo_background[pixels], 1))
        )
        if fluo_relative <= params.blob_fluo_relative_max:
            automatic_ids.add(component_id)

    reviewed_ids: set[int] = set()
    for ellipse in review_ellipses:
        center_x, center_y = ellipse["center_xy"]
        radius_x, radius_y = ellipse["radius_xy"]
        if radius_x <= 0 or radius_y <= 0:
            raise ValueError("Review ellipse radii must be positive")
        minimum_area = int(ellipse.get("component_area_min", 60))
        minimum_solidity = float(ellipse.get("solidity_min", 0.75))
        for region in regions.values():
            y, x = region.centroid
            if (
                ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 < 1
                and region.area >= minimum_area
                and region.solidity >= minimum_solidity
            ):
                reviewed_ids.add(region.label)

    removed_ids = automatic_ids | reviewed_ids
    foreground = initial & ~np.isin(components, list(removed_ids))
    mask = np.where(foreground, 255, 0).astype(np.uint8)
    audit = {
        "initial_pixels": int(initial.sum()),
        "final_pixels": int(foreground.sum()),
        "automatic_removed_component_ids": sorted(automatic_ids),
        "review_removed_component_ids": sorted(reviewed_ids - automatic_ids),
    }
    return mask, audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover_pairs(input_root: Path) -> list[tuple[Path, Path, Path]]:
    if (input_root / "ER.tif").is_file():
        er_files = [input_root / "ER.tif"]
    else:
        er_files = sorted(input_root.rglob("ER.tif"))
    if not er_files:
        raise FileNotFoundError(f"No ER.tif files under {input_root}")
    pairs = []
    for er_path in er_files:
        fluo_path = er_path.with_name("fluo.tif")
        if not fluo_path.is_file():
            raise FileNotFoundError(f"Missing paired fluo.tif: {er_path.parent}")
        pairs.append((er_path, fluo_path, er_path.parent.relative_to(input_root)))
    return pairs


def run_batch(
    input_root: Path,
    output_root: Path,
    *,
    z_start: int = 0,
    z_stop: int | None = None,
    params: ERParameters = ERParameters(),
    review_rois: dict | None = None,
) -> list[Path]:
    """Process aligned stacks slice by slice into a new mirrored output tree.

    Output root is exclusive. Temporary TIFFs are renamed only after all
    slices finish, so interrupted stacks cannot be mistaken for final masks.
    """
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    pairs = _discover_pairs(input_root)
    if z_start < 0 or (z_stop is not None and z_stop <= z_start):
        raise ValueError("Require 0 <= z_start < z_stop")
    review_rois = review_rois or {}
    output_root.mkdir(parents=True)
    completed = []
    for er_path, fluo_path, relative in pairs:
        with tifffile.TiffFile(er_path) as er_file, tifffile.TiffFile(fluo_path) as fluo_file:
            er_shape = er_file.series[0].shape
            fluo_shape = fluo_file.series[0].shape
            if len(er_shape) != 3 or er_shape != fluo_shape:
                raise ValueError(f"Stack shape mismatch: {er_path}: {er_shape} vs {fluo_shape}")
            depth, height, width = er_shape
            stop = depth if z_stop is None else z_stop
            if stop > depth:
                raise ValueError(f"Requested z_stop={stop} exceeds depth={depth}: {er_path}")
            if len(er_file.pages) != depth or len(fluo_file.pages) != depth:
                raise ValueError(f"Expected one TIFF page per Z slice: {er_path}")

            destination = output_root / relative
            destination.mkdir(parents=True, exist_ok=True)
            final_path = destination / "ER_mask.tif"
            partial_path = destination / "ER_mask.partial.tif"
            if final_path.exists() or partial_path.exists():
                raise FileExistsError(f"Mask already exists: {destination}")
            output = tifffile.memmap(
                partial_path, shape=(stop - z_start, height, width), dtype=np.uint8, imagej=True
            )
            audits = []
            for output_index, z_index in enumerate(range(z_start, stop)):
                ellipses = tuple(review_rois.get(relative.as_posix(), {}).get(str(z_index), ()))
                mask, audit = segment_er_slice(
                    er_file.pages[z_index].asarray(),
                    fluo_file.pages[z_index].asarray(),
                    params,
                    review_ellipses=ellipses,
                )
                output[output_index] = mask
                audits.append({"source_z_index_zero_based": z_index, **audit})
            output.flush()
            del output
            partial_path.rename(final_path)
            manifest = {
                "method": "exploratory ER/Fluo classical image processing; not U-Net",
                "source_er": str(er_path),
                "source_fluo": str(fluo_path),
                "source_shape": list(er_shape),
                "output_shape": [stop - z_start, height, width],
                "source_z_start_zero_based": z_start,
                "source_z_stop_exclusive": stop,
                "dtype": "uint8",
                "background": 0,
                "foreground": 255,
                "parameters": asdict(params),
                "review_rois_applied": any(bool(review_rois.get(relative.as_posix(), {}).get(str(z))) for z in range(z_start, stop)),
                "output_sha256": _sha256(final_path),
                "slices": audits,
            }
            (destination / "ER_mask_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            completed.append(final_path)
            print(f"{relative}: {stop - z_start} slices -> {final_path}", flush=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="ER root or one cell folder")
    parser.add_argument("--output-root", type=Path, required=True, help="Must be a new directory")
    parser.add_argument("--z-start", type=int, default=0, help="Zero-based inclusive slice index")
    parser.add_argument("--z-stop", type=int, help="Zero-based exclusive slice index")
    parser.add_argument("--review-rois", type=Path, help="Optional JSON for specifically reviewed slices")
    args = parser.parse_args()
    rois = json.loads(args.review_rois.read_text(encoding="utf-8")) if args.review_rois else None
    run_batch(args.input_root, args.output_root, z_start=args.z_start, z_stop=args.z_stop, review_rois=rois)


if __name__ == "__main__":
    main()
