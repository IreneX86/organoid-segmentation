"""Batch ER/Fluo candidate masks with optional review and gentle line refinement.

This is classical image processing. It is not a trained U-Net or a validated
ground-truth label generator. Existing TIFFs, predictions and checkpoints are
never modified. See docs/er_bright_line_pipeline.md before batch use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, find_objects, gaussian_filter, label
from skimage.filters import sato
from skimage.measure import regionprops
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize

from er_ridge_batch_predict import pairs, predict_stack


@dataclass(frozen=True)
class ERBrightLineSettings:
    """Frozen defaults from the exploratory eight-slice preview.

    These numbers are pixel-scale dependent; evaluate anew for other imaging
    resolutions or microscope settings before scientific quantification.
    """

    er_background_sigma: float = 25.0
    # The defaults preserve the original spheroid candidate exactly. A dim ER
    # imaging domain can use smaller floors in an explicit, separate profile.
    relative_background_floor: float = 250.0
    weak_ridge_floor: float = 0.012
    strong_ridge_floor: float = 0.025
    foreground_percentile: float = 80.0
    bright_percentile: float = 90.0
    highpass_fraction: float = 0.06
    minimum_highpass: float = 12.0
    strong_highpass_floor: float = 25.0
    group_min_dilated_area: int = 300
    group_min_raw_area: int = 65
    fleck_min_area: int = 45
    fleck_small_area: int = 160
    line_half_width: int = 4


def settings_for_profile(name: str) -> ERBrightLineSettings:
    """Select an explicit imaging-domain profile; never infer biology from file names.

    The dim-organoid values are exploratory. They relax only intensity floors,
    leaving fragment filtering and line refinement unchanged. They must not
    silently replace the previously reviewed spheroid/default outputs.
    """
    default = ERBrightLineSettings()
    if name == "spheroid_default":
        return default
    if name == "dim_organoid_candidate":
        return replace(default, relative_background_floor=150.0,
                       minimum_highpass=6.0, strong_highpass_floor=12.0)
    raise ValueError(f"Unknown ER imaging profile: {name}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ellipse_region(shape: tuple[int, int], items: list[dict]) -> np.ndarray:
    """Create a coarse review region, not a biological object contour."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    result = np.zeros(shape, dtype=bool)
    for item in items:
        x, y = item["center_xy"]
        rx, ry = item["radius_xy"]
        if rx <= 0 or ry <= 0:
            raise ValueError("Review ellipse radii must be positive")
        result |= ((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2 <= 1
    return result


def grouped_filter(
    raw: np.ndarray,
    base: np.ndarray,
    anchor: np.ndarray,
    settings: ERBrightLineSettings,
) -> np.ndarray:
    """Retain coherent, supported ER candidates and reject compact bright spots."""
    raw = remove_small_objects(raw, min_size=40)
    groups, n = label(binary_dilation(raw, structure=disk(4)))
    group_area = np.bincount(groups.ravel(), minlength=n + 1)
    raw_area = np.bincount(groups.ravel(), weights=raw.ravel(), minlength=n + 1)
    base_area = np.bincount(groups.ravel(), weights=base.ravel(), minlength=n + 1)
    anchor_area = np.bincount(groups.ravel(), weights=anchor.ravel(), minlength=n + 1)
    keep = (
        (group_area >= settings.group_min_dilated_area)
        & (raw_area >= settings.group_min_raw_area)
        & ((base_area >= 15) | (anchor_area >= 30))
    )
    keep[0] = False
    for region in regionprops(groups):
        if keep[region.label] and region.area < 2500 and region.solidity > 0.88 and region.eccentricity < 0.72:
            keep[region.label] = False
    return raw & keep[groups]


def bright_candidate(
    er_image: np.ndarray,
    fluo_image: np.ndarray,
    base_mask: np.ndarray,
    settings: ERBrightLineSettings,
    artifact_bbox_yxyx: list[int] | None = None,
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """Add high-confidence ridges/bright ER within broad Fluo tissue support."""
    er = np.asarray(er_image, dtype=np.float32)
    fluo = np.asarray(fluo_image, dtype=np.float32)
    base = np.asarray(base_mask) > 0
    if er.ndim != 2 or er.shape != fluo.shape or er.shape != base.shape:
        raise ValueError("ER, Fluo and base mask must be aligned 2D arrays")
    background = gaussian_filter(er, settings.er_background_sigma)
    highpass = gaussian_filter(er, 1) - background
    relative = np.clip(
        (er - background) / np.maximum(background, settings.relative_background_floor),
        0, 2,
    )
    ridge = sato(relative, sigmas=(1, 2, 4), black_ridges=False)
    p50, p80, p90, p99 = np.percentile(er, [50, settings.foreground_percentile, settings.bright_percentile, 99])
    minimum_contrast = max(settings.minimum_highpass, settings.highpass_fraction * (p99 - p50))
    fluo_detail = gaussian_filter(fluo, 3) - gaussian_filter(fluo, 60)
    support = remove_small_objects(
        binary_dilation(binary_closing(fluo_detail > 30, disk(10)), structure=disk(25)),
        min_size=5000,
    )
    anchor = (ridge > settings.strong_ridge_floor) & (
        highpass > max(settings.strong_highpass_floor, minimum_contrast)
    )
    weak = (ridge > settings.weak_ridge_floor) & (highpass > minimum_contrast) & (er > p80) & support
    broad = (er > p90) & (highpass > minimum_contrast) & support
    if artifact_bbox_yxyx is not None:
        y0, x0, y1, x1 = artifact_bbox_yxyx
        if not (0 <= y0 <= y1 <= er.shape[0] and 0 <= x0 <= x1 <= er.shape[1]):
            raise ValueError("Artifact bounding box is outside image")
        weak[y0:y1, x0:x1] = False
        broad[y0:y1, x0:x1] = False
    result = grouped_filter(base | weak | broad, base, anchor, settings)
    audit = {
        "ER_p80": float(p80), "ER_p90": float(p90),
        "minimum_contrast": float(minimum_contrast),
        "base_pixels": int(base.sum()), "bright_candidate_pixels": int(result.sum()),
        "artifact_bbox_applied": artifact_bbox_yxyx,
    }
    return result, audit, highpass, ridge


def reviewed_rescue(
    er_image: np.ndarray,
    current: np.ndarray,
    highpass: np.ndarray,
    ridge: np.ndarray,
    ellipses: list[dict],
) -> np.ndarray:
    """Find ER-like pixels near reviewed misses; never fill the ellipse itself."""
    if not ellipses:
        return np.zeros_like(current, dtype=bool)
    roi = ellipse_region(current.shape, ellipses)
    p50, p99 = np.percentile(er_image, [50, 99])
    hp_floor = max(2.5, 0.018 * (p99 - p50))
    ridge_floor = 0.006 if p99 < 500 else 0.013
    relaxed = ((ridge > ridge_floor) & (highpass > hp_floor)) | (
        (ridge > ridge_floor * 1.5) & (highpass > hp_floor * 0.7)
    )
    relaxed = remove_small_objects(relaxed, min_size=18)
    groups, n = label(binary_dilation(relaxed, structure=disk(3)))
    group_raw = np.bincount(groups.ravel(), weights=relaxed.ravel(), minlength=n + 1)
    group_roi = np.bincount(groups.ravel(), weights=(relaxed & roi).ravel(), minlength=n + 1)
    keep = (group_raw >= 65) & (group_roi >= 25)
    keep[0] = False
    return relaxed & keep[groups] & binary_dilation(roi, structure=disk(14)) & ~current


def filter_weak_flecks(
    mask: np.ndarray, er_image: np.ndarray, settings: ERBrightLineSettings
) -> tuple[np.ndarray, int]:
    """Remove only isolated tiny/weak pieces; avoid pruning long fine fibers."""
    er = np.asarray(er_image, dtype=np.float32)
    highpass = gaussian_filter(er, 1) - gaussian_filter(er, settings.er_background_sigma)
    p50, p99 = np.percentile(er, [50, 99])
    bright_floor = max(4.0, 0.028 * (p99 - p50))
    groups, n = label(binary_dilation(mask, structure=disk(2)))
    raw_area = np.bincount(groups.ravel(), weights=mask.ravel(), minlength=n + 1)
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    for index, box in enumerate(find_objects(groups), start=1):
        if box is None or raw_area[index] >= settings.fleck_small_area:
            continue
        values = highpass[box][(groups[box] == index) & mask[box]]
        if raw_area[index] < settings.fleck_min_area or values.size == 0 or np.percentile(values, 90) < bright_floor:
            keep[index] = False
    result = mask & keep[groups]
    return result, int((mask & ~result).sum())


def refine_lines(mask: np.ndarray, half_width: int) -> tuple[np.ndarray, int]:
    """Narrow existing thick areas, restoring a component if thinning splits it."""
    if half_width < 1:
        raise ValueError("line_half_width must be >= 1")
    centerline = skeletonize(mask)
    candidate = mask & binary_dilation(centerline, structure=disk(half_width))
    structure = np.ones((3, 3), dtype=bool)
    original_ids, original_count = label(mask, structure=structure)
    narrowed_ids, _ = label(candidate, structure=structure)
    restored = 0
    for component, box in enumerate(find_objects(original_ids), start=1):
        if box is None:
            continue
        source_region = original_ids[box] == component
        retained = np.unique(narrowed_ids[box][source_region])
        if np.count_nonzero(retained) > 1:
            candidate[box][source_region] = True
            restored += 1
    if label(candidate, structure=structure)[1] != original_count:
        raise RuntimeError("Line refinement changed the number of connected regions")
    return candidate, restored


def narrow_existing_lines(
    bright: np.ndarray, reference_line: np.ndarray, half_width: int,
) -> tuple[np.ndarray, int]:
    """Thin the radius-4 line without adding pixels or breaking its regions.

    The historical ``refine_lines`` can restore an entire bright component if
    a narrow radius splits it. For a visual thin-line option, restore only the
    *reference line* in that case, never the wider bright candidate.
    """
    if not 1 <= half_width <= 4:
        raise ValueError("half_width must be between 1 and the historical radius 4")
    if bright.shape != reference_line.shape or np.any(reference_line & ~bright):
        raise ValueError("reference_line must be a subset of bright, with matching shape")
    candidate = reference_line & binary_dilation(skeletonize(bright), structure=disk(half_width))
    structure = np.ones((3, 3), dtype=bool)
    reference_ids, reference_count = label(reference_line, structure=structure)
    candidate_ids, _ = label(candidate, structure=structure)
    restored = 0
    for component, box in enumerate(find_objects(reference_ids), start=1):
        if box is None:
            continue
        region = reference_ids[box] == component
        retained = np.unique(candidate_ids[box][region])
        retained = retained[retained != 0]
        if retained.size != 1:
            candidate[box][region] = True
            restored += 1
    if label(candidate, structure=structure)[1] != reference_count:
        raise RuntimeError("Narrowing changed the number of connected regions")
    return candidate, restored


def process_slice(
    er: np.ndarray,
    fluo: np.ndarray,
    base: np.ndarray,
    settings: ERBrightLineSettings = ERBrightLineSettings(),
    review: dict | None = None,
    artifact_bbox_yxyx: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return bright and thinner masks, both uint8 0/255 in source coordinates."""
    review = review or {}
    allowed = {"exclude_ellipses", "rescue_ellipses"}
    if set(review) - allowed:
        raise ValueError(f"Unknown review keys: {sorted(set(review)-allowed)}")
    bright, audit, highpass, ridge = bright_candidate(er, fluo, base, settings, artifact_bbox_yxyx)
    rescue = reviewed_rescue(er, bright, highpass, ridge, review.get("rescue_ellipses", []))
    bright |= rescue
    excluded = ellipse_region(bright.shape, review.get("exclude_ellipses", []))
    review_removed = int((bright & excluded).sum())
    bright &= ~excluded
    bright, fleck_removed = filter_weak_flecks(bright, er, settings)
    line, restored = refine_lines(bright, settings.line_half_width)
    audit.update({
        "review_rescue_pixels": int(rescue.sum()),
        "review_excluded_pixels": review_removed,
        "weak_fleck_removed_pixels": fleck_removed,
        "bright_pixels_after_review": int(bright.sum()),
        "line_pixels": int(line.sum()),
        "components_restored_to_prevent_new_splits": restored,
    })
    return bright.astype(np.uint8)*255, line.astype(np.uint8)*255, audit


def _load_stack(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim == 2:
        return image[None]
    if image.ndim == 3:
        return image
    raise ValueError(f"Expected 2D or ZYX TIFF: {path}, got {image.shape}")


def _review_for_z(
    review_json: dict, relative: Path, source_z: int, direct_cell_name: str | None = None
) -> dict:
    """Resolve review keys for a tree or for a directly supplied cell folder.

    A direct cell folder has relative path ``.``. Keep that historical key
    working, while also accepting the more readable real cell name.
    """
    keys = [relative.as_posix(), relative.name]
    if relative == Path('.') and direct_cell_name is not None:
        keys.append(direct_cell_name)
    for key in keys:
        if key in review_json:
            return review_json[key].get(str(source_z), {})
    return {}


def run_batch(
    input_root: Path,
    output_root: Path,
    *,
    base_mask_root: Path | None = None,
    artifact_seeds: dict | None = None,
    reviews: dict | None = None,
    z_start: int = 0,
    z_stop: int | None = None,
    settings: ERBrightLineSettings = ERBrightLineSettings(),
    profile_map: dict[str, str] | None = None,
    line_half_width_override: int | None = None,
) -> list[Path]:
    """Process paired images into a new, exclusively owned mirrored directory."""
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if output_root.exists():
        raise FileExistsError(f"Output exists; refusing overwrite: {output_root}")
    tasks = pairs(input_root)
    if profile_map is not None:
        # A mixed imaging batch needs an explicit choice for *every* cell. Do
        # not infer the profile from brightness or silently use a fallback.
        if settings != ERBrightLineSettings():
            raise ValueError("Use either uniform settings or profile_map, not both")
        cell_names = [er_path.parent.name for er_path, _, _ in tasks]
        if len(cell_names) != len(set(cell_names)):
            raise ValueError("Duplicate cell folder names make profile_map ambiguous")
        if not isinstance(profile_map, dict) or set(profile_map) != set(cell_names):
            raise ValueError(f"profile_map must name exactly these cells: {sorted(cell_names)}")
        for name in profile_map.values():
            settings_for_profile(name)
    if base_mask_root is not None:
        base_mask_root = base_mask_root.expanduser().resolve()
    artifact_seeds = artifact_seeds or {}
    reviews = reviews or {}
    if z_start < 0 or (z_stop is not None and z_stop <= z_start):
        raise ValueError("Invalid Z range")
    if line_half_width_override is not None and not 1 <= line_half_width_override <= 4:
        raise ValueError("line_half_width_override must be between 1 and 4")
    if (line_half_width_override is not None and profile_map is None
            and line_half_width_override > settings.line_half_width):
        raise ValueError("line_half_width_override cannot widen the configured line")
    # Validate paths/shape before creating any output.
    for er_path, fluo_path, relative in tasks:
        with tifffile.TiffFile(er_path) as er_file, tifffile.TiffFile(fluo_path) as fluo_file:
            er_shape = er_file.series[0].shape
            if er_shape != fluo_file.series[0].shape or len(er_shape) not in (2, 3):
                raise ValueError(f"Misaligned ER/Fluo TIFFs: {er_path.parent}")
            depth = 1 if len(er_shape) == 2 else er_shape[0]
            if not (0 <= z_start < (depth if z_stop is None else z_stop) <= depth):
                raise ValueError(f"Z range outside {er_path}")
        if base_mask_root is not None:
            mask_path = base_mask_root/relative/'ER_mask.tif'
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            with tifffile.TiffFile(mask_path) as mask_file:
                if mask_file.series[0].shape != er_shape:
                    raise ValueError(f"Base mask shape mismatch: {mask_path}")
    output_root.mkdir(parents=True)
    completed = []
    for er_path, fluo_path, relative in tasks:
        with tifffile.TiffFile(er_path) as source_file:
            source_shape = source_file.series[0].shape
        cell_profile = profile_map[er_path.parent.name] if profile_map is not None else None
        cell_settings = settings_for_profile(cell_profile) if cell_profile is not None else settings
        reported_settings = (replace(cell_settings, line_half_width=line_half_width_override)
                             if line_half_width_override is not None else cell_settings)
        if cell_profile is None:
            cell_profile = next(
                (name for name in ('spheroid_default', 'dim_organoid_candidate')
                 if cell_settings == settings_for_profile(name)), 'custom_settings'
            )
        direct_cell_name = input_root.name if relative == Path('.') else None
        source_ndim = len(source_shape)
        depth = 1 if source_ndim == 2 else source_shape[0]
        stop = depth if z_stop is None else z_stop
        base_path = base_mask_root/relative/'ER_mask.tif' if base_mask_root is not None else None
        if base_mask_root is None:
            seed_record = artifact_seeds.get(relative.as_posix(), artifact_seeds.get(relative.name))
            seed = tuple(seed_record["seed_xy"]) if seed_record else None
            if depth == 1:
                # The legacy 3D ridge entry point needs at least a stack.
                from er_ridge_batch_predict import _candidate_no_artifact
                single_er = tifffile.imread(er_path)
                single_fluo = tifffile.imread(fluo_path)
                base_stack = _candidate_no_artifact(single_er.squeeze(), single_fluo.squeeze())[0][None]
                base_manifest = None
            else:
                base_stack, base_manifest = predict_stack(er_path, fluo_path, z_start, stop, seed)
        else:
            base_stack = _load_stack(base_path)[z_start:stop]
            base_manifest_path = base_path.with_name('ER_mask_manifest.json')
            base_manifest = json.loads(base_manifest_path.read_text()) if base_manifest_path.is_file() else None
            if base_manifest is not None and base_manifest.get('source_shape') != list(source_shape):
                raise ValueError(f'Base manifest source shape mismatch: {base_manifest_path}')
        # Load source data after the ridge pass so its intermediate volumes can
        # be released first. Each organoid is processed separately.
        er_stack = _load_stack(er_path)
        fluo_stack = _load_stack(fluo_path)
        bright_stack = np.empty_like(base_stack, dtype=np.uint8)
        line_stack = np.empty_like(base_stack, dtype=np.uint8)
        audits = []
        for local_z, source_z in enumerate(range(z_start, stop)):
            bbox = None
            if base_manifest is not None:
                for entry in base_manifest.get('slices', []):
                    if entry.get('source_z_zero_based') == source_z:
                        bbox = entry.get('strong', {}).get('artifact_group_bbox_yxyx')
                        break
            bright, line, audit = process_slice(
                er_stack[source_z], fluo_stack[source_z], base_stack[local_z],
                settings=cell_settings, review=_review_for_z(reviews, relative, source_z, direct_cell_name),
                artifact_bbox_yxyx=bbox,
            )
            if line_half_width_override is not None and line_half_width_override < cell_settings.line_half_width:
                narrowed, restored_reference = narrow_existing_lines(
                    bright > 0, line > 0, line_half_width_override,
                )
                line = narrowed.astype(np.uint8) * 255
                audit['line_pixels'] = int(narrowed.sum())
                audit['reference_components_restored_to_preserve_continuity'] = restored_reference
            bright_stack[local_z] = bright
            line_stack[local_z] = line
            audits.append({'source_z_zero_based': source_z, **audit})
        target = output_root/relative
        if target != output_root:
            target.mkdir(parents=True, exist_ok=False)
        bright_path = target/'ER_mask_bright.tif'
        line_path = target/'ER_mask_line.tif'
        output_shape = None
        for path, stack in ((bright_path, bright_stack), (line_path, line_stack)):
            temp = path.with_suffix('.partial.tif')
            # A one-plane selection is a 2D TIFF; its global Z coordinate is
            # retained in the manifest. Multi-plane results are ImageJ ZYX.
            payload = stack[0] if stop-z_start == 1 else stack
            output_shape = list(payload.shape)
            # ImageJ metadata preserves ZYX axes for Fiji/napari stack browsing.
            tifffile.imwrite(
                temp, payload, imagej=True, compression='zlib',
                metadata={'axes': 'ZYX'} if payload.ndim == 3 else None,
            )
            temp.replace(path)
        manifest = {
            'status': 'exploratory candidate; not validated ground truth',
            'source_er': str(er_path), 'source_fluo': str(fluo_path),
            'source_er_sha256': sha256(er_path), 'source_fluo_sha256': sha256(fluo_path),
            'source_shape': list(source_shape),
            'output_shape': output_shape,
            'source_z_start_zero_based': z_start, 'source_z_stop_exclusive': stop,
            'base_mask_source': str(base_path) if base_path else 'er_ridge_batch_predict',
            'base_mask_sha256': sha256(base_path) if base_path else None,
            'profile': cell_profile,
            'settings': asdict(reported_settings), 'manual_review_applied': any(
                _review_for_z(reviews, relative, z, direct_cell_name)
                for z in range(z_start, stop)
            ),
            'review_config_sha256': hashlib.sha256(json.dumps(reviews, sort_keys=True).encode()).hexdigest(),
            'artifact_seed_config_sha256': hashlib.sha256(json.dumps(artifact_seeds, sort_keys=True).encode()).hexdigest(),
            'foreground': 255, 'background': 0,
            'bright_sha256': sha256(bright_path), 'line_sha256': sha256(line_path),
            'slice_audits': audits,
        }
        (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        completed.append(line_path)
        print(f'{relative}: COMPLETE {line_path}', flush=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True, help='Folder of paired ER.tif and fluo.tif')
    parser.add_argument('--output-root', type=Path, required=True, help='New, nonexistent output directory')
    parser.add_argument('--base-mask-root', type=Path, help='Existing mirrored ER_mask.tif tree; otherwise run ridge predictor')
    parser.add_argument('--artifact-seeds', type=Path, help='Optional reviewed artifact seeds for ridge predictor')
    parser.add_argument('--review-rois', type=Path, help='Optional per-slice review ellipses in source pixel coordinates')
    parser.add_argument('--z-start', type=int, default=0, help='Zero-based inclusive Z index')
    parser.add_argument('--z-stop', type=int, help='Zero-based exclusive Z index')
    profile_options = parser.add_mutually_exclusive_group()
    profile_options.add_argument('--profile', choices=('spheroid_default', 'dim_organoid_candidate'),
                                 help='One explicit imaging-domain profile for every cell')
    profile_options.add_argument('--profile-map', type=Path,
                                 help='JSON mapping every cell folder name to an explicit profile')
    parser.add_argument('--line-half-width', type=int,
                        help='Opt-in line radius (default 4 px; try 3 for slightly thinner masks)')
    args = parser.parse_args()
    seeds = json.loads(args.artifact_seeds.read_text()) if args.artifact_seeds else None
    reviews = json.loads(args.review_rois.read_text()) if args.review_rois else None
    profile_map = json.loads(args.profile_map.read_text()) if args.profile_map else None
    run_batch(args.input_root, args.output_root, base_mask_root=args.base_mask_root,
              artifact_seeds=seeds, reviews=reviews, z_start=args.z_start, z_stop=args.z_stop,
              settings=settings_for_profile(args.profile or 'spheroid_default'),
              profile_map=profile_map, line_half_width_override=args.line_half_width)


if __name__ == '__main__':
    main()
