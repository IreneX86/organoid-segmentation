"""Frozen, user-selected ER fiber refinement with adjacent-Z support.

This produces an exploratory image-processing candidate, not a ground-truth
label. Parameters reproduce the topology-safe five-plane review of 2026-10-28.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, gaussian_filter, label
from skimage.filters import apply_hysteresis_threshold, meijering
from skimage.morphology import disk, skeletonize


@dataclass(frozen=True)
class CoherentFiberSettings:
    camera_smoothing_sigma: float = 0.65
    background_sigma: float = 18.0
    ridge_sigmas: tuple[float, ...] = (1.0, 1.8, 2.8)
    hysteresis_low: float = 0.055
    hysteresis_high: float = 0.17
    old_support_radius: int = 30
    z_support_radius: int = 3
    group_radius: int = 1
    minimum_center_pixels: int = 24
    minimum_old_contact_pixels: int = 3
    old_contact_radius: int = 6
    old_body_radius: int = 2
    added_line_radius: int = 1


SETTINGS = CoherentFiberSettings()


def ridge_network(er: np.ndarray, settings: CoherentFiberSettings = SETTINGS) -> np.ndarray:
    """Return a coherent 2D ER ridge map; no batch-dependent normalization."""
    if er.ndim != 2:
        raise ValueError("ER plane must be 2D")
    image = gaussian_filter(er.astype(np.float32), settings.camera_smoothing_sigma)
    background = gaussian_filter(image, settings.background_sigma)
    highpass = image - background
    p50, p99 = np.percentile(er, (50, 99))
    relative = np.clip(highpass / np.maximum(background, max(60, 0.25 * p50)), 0, 1)
    ridge = meijering(relative, sigmas=settings.ridge_sigmas, black_ridges=False)
    contrast = highpass > max(4, 0.025 * (p99 - p50))
    return apply_hysteresis_threshold(
        np.where(contrast, ridge, 0),
        low=settings.hysteresis_low,
        high=settings.hysteresis_high,
    )


def refine_coherent_fiber(
    current: np.ndarray,
    old_line: np.ndarray,
    adjacent: np.ndarray,
    settings: CoherentFiberSettings = SETTINGS,
) -> tuple[np.ndarray, dict]:
    """Add Z-supported fibers while preserving each old 8-connected region."""
    if current.ndim != 2 or current.shape != old_line.shape or current.shape != adjacent.shape:
        raise ValueError("current, old_line and adjacent must be aligned 2D arrays")
    current = current.astype(bool)
    old = old_line.astype(bool)
    adjacent = adjacent.astype(bool)
    zone = binary_dilation(old, structure=disk(settings.old_support_radius))
    z_support = binary_dilation(adjacent, structure=disk(settings.z_support_radius))
    raw = current & zone & z_support
    center = skeletonize(raw)
    structure = np.ones((3, 3), dtype=bool)
    ids, count = label(binary_dilation(center, structure=disk(settings.group_radius)),
                       structure=structure)
    sizes = np.bincount(ids.ravel(), weights=center.ravel(), minlength=count + 1)
    contact = np.bincount(
        ids.ravel(),
        weights=(center & binary_dilation(old, structure=disk(settings.old_contact_radius))).ravel(),
        minlength=count + 1,
    )
    keep = (sizes >= settings.minimum_center_pixels) & (
        contact >= settings.minimum_old_contact_pixels
    )
    keep[0] = False
    lines = center & keep[ids]
    old_center = skeletonize(old)
    old_body = old & binary_dilation(old_center, structure=disk(settings.old_body_radius))
    detail = binary_dilation(lines, structure=disk(settings.added_line_radius))
    candidate = old_body | detail
    old_ids, old_count = label(old, structure=structure)
    new_ids, _ = label(candidate, structure=structure)
    restored = 0
    # Only an old region whose narrowed pixels split is restored. New detail
    # does not replace or erase a previously continuous region.
    for index, box in enumerate(find_objects(old_ids), start=1):
        if box is None:
            continue
        region = old_ids[box] == index
        found = np.unique(new_ids[box][region])
        if np.count_nonzero(found) != 1:
            candidate[box][region] = True
            restored += 1
    new_ids, _ = label(candidate, structure=structure)
    for index, box in enumerate(find_objects(old_ids), start=1):
        if box is None:
            continue
        if np.count_nonzero(np.unique(new_ids[box][old_ids[box] == index])) != 1:
            raise RuntimeError(f"Old ER component {index} split after restoration")
    if not np.all(candidate[old_center]):
        raise RuntimeError("Old ER centerline was removed")
    audit = {
        "coherent_pixels": int(current.sum()),
        "z_supported_pixels": int(raw.sum()),
        "line_center_pixels": int(lines.sum()),
        "line_groups_kept": int(keep.sum()),
        "old_pixels": int(old.sum()),
        "candidate_pixels": int(candidate.sum()),
        "ridge_additions_outside_old": int((detail & ~old).sum()),
        "old_components_restored_for_connectivity": restored,
    }
    return candidate.astype(np.uint8) * 255, audit


def settings_record() -> dict:
    return asdict(SETTINGS)
