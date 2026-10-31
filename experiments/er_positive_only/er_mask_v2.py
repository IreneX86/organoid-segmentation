"""ER mask v2 from ER.tif (algorithmic pseudo-label, NOT manual ground truth).

Outputs per plane (same 1200x1200 grid as the source):
  ER_mask_v2.tif        uint8 0/255  ER tubule/sheet pixels
  ER_positive_region.tif uint8 0/255 cells where the ER reporter is visible.
                         Outside this region ER status is UNKNOWN (reporter absent),
                         so training must IGNORE those pixels, not treat them as background.
Design vs the previous coherent-fiber label:
  * positive-region is explicit, so non-expressing cells are not silent negatives;
  * ridge strength is normalised by local contrast, so dim expressing cells are not
    under-labelled relative to bright ones (previous WT recall was 25-51% of ER ridges);
  * hysteresis keeps weak tubules only when connected to strong ones.
Fluo is used only for the broad tissue support (as in the old pipeline).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter
from skimage.filters import apply_hysteresis_threshold, sato, threshold_otsu
from skimage.morphology import disk, remove_small_objects

P = dict(smooth=0.7, bg_sigma=12.0, contrast_sigma=16.0, region_sigma=10.0,
         region_min_px=3000, region_dilate=4, ridge_sigmas=(1.0, 1.5, 2.0),
         hi_pct=90.0, lo_pct=72.0, min_obj=15)


def positive_region(E, tissue):
    """Reporter-positive cells = where tubule-scale (1-4 px) energy clearly exceeds
    pixel-noise energy. Brightness alone confuses out-of-focus haze with expressing
    cells and misses dim expressing cells; structure-to-noise does neither.
    Threshold: Otsu on the (bimodal) in-tissue log ratio, fixed rule, no tuning per image."""
    band = gaussian_filter(E, 1.0) - gaussian_filter(E, 4.0)
    fine = E - gaussian_filter(E, 1.0)
    ratio = np.log(gaussian_filter(band**2, P["region_sigma"]) / (gaussian_filter(fine**2, P["region_sigma"]) + 1e-6) + 1e-12)
    R = binary_fill_holes((ratio > threshold_otsu(ratio[tissue])) & tissue)
    R = remove_small_objects(R, max_size=P["region_min_px"])
    return binary_dilation(R, disk(P["region_dilate"])) & tissue


def er_mask(E, tissue):
    R = positive_region(E, tissue)
    e = gaussian_filter(E, P["smooth"])
    hp = e - gaussian_filter(e, P["bg_sigma"])
    local = np.sqrt(gaussian_filter(hp**2, P["contrast_sigma"])) + 1e-6
    z = hp / local                                   # contrast-normalised detail
    ridge = sato(z, sigmas=P["ridge_sigmas"], black_ridges=False)
    ridge[z <= 0] = 0                                 # tubules must be locally bright
    if not R.any():
        return np.zeros_like(R), R
    hi, lo = np.percentile(ridge[R], [P["hi_pct"], P["lo_pct"]])
    M = apply_hysteresis_threshold(ridge, lo, hi) & R
    M = remove_small_objects(M, max_size=P["min_obj"])
    return M, R
