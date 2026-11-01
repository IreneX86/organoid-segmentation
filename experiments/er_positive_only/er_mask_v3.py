"""ER mask v3 = v2 + two fixed post-processing rules (algorithmic pseudo-label, NOT GT).

1. Puncta removal (revised 2026-11-02 after v3a removed network junctions):
   candidates = round bright spots (both Hessian eigenvalues < 0, ratio >= 0.6) at
   sigma 1.5 or 2.5 inside the positive region. A candidate is a PUNCTUM only if its
   background-subtracted peak is >= 2x the 90th-percentile background-subtracted
   intensity of the surrounding tubule mask (within 30 px), i.e. clearly brighter than
   the brightest nearby tubules. (A median reference was tried first and flagged
   hundreds of ordinary junctions per image; rejected.) Network junctions are about as bright as
   the tubules they join, so they are kept. Each punctum is removed with a disk of
   radius 2*sigma+1 around its peak plus the candidate itself; fragments < 30 px dropped.
2. Width cap: mask pixels farther than 2.5 px from the mask skeleton are removed
   (no structure wider than ~5 px; thin tubules unchanged).
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label, find_objects
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import disk, remove_small_objects, skeletonize
from er_mask_v2 import er_mask as er_mask_v2, P as P2

P3 = dict(blob_sigmas=(1.5, 2.5), blob_ratio=0.6, peak_ratio=2.0, ref_pct=90.0, local_px=30, min_obj_after=30, max_halfwidth=2.5)


def puncta(E, M, R):
    e = gaussian_filter(E.astype(np.float32), P2["smooth"])
    hp = e - gaussian_filter(e, P2["bg_sigma"])
    remove = np.zeros(E.shape, bool); n = 0
    H, W = E.shape; L = P3["local_px"]
    for s in P3["blob_sigmas"]:
        l1, l2 = hessian_matrix_eigvals(hessian_matrix(e, sigma=s, order="rc", use_gaussian_derivatives=False))
        cand = (l1 < 0) & (l2 < 0) & (l1 / np.minimum(l2, -1e-9) >= P3["blob_ratio"]) & R & (hp > 0)
        lab, k = label(cand)
        for i, sl in enumerate(find_objects(lab), 1):
            if sl is None:
                continue
            comp = lab[sl] == i
            if comp.sum() < 3:
                continue
            sub = np.where(comp, hp[sl], -np.inf); py, px = np.unravel_index(np.argmax(sub), sub.shape)
            py += sl[0].start; px += sl[1].start; peak = hp[py, px]
            y0, y1, x0, x1 = max(py - L, 0), min(py + L + 1, H), max(px - L, 0), min(px + L + 1, W)
            win_m = M[y0:y1, x0:x1] & ~binary_dilation(lab[y0:y1, x0:x1] == i, disk(int(2 * s + 1)))
            ref = np.percentile(hp[y0:y1, x0:x1][win_m], P3["ref_pct"]) if win_m.sum() >= 30 else np.percentile(hp[y0:y1, x0:x1][R[y0:y1, x0:x1]], 99)
            if ref > 0 and peak >= P3["peak_ratio"] * ref:
                r = int(round(2 * s + 1))
                yy, xx = np.ogrid[:H, :W]
                yslc, xslc = slice(max(py - r, 0), min(py + r + 1, H)), slice(max(px - r, 0), min(px + r + 1, W))
                remove[yslc, xslc] |= (yy[yslc] - py) ** 2 + (xx[:, xslc] - px) ** 2 <= r * r
                remove[sl] |= comp
                n += 1
    return remove, n


def er_mask(E, tissue):
    M, R = er_mask_v2(E, tissue)
    if not M.any():
        return M, R, {}
    B, n = puncta(E, M, R)
    M1 = remove_small_objects(M & ~B, max_size=P3["min_obj_after"])
    S = skeletonize(M1)
    M2 = remove_small_objects(M1 & (distance_transform_edt(~S) <= P3["max_halfwidth"]), max_size=P3["min_obj_after"])
    info = dict(v2_px=int(M.sum()), puncta=n, removed_puncta_px=int((M & ~M1).sum()), removed_width_px=int((M1 & ~M2).sum()), v3_px=int(M2.sum()))
    return M2, R, info
