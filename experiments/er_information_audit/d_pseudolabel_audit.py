"""Analysis D: pseudo-label (ER_mask_line) audit against the ER channel itself.

Reference used here: an independent, simple ER ridge map (Sato, sigmas 1-2 px,
top 12% of in-tissue response) computed directly from ER.tif. It is ANOTHER
algorithmic reference, not ground truth; disagreement measures inconsistency,
not which side is right. All statistics are within Fluo-defined tissue.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from skimage.filters import sato
from skimage.morphology import disk, skeletonize

from common import BATCH, SPECIMENS, load_plane, robust_norm, sample_planes, tissue_mask


def ridge_ref(E, m):
    e = robust_norm(E, m)
    e = e - gaussian_filter(e, 12)
    r = sato(e, sigmas=(1.0, 1.5, 2.0), black_ridges=False)
    thr = np.percentile(r[m], 88)
    return (r > thr) & m, e


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main():
    out = {}
    for s in SPECIMENS:
        rows = []
        for z in sample_planes(s):
            E, F, PLr = load_plane(s, z, "er"), load_plane(s, z, "fluo"), load_plane(s, z, "pl")
            E1, PL1 = load_plane(s, z + 1, "er"), load_plane(s, z + 1, "pl") > 0
            m = tissue_mask(F)
            PL = (PLr > 0) & m
            R, ehp = ridge_ref(E, m)
            R1, _ = ridge_ref(E1, m)
            sk = skeletonize(PL)
            width = 2 * distance_transform_edt(PL)[sk] - 1 if sk.any() else np.array([np.nan])
            tol = disk(2)
            Rd, PLd = binary_dilation(R, tol), binary_dilation(PL, tol)
            ring = binary_dilation(PL, disk(6)) & ~binary_dilation(PL, disk(2)) & m
            en, fn = robust_norm(E, m), robust_norm(F, m)
            rows.append({
                "z": z,
                "pl_frac_tissue": float(PL.sum() / m.sum()),
                "pl_width_median": float(np.median(width)),
                "pl_width_p90": float(np.percentile(width, 90)),
                "pl_on_er_ridge": float((PL & Rd).sum() / max(PL.sum(), 1)),
                "er_ridge_covered": float((R & PLd).sum() / max(R.sum(), 1)),
                "pl_skel_on_ridge": float((sk & Rd).sum() / max(sk.sum(), 1)),
                "er_hp_in_pl": float(ehp[PL].mean()), "er_hp_ring": float(ehp[ring].mean()),
                "er_in_pl": float(en[PL].mean()), "er_out_pl": float(en[m & ~PL].mean()),
                "fluo_in_pl": float(fn[PL].mean()), "fluo_out_pl": float(fn[m & ~PL].mean()),
                "pl_z_iou": iou(PL, PL1 & m), "ridge_z_iou": iou(R, R1),
            })
        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "z"}
        out[s] = {"batch": BATCH[s], "planes": rows, "mean": agg}
        print(s, BATCH[s], {k: round(v, 3) for k, v in agg.items()})
    json.dump(out, open("/home/claude/er_audit/out/d_pseudolabel_audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
