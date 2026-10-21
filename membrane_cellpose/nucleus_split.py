# -*- coding: utf-8 -*-
"""
Use fluo.tif's nuclear "holes" (this is a lipid-dye stain of the whole cell;
the nucleus itself doesn't take up the dye, so it shows as a dark hole
surrounded by bright punctate cytoplasmic signal) as a one-nucleus-per-cell
prior to catch and fix merged/under-segmented instances in a cellpose
prediction.

Works in full 3D, not per-2D-slice: a real nucleus is a compact blob that
persists coherently across many consecutive z-slices, while texture noise
in the fluo channel is not spatially consistent slice-to-slice. An earlier
per-2D-slice version of this script tried to find "the darkest N% of each
slice" independently on every slice and mostly just rediscovered noise
(tested on ground truth: no recall gain, net negative on precision). Doing
real 3D connected-component analysis on the dark voxels naturally filters
that noise out, since only a genuine nucleus accumulates enough contiguous
3D volume across z to pass the size/contrast filters below.

For every 3D instance in a (stitched) cellpose prediction volume: find dark
3D nucleus blob(s) inside its 3D mask. 0 or 1 -> left alone. 2+ -> very
likely two touching cells merged into one instance; split with a
marker-controlled 3D watershed seeded at each nucleus blob's centroid.
Because this operates on the whole 3D mask at once, the split result is
inherently z-consistent (no separate re-stitching step needed).

Usage:
    python nucleus_split.py --cell-dir <dir> --pred pred_full330_finetuned_v4.tif \
        --out pred_full330_v5.tif --report split_report_v5.csv
"""

import argparse
import csv

import numpy as np
import scipy.ndimage as ndi
import tifffile as tiff
from skimage.segmentation import watershed

from cellpose_instance_seg import load_channels


def detect_3d_holes_in_instance(fluo_vol, inst_mask, min_hole_frac=0.06, max_hole_frac=0.60,
                                 min_hole_vox=300, dark_percentile=30, min_contrast_ratio=1.25,
                                 min_hole_vol_ratio=0.4):
    """3D analogue of hole detection: find dark nucleus blobs inside a single
    instance's 3D mask, using 3D connectivity so single-slice noise can't
    pass the volume threshold on its own.

    Returns [(voxel_count, (z, y, x)), ...] sorted largest first."""
    vals = fluo_vol[inst_mask]
    if vals.size < min_hole_vox * 2:
        return []
    thresh = np.percentile(vals, dark_percentile)
    dark = inst_mask & (fluo_vol <= thresh)
    dark = ndi.binary_opening(dark, structure=np.ones((3, 3, 3)))  # drop 1-2 voxel noise
    lbl, n = ndi.label(dark, structure=np.ones((3, 3, 3)))  # 26-connectivity
    if n == 0:
        return []

    inst_vox = inst_mask.sum()
    min_vox = max(min_hole_vox, min_hole_frac * inst_vox)
    max_vox = max_hole_frac * inst_vox
    bright_median = np.median(fluo_vol[inst_mask & ~dark]) if (inst_mask & ~dark).any() else None

    holes = []
    sizes = ndi.sum(dark, lbl, index=np.arange(1, n + 1))
    centroids = ndi.center_of_mass(dark, lbl, index=np.arange(1, n + 1))
    for i, (vox, (cz, cy, cx)) in enumerate(zip(sizes, centroids), start=1):
        if not (min_vox <= vox <= max_vox):
            continue
        if bright_median is None or bright_median <= 0:
            continue
        blob_median = np.median(fluo_vol[lbl == i])
        if bright_median / max(blob_median, 1e-6) < min_contrast_ratio:
            continue  # not actually dark enough relative to the rest of the cell
        holes.append((vox, (cz, cy, cx)))
    holes.sort(key=lambda h: -h[0])

    if len(holes) >= 2 and holes[1][0] < min_hole_vol_ratio * holes[0][0]:
        holes = holes[:1]  # 2nd blob too small to be a real second nucleus -> noise fragment
    return holes


def split_merged_instances_3d(label_vol, fluo_vol, next_id, max_holes=4, min_instance_vox=3000):
    """Split any 3D instance in label_vol with >=2 detected 3D nucleus blobs.
    Returns (corrected_vol, next_id, split_log) where split_log is a list of
    (orig_id, n_holes, [resulting_ids])."""
    out = label_vol.copy()
    split_log = []
    for inst_id in np.unique(label_vol):
        if inst_id == 0:
            continue
        inst_mask = label_vol == inst_id
        vox = inst_mask.sum()
        if vox < min_instance_vox:  # too small in 3D to plausibly contain 2 nuclei
            continue

        zs = np.where(inst_mask.any(axis=(1, 2)))[0]
        z0, z1 = zs.min(), zs.max() + 1
        sub_mask = inst_mask[z0:z1]
        sub_fluo = fluo_vol[z0:z1]

        holes = detect_3d_holes_in_instance(sub_fluo, sub_mask)
        if len(holes) < 2:
            continue
        holes = holes[:max_holes]

        markers = np.zeros(sub_mask.shape, dtype=np.int32)
        for k, (_vox, (cz, cy, cx)) in enumerate(holes, start=1):
            markers[int(round(cz)), int(round(cy)), int(round(cx))] = k

        distance = ndi.distance_transform_edt(sub_mask)
        labels_ws = watershed(-distance, markers, mask=sub_mask)

        result_ids = []
        for k in range(1, len(holes) + 1):
            gid = inst_id if k == 1 else next_id
            if k != 1:
                next_id += 1
            out[z0:z1][labels_ws == k] = gid
            result_ids.append(int(gid))
        split_log.append((int(inst_id), len(holes), result_ids, int(z0), int(z1)))
    return out, next_id, split_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-dir", required=True)
    p.add_argument("--pred", required=True, help="cellpose prediction tif to correct (a stitched 3D volume)")
    p.add_argument("--out", required=True, help="output tif path for the corrected labels")
    p.add_argument("--report", default=None, help="optional CSV path listing every split made")
    p.add_argument("--z-start", type=int, default=None,
                    help="if --pred was generated with --z-start, pass it here so fluo slices align")
    args = p.parse_args()

    _memb, fluo = load_channels(args.cell_dir)
    pred = tiff.imread(args.pred)

    z0 = args.z_start or 0
    fluo_matched = fluo[z0:z0 + pred.shape[0]]

    corrected, next_id, log = split_merged_instances_3d(pred, fluo_matched, int(pred.max()) + 1)

    tiff.imwrite(args.out, corrected.astype(np.uint32), compression="zlib")
    print(f"saved corrected labels -> {args.out}")
    print(f"3D instances split: {len(log)} (out of {len(np.unique(pred)) - 1} instances scanned)")

    if args.report:
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["original_id", "n_holes_detected", "resulting_ids", "z_start", "z_end"])
            for orig_id, n_holes, result_ids, zs, ze in log:
                w.writerow([orig_id, n_holes, " ".join(map(str, result_ids)), zs + z0, ze + z0])
        print(f"split report -> {args.report}")


if __name__ == "__main__":
    main()
