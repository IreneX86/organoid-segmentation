# -*- coding: utf-8 -*-
"""
Evaluate a prediction against a dense hand-corrected ground truth region
(from napari_correct.py), instead of the sparse label.tif. Use this
whenever a corrected/region_*.tif exists for the z-range you care about --
label.tif is a sparse partial annotation and its FP counts are not
trustworthy (see README.md).

Usage:
    python eval_dense.py --pred pred_z1-96_finetuned_v2.tif \
        --gt corrected/region_z1-70.tif --pred-z-offset 0
"""

import argparse

import tifffile as tiff
from cellpose.metrics import average_precision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="prediction tif")
    p.add_argument("--gt", required=True,
                    help="dense ground-truth tif produced by napari_correct.py "
                         "(reads the matching .json sidecar for its z-range)")
    p.add_argument("--pred-z-offset", type=int, default=0,
                    help="z index in --pred that corresponds to the ground truth's z_start "
                         "(0 if --pred was generated with the same --z-start as the gt region)")
    p.add_argument("--thresholds", default="0.5,0.75,0.9")
    args = p.parse_args()

    pred = tiff.imread(args.pred)
    gt = tiff.imread(args.gt)
    n = gt.shape[0]

    pred_slices = [pred[args.pred_z_offset + i] for i in range(n)]
    gt_slices = [gt[i] for i in range(n)]

    thresholds = [float(t) for t in args.thresholds.split(",")]
    ap, tp, fp, fn = average_precision(gt_slices, pred_slices, threshold=thresholds)

    for i, thr in enumerate(thresholds):
        tp_sum, fp_sum, fn_sum = tp[:, i].sum(), fp[:, i].sum(), fn[:, i].sum()
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else float("nan")
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else float("nan")
        print(f"IoU={thr:.2f}  TP={tp_sum:.0f}  FP={fp_sum:.0f}  FN={fn_sum:.0f}  "
              f"precision={precision:.3f}  recall={recall:.3f}")


if __name__ == "__main__":
    main()
