# -*- coding: utf-8 -*-
"""
Cellpose-based single-cell instance segmentation for Memb.tif volumes.

Strategy: run Cellpose per Z-slice (2D) with two channels
[membrane, nuclear-fluo], then stitch masks across Z using IoU overlap
(stitch_threshold). This avoids the large GPU memory footprint of true
3D inference on a (330, 1200, 1200) volume and does not require known
voxel spacing / anisotropy.

A sparse ground-truth annotation (label.tif, ~23 cells over z=1..96) is
available for this dataset and is used to (1) fine-tune the pretrained
model and (2) evaluate segmentation quality.

Usage:
    python cellpose_instance_seg.py predict  --cell-dir <dir> --out out.tif
    python cellpose_instance_seg.py finetune --cell-dir <dir> --out custom_model
    python cellpose_instance_seg.py eval     --cell-dir <dir> --pred out.tif
"""

import argparse
import json
import os

import numpy as np
import tifffile as tiff


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_channels(cell_dir: str):
    """Load membrane (cyto) and fluo (nuclear) volumes as (Z, Y, X) arrays."""
    memb_path = os.path.join(cell_dir, "Memb.tif")
    fluo_path = os.path.join(cell_dir, "fluo.tif")
    memb = tiff.imread(memb_path)
    fluo = tiff.imread(fluo_path)
    if memb.shape != fluo.shape:
        raise ValueError(f"shape mismatch: Memb {memb.shape} vs fluo {fluo.shape}")
    return memb, fluo


def load_label(cell_dir: str):
    label_path = os.path.join(cell_dir, "label.tif")
    return tiff.imread(label_path)


def labeled_z_range(label: np.ndarray):
    """Z indices where label.tif actually has annotated instances."""
    nonzero_z = np.where(label.reshape(label.shape[0], -1).max(axis=1) > 0)[0]
    return nonzero_z


# ---------------------------------------------------------------------------
# Inference: 2D per-slice + Z stitching
# ---------------------------------------------------------------------------

def run_cellpose_2d_stitch(
    memb: np.ndarray,
    fluo: np.ndarray,
    pretrained_model: str = "cpsam_v2",
    diameter: float = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    stitch_threshold: float = 0.25,
    gpu: bool = True,
    z_slice: slice = None,
) -> np.ndarray:
    """Segment a (Z, Y, X) membrane volume into 3D instance labels.

    Cellpose v4 (CPSAM) takes arbitrary channel order/count instead of the
    old channels=[cyto, nucleus] indices; we just stack membrane + fluo on
    a trailing channel axis (channel_axis=-1) and it pads to 3 channels
    internally. Slices are run independently in 2D (z_axis=0, do_3D=False),
    then linked across Z by mask overlap (stitch_threshold).
    """
    from cellpose import models

    if z_slice is not None:
        memb = memb[z_slice]
        fluo = fluo[z_slice]

    stack = np.stack([memb, fluo], axis=-1)  # (Z, Y, X, 2)

    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    masks, flows, styles = model.eval(
        stack,
        channel_axis=-1,
        z_axis=0,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        do_3D=False,
        stitch_threshold=stitch_threshold,
    )[:3]

    return np.asarray(masks).astype(np.uint32)


# ---------------------------------------------------------------------------
# Fine-tuning on the existing sparse ground truth (label.tif)
# ---------------------------------------------------------------------------

def build_finetune_set(cell_dir: str, extra_region_paths: list = None, use_base_label: bool = True):
    """Build 2D (image, mask) training pairs from the annotated Z range,
    plus any extra hand-corrected regions from napari_correct.py (each a
    tif + matching .json sidecar recording its crop bounds).

    If an extra region's z-range overlaps label.tif's annotated z-range,
    the extra region REPLACES the base label.tif slices for those z's
    (assumed to be a denser/corrected version of the same slice) instead
    of adding a second, conflicting set of labels for the same image.

    use_base_label=False skips label.tif's remaining (still-sparse,
    unverified) slices entirely and trains only on corrected regions --
    use this once you know label.tif is incomplete outside the regions
    you've already densified, so the leftover sparse slices don't teach
    the model to under-detect."""
    memb, fluo = load_channels(cell_dir)

    regions = []
    covered_z = set()
    for region_path in extra_region_paths or []:
        with open(region_path + ".json") as f:
            b = json.load(f)
        regions.append((region_path, b))
        covered_z.update(range(b["z_start"], b["z_end"]))

    train_imgs, train_masks = [], []

    if use_base_label:
        label = load_label(cell_dir)
        z_idx = labeled_z_range(label)
        for z in z_idx:
            if z in covered_z:
                continue  # superseded by a corrected region below
            img2 = np.stack([memb[z], fluo[z]], axis=-1)  # (Y, X, 2)
            train_imgs.append(img2)
            train_masks.append(label[z].astype(np.uint16))

    if not train_imgs and not extra_region_paths:
        raise ValueError("no training data: label.tif unused/empty and no extra regions given")

    for region_path, b in regions:
        region_labels = tiff.imread(region_path)
        region_memb, region_fluo = load_channels(b["cell_dir"])
        for i, z in enumerate(range(b["z_start"], b["z_end"])):
            lbl_slice = region_labels[i]
            if lbl_slice.max() == 0:
                continue
            memb_slice = region_memb[z, b["y_start"]:b["y_end"], b["x_start"]:b["x_end"]]
            fluo_slice = region_fluo[z, b["y_start"]:b["y_end"], b["x_start"]:b["x_end"]]
            train_imgs.append(np.stack([memb_slice, fluo_slice], axis=-1))
            train_masks.append(lbl_slice.astype(np.uint16))

    return train_imgs, train_masks


def finetune_model(
    cell_dir: str,
    out_model_path: str,
    base_model: str = "cpsam_v2",
    n_epochs: int = 60,
    learning_rate: float = 1e-5,
    val_fraction: float = 0.2,
    gpu: bool = True,
    extra_region_paths: list = None,
    use_base_label: bool = True,
):
    """Fine-tune a pretrained Cellpose model on the annotated slices.

    Splits the annotated Z-slices into train/val so progress can be
    monitored; the resulting custom model is saved to out_model_path
    and can be passed to run_cellpose_2d_stitch(pretrained_model=...).
    """
    import logging
    from cellpose import models, train
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("cellpose").setLevel(logging.INFO)

    imgs, masks = build_finetune_set(cell_dir, extra_region_paths=extra_region_paths,
                                      use_base_label=use_base_label)
    n_val = max(1, int(len(imgs) * val_fraction))
    train_imgs, val_imgs = imgs[:-n_val], imgs[-n_val:]
    train_masks, val_masks = masks[:-n_val], masks[-n_val:]

    model = models.CellposeModel(gpu=gpu, pretrained_model=base_model)

    # cellpose's train_seg always creates its own "models" subfolder under
    # save_path, so if out_model_path already ends in ".../models/<name>",
    # strip that trailing "models" to avoid a "models/models/<name>" nest.
    save_path = os.path.dirname(out_model_path) or "."
    if os.path.basename(save_path.rstrip("\\/")) == "models":
        save_path = os.path.dirname(save_path) or "."

    model_path = train.train_seg(
        model.net,
        train_data=train_imgs,
        train_labels=train_masks,
        test_data=val_imgs,
        test_labels=val_masks,
        channel_axis=-1,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        save_path=save_path,
        model_name=os.path.basename(out_model_path),
    )
    print("saved fine-tuned model to:", model_path)
    return model_path


# ---------------------------------------------------------------------------
# Evaluation against label.tif ground truth
# ---------------------------------------------------------------------------

def evaluate_against_gt(cell_dir: str, pred: np.ndarray, iou_thresholds=(0.5, 0.75, 0.9), z_offset: int = 0):
    """z_offset: z index in `pred` that corresponds to z=0 in label.tif.
    Use this if `pred` was produced from a cropped z-range (--z-start)."""
    from cellpose.metrics import average_precision

    label = load_label(cell_dir)
    z_idx = labeled_z_range(label)

    gt_slices = [label[z] for z in z_idx]
    pred_slices = [pred[z - z_offset] for z in z_idx]

    ap, tp, fp, fn = average_precision(gt_slices, pred_slices, threshold=list(iou_thresholds))
    for i, thr in enumerate(iou_thresholds):
        print(f"IoU={thr:.2f}  AP={ap[:, i].mean():.3f}  "
              f"TP={tp[:, i].sum():.0f}  FP={fp[:, i].sum():.0f}  FN={fn[:, i].sum():.0f}")
    return ap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["predict", "finetune", "eval"])
    p.add_argument("--cell-dir", required=True)
    p.add_argument("--out", default="instance_labels.tif")
    p.add_argument("--pred", default=None, help="existing prediction tif (for eval mode)")
    p.add_argument("--model", default="cpsam_v2",
                    help="cellpose pretrained_model name (e.g. cpsam_v2) or path to a custom/fine-tuned model")
    p.add_argument("--diameter", type=float, default=None)
    p.add_argument("--stitch-threshold", type=float, default=0.25)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--z-start", type=int, default=None)
    p.add_argument("--z-end", type=int, default=None)
    p.add_argument("--epochs", type=int, default=60, help="finetune mode only")
    p.add_argument("--extra-regions", default=None,
                    help="finetune mode only: comma-separated paths to napari_correct.py output tifs "
                         "(each needs a matching .json sidecar)")
    p.add_argument("--no-base-label", action="store_true",
                    help="finetune mode only: skip label.tif's remaining (still-sparse) slices, "
                         "train only on --extra-regions")
    args = p.parse_args()

    gpu = not args.no_gpu
    z_slice = slice(args.z_start, args.z_end) if (args.z_start or args.z_end) else None

    if args.mode == "predict":
        memb, fluo = load_channels(args.cell_dir)
        masks = run_cellpose_2d_stitch(
            memb, fluo,
            pretrained_model=args.model,
            diameter=args.diameter,
            stitch_threshold=args.stitch_threshold,
            gpu=gpu,
            z_slice=z_slice,
        )
        tiff.imwrite(args.out, masks, compression="zlib")
        print(f"saved {masks.shape} instance labels ({masks.max()} cells) -> {args.out}")

    elif args.mode == "finetune":
        extra_regions = args.extra_regions.split(",") if args.extra_regions else None
        finetune_model(args.cell_dir, args.out, base_model=args.model, gpu=gpu, n_epochs=args.epochs,
                        extra_region_paths=extra_regions, use_base_label=not args.no_base_label)

    elif args.mode == "eval":
        if args.pred is None:
            raise ValueError("--pred is required for eval mode")
        pred = tiff.imread(args.pred)
        evaluate_against_gt(args.cell_dir, pred, z_offset=args.z_start or 0)


if __name__ == "__main__":
    main()
