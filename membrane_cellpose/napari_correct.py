# -*- coding: utf-8 -*-
"""
Human-in-the-loop label correction: open a region of Memb.tif/fluo.tif in
napari, pre-seeded with the model's own prediction (if available) so you
correct mistakes instead of drawing every cell from scratch. Save the
corrected region back out for use as extra fine-tuning data.

Pick regions that add DIVERSITY (different z-depth, different part of the
spheroid) rather than re-labeling the same z=1..96 patch that label.tif
already covers -- the model already knows that patch well.

Usage:
    python napari_correct.py --cell-dir <dir> --pred pred_full330_cpsam.tif ^
        --z-start 150 --z-end 180 --out corrected/region_z150.tif

    # no existing prediction to seed from -> starts from blank labels
    python napari_correct.py --cell-dir <dir> ^
        --z-start 250 --z-end 280 --y-start 0 --y-end 600 --x-start 0 --x-end 600 ^
        --out corrected/region_z250_corner.tif

In napari:
    - Scroll the z-slider (bottom) to move through slices; correct each
      slice in 2D (this matches how the mask data is used for training).
      Painting only ever touches the current slice (n_edit_dimensions=2),
      so you can't accidentally bleed a correction into neighboring z.
    - The `labels` layer starts in CONTOUR mode (outlines only) so the
      membrane signal underneath stays visible -- much easier to align a
      boundary to the real membrane than a solid fill. Press 'l' (Labels
      layer > "n" render) or toggle the "contour" spinner in the layer
      controls (top-left) to switch back to solid fill if you prefer.
    - Tools: '2' = paint brush, '3' = fill bucket, '1' = pan/zoom,
      'm' = pick a fresh unused label id (use this before painting a cell
      the model missed), Ctrl+Z = undo. Paint with label 0 (or the eraser)
      to delete a false positive.
    - Press 's' at any time to checkpoint your progress WITHOUT closing
      the window (recommended every few slices -- don't rely on only the
      close-on-exit save for a long session).
    - Close the napari window when done -- the labels + a sidecar .json
      (recording where this crop sits in the full volume) are saved
      automatically to --out.
"""

import argparse
import json
import os

import numpy as np
import tifffile as tiff

from cellpose_instance_seg import load_channels


def crop(arr, z0, z1, y0, y1, x0, x1):
    return arr[z0:z1, y0:y1, x0:x1]


def count_instances(labels):
    """Number of distinct nonzero label ids actually present.

    NOT the same as labels.max(): when seeding from a crop of a
    globally-labeled full-volume prediction, label ids are not
    renumbered 1..N within the crop, so max() reflects an arbitrary
    global id, not how many cells are in this region."""
    return int(np.count_nonzero(np.unique(labels)))


def apply_overlay(seed_labels, bounds, overlay_path):
    """Stamp an already-corrected region (from a previous napari_correct.py
    run) on top of seed_labels wherever it overlaps the new crop `bounds`,
    so re-opening a larger/shifted range doesn't discard prior work."""
    z0, z1, y0, y1, x0, x1 = bounds
    with open(overlay_path + ".json") as f:
        ob = json.load(f)
    oz0, oz1, oy0, oy1, ox0, ox1 = (ob["z_start"], ob["z_end"], ob["y_start"],
                                     ob["y_end"], ob["x_start"], ob["x_end"])

    iz0, iz1 = max(z0, oz0), min(z1, oz1)
    iy0, iy1 = max(y0, oy0), min(y1, oy1)
    ix0, ix1 = max(x0, ox0), min(x1, ox1)
    if iz0 >= iz1 or iy0 >= iy1 or ix0 >= ix1:
        print(f"warning: --overlay-region {overlay_path} does not overlap the requested crop, skipping")
        return

    overlay_data = tiff.imread(overlay_path)
    seed_labels[iz0 - z0:iz1 - z0, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
        overlay_data[iz0 - oz0:iz1 - oz0, iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0]
    print(f"applied overlay from {overlay_path}: z=[{iz0},{iz1}) y=[{iy0},{iy1}) x=[{ix0},{ix1})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-dir", required=True)
    p.add_argument("--pred", default=None, help="existing full-volume prediction tif to seed labels from")
    p.add_argument("--out", required=True, help="output tif path for the corrected region")
    p.add_argument("--z-start", type=int, required=True)
    p.add_argument("--z-end", type=int, required=True)
    p.add_argument("--y-start", type=int, default=0)
    p.add_argument("--y-end", type=int, default=None)
    p.add_argument("--x-start", type=int, default=0)
    p.add_argument("--x-end", type=int, default=None)
    p.add_argument("--brush-size", type=int, default=12, help="starting paint-brush diameter in pixels")
    p.add_argument("--overlay-regions", default=None,
                    help="comma-separated paths to previously corrected region tifs (with .json sidecars) "
                         "whose overlap with this crop should seed from the corrected data instead of "
                         "--pred -- use this when extending/re-opening a range that includes already-"
                         "corrected z-slices, so you don't lose or redo that work")
    args = p.parse_args()

    memb, fluo = load_channels(args.cell_dir)
    y_end = args.y_end or memb.shape[1]
    x_end = args.x_end or memb.shape[2]
    bounds = (args.z_start, args.z_end, args.y_start, y_end, args.x_start, x_end)

    memb_c = crop(memb, *bounds)
    fluo_c = crop(fluo, *bounds)

    if args.pred is not None:
        pred = tiff.imread(args.pred)
        seed_labels = crop(pred, *bounds).astype(np.uint32)
    else:
        seed_labels = np.zeros_like(memb_c, dtype=np.uint32)

    for overlay_path in (args.overlay_regions.split(",") if args.overlay_regions else []):
        apply_overlay(seed_labels, bounds, overlay_path)

    import napari

    # Percentile contrast so the membrane signal is actually visible instead
    # of washed-out raw uint16 range (min/max in a spheroid image are often
    # dominated by a few very bright/dark outlier pixels).
    memb_lo, memb_hi = np.percentile(memb_c, [1, 99.5])
    fluo_lo, fluo_hi = np.percentile(fluo_c, [1, 99.5])

    viewer = napari.Viewer()
    viewer.add_image(memb_c, name="Memb", colormap="gray",
                      contrast_limits=(float(memb_lo), float(memb_hi)))
    viewer.add_image(fluo_c, name="fluo", colormap="green", blending="additive",
                      visible=False, contrast_limits=(float(fluo_lo), float(fluo_hi)))
    labels_layer = viewer.add_labels(seed_labels, name="labels")
    labels_layer.contour = 1            # outlines only -> membrane stays visible underneath
    labels_layer.n_edit_dimensions = 2  # never paint across z by accident
    labels_layer.brush_size = args.brush_size

    def save_current(*_):
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tiff.imwrite(args.out, labels_layer.data.astype(np.uint32), compression="zlib")
        sidecar = {
            "cell_dir": args.cell_dir,
            "z_start": bounds[0], "z_end": bounds[1],
            "y_start": bounds[2], "y_end": bounds[3],
            "x_start": bounds[4], "x_end": bounds[5],
        }
        with open(args.out + ".json", "w") as f:
            json.dump(sidecar, f, indent=2)
        print(f"saved corrected labels -> {args.out}  ({count_instances(labels_layer.data)} instances)")

    @viewer.bind_key("s")
    def _save_checkpoint(_viewer):
        save_current()

    viewer.window._qt_window.destroyed.connect(save_current)

    print(__doc__)
    print(f"starting instances in this crop: {count_instances(seed_labels)}")
    napari.run()
    save_current()


if __name__ == "__main__":
    main()
