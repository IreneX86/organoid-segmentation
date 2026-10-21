# -*- coding: utf-8 -*-
"""
Render a paper-quality 3D screenshot of an instance segmentation result
using napari (offscreen, no interactive window). Produces two views:
  1. "whole"   -- the full volume (or a chosen z/xy crop), instance labels
                  colored, translucent rendering, to show overall context.
  2. "cutaway" -- a smaller, denser sub-region rendered at higher zoom with
                  iso_categorical rendering, to clearly show individual
                  cells as separated 3D objects (a full-volume view of a
                  densely packed spheroid mostly just looks like a solid
                  blob from outside -- a cropped close-up communicates
                  "these are separate 3D cells" much better).

Usage:
    python render_3d_demo.py --cell-dir <dir> --labels memb_instances_v4.tif \
        --out-prefix demo
"""

import argparse
import os

import numpy as np
import tifffile as tiff

from cellpose_instance_seg import load_channels


def render_view(memb, labels, out_path, voxel_scale, camera_angles=(20, -35, 0), zoom_factor=1.0,
                 labels_rendering="translucent", show_memb=True, memb_opacity=0.15):
    """zoom_factor is relative to napari's own auto-fit zoom (from reset_view()),
    not an absolute value -- the fit zoom differs a lot depending on the data's
    actual extent (e.g. a 330x1200x1200 volume vs a small crop), so a fixed
    absolute zoom looks wildly different depending on what's loaded.

    voxel_scale = (z, y, x) physical size per voxel -- without this the tif
    has no calibration and napari renders z-slices as if they were spaced
    the same as xy pixels, which flattens a real (roughly spherical) spheroid
    into a squashed disc."""
    import napari

    viewer = napari.Viewer(ndisplay=3, show=True)

    if show_memb:
        lo, hi = np.percentile(memb, [50, 99.5])
        viewer.add_image(memb, name="Memb", colormap="gray", rendering="attenuated_mip",
                          contrast_limits=(float(lo), float(hi)), opacity=memb_opacity,
                          scale=voxel_scale)

    labels_layer = viewer.add_labels(labels, name="instances", rendering=labels_rendering,
                                      opacity=1.0 if labels_rendering == "iso_categorical" else 0.85,
                                      scale=voxel_scale)
    labels_layer.contour = 0

    viewer.reset_view()
    viewer.camera.angles = camera_angles
    viewer.camera.zoom = viewer.camera.zoom * zoom_factor

    viewer.screenshot(out_path, canvas_only=True, flash=False, size=(1600, 1600))
    viewer.close()
    print("saved", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-dir", required=True)
    p.add_argument("--labels", required=True, help="instance-label tif (e.g. memb_instances_v4.tif)")
    p.add_argument("--out-prefix", default="demo_3d")
    p.add_argument("--cutaway-z0", type=int, default=10)
    p.add_argument("--cutaway-z1", type=int, default=95)
    p.add_argument("--cutaway-y0", type=int, default=400)
    p.add_argument("--cutaway-y1", type=int, default=700)
    p.add_argument("--cutaway-x0", type=int, default=400)
    p.add_argument("--cutaway-x1", type=int, default=700)
    p.add_argument("--voxel-xy-um", type=float, default=0.085, help="physical size of one xy pixel, in microns")
    p.add_argument("--voxel-z-um", type=float, default=0.3, help="physical z-step between slices, in microns")
    args = p.parse_args()

    voxel_scale = (args.voxel_z_um, args.voxel_xy_um, args.voxel_xy_um)

    memb, _fluo = load_channels(args.cell_dir)
    labels = tiff.imread(args.labels)

    render_view(memb, labels, f"{args.out_prefix}_whole.png", voxel_scale,
                camera_angles=(-25, -35, -140), zoom_factor=0.9,
                labels_rendering="translucent", memb_opacity=0.12)

    z0, z1 = args.cutaway_z0, args.cutaway_z1
    y0, y1 = args.cutaway_y0, args.cutaway_y1
    x0, x1 = args.cutaway_x0, args.cutaway_x1
    memb_c = memb[z0:z1, y0:y1, x0:x1]
    labels_c = labels[z0:z1, y0:y1, x0:x1]

    render_view(memb_c, labels_c, f"{args.out_prefix}_cutaway.png", voxel_scale,
                camera_angles=(-25, -35, -140), zoom_factor=0.9,
                labels_rendering="iso_categorical", show_memb=False)


if __name__ == "__main__":
    main()
