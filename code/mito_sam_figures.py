"""Plot locked local SAM pilot; raw biological panels stay in ignored output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mito_sam_feasibility import normalize

METHODS = [
    "baseline",
    "otsu_control",
    "microsam_lm_otsu",
    "microsam_lm_baseline",
    "sam2_tiny_otsu",
    "sam2_tiny_baseline",
]
LABELS = [
    "Frozen U-Net",
    "Otsu control",
    "micro-SAM / Otsu",
    "micro-SAM / U-Net",
    "SAM2 Tiny / Otsu",
    "SAM2 Tiny / U-Net",
]
COLORS = ["#111111", "#999999", "#16758d", "#16758d", "#b56627", "#b56627"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["output", "image-dir", "mask-dir", "baseline-root", "figure"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    rows = list(csv.DictReader((a.output / "metrics.csv").open()))
    manifest = json.loads((a.output / "manifest.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, key, title in zip(
        axes.ravel(),
        ["dice", "precision", "recall", "boundary_f1_2px"],
        ["Full-image Dice", "Pixel precision", "Pixel recall", "Boundary F1 (2 px tolerance)"],
        strict=True,
    ):
        for i, method in enumerate(METHODS):
            group = [r for r in rows if r["method"] == method]
            vals = [float(r[key]) for r in group]
            for j, (r, v) in enumerate(zip(group, vals, strict=True)):
                low = float(r["foreground_fraction"]) <= 0.05
                ax.scatter(
                    i + (j - 1.5) * 0.06,
                    v,
                    c=COLORS[i],
                    marker="o" if low else "^",
                    s=36,
                    alpha=0.7,
                )
            ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, c=COLORS[i], lw=3)
        ax.set_xticks(range(len(METHODS)), LABELS, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("Score (0–1)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Mito SAM CPU pilot: 4 fixed images, not five-fold CV\n"
        "Circles: foreground ≤5% (n=2); triangles: >5% (n=2); horizontal bars: image mean",
        fontsize=13,
    )
    fig.supxlabel(
        "Source: locked pretrained inference, 2026-10-08. "
        "U-Net prompts come from frozen OOF predictions, never GT.",
        fontsize=9,
    )
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figure)
    fig.savefig(a.figure.with_suffix(".png"), dpi=160)
    plt.close(fig)
    panels = a.output / "panels"
    panels.mkdir(exist_ok=True)
    for case in manifest["cases"]:
        name = case["filename"]
        image = normalize(tifffile.imread(a.image_dir / name))
        gt = tifffile.imread(a.mask_dir / name) > 0
        predictions = {
            m: tifffile.imread(
                (a.baseline_root / f"fold_{case['fold']}" / "predictions" / "validation" / name)
                if m == "baseline"
                else a.output / m / name
            )
            > 0
            for m in METHODS
        }
        # A single input-only ROI chosen by the Otsu proposal's median location.
        points = np.argwhere(predictions["otsu_control"])
        cy, cx = np.median(points, axis=0).astype(int) if len(points) else (600, 600)
        y = int(np.clip(cy - 160, 0, 880))
        x = int(np.clip(cx - 160, 0, 880))
        roi = np.s_[y : y + 320, x : x + 320]
        fig, axs = plt.subplots(3, 8, figsize=(24, 9), layout="constrained")
        data = [image, gt] + [predictions[m] for m in METHODS]
        titles = ["Input", "Manual GT"] + LABELS
        for col, (im, title) in enumerate(zip(data, titles, strict=True)):
            axs[0, col].imshow(im, cmap="gray", vmin=0, vmax=1)
            axs[0, col].set_title(title, fontsize=10)
            axs[1, col].imshow(im[roi], cmap="gray", vmin=0, vmax=1)
            if col < 2:
                axs[2, col].imshow(im[roi], cmap="gray", vmin=0, vmax=1)
            else:
                pred = im.astype(bool)
                overlay = np.repeat((image * 0.45)[..., None], 3, axis=2)
                overlay[pred & ~gt] = [1, 0.18, 0.15]
                overlay[~pred & gt] = [0.1, 0.45, 1]
                overlay[pred & gt] = [0.1, 0.8, 0.35]
                axs[2, col].imshow(overlay[roi])
            for ax in axs[:, col]:
                ax.axis("off")
        fig.suptitle(
            f"{name} | Native 1200×1200 predictions; shown ROI x={x}, y={y}, 320×320 | "
            "Red=FP, Blue=FN, Green=TP\nLocal-only biological images. "
            "ROI from input Otsu median, not GT/error selection.",
            fontsize=12,
        )
        fig.savefig(panels / (Path(name).stem + "_comparison.png"), dpi=140)
        plt.close(fig)
    print("Figures saved; raw-image panels remain local-only.")


if __name__ == "__main__":
    main()
