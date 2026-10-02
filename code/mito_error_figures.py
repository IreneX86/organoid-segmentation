"""Create local-only visual diagnostics for eight frozen Mito OOF examples.

The rendered panels contain biological image pixels and must stay in the
Git-ignored output directory. Colors: green TP, red FP, cyan FN. Probability
maps come from the frozen checkpoint and were verified to reproduce the
archived OOF masks exactly at each fold's validation-selected threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from tiff_io import normalize_image, read_binary_mask, read_tiff_2d


def _overlay(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*truth.shape, 4), dtype=np.float32)
    tp = truth & prediction
    fp = ~truth & prediction
    fn = truth & ~prediction
    rgba[tp] = (0.05, 0.9, 0.25, 0.48)
    rgba[fp] = (1.0, 0.1, 0.1, 0.82)
    rgba[fn] = (0.0, 0.85, 1.0, 0.82)
    return rgba


def _fp_crop(fp: np.ndarray, side: int = 320) -> tuple[slice, slice]:
    density = ndi.uniform_filter(fp.astype(np.float32), size=side)
    y, x = np.unravel_index(int(np.argmax(density)), density.shape)
    y0 = min(max(y - side // 2, 0), fp.shape[0] - side)
    x0 = min(max(x - side // 2, 0), fp.shape[1] - side)
    return slice(y0, y0 + side), slice(x0, x0 + side)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads((args.probability_dir / "cases.json").read_text())
    with args.audit_csv.open(newline="", encoding="utf-8") as handle:
        metrics = {row["filename"]: row for row in csv.DictReader(handle)}

    for case in cases:
        filename = case["filename"]
        fold = int(case["fold"])
        raw = read_tiff_2d(args.image_dir / filename)
        image = normalize_image(raw)
        truth = read_binary_mask(args.mask_dir / filename).astype(bool)
        prediction = read_binary_mask(
            args.baseline_root / f"fold_{fold}" / "predictions" / "validation" / filename
        ).astype(bool)
        probability = np.load(
            args.probability_dir / f"{Path(filename).stem}_probability.npy"
        )
        if any(x.shape != image.shape for x in (truth, prediction, probability)):
            raise ValueError(f"Geometry mismatch for {filename}")
        fp = prediction & ~truth
        zoom = _fp_crop(fp)
        color = _overlay(truth, prediction)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("Input · per-image p1–p99.8 normalization")
        axes[0, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("Manual GT · 0/1")
        axes[0, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[0, 2].set_title("Frozen OOF prediction · 0/1")
        heat = axes[1, 0].imshow(probability, cmap="viridis", vmin=0, vmax=1)
        axes[1, 0].set_title("Frozen-model probability · 0–1")
        fig.colorbar(heat, ax=axes[1, 0], fraction=0.045, pad=0.02)
        axes[1, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[1, 1].imshow(color)
        axes[1, 1].set_title("Full overlay · TP green / FP red / FN cyan")
        axes[1, 2].imshow(image[zoom], cmap="gray", vmin=0, vmax=1)
        axes[1, 2].imshow(color[zoom])
        axes[1, 2].set_title(
            f"320×320 native-pixel crop with most FP · x={zoom[1].start}, y={zoom[0].start}"
        )
        row = metrics[filename]
        fig.suptitle(
            f"{filename} · fold {fold} · {case['stratum']} / {case['dice_group']}"
            f" · FG {100 * float(row['foreground_fraction']):.2f}%"
            f" · Dice {float(row['dice']):.3f} · precision {float(row['precision']):.3f}"
        )
        fig.savefig(args.output_dir / f"{Path(filename).stem}_panel.png", dpi=135)
        plt.close(fig)

    # Aggregate chart is descriptive; the 5% group is based on manual masks
    # and must not be used to choose an image-specific deployment threshold.
    rows = list(metrics.values())
    fg = np.array([float(r["foreground_fraction"]) for r in rows]) * 100
    dice = np.array([float(r["dice"]) for r in rows])
    precision = np.array([float(r["precision"]) for r in rows])
    near = np.array([float(r["fp_within_2px"]) for r in rows])
    low = fg <= 5
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for ax, values, label in zip(
        axes, (dice, precision, near),
        ("OOF Dice", "OOF precision", "Fraction of FP within 2 px of GT"),
    ):
        ax.scatter(fg[~low], values[~low], label="FG >5% (n=29)", c="#4a6b83")
        ax.scatter(fg[low], values[low], label="FG ≤5% (n=7)", c="#b03a3a")
        ax.axvline(5, color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel="Manual foreground fraction (%)", ylabel=label)
        ax.set_title(f"{label} vs foreground fraction")
        ax.legend(fontsize=8)
    fig.suptitle("Frozen Mito five-fold OOF · 36 labelled development images · 2026-10-02")
    fig.savefig(args.output_dir / "oof_error_overview.png", dpi=160)
    plt.close(fig)
    confidence_path = args.output_dir.parent / "confidence_summary.json"
    if confidence_path.is_file():
        confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
        groups = {
            "FG ≤5% (7 images)": [r for r in rows if float(r["foreground_fraction"]) <= .05],
            "FG >5% (29 images)": [r for r in rows if float(r["foreground_fraction"]) > .05],
        }
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for index, (label, group) in enumerate(groups.items()):
            near = np.mean([float(r["fp_within_2px"]) for r in group])
            far = np.mean([float(r["fp_farther_10px"]) for r in group])
            middle = 1 - near - far
            labels = (
                ("FP ≤2 px from GT", "FP 2–10 px", "FP >10 px")
                if index == 0 else (None, None, None)
            )
            axes[0].bar(index, near, label=labels[0], color="#45866c")
            axes[0].bar(index, middle, bottom=near, label=labels[1], color="#d0a05a")
            axes[0].bar(index, far, bottom=near + middle, label=labels[2], color="#ba5d5d")
        axes[0].set(
            title="Mean FP distance composition (per image)",
            ylabel="Fraction of false-positive pixels (0–1)",
            xlabel="Manual foreground group",
            xticks=[0, 1], xticklabels=list(groups), ylim=(0, 1),
        )
        axes[0].legend(fontsize=8)
        near_bright = [
            np.mean([float(r["fp_local_contrast_above_tn_p90"]) for r in group])
            for group in groups.values()
        ]
        fp_confident = [
            confidence[name]["fp_probability_ge_0_9"]
            for name in ("low_foreground", "higher_foreground")
        ]
        axis_x = np.arange(2)
        axes[1].bar(axis_x - .18, near_bright, width=.36,
                    label="FP above TN local-contrast p90", color="#4a6b83")
        axes[1].bar(axis_x + .18, fp_confident, width=.36,
                    label="FP with model p ≥0.9", color="#a76b56")
        axes[1].axhline(.1, color="gray", linestyle="--", linewidth=1,
                        label="TN p90 reference: 0.10")
        axes[1].set(
            title="Locally bright and high-confidence FP (image means)",
            ylabel="Fraction of false-positive pixels (0–1)",
            xlabel="Manual foreground group",
            xticks=axis_x, xticklabels=list(groups), ylim=(0, 1),
        )
        axes[1].legend(fontsize=8)
        fig.suptitle("Source: frozen five-fold OOF masks and CPU-recomputed probabilities · 2026-10-02")
        fig.savefig(args.output_dir / "oof_fp_mechanisms.png", dpi=160)
        plt.close(fig)
    print(f"Saved {len(cases)} six-panel case images and aggregate chart to {args.output_dir}")


if __name__ == "__main__":
    main()
