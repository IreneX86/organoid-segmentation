"""Plot an input-only Nuc intensity audit without reading any segmentation masks.

This figure describes the existing image domain, not model accuracy. The
training/validation membership and source-image hashes come from the frozen
internal split. The ten target images are unlabeled and never enter training.

Example::

    python code/nuc_input_contrast_figure.py \
      --data-root ~/Desktop/Organoid \
      --output-prefix reports/assets/nuc_input_contrast_20261016
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile


GROUPS = ("training", "validation", "target_first_10")
COLORS = ("#207c90", "#aa6b2d", "#ad4563")
MID_INTENSITY_MIN = 0.15
MID_INTENSITY_MAX = 0.85
HIGH_PASS_SIGMA = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(image: np.ndarray) -> np.ndarray:
    """Match the existing per-image p1–p99.8 preprocessing exactly."""

    image_float = image.astype(np.float32, copy=False)
    low, high = np.percentile(image_float, [1.0, 99.8])
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image_float - low) / (high - low), 0, 1).astype(np.float32)


def _measure(path: Path) -> tuple[float, float]:
    image = tifffile.imread(path)
    if image.ndim != 2 or image.shape != (1200, 1200):
        raise ValueError(f"Expected 1200×1200 single-plane TIFF: {path}")
    normalized = _normalize(image)
    image_q90 = float(np.percentile(normalized, 90))
    mid = (normalized >= MID_INTENSITY_MIN) & (normalized <= MID_INTENSITY_MAX)
    if not np.any(mid):
        raise ValueError(f"No middle-intensity pixels in {path}")
    # The residual is an input-texture measurement, not a nucleus prediction.
    blurred = gaussian_filter(normalized, sigma=HIGH_PASS_SIGMA, mode="reflect")
    detail_q90 = float(np.percentile(np.abs(normalized - blurred)[mid], 90))
    return image_q90, detail_q90


def _collect(project_root: Path, data_root: Path) -> list[dict[str, object]]:
    manifest_path = project_root / "experiments/results/nuc_internal_split_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = manifest["train"]
    validation = manifest["validation"]
    if (len(training), len(validation)) != (133, 34):
        raise ValueError("The frozen Nuc split is not the expected 133/34 split")
    if len(set(training + validation)) != 167:
        raise ValueError("The frozen split contains duplicate image names")

    source_dir = data_root / "label_ref/Nuc/Nuc_input"
    target_dir = data_root / "organoid_data/cell_22375_NZ_100/fluo"
    entries = [
        ("training", name, source_dir / name) for name in training
    ] + [
        ("validation", name, source_dir / name) for name in validation
    ] + [
        ("target_first_10", f"fluo_{index:03}.tif",
         target_dir / f"fluo_{index:03}.tif") for index in range(1, 11)
    ]

    rows: list[dict[str, object]] = []
    for group, name, path in entries:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = _sha256(path)
        if group != "target_first_10" and actual_hash != manifest["input_sha256"][name]:
            raise ValueError(f"Frozen source-image hash mismatch: {name}")
        q90, detail = _measure(path)
        rows.append({
            "group": group,
            "filename": name,
            "input_sha256": actual_hash,
            "normalized_intensity_q90": q90,
            "midintensity_abs_highpass_q90": detail,
        })
    return rows


def _plot(rows: list[dict[str, object]], output_path: Path) -> None:
    labels = ("Training\n(n=133)", "Validation\n(n=34)", "Target, first 10\n(unlabeled)")
    measures = (
        ("normalized_intensity_q90", "Image intensity (90th percentile)"),
        ("midintensity_abs_highpass_q90", "Local detail (90th percentile)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), dpi=180)
    fig.patch.set_facecolor("#f8fafb")
    rng = np.random.default_rng(20261016)  # Jitter affects appearance only.
    for axis, (field, ylabel) in zip(axes, measures):
        axis.set_facecolor("white")
        values_by_group = [
            np.asarray([float(row[field]) for row in rows if row["group"] == group])
            for group in GROUPS
        ]
        for index, (values, color) in enumerate(zip(values_by_group, COLORS), start=1):
            x = index + rng.uniform(-0.18, 0.18, size=len(values))
            axis.scatter(x, values, s=18, color=color, alpha=0.58,
                         edgecolors="none", zorder=2)
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            axis.plot([index - 0.22, index + 0.22], [median, median],
                      color="#162b36", linewidth=2.4, zorder=4)
            axis.plot([index, index], [q25, q75], color="#162b36",
                      linewidth=5.0, alpha=0.65, zorder=3)
            axis.annotate(f"median {median:.4f}", (index, median),
                          xytext=(0, 9), textcoords="offset points",
                          ha="center", color="#162b36", fontsize=8.5,
                          fontweight="semibold")
        axis.set_xticks([1, 2, 3], labels)
        axis.set_xlim(0.55, 3.45)
        axis.set_ylabel(ylabel, labelpad=8)
        axis.grid(axis="y", color="#e3e9ed", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#b5c3ca")
        axis.tick_params(axis="both", labelsize=9)

    axes[0].set_title("Overall image brightness after normalization", loc="left",
                      fontweight="semibold", fontsize=11)
    axes[1].set_title("Fine-scale intensity structure", loc="left",
                      fontweight="semibold", fontsize=11)
    fig.suptitle("Nuc input-domain contrast audit", x=0.06, ha="left",
                 fontsize=17, fontweight="bold", color="#162b36")
    fig.text(0.06, 0.915,
             "Per-image p1–p99.8 normalization · local detail = |image − Gaussian σ2| "
             "on pixels with normalized intensity 0.15–0.85",
             fontsize=9.8, color="#40525c")
    fig.text(0.06, 0.045,
             "Input images only. Target images have no masks; these plots do not "
             "measure segmentation accuracy or establish a causal mechanism.",
             fontsize=9.3, color="#9b3c4f", fontweight="semibold")
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.08, right=0.97, wspace=0.27)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = _collect(args.project_root, args.data_root)
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    if csv_path.exists() or png_path.exists():
        raise FileExistsError("Audit outputs already exist; use a fresh output prefix")
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, png_path)
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        print(group, len(subset),
              "intensity_q90_median=", np.median([row["normalized_intensity_q90"]
                                                   for row in subset]),
              "local_detail_q90_median=", np.median([row["midintensity_abs_highpass_q90"]
                                                      for row in subset]))
    print("PNG:", png_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    sys.exit(main())
