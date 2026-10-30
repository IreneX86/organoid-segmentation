"""Plot paired fluorescence correlations from the eight-slice ER pilot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/er_meshline_pilot_20261028"


def main() -> None:
    rows = list(csv.DictReader((RUN / "correlation_metrics.csv").open()))
    rows = [r for r in rows if r["scope"] == "fluo_tissue_support"]
    keys = list(dict.fromkeys((r["cell"], r["z_one_based"]) for r in rows))
    labels = [f"{cell.split('_')[1][-5:]} Z{z}" for cell, z in keys]
    old = [next(r for r in rows if (r["cell"], r["z_one_based"], r["mask"]) == (*key, "old")) for key in keys]
    new = [next(r for r in rows if (r["cell"], r["z_one_based"], r["mask"]) == (*key, "new")) for key in keys]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    metrics = [("mask_er_pearson", "Mask vs ER intensity: Pearson r"),
               ("mask_fluo_pearson", "Mask vs Fluo intensity: Pearson r"),
               ("mask_fraction", "Mask coverage within Fluo tissue support")]
    x = np.arange(len(keys))
    for ax, (field, title) in zip(axes, metrics):
        a = [float(r[field]) for r in old]
        b = [float(r[field]) for r in new]
        ax.plot(x, a, marker="o", color="#33658A", label="Existing mask")
        ax.plot(x, b, marker="s", color="#B44593", label="Mesh-line candidate")
        for i in range(len(keys)):
            ax.plot([i, i], [a[i], b[i]], color="#999999", linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("Fraction" if field == "mask_fraction" else "r (unitless)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xticks(x, labels, rotation=30, ha="right")
    axes[-1].set_xlabel("Cell ID suffix and Z plane")
    fig.suptitle("Eight ER review slices · within Fluo-derived tissue support", fontsize=13)
    fig.savefig(RUN / "paired_correlation.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
