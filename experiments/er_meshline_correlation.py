"""Read-only correlations for exploratory ER masks and source fluorescence.

Binary-mask/continuous-intensity Pearson is the point-biserial correlation.
Neither this quantity nor Spearman measures segmentation accuracy without GT.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "outputs/er_random5_profiled_20261028/originals_masks_correlation"
RUN = ROOT / "outputs/er_meshline_pilot_20261028"
sys.path.insert(0, str(ROOT / "outputs/er_random5_profiled_20261028"))
from export_sources_and_correlation import tissue_support  # noqa: E402


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 3:
        raise ValueError("Expected matching arrays with at least 3 pixels")
    xm, ym = x.mean(), y.mean()
    xd, yd = x - xm, y - ym
    denominator = np.sqrt(np.dot(xd, xd) * np.dot(yd, yd))
    return float(np.dot(xd, yd) / denominator) if denominator else None


def mean_tied_ranks(values: np.ndarray) -> np.ndarray:
    """Exact rank transform for integer microscopy pixels, including ties."""
    a = np.asarray(values)
    if a.dtype.kind not in "ui" or int(a.min()) < 0:
        raise TypeError("Expected nonnegative integer TIFF intensity")
    counts = np.bincount(a.ravel())
    starts = np.cumsum(counts, dtype=np.int64) - counts
    midpoint = starts + (counts + 1) / 2.0
    return midpoint[a]


def summarize(rows: list[dict], key: str) -> dict:
    values = np.array([row[key] for row in rows if row[key] is not None], dtype=float)
    return {"n": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max())}


def main() -> None:
    manifest = json.loads((RUN / "manifest.json").read_text())
    export = json.loads((EXPORT / "correlation_results.json").read_text())
    records = {(x["cell"], x["z_one_based"]): x for x in export["slices"]}
    rows: list[dict] = []
    for candidate in manifest["results"]:
        cell, z = candidate["cell"], candidate["z_one_based"]
        original = records[cell, z]
        er = tifffile.imread(original["er_original"])
        fluo = tifffile.imread(original["fluo_original"])
        masks = {"old": tifffile.imread(original["line_mask"]) > 0,
                 "new": tifffile.imread(candidate["mask"]) > 0}
        if not (er.shape == fluo.shape == masks["old"].shape == masks["new"].shape):
            raise ValueError(f"Shape mismatch: {cell} z{z}")
        scopes = {"full_image": np.ones(er.shape, bool),
                  "fluo_tissue_support": tissue_support(fluo)}
        for scope_name, scope in scopes.items():
            images = {"er": er[scope], "fluo": fluo[scope]}
            ranks = {name: mean_tied_ranks(img) for name, img in images.items()}
            er_fluo_pearson = pearson(images["er"], images["fluo"])
            er_fluo_spearman = pearson(ranks["er"], ranks["fluo"])
            for mask_name, whole_mask in masks.items():
                mask = whole_mask[scope]
                row = {"cell": cell, "z_one_based": z, "profile": candidate["profile"],
                       "scope": scope_name, "mask": mask_name, "n_pixels": int(scope.sum()),
                       "mask_fraction": float(mask.mean()),
                       "er_fluo_pearson": er_fluo_pearson,
                       "er_fluo_spearman": er_fluo_spearman}
                for name, image in images.items():
                    row[f"mask_{name}_pearson"] = pearson(mask, image)
                    row[f"mask_{name}_spearman"] = pearson(mask, ranks[name])
                    row[f"{name}_mean_mask"] = float(image[mask].mean()) if mask.any() else None
                    row[f"{name}_mean_other"] = float(image[~mask].mean()) if (~mask).any() else None
                rows.append(row)
        print(f"{cell} z{z} complete", flush=True)
    csv_path = RUN / "correlation_metrics.csv"
    json_path = RUN / "correlation_summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Correlation outputs already exist; refusing overwrite")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"definition": "Pearson(mask, intensity) is point-biserial r; Spearman uses exact tied intensity ranks",
               "caveat": "Spatial intensity association is not ER segmentation accuracy; pixels and nearby slices are not independent replicates",
               "n_slices": len(manifest["results"]), "metrics": {}}
    for scope in ("full_image", "fluo_tissue_support"):
        for mask in ("old", "new"):
            group = [r for r in rows if r["scope"] == scope and r["mask"] == mask]
            summary["metrics"][f"{scope}/{mask}"] = {
                key: summarize(group, key) for key in
                ("mask_er_pearson", "mask_er_spearman", "mask_fluo_pearson",
                 "mask_fluo_spearman", "mask_fraction", "er_fluo_pearson")}
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
