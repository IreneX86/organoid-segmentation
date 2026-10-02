"""Describe frozen OOF probability confidence at existing FP/FN pixels.

This is diagnostic only: it never scans or selects a new threshold. Probability
maps must come from each image's frozen outer-fold checkpoint. Outputs are
derived per-image statistics, never raw biological image pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from tiff_io import read_binary_mask


def _mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.audit_csv.open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    rows = []
    for audited in audit_rows:
        filename = audited["filename"]
        fold = int(audited["fold"])
        truth = read_binary_mask(args.mask_dir / filename).astype(bool)
        predicted = read_binary_mask(
            args.baseline_root / f"fold_{fold}" / "predictions" / "validation" / filename
        ).astype(bool)
        p = np.load(args.probability_dir / f"{Path(filename).stem}_probability.npy")
        if p.shape != truth.shape or p.shape != predicted.shape:
            raise ValueError(f"Geometry mismatch for {filename}")
        fp = predicted & ~truth
        fn = truth & ~predicted
        fp_distance = ndi.distance_transform_edt(~truth)
        fp_p = p[fp]
        fn_p = p[fn]
        distant_fp_p = p[fp & (fp_distance > 10)]
        near_fp_p = p[fp & (fp_distance <= 2)]
        rows.append({
            "filename": filename,
            "fold": fold,
            "foreground_fraction": float(audited["foreground_fraction"]),
            "dice": float(audited["dice"]),
            "fp_probability_median": float(np.median(fp_p)) if fp_p.size else 0.0,
            "fn_probability_median": float(np.median(fn_p)) if fn_p.size else 0.0,
            "fp_probability_ge_0_9": _mean_or_zero(fp_p >= .9),
            "fp_probability_ge_0_99": _mean_or_zero(fp_p >= .99),
            "fn_probability_le_0_1": _mean_or_zero(fn_p <= .1),
            "near_fp_probability_ge_0_9": _mean_or_zero(near_fp_p >= .9),
            "distant_fp_probability_ge_0_9": _mean_or_zero(distant_fp_p >= .9),
        })
    if len(rows) != 36:
        raise ValueError("Expected exactly 36 frozen OOF cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "confidence_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields = [field for field in rows[0] if field not in {
        "filename", "fold", "foreground_fraction", "dice"
    }]
    summary = {}
    for name, members in (
        ("low_foreground", [r for r in rows if r["foreground_fraction"] <= .05]),
        ("higher_foreground", [r for r in rows if r["foreground_fraction"] > .05]),
    ):
        summary[name] = {
            "image_count": len(members),
            **{field: float(np.mean([r[field] for r in members])) for field in fields},
        }
    (args.output_dir / "confidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
