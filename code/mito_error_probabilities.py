"""CPU-only probability maps for fixed representative frozen-OOF Mito cases.

Case selection is descriptive and never changes a checkpoint or threshold. The
saved probability arrays and raw-image panels belong in an ignored local
directory; do not add them to Git or use them to tune on these held-out images.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sliding_window import predict_sliding_window
from tiff_io import normalize_image, read_binary_mask, read_tiff_2d


def select_cases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Take two Dice extremes from each foreground stratum, eight in total."""

    selected = []
    for stratum, members in (
        ("low_foreground", [r for r in rows if float(r["foreground_fraction"]) <= .05]),
        ("higher_foreground", [r for r in rows if float(r["foreground_fraction"]) > .05]),
    ):
        ordered = sorted(members, key=lambda r: float(r["dice"]))
        for kind, examples in (("lower_dice", ordered[:2]), ("higher_dice", ordered[-2:])):
            for row in examples:
                selected.append({
                    "filename": row["filename"],
                    "fold": row["fold"],
                    "stratum": stratum,
                    "dice_group": kind,
                    "observed_oof_dice": row["dice"],
                })
    if len(selected) != 8 or len({r["filename"] for r in selected}) != 8:
        raise ValueError("Representative case selection did not yield eight unique images")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--all-oof", action="store_true",
        help="Also export all 36 OOF probabilities for aggregate confidence diagnostics.",
    )
    args = parser.parse_args()
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("This audit must run locally on CPU, not use GPU workload")
    with args.audit_csv.open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    cases = (
        [{"filename": r["filename"], "fold": r["fold"]} for r in audit_rows]
        if args.all_oof else select_cases(audit_rows)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    active_fold = None
    for case in cases:
        fold = int(case["fold"])
        if fold != active_fold:
            model = tf.keras.models.load_model(
                args.baseline_root / f"fold_{fold}" / "baseline_unet.keras",
                compile=False,
            )
            active_fold = fold
        filename = case["filename"]
        image = normalize_image(read_tiff_2d(args.image_dir / filename))
        probability_path = args.output_dir / f"{Path(filename).stem}_probability.npy"
        probability = (
            np.load(probability_path)
            if probability_path.exists()
            else predict_sliding_window(
                model, image, tile_size=512, overlap=128, prediction_batch_size=1
            )
        )
        if probability.shape != image.shape or not np.isfinite(probability).all():
            raise ValueError(f"Invalid probability map for {filename}")
        threshold_path = args.baseline_root / f"fold_{fold}" / "threshold.json"
        threshold = float(json.loads(threshold_path.read_text())["threshold"])
        archived_mask = read_binary_mask(
            args.baseline_root / f"fold_{fold}" / "predictions" / "validation" / filename
        ).astype(bool)
        reconstructed = probability >= threshold
        # CPU and GPU TensorFlow kernels may differ around the threshold. Keep
        # archived OOF masks as the ground truth of the frozen evaluation.
        mismatch = int(np.count_nonzero(reconstructed != archived_mask))
        case["threshold"] = threshold
        case["cpu_vs_archived_mask_disagreement_pixels"] = mismatch
        if not probability_path.exists():
            np.save(probability_path, probability)
        print(filename, "fold", fold, "CPU/archived disagreement", mismatch, flush=True)
    manifest_name = "all_cases.json" if args.all_oof else "cases.json"
    (args.output_dir / manifest_name).write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
