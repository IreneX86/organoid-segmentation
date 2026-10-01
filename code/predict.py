"""Batch prediction for a TIFF folder with same-name, same-size mask output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from sliding_window import predict_sliding_window
from tiff_io import (
    list_tiff_files,
    normalize_image,
    read_tiff_2d,
    write_mask_tiff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict same-name 2D semantic masks for every TIFF in a folder."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Defaults to threshold.json next to the model, otherwise 0.5.",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.8)
    parser.add_argument("--foreground-value", type=int, choices=(1, 255), default=255)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing existing output masks.",
    )
    return parser.parse_args()


def _load_threshold(model_path: Path, explicit_threshold: float | None) -> float:
    if explicit_threshold is not None:
        threshold = explicit_threshold
    else:
        threshold_path = model_path.parent / "threshold.json"
        if threshold_path.is_file():
            with threshold_path.open(encoding="utf-8") as handle:
                threshold = float(json.load(handle)["threshold"])
        else:
            threshold = 0.5
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    return threshold


def _preflight_outputs(
    input_files: list[Path],
    output_dir: Path,
    overwrite: bool,
) -> None:
    if not input_files:
        raise ValueError("No TIFF files found in the input directory")
    existing = [output_dir / path.name for path in input_files if (output_dir / path.name).exists()]
    if existing and not overwrite:
        preview = ", ".join(path.name for path in existing[:10])
        raise FileExistsError(
            f"{len(existing)} output file(s) already exist ({preview}). "
            "Use --overwrite to replace them explicitly."
        )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model.resolve()
    if input_dir == output_dir:
        raise ValueError("input-dir and output-dir must be different directories")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    input_files = list_tiff_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _preflight_outputs(input_files, output_dir, args.overwrite)
    threshold = _load_threshold(model_path, args.threshold)
    model = tf.keras.models.load_model(model_path, compile=False)

    manifest_entries = []
    for input_path in input_files:
        image = read_tiff_2d(input_path)
        normalized = normalize_image(
            image,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
        )
        probability = predict_sliding_window(
            model,
            normalized,
            tile_size=args.tile_size,
            overlap=args.overlap,
            prediction_batch_size=args.prediction_batch_size,
        )
        binary_mask = (probability >= threshold).astype(np.uint8)
        if binary_mask.shape != image.shape:
            raise RuntimeError(
                f"Prediction shape changed for {input_path.name}: "
                f"input={image.shape}, output={binary_mask.shape}"
            )

        output_path = output_dir / input_path.name
        write_mask_tiff(output_path, binary_mask, args.foreground_value)
        saved = read_tiff_2d(output_path)
        if saved.shape != image.shape:
            raise RuntimeError(
                f"Saved TIFF shape mismatch for {input_path.name}: "
                f"input={image.shape}, saved={saved.shape}"
            )
        allowed_values = {0, args.foreground_value}
        if not set(np.unique(saved).tolist()).issubset(allowed_values):
            raise RuntimeError(f"Saved TIFF has invalid mask values: {output_path}")

        manifest_entries.append(
            {
                "filename": input_path.name,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "foreground_fraction": float(binary_mask.mean()),
            }
        )
        print(f"Saved {output_path} with shape {binary_mask.shape}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": str(model_path),
        "threshold": threshold,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "foreground_value": args.foreground_value,
        "files": manifest_entries,
    }
    with (output_dir / "prediction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

