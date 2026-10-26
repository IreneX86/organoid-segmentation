"""Predict same-name, full-resolution Golgi masks with the 2026-10-26 pilot.

The checkpoint's validation threshold and image normalization are fixed here.
This is a 2D semantic mask, not Golgi instance segmentation or 3D tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sliding_window import predict_sliding_window
from tiff_io import list_tiff_files, normalize_image, read_tiff_2d, write_mask_tiff


PILOT_SHA256 = "6f248ca44be752ccef3425e8cf7c2f953c2b1575f9eda7e719372ae83a263772"
PATCH_SIZE = 512
OVERLAP = 128
THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.prediction_batch_size < 1:
        raise ValueError("prediction-batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != PILOT_SHA256:
        raise ValueError("Checkpoint SHA256 differs from the documented Golgi pilot")
    sources = [
        path for path in list_tiff_files(args.input_dir)
        if not path.name.startswith("._")
    ]
    if not sources:
        raise ValueError("No input TIFF files found")
    if args.output_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing output directory")

    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = tf.keras.models.load_model(args.checkpoint, compile=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    records = []
    for source in sources:
        image = read_tiff_2d(source)
        normalized = normalize_image(image, 1.0, 99.8)
        probability = predict_sliding_window(
            model,
            normalized,
            tile_size=PATCH_SIZE,
            overlap=OVERLAP,
            prediction_batch_size=args.prediction_batch_size,
        )
        if probability.shape != image.shape or not np.isfinite(probability).all():
            raise RuntimeError(f"Invalid probability map for {source.name}")
        mask = probability >= THRESHOLD
        output = args.output_dir / source.name
        write_mask_tiff(output, mask, foreground_value=255)
        records.append({
            "filename": source.name,
            "shape": list(image.shape),
            "foreground_pixels": int(np.count_nonzero(mask)),
            "input_sha256": sha256(source),
            "output_sha256": sha256(output),
        })
        print(f"PREDICTED {source.name}", flush=True)

    with (args.output_dir / "prediction_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "checkpoint_sha256": checkpoint_sha,
            "normalization": "per-image p1-p99.8",
            "patch_size": PATCH_SIZE,
            "overlap": OVERLAP,
            "threshold": THRESHOLD,
            "mask_dtype": "uint8",
            "background": 0,
            "foreground": 255,
            "image_count": len(records),
            "images": records,
        }, stream, indent=2)


if __name__ == "__main__":
    main()
