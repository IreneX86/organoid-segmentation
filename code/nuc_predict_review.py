"""Generate same-grid Nuc review masks with the approved 2D postprocessing.

The low threshold is an expert-chosen review setting, not a validation-selected
accuracy claim. The frozen model also supplies high-confidence cores used only
to separate a few touching regions. Each TIFF is inferred independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

from nuc_postprocess import postprocess_nucleus_masks
from sliding_window import predict_sliding_window
from tiff_io import list_tiff_files, normalize_image, read_tiff_2d


MODEL_SHA256 = "c1191f00e327d6f73acc73f202a3a8e44895cb669bbfa2c6a1f3f664cc986b9e"
LOW_THRESHOLD = 0.03
CORE_THRESHOLD = 0.45
TILE_SIZE = 512
OVERLAP = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_inputs(input_dir: Path, limit: int | None) -> list[Path]:
    """Select deterministic TIFF names; ignore macOS AppleDouble sidecars."""

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer")
    files = [path for path in list_tiff_files(input_dir) if not path.name.startswith("._")]
    if not files:
        raise ValueError(f"No usable TIFF files found in {input_dir}")
    names = [path.name.lower() for path in files]
    if len(names) != len(set(names)):
        raise ValueError("Input filenames collide on case-insensitive filesystems")
    return files if limit is None else files[:limit]


def predict_folder(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    limit: int | None = None,
    prediction_batch_size: int = 4,
) -> Path:
    """Write a fresh, complete-on-success folder of uint8 0/255 review masks.

    An interrupted run leaves its new folder without a completion manifest.
    It is never resumed by replacing files; use another output directory.
    """

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("Output directory must be separate from the input TIFF directory")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Nuc checkpoint not found: {model_path}")
    if prediction_batch_size <= 0:
        raise ValueError("prediction_batch_size must be positive")

    files = select_inputs(input_dir, limit)
    model_sha256 = sha256_file(model_path)
    if model_sha256 != MODEL_SHA256:
        raise ValueError(
            "Checkpoint SHA256 differs from the reviewed Nuc model; "
            f"expected {MODEL_SHA256}, received {model_sha256}"
        )

    # Delay the heavy TensorFlow import until all cheap path/model checks pass.
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []

    for source in files:
        image = read_tiff_2d(source)
        normalized = normalize_image(image, lower_percentile=1.0, upper_percentile=99.8)
        probability = predict_sliding_window(
            model,
            normalized,
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            prediction_batch_size=prediction_batch_size,
        )
        if probability.shape != image.shape or not np.isfinite(probability).all():
            raise RuntimeError(f"Invalid probability map for {source.name}")

        # One probability map supplies both masks, so the 0.45 cores and 0.03
        # candidate pixels always share identical image coordinates.
        final_mask, processing = postprocess_nucleus_masks(
            probability >= LOW_THRESHOLD,
            probability >= CORE_THRESHOLD,
        )
        if final_mask.shape != image.shape or final_mask.dtype != np.bool_:
            raise RuntimeError(f"Invalid Nuc postprocessing result for {source.name}")
        encoded = final_mask.astype(np.uint8) * 255
        destination = output_dir / source.name
        with destination.open("xb") as stream:
            tifffile.imwrite(stream, encoded, photometric="minisblack", metadata=None)
        readback = read_tiff_2d(destination)
        if readback.dtype != np.uint8 or not np.array_equal(readback, encoded):
            raise RuntimeError(f"Saved mask failed readback verification: {destination}")

        rows.append(
            {
                "filename": source.name,
                "input_sha256": sha256_file(source),
                "output_sha256": sha256_file(destination),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "foreground_pixels": int(final_mask.sum()),
                "postprocessing": processing,
            }
        )
        print(f"Saved {destination} ({len(rows)}/{len(files)})", flush=True)

    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "nuc_review_2d_low003_core045_min500_close4_curved_split_v1",
        "scope": "automatic masks for expert review, not manual ground truth",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_sha256": model_sha256,
        "low_threshold": LOW_THRESHOLD,
        "core_threshold": CORE_THRESHOLD,
        "normalization_percentiles": [1.0, 99.8],
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "prediction_batch_size": prediction_batch_size,
        "foreground_value": 255,
        "files": rows,
    }
    manifest_path = output_dir / "prediction_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional deterministic first-N preview; omit to process every TIFF.",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    args = parser.parse_args()
    manifest_path = predict_folder(
        args.input_dir,
        args.output_dir,
        args.model,
        limit=args.limit,
        prediction_batch_size=args.prediction_batch_size,
    )
    print(f"Completed Nuc review prediction: {manifest_path}")


if __name__ == "__main__":
    main()
