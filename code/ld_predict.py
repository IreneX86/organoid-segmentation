"""Batch-infer independent LD TIFFs on their original pixel grid.

The model manifest fixes normalization, tiling, and threshold. No 3D ordering,
whole-image resizing, or model/threshold selection occurs during inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

from sliding_window import predict_sliding_window
from tiff_io import list_tiff_files, normalize_image, read_tiff_2d, write_mask_tiff


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_manifest(model_dir: Path) -> tuple[Path, dict[str, object]]:
    manifest_path = model_dir / "final_model_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (manifest.get("target") != "ld" or manifest.get("status") != "complete"
            or manifest.get("semantic_labels") != {"background": 0, "foreground": 1}):
        raise ValueError("Model directory does not hold a completed binary LD model")
    model_name = manifest.get("model_file")
    if not isinstance(model_name, str) or Path(model_name).name != model_name:
        raise ValueError("Model manifest contains an unsafe model filename")
    model_path = model_dir / model_name
    if _sha256(model_path) != manifest.get("model_sha256"):
        raise ValueError("LD checkpoint differs from its final model manifest")
    policy_path = model_dir / "production_policy.json"
    if _sha256(policy_path) != manifest.get("policy_sha256"):
        raise ValueError("LD production policy differs from its final model manifest")
    with policy_path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    with (model_dir / "threshold.json").open(encoding="utf-8") as handle:
        saved_threshold = json.load(handle)
    if (manifest.get("threshold") != policy.get("deployment_threshold")
            or saved_threshold.get("threshold") != manifest.get("threshold")):
        raise ValueError("LD inference threshold differs from the validation-only policy")
    settings = policy.get("scientific_settings")
    if not isinstance(settings, dict) or any(
        manifest.get(model_key) != settings.get(setting_key)
        for model_key, setting_key in (
            ("tile_size", "patch_size"), ("overlap", "overlap"),
            ("prediction_batch_size", "prediction_batch_size"),
            ("lower_percentile", "lower_percentile"),
            ("upper_percentile", "upper_percentile"),
        )
    ):
        raise ValueError("LD inference settings differ from the trained policy")
    # The predictor's geometry and TIFF encoding also live in these two
    # modules. Lock them along with this entry point so later unrelated code
    # edits cannot silently change the saved model's output pixel grid.
    for source_name in ("ld_predict.py", "sliding_window.py", "tiff_io.py"):
        source_snapshot = model_dir / "source_snapshot" / "code" / source_name
        current_source = Path(__file__).resolve().with_name(source_name)
        if _sha256(source_snapshot) != _sha256(current_source):
            raise ValueError(
                f"LD inference source {source_name} differs from the archived production code"
            )
    threshold = float(manifest["threshold"])
    tile_size = int(manifest["tile_size"])
    overlap = int(manifest["overlap"])
    prediction_batch_size = int(manifest["prediction_batch_size"])
    if not 0 < threshold < 1 or tile_size <= 0 or tile_size % 16:
        raise ValueError("Invalid model threshold or tile size")
    if not 0 <= overlap < tile_size:
        raise ValueError("Invalid model tiling overlap")
    if prediction_batch_size < 1:
        raise ValueError("Invalid model prediction batch size")
    return model_path, manifest


def predict_folder(
    model: object,
    input_dir: Path,
    output_dir: Path,
    manifest: dict[str, object],
    save_probabilities: bool = False,
    foreground_value: int = 255,
) -> dict[str, object]:
    """Save one same-named, same-sized mask for every real 2D TIFF in a folder."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir or output_dir.exists():
        raise FileExistsError("LD output directory must be new and distinct from input")
    if foreground_value not in (1, 255):
        raise ValueError("foreground_value must be 1 or 255")
    all_tiff = list_tiff_files(input_dir)
    # AppleDouble `._name.tif` files are macOS metadata, not microscopy images.
    images = [path for path in all_tiff if not path.name.startswith("._")]
    if not images:
        raise ValueError("No real TIFF images found in the input folder")
    threshold = float(manifest["threshold"])
    tile_size = int(manifest["tile_size"])
    overlap = int(manifest["overlap"])
    prediction_batch_size = int(manifest["prediction_batch_size"])
    lower = float(manifest["lower_percentile"])
    upper = float(manifest["upper_percentile"])
    if not 0 < threshold < 1 or not 0 <= lower < upper <= 100:
        raise ValueError("Invalid locked inference normalization or threshold")
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for path in images:
        image = read_tiff_2d(path)
        normalized = normalize_image(image, lower, upper)
        probability = predict_sliding_window(
            model, normalized, tile_size=tile_size, overlap=overlap,
            prediction_batch_size=prediction_batch_size,
        )
        if probability.shape != image.shape or not np.isfinite(probability).all():
            raise RuntimeError(f"LD probability grid is invalid: {path.name}")
        mask = (probability >= threshold).astype(np.uint8)
        mask_path = output_dir / path.name
        write_mask_tiff(mask_path, mask, foreground_value=foreground_value)
        saved = read_tiff_2d(mask_path)
        if (saved.shape != image.shape or saved.dtype != np.uint8
                or not np.array_equal(saved, mask * foreground_value)):
            raise RuntimeError(f"Saved LD mask failed read-back: {mask_path}")
        record: dict[str, object] = {
            "filename": path.name, "height": int(image.shape[0]),
            "width": int(image.shape[1]), "input_sha256": _sha256(path),
            "mask_sha256": _sha256(mask_path),
            "foreground_fraction": float(mask.mean()),
        }
        if save_probabilities:
            probability_path = output_dir / "probabilities" / path.name
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(probability_path, probability, metadata=None)
            if not np.array_equal(read_tiff_2d(probability_path), probability):
                raise RuntimeError(f"Saved LD probabilities failed read-back: {path.name}")
            record["probability_sha256"] = _sha256(probability_path)
        records.append(record)

    result: dict[str, object] = {
        "target": "ld", "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "model_sha256": manifest.get("model_sha256"),
        "inference_code_sha256": _sha256(Path(__file__).resolve()),
        "threshold": threshold, "tile_size": tile_size, "overlap": overlap,
        "prediction_batch_size": prediction_batch_size,
        "lower_percentile": lower, "upper_percentile": upper,
        "foreground_value": foreground_value,
        "semantic_labels_internal": {"background": 0, "foreground": 1},
        "image_count": len(records),
        "ignored_appledouble_files": len(all_tiff) - len(images),
        "probabilities_saved": save_probabilities,
        "files": records,
    }
    with (output_dir / "prediction_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--foreground-value", type=int, choices=(1, 255), default=255)
    args = parser.parse_args()
    model_path, manifest = _validate_model_manifest(args.model_dir.resolve())
    if args.output_dir.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    result = predict_folder(
        model, args.input_dir, args.output_dir, manifest,
        save_probabilities=args.save_probabilities,
        foreground_value=args.foreground_value,
    )
    print(f"LD_BATCH_PREDICTION_COMPLETE images={result['image_count']} "
          f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
