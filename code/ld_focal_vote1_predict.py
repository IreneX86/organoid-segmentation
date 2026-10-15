"""Batch-predict independent 2D LD TIFFs with the review-selected 1/5 vote.

Five archived focal checkpoints each vote at their own *locked inner-validation*
threshold. Any positive vote produces foreground (one of five). This cutoff
was preferred by the user in a visual review, but has not passed independent
ensemble validation; it does not replace the frozen LD production baseline.

The output directory must be new. Re-running against a completed output only
verifies its manifest and returns without inference. Partial or mismatched
outputs are never adopted or overwritten. One model is loaded at a time to
keep peak inference memory low; hidden temporary vote maps preserve the exact
input pixel grid until the final output directory is published.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sliding_window import predict_sliding_window
from tiff_io import list_tiff_files, normalize_image, read_tiff_2d, write_mask_tiff


MODEL_ID = "ld_focal_dice_gamma2_v1"
METHOD = "five_fold_binary_vote_at_locked_validation_thresholds_one_of_five"
SETTINGS = {
    "patch_size": 512,
    "overlap": 128,
    "prediction_batch_size": 4,
    "lower_percentile": 1.0,
    "upper_percentile": 99.8,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    checkpoint: Path
    checkpoint_sha256: str
    threshold: float
    threshold_sha256: str
    selection_lock_sha256: str
    config_sha256: str

    def public_record(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "checkpoint_sha256": self.checkpoint_sha256,
            "threshold": self.threshold,
            "threshold_sha256": self.threshold_sha256,
            "selection_lock_sha256": self.selection_lock_sha256,
            "config_sha256": self.config_sha256,
        }


def load_fold_specs(runs_root: Path) -> list[FoldSpec]:
    """Verify all five archived checkpoints and their validation-only locks."""

    root = runs_root.resolve()
    if (root / "ld").is_dir():
        root = root / "ld"
    folds: list[FoldSpec] = []
    for fold in range(5):
        run = root / f"fold_{fold}"
        checkpoint = run / "focal_unet.keras"
        threshold_file = run / "threshold.json"
        lock_file = run / "selection.lock.json"
        config_file = run / "experiment_config.json"
        split_file = run / "split.json"
        lock, config = _json(lock_file), _json(config_file)
        threshold_record, split = _json(threshold_file), _json(split_file)
        if (lock.get("target") != "ld" or lock.get("fold") != fold
                or config.get("target") != "ld" or config.get("fold") != fold
                or config.get("candidate") != MODEL_ID
                or config.get("loss") != "focal_bce_gamma2_plus_soft_dice"
                or config.get("num_folds") != 5
                or split.get("target") != "ld" or split.get("fold") != fold
                or split.get("num_folds") != 5):
            raise ValueError(f"Fold {fold} is not the locked LD focal run")
        if any(config.get(key) != value for key, value in SETTINGS.items()):
            raise ValueError(f"Fold {fold} uses different inference settings")
        hashes = {
            "checkpoint_sha256": _sha256(checkpoint),
            "threshold_sha256": _sha256(threshold_file),
            "experiment_config_sha256": _sha256(config_file),
            "split_sha256": _sha256(split_file),
        }
        if any(lock.get(field) != value for field, value in hashes.items()):
            raise ValueError(f"Fold {fold} differs from its selection lock")
        threshold = float(threshold_record["threshold"])
        if not 0 < threshold < 1 or lock.get("threshold") != threshold:
            raise ValueError(f"Fold {fold} has a mismatched validation threshold")
        folds.append(FoldSpec(
            fold=fold,
            checkpoint=checkpoint,
            checkpoint_sha256=hashes["checkpoint_sha256"],
            threshold=threshold,
            threshold_sha256=hashes["threshold_sha256"],
            selection_lock_sha256=_sha256(lock_file),
            config_sha256=hashes["experiment_config_sha256"],
        ))
    if len({fold.checkpoint_sha256 for fold in folds}) != 5:
        raise ValueError("Expected five distinct focal fold checkpoints")
    return folds


def _input_images(input_dir: Path) -> tuple[list[Path], int]:
    files = list_tiff_files(input_dir)
    images = [path for path in files if not path.name.startswith("._")]
    if not images or len({path.name.lower() for path in images}) != len(images):
        raise ValueError("Expected distinct real TIFF input filenames")
    return images, len(files) - len(images)


def _verify_complete_output(
    output_dir: Path,
    input_dir: Path,
    images: list[Path],
    fold_records: list[dict[str, object]],
    code_hash: str,
) -> dict[str, Any]:
    """Only a complete, fully matching run may be skipped on a repeat call."""

    manifest = _json(output_dir / "prediction_manifest.json")
    expected = {
        "status": "complete",
        "target": "ld",
        "scientific_status": "exploratory_human_review_preferred_not_independently_validated",
        "method": METHOD,
        "not_validated_as_ensemble": True,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "folds": fold_records,
        "settings": SETTINGS,
        "foreground_value": 255,
        "inference_code_sha256": code_hash,
        "image_count": len(images),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing LD ensemble output has a different or incomplete manifest")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != len(images):
        raise ValueError("Existing LD ensemble output manifest is incomplete")
    by_name = {item.get("filename"): item for item in records if isinstance(item, dict)}
    if set(by_name) != {path.name for path in images}:
        raise ValueError("Existing LD ensemble output filenames differ from inputs")
    actual_outputs = {p.name for p in list_tiff_files(output_dir)
                      if not p.name.startswith("._")}
    if actual_outputs != set(by_name):
        raise ValueError("Existing LD ensemble mask files differ from manifest")
    for path in images:
        record = by_name[path.name]
        mask_path = output_dir / path.name
        if (record.get("input_sha256") != _sha256(path)
                or record.get("mask_sha256") != _sha256(mask_path)):
            raise ValueError(f"Existing LD ensemble output failed verification: {path.name}")
        image = read_tiff_2d(path)
        mask = read_tiff_2d(mask_path)
        if (mask.shape != image.shape
                or mask.dtype != np.uint8
                or not np.isin(mask, (0, 255)).all()
                or record.get("height") != image.shape[0]
                or record.get("width") != image.shape[1]
                or record.get("foreground_pixels") != int(np.count_nonzero(mask))):
            raise ValueError(f"Existing LD ensemble output failed verification: {path.name}")
    return manifest


def predict_folder(
    runs_root: Path,
    input_dir: Path,
    output_dir: Path,
    model_loader: Callable[[Path], object],
    cleanup_model: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run/verify the 5-fold one-vote rule without touching prior masks."""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("LD output must be distinct from and outside the input directory")
    images, ignored_appledouble = _input_images(input_dir)
    folds = load_fold_specs(runs_root)
    fold_records = [fold.public_record() for fold in folds]
    code_hash = _sha256(Path(__file__).resolve())
    if output_dir.exists():
        return _verify_complete_output(
            output_dir, input_dir, images, fold_records, code_hash
        )
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {output_dir.parent}")

    # Keep a partial run hidden. The public destination appears only after all
    # same-size uint8 masks and provenance records have passed read-back.
    scratch = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.in_progress_", dir=output_dir.parent
    ))
    published = False
    try:
        vote_dir = scratch / "_votes"
        vote_dir.mkdir()
        for fold in folds:
            model = model_loader(fold.checkpoint)
            try:
                for number, path in enumerate(images, start=1):
                    raw = read_tiff_2d(path)
                    normalized = normalize_image(
                        raw, SETTINGS["lower_percentile"], SETTINGS["upper_percentile"]
                    )
                    probability = predict_sliding_window(
                        model, normalized,
                        tile_size=SETTINGS["patch_size"],
                        overlap=SETTINGS["overlap"],
                        prediction_batch_size=SETTINGS["prediction_batch_size"],
                    )
                    if probability.shape != raw.shape or not np.isfinite(probability).all():
                        raise RuntimeError(f"Invalid focal probability grid: {path.name}")
                    vote_file = vote_dir / f"{path.name}.npy"
                    if fold.fold == 0:
                        votes = np.lib.format.open_memmap(
                            vote_file, mode="w+", dtype=np.uint8, shape=raw.shape
                        )
                        votes[:] = 0
                    else:
                        votes = np.load(vote_file, mmap_mode="r+")
                        if votes.shape != raw.shape or votes.dtype != np.uint8:
                            raise RuntimeError(f"Vote grid changed: {path.name}")
                    votes[:] += probability >= fold.threshold
                    votes.flush()
                    del votes
                    print(f"LD_FOCAL_VOTE1 fold={fold.fold} image={number}/{len(images)} "
                          f"file={path.name}", flush=True)
            finally:
                del model
                gc.collect()
                if cleanup_model is not None:
                    cleanup_model()

        records: list[dict[str, object]] = []
        for path in images:
            raw = read_tiff_2d(path)
            votes = np.load(vote_dir / f"{path.name}.npy", mmap_mode="r")
            if votes.shape != raw.shape or np.any(votes > 5):
                raise RuntimeError(f"Invalid final vote count: {path.name}")
            # Review-selected union of the five validation-locked binary masks.
            # TIFF writer maps internal foreground=1 to white=255; background=0
            # remains black. Keep the original grid and do not resize the image.
            binary = (votes >= 1).astype(np.uint8)
            del votes
            destination = scratch / path.name
            write_mask_tiff(destination, binary, foreground_value=255)
            saved = read_tiff_2d(destination)
            if (saved.shape != raw.shape or saved.dtype != np.uint8
                    or not np.array_equal(saved, binary * 255)):
                raise RuntimeError(f"LD mask failed read-back: {path.name}")
            records.append({
                "filename": path.name,
                "height": int(raw.shape[0]),
                "width": int(raw.shape[1]),
                "input_sha256": _sha256(path),
                "mask_sha256": _sha256(destination),
                "foreground_pixels": int(binary.sum()),
            })
        shutil.rmtree(vote_dir)
        manifest: dict[str, Any] = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target": "ld",
            "scientific_status": "exploratory_human_review_preferred_not_independently_validated",
            "method": METHOD,
            "not_validated_as_ensemble": True,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "folds": fold_records,
            "settings": SETTINGS,
            "foreground_value": 255,
            "inference_code_sha256": code_hash,
            "image_count": len(images),
            "ignored_appledouble_files": ignored_appledouble,
            "files": records,
        }
        with (scratch / "prediction_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        if output_dir.exists():
            raise FileExistsError(f"Output appeared while predicting: {output_dir}")
        os.rename(scratch, output_dir)
        published = True
        return manifest
    finally:
        if not published:
            # Retain failed scratch output for diagnosis; never publish or
            # silently reuse it as a completed segmentation result.
            print(f"LD_FOCAL_VOTE1_PARTIAL_RETAINED path={scratch}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Archive root containing ld/fold_0 ... ld/fold_4.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Import TensorFlow only for a new inference run. A completed, matching
    # output can be verified and skipped without loading any checkpoints.
    if args.output_dir.resolve().exists():
        result = predict_folder(
            args.runs_root, args.input_dir, args.output_dir,
            model_loader=lambda _: (_ for _ in ()).throw(
                RuntimeError("Completed output verification must not load a model")
            ),
        )
    else:
        import tensorflow as tf

        result = predict_folder(
            args.runs_root, args.input_dir, args.output_dir,
            model_loader=lambda path: tf.keras.models.load_model(path, compile=False),
            cleanup_model=tf.keras.backend.clear_session,
        )
    print(f"LD_FOCAL_VOTE1_COMPLETE images={result['image_count']} "
          f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
