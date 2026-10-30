"""Exploratory Fluo-only U-Net trained against ER-derived pseudo-labels.

The ER source is deliberately absent from the training payload. Entire
organoids are assigned to train, validation, or test before fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import tarfile
from functools import lru_cache

import numpy as np
import tifffile


# Fixed before any student-model metric is observed. Both acquisition domains
# occur in training; validation and test are disjoint complete organoids.
VALIDATION_CELL = "cell_20260904_23963_NZ_40"
TEST_CELL = "cell_20260814_23300_NZ_100"
SEED = 20261028


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source_root: Path, mask_root: Path, destination: Path) -> None:
    """Package only Fluo + completed masks, without sending ER source TIFFs."""
    from er_ridge_batch_predict import pairs

    if destination.exists():
        raise FileExistsError(destination)
    discovered = pairs(source_root)
    if len(discovered) != 8:
        raise ValueError(f"Expected 8 aligned organoids, found {len(discovered)}")
    records = []
    for _er, fluo, relative in discovered:
        mask = mask_root / relative / "ER_mask_line.tif"
        run_manifest = mask.parent / "manifest.json"
        if not mask.is_file() or not run_manifest.is_file():
            raise FileNotFoundError(f"Completed mask and manifest required: {mask}")
        record = json.loads(run_manifest.read_text())
        if record["source_z_start_zero_based"] != 0 or record["source_z_stop_exclusive"] != record["source_shape"][0]:
            raise ValueError(f"Mask is not a complete stack: {mask}")
        if sha256(fluo) != record["source_fluo_sha256"] or sha256(mask) != record["outputs"]["ER_mask_line"]["sha256"]:
            raise ValueError(f"Source or mask hash changed: {relative}")
        with tifffile.TiffFile(fluo) as image_file, tifffile.TiffFile(mask) as mask_file:
            shape = image_file.series[0].shape
            if shape != mask_file.series[0].shape or len(shape) != 3:
                raise ValueError(f"Shape mismatch: {relative}")
            if len(image_file.pages) != shape[0] or len(mask_file.pages) != shape[0]:
                raise ValueError(f"Expected one TIFF page per Z: {relative}")
        name = fluo.parent.name
        split = "validation" if name == VALIDATION_CELL else "test" if name == TEST_CELL else "train"
        records.append({"cell": name, "domain": relative.parts[0].strip(), "split": split,
                        "shape_zyx": list(shape), "fluo": f"cells/{name}/fluo.tif",
                        "mask": f"cells/{name}/mask.tif", "fluo_sha256": record["source_fluo_sha256"],
                        "mask_sha256": record["outputs"]["ER_mask_line"]["sha256"],
                        "_local_fluo": str(fluo), "_local_mask": str(mask)})
    if [sum(row["split"] == split for row in records) for split in ("train", "validation", "test")] != [6, 1, 1]:
        raise ValueError("Unexpected grouped split")
    destination.parent.mkdir(parents=True, exist_ok=True)
    public_records = [{key: value for key, value in row.items() if not key.startswith("_local")}
                      for row in records]
    manifest = {"task": "Fluo-only ER pseudo-label imitation; not manual ground truth",
                "seed": SEED, "split_unit": "whole organoid", "records": public_records}
    temp = destination.with_suffix(destination.suffix + ".partial")
    try:
        with tarfile.open(temp, "w:gz", compresslevel=1) as archive:
            payload = json.dumps(manifest, indent=2).encode()
            import io
            info = tarfile.TarInfo("dataset_manifest.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            for row in records:
                archive.add(row["_local_fluo"], arcname=row["fluo"], recursive=False)
                archive.add(row["_local_mask"], arcname=row["mask"], recursive=False)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"archive": str(destination), "sha256": sha256(destination),
                      "cells": len(records), "slices": sum(r["shape_zyx"][0] for r in records),
                      "splits": {s: [r["cell"] for r in records if r["split"] == s]
                                 for s in ("train", "validation", "test")}}, indent=2))


@lru_cache(maxsize=1024)
def _plane(image_path: str, mask_path: str, z: int) -> tuple[np.ndarray, np.ndarray]:
    from tiff_io import normalize_image
    image = tifffile.imread(image_path, key=z)
    mask = tifffile.imread(mask_path, key=z)
    if image.shape != mask.shape or mask.dtype != np.uint8 or np.any((mask != 0) & (mask != 255)):
        raise ValueError(f"Invalid aligned binary mask: {mask_path} z={z}")
    return normalize_image(image), (mask > 0).astype(np.float32)


def weighted_bce_dice_5_loss(y_true, y_pred):
    """Prespecified recall-focused loss for ER's 2–5% foreground masks.

    Weight five is a rounded square-root class-ratio from *training* masks,
    not a parameter chosen on validation or test. Normalizing by total weight
    keeps the BCE scale stable across foreground-rich and sparse patches.
    """
    import tensorflow as tf
    from losses import soft_dice_coefficient

    truth = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(truth, y_pred)
    weights = 1.0 + 4.0 * truth[..., 0]
    balanced_bce = tf.reduce_sum(weights * bce) / tf.reduce_sum(weights)
    return balanced_bce + (1.0 - soft_dice_coefficient(truth, y_pred))


def train(data_root: Path, output_root: Path, *, max_epochs: int = 40,
          loss_variant: str = "baseline", validation_only: bool = False) -> None:
    import tensorflow as tf
    from model.unet import build_unet_2d
    from losses import binary_crossentropy_dice_loss, soft_dice_coefficient
    from patch_dataset import _augment_patch, _sample_patch
    from sliding_window import predict_sliding_window

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    rows = manifest["records"]
    if len(rows) != 8 or {r["split"] for r in rows} != {"train", "validation", "test"}:
        raise ValueError("Expected immutable eight-organoid grouped split")
    for row in rows:
        if validation_only and row["split"] == "test":
            # Test files stay in the immutable bundle, but this exploratory
            # candidate run must not inspect their pixels or labels.
            continue
        for kind in ("fluo", "mask"):
            path = data_root / row[kind]
            if not path.is_file() or sha256(path) != row[f"{kind}_sha256"]:
                raise ValueError(f"Dataset hash mismatch: {path}")
    gpu = tf.config.list_physical_devices("GPU")
    if len(gpu) != 1:
        raise RuntimeError(f"Exactly one assigned GPU required, saw {gpu}")
    tf.config.experimental.set_memory_growth(gpu[0], True)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:
        print(f"Determinism request unavailable: {exc}", flush=True)
    with tf.device("/GPU:0"):
        check = tf.matmul(tf.ones((128, 128)), tf.ones((128, 128)))
    if "GPU" not in check.device or float(check[0, 0]) != 128.0:
        raise RuntimeError(f"GPU matrix preflight failed: {check.device}")
    print("GPU_PREFLIGHT_PASS", gpu, check.device, flush=True)

    output_root.mkdir(parents=True)
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if loss_variant not in {"baseline", "weighted_bce_dice_5"}:
        raise ValueError(f"Unknown ER student loss: {loss_variant}")
    config = {"input": "Fluo only", "target": "full ER-derived pseudo-label",
              "model": "build_unet_2d", "base_filters": 16, "patch_size": 384,
              "batch_size": 4, "steps_per_epoch": 48, "max_epochs": max_epochs,
              "early_stopping_patience": 8, "learning_rate": 0.001,
              "loss": loss_variant, "threshold": 0.5,
              "validation_only": validation_only,
              "normalization": "per-plane percentiles 1-99.8", "seed": SEED,
              "split_unit": "whole organoid"}
    (output_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_root / "environment.json").write_text(json.dumps({
        "python": sys.version, "tensorflow": tf.__version__, "numpy": np.__version__,
        "tifffile": tifffile.__version__, "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [str(g) for g in gpu], "tf_build": tf.sysconfig.get_build_info()
    }, indent=2, default=str) + "\n")

    class Patches(tf.keras.utils.Sequence):
        def __init__(self, split: str, length: int):
            super().__init__()
            self.records = [r for r in rows if r["split"] == split]
            self.planes = [(r, z) for r in self.records for z in range(r["shape_zyx"][0])]
            self.split = split
            self.length = length
            self.epoch = 0

        def __len__(self) -> int:
            return self.length

        def on_epoch_end(self) -> None:
            if self.split == "train":
                self.epoch += 1

        def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
            rng = np.random.default_rng(SEED + index + (self.epoch * 100003 if self.split == "train" else 9917))
            images, masks = [], []
            for item in range(4):
                if self.split == "train":
                    row, z = self.planes[int(rng.integers(len(self.planes)))]
                else:
                    row, z = self.planes[(index * 4 + item) % len(self.planes)]
                image, mask = _plane(str(data_root / row["fluo"]), str(data_root / row["mask"]), z)
                image, mask = _sample_patch(image, mask, 384, rng, 0.5)
                if self.split == "train":
                    image, mask = _augment_patch(image, mask, rng)
                images.append(image[..., None])
                masks.append(mask[..., None])
            return np.stack(images), np.stack(masks)

    chosen_loss = (binary_crossentropy_dice_loss if loss_variant == "baseline"
                   else weighted_bce_dice_5_loss)
    model = build_unet_2d(1, 16)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=chosen_loss,
                  metrics=[soft_dice_coefficient])
    checkpoint = output_root / "er_fluo_student.keras"
    history = model.fit(
        Patches("train", 48), validation_data=Patches("validation", 10),
        epochs=max_epochs, verbose=2,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss",
                                                     save_best_only=True),
                   tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                     restore_best_weights=True),
                   tf.keras.callbacks.CSVLogger(str(output_root / "training_history.csv"))])
    model = tf.keras.models.load_model(checkpoint, compile=False)
    print("BEST_CHECKPOINT_LOCKED", flush=True)
    preview_root = output_root / "test_previews"
    preview_root.mkdir()
    with (output_root / "per_slice_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "cell", "z_zero_based", "dice", "iou", "precision", "recall", "foreground_ratio"])
        writer.writeheader()
        summary = {}
        validation_thresholds = {str(t): [] for t in (0.1, 0.2, 0.3, 0.4, 0.5)} if validation_only else {}
        for split in (("validation",) if validation_only else ("validation", "test")):
            metrics = []
            for row in rows:
                if row["split"] != split:
                    continue
                for z in range(row["shape_zyx"][0]):
                    image, target = _plane(str(data_root / row["fluo"]), str(data_root / row["mask"]), z)
                    probability = predict_sliding_window(model, image, tile_size=384,
                                                         overlap=96, prediction_batch_size=4)
                    predicted = probability >= 0.5
                    truth = target > 0
                    if split == "validation" and validation_only:
                        for threshold_text, scores in validation_thresholds.items():
                            candidate = probability >= float(threshold_text)
                            ctp = int(np.count_nonzero(candidate & truth))
                            cfp = int(np.count_nonzero(candidate & ~truth))
                            cfn = int(np.count_nonzero(~candidate & truth))
                            scores.append(1.0 if ctp + cfp + cfn == 0 else
                                          2 * ctp / (2 * ctp + cfp + cfn))
                    if split == "test" and z in {row["shape_zyx"][0] // 4,
                                                 row["shape_zyx"][0] // 2,
                                                 3 * row["shape_zyx"][0] // 4}:
                        tifffile.imwrite(preview_root / f"{row['cell']}_z{z:03d}_mask.tif",
                                         predicted.astype(np.uint8) * 255)
                        tifffile.imwrite(preview_root / f"{row['cell']}_z{z:03d}_probability.tif",
                                         probability.astype(np.float32))
                    tp = int(np.count_nonzero(predicted & truth))
                    fp = int(np.count_nonzero(predicted & ~truth))
                    fn = int(np.count_nonzero(~predicted & truth))
                    # Correctly score an empty prediction against an empty
                    # target instead of treating it as a missed ER network.
                    both_empty = tp + fp + fn == 0
                    dice = 1.0 if both_empty else (2 * tp) / (2 * tp + fp + fn)
                    result = {"split": split, "cell": row["cell"], "z_zero_based": z,
                              "dice": dice, "iou": 1.0 if both_empty else tp / (tp + fp + fn),
                              "precision": 1.0 if tp + fp == 0 and fn == 0 else tp / max(1, tp + fp),
                              "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
                              "foreground_ratio": float(np.mean(truth))}
                    writer.writerow(result)
                    metrics.append(result)
                print(f"EVAL_{split.upper()}_COMPLETE {row['cell']}", flush=True)
            summary[split] = {k: float(np.mean([m[k] for m in metrics]))
                              for k in ("dice", "iou", "precision", "recall")}
    if validation_only:
        threshold_means = {key: float(np.mean(scores)) for key, scores in validation_thresholds.items()}
        selected = max(threshold_means, key=lambda key: (threshold_means[key], float(key)))
        summary["validation_threshold_comparison"] = {
            "mean_per_slice_dice": threshold_means,
            "selected_threshold": float(selected),
            "selection_rule": "highest mean validation Dice; ties choose higher threshold",
        }
    summary["epochs_completed"] = len(history.history["loss"])
    summary["best_epoch_by_val_loss"] = int(np.argmin(history.history["val_loss"]) + 1)
    summary["checkpoint_sha256"] = sha256(checkpoint)
    summary["interpretation"] = ("Validation pseudo-label agreement only; locked test not read"
                                  if validation_only else
                                  "Pseudo-label agreement only; no independent manual ER test")
    (output_root / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ER_FLUO_STUDENT_COMPLETE", json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("prepare")
    pack.add_argument("--source-root", type=Path, required=True)
    pack.add_argument("--mask-root", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data-root", type=Path, required=True)
    fit.add_argument("--output-root", type=Path, required=True)
    fit.add_argument("--max-epochs", type=int, default=40)
    fit.add_argument("--loss-variant", choices=("baseline", "weighted_bce_dice_5"),
                     default="baseline")
    fit.add_argument("--validation-only", action="store_true",
                     help="Do not read the locked test organoid during model selection")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_root, args.mask_root, args.archive)
    else:
        train(args.data_root, args.output_root, max_epochs=args.max_epochs,
              loss_variant=args.loss_variant, validation_only=args.validation_only)


if __name__ == "__main__":
    main()
