"""Fit one final LD U-Net from scratch on all 36 labelled images.

Epoch count and deployment threshold come from a separately locked,
validation-only five-fold policy. No training-set score or outer-test result
is used to choose either value during this full-data fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tifffile

from ld_final_policy import BASELINE_CANDIDATE_ID
from ld_train import CODE_RELATIVE_PATHS, _code_sha256_manifest, _read_frozen_map
from pipeline_config import PipelineConfig
from tiff_io import pair_tiff_directories, validate_pairs


FINAL_SOURCE_FILES = tuple(dict.fromkeys((
    *CODE_RELATIVE_PATHS,
    "code/ld_final_policy.py", "code/ld_train_final.py", "code/ld_predict.py",
    "code/ld_review_queue.py",
)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _validate_keras_archive(path: Path) -> None:
    """Reject a truncated or CRC-damaged Keras archive before publication."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Final LD checkpoint was not saved")
    try:
        with zipfile.ZipFile(path) as archive:
            required = {"metadata.json", "config.json", "model.weights.h5"}
            if not required.issubset(archive.namelist()):
                raise RuntimeError("Final LD checkpoint lacks Keras model entries")
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise RuntimeError(f"Final LD checkpoint has a corrupt member: {corrupt_member}")
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Final LD checkpoint is not a readable Keras archive") from exc


def _load_and_validate_policy(
    policy_path: Path, fold_map_path: Path, label_manifest_path: Path
) -> dict[str, object]:
    """Reject a policy unrelated to these exact 36 images or code version."""
    with policy_path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    if (policy.get("target") != "ld"
            or policy.get("candidate_id") != BASELINE_CANDIDATE_ID
            or policy.get("policy_version") != "ld_validation_median_v1"
            or policy.get("selection_basis") != "five_fold_inner_validation_only"
            or policy.get("outer_test_used_for_selection") is not False
            or policy.get("image_count") != 36):
        raise ValueError("Expected a locked validation-only LD production policy")
    if (policy.get("fold_map_sha256") != _sha256(fold_map_path)
            or policy.get("label_manifest_sha256") != _sha256(label_manifest_path)):
        raise ValueError("Policy was derived from a different LD fold map or label version")
    settings = policy.get("scientific_settings")
    if not isinstance(settings, dict):
        raise ValueError("Production policy lacks scientific settings")
    if settings.get("architecture") != "unet_2d_baseline" or settings.get("loss") != "bce_dice":
        raise ValueError("This final-fit entry point supports the LD baseline only")
    if settings.get("code_sha256") != _code_sha256_manifest():
        raise ValueError("Current baseline code differs from the validation-locked code")
    epochs = policy.get("final_epoch_count")
    threshold = policy.get("deployment_threshold")
    if type(epochs) is not int or epochs < 1 or epochs > int(settings["epochs"]):
        raise ValueError("Invalid fixed final epoch count")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("Invalid validation-derived deployment threshold")
    records = policy.get("fold_validation_records")
    if (not isinstance(records, list) or len(records) != 5
            or {item.get("fold") for item in records} != set(range(5))):
        raise ValueError("Production policy requires all five validation folds")
    # Histories were hashed when the validation-only policy was created. The
    # policy does not trust outer-test scores or take epochs from the final fit.
    for item in records:
        if not isinstance(item.get("training_history_sha256"), str) or len(
            item["training_history_sha256"]
        ) != 64:
            raise ValueError("Production policy lacks a fold history hash")
    return policy


def _validate_all_labels(
    pairs: list, fold_map_path: Path, label_manifest_path: Path
) -> list[dict[str, int | float | str]]:
    """Validate all 36 pairs because no held-out labels exist in final fitting."""
    frozen = json.loads(fold_map_path.read_text(encoding="utf-8"))
    _read_frozen_map(fold_map_path, {pair.filename for pair in pairs}, 5)
    with label_manifest_path.open(encoding="utf-8") as handle:
        derived = json.load(handle)
    if (derived.get("rule_id") != "ld_antialias_midpoint_v1"
            or derived.get("label_semantics") != {"background": 0, "foreground": 1}):
        raise ValueError("Final LD fit requires the frozen midpoint binary labels")
    records = derived.get("files")
    if not isinstance(records, list):
        raise ValueError("Derived-label manifest lacks file records")
    by_name = {item["filename"]: item for item in records}
    if len(by_name) != 36 or set(by_name) != {pair.filename for pair in pairs}:
        raise ValueError("Derived-label manifest does not cover exactly the 36 pairs")
    for pair in pairs:
        name = pair.filename
        if _sha256(pair.image_path) != frozen["input_sha256"][name]:
            raise ValueError(f"Input TIFF differs from the frozen LD dataset: {name}")
        if _sha256(pair.label_path) != by_name[name]["derived_mask_sha256"]:
            raise ValueError(f"Derived LD label differs from its manifest: {name}")
    return validate_pairs(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A fresh directory, independent of CV and Mito outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    label_dir = args.label_dir.resolve()
    if label_dir == (input_dir.parent / "LD_output").resolve() or input_dir == label_dir:
        raise ValueError("Use versioned binary LD labels, not original Photoshop masks")
    policy_path = args.policy.resolve()
    fold_map_path = args.fold_map.resolve()
    label_manifest_path = args.label_manifest.resolve()
    policy = _load_and_validate_policy(policy_path, fold_map_path, label_manifest_path)
    pairs = pair_tiff_directories(input_dir, label_dir)
    if len(pairs) != 36:
        raise ValueError("Final LD fit needs exactly 36 image/label pairs")
    dataset_report = _validate_all_labels(pairs, fold_map_path, label_manifest_path)
    settings = policy["scientific_settings"]
    config = PipelineConfig(
        patch_size=int(settings["patch_size"]), overlap=int(settings["overlap"]),
        lower_percentile=float(settings["lower_percentile"]),
        upper_percentile=float(settings["upper_percentile"]),
        random_seed=int(settings["seed_base"]),
    )
    config.validate()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing LD run: {output_dir}")

    # Keep TensorFlow unavailable to read-only validation and --help. The
    # production fit itself requires exactly one visible, allocated GPU.
    import tensorflow as tf

    from losses import binary_crossentropy_dice_loss, soft_dice_coefficient
    from model import build_unet_2d
    from patch_dataset import RandomPatchSequence

    if len(tf.config.list_physical_devices("GPU")) != 1:
        raise RuntimeError("Final LD fit requires exactly one visible GPU")
    with tf.device("/GPU:0"):
        probe = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
    if "GPU" not in probe.device.upper():
        raise RuntimeError(f"TensorFlow GPU preflight failed: {probe.device}")

    output_dir.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parent.parent
    final_code_sha = {}
    for name in FINAL_SOURCE_FILES:
        source = project_root / name
        destination = output_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        final_code_sha[name] = _sha256(source)
        if _sha256(destination) != final_code_sha[name]:
            raise RuntimeError(f"Final-fit source changed while archiving: {name}")
    shutil.copyfile(policy_path, output_dir / "production_policy.json")
    _write_json(output_dir / "dataset_report.json", {"all_36": dataset_report})
    _write_json(output_dir / "final_fit_config.json", {
        "target": "ld", "image_count": 36, "policy_sha256": _sha256(policy_path),
        "fold_map_sha256": _sha256(fold_map_path),
        "label_manifest_sha256": _sha256(label_manifest_path),
        "input_dir": str(input_dir), "label_dir": str(label_dir),
        "final_source_sha256": final_code_sha,
        "epochs": int(policy["final_epoch_count"]),
        "scientific_settings": settings,
    })
    seed = config.random_seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sequence = RandomPatchSequence(
        pairs, batch_size=int(settings["batch_size"]),
        patch_size=config.patch_size,
        steps_per_epoch=int(settings["steps_per_epoch"]),
        lower_percentile=config.lower_percentile,
        upper_percentile=config.upper_percentile,
        positive_patch_probability=float(settings["positive_patch_probability"]),
        random_seed=seed, augment=True,
    )
    model = build_unet_2d()  # Fresh LD initialization; never reuse Mito weights.
    model.compile(optimizer=tf.keras.optimizers.Adam(float(settings["learning_rate"])),
                  loss=binary_crossentropy_dice_loss,
                  metrics=[soft_dice_coefficient])
    history = model.fit(
        sequence, epochs=int(policy["final_epoch_count"]),
        callbacks=[tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv"))],
    )
    if len(history.epoch) != policy["final_epoch_count"] or any(
        not np.isfinite(values).all() for values in history.history.values()
    ):
        raise RuntimeError("Final LD fit did not finish with finite metrics")
    model_path = output_dir / "final_ld_unet.keras"
    model.save(model_path)
    _validate_keras_archive(model_path)
    # The archive must also load and reproduce the in-memory model's output.
    # Use a tiny, fixed input to avoid an unnecessary full-image GPU workload.
    probe_image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(1, 32, 32, 1)
    original_output = np.asarray(model(probe_image, training=False))
    reloaded_model = tf.keras.models.load_model(model_path, compile=False)
    reloaded_output = np.asarray(reloaded_model(probe_image, training=False))
    if (reloaded_output.shape != probe_image.shape
            or not np.isfinite(reloaded_output).all()
            or not np.allclose(original_output, reloaded_output, rtol=1e-5, atol=1e-6)):
        raise RuntimeError("Final LD checkpoint failed model reload or inference verification")
    del reloaded_model
    threshold = float(policy["deployment_threshold"])
    _write_json(output_dir / "threshold.json", {
        "threshold": threshold, "selection_basis": policy["selection_basis"],
        "policy_sha256": _sha256(policy_path),
    })
    _write_json(output_dir / "environment.json", {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "numpy": np.__version__, "tifffile": tifffile.__version__,
        "gpu": [device.name for device in tf.config.list_physical_devices("GPU")],
    })
    _write_json(output_dir / "final_model_manifest.json", {
        "target": "ld", "status": "complete", "model_file": model_path.name,
        "model_sha256": _sha256(model_path),
        "policy_sha256": _sha256(output_dir / "production_policy.json"),
        "training_history_sha256": _sha256(output_dir / "training_history.csv"),
        "threshold": threshold,
        "tile_size": config.patch_size, "overlap": config.overlap,
        "prediction_batch_size": int(settings["prediction_batch_size"]),
        "lower_percentile": config.lower_percentile,
        "upper_percentile": config.upper_percentile,
        "semantic_labels": {"background": 0, "foreground": 1},
        "label_rule": "ld_antialias_midpoint_v1", "training_images": 36,
        "epochs_completed": len(history.epoch),
        "final_loss": float(history.history["loss"][-1]),
        "final_soft_dice_on_training_patches": float(
            history.history["soft_dice_coefficient"][-1]
        ),
    })
    print(f"LD_FINAL_TRAINING_COMPLETE epochs={len(history.epoch)} model={model_path}")


if __name__ == "__main__":
    main()
