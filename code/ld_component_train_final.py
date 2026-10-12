"""Fit the selected LD component-uniform method on all 36 labelled images.

This entry point is deliberately separate from the frozen baseline final fit.
It accepts only the validation-locked global policy selecting the component
candidate. The sole scientific change is its positive-crop center sampler;
no CV checkpoint, outer-test score, or Mito weight is used in this fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import zipfile
from pathlib import Path

import numpy as np
import tifffile

from ld_component_train import CODE_RELATIVE_PATHS, _code_sha256_manifest, _read_frozen_map
from ld_method_policy import build_method_policy
from pipeline_config import PipelineConfig
from tiff_io import pair_tiff_directories, validate_pairs


CANDIDATE_ID = "component_uniform_positive_center_v1"
PROTOCOL_PATH = "experiments/ld_production_selection_policy.md"
FINAL_SOURCE_FILES = tuple(dict.fromkeys((
    *CODE_RELATIVE_PATHS,
    "code/ld_component_train_final.py", "code/ld_method_policy.py",
    "code/ld_predict.py", "experiments/ld_production_selection_policy.md",
)))
REQUIRED_GLOBAL_CHECKS = (
    "candidate_gate_wins_at_least_4_of_5",
    "all_36_mean_paired_dice_delta_at_least_0_01",
    "each_fold_mean_precision_delta_at_least_minus_0_03",
    "no_individual_dice_delta_below_minus_0_10",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validated_policy(
    policy_path: Path,
    baseline_cv_root: Path,
    candidate_cv_root: Path,
    fold_map_path: Path,
    label_manifest_path: Path,
    label_dir: Path,
) -> dict[str, object]:
    """Fail closed unless the locked validation rule selected this candidate."""
    policy = _read_json(policy_path)
    if (policy.get("policy_version") != "ld_global_validation_method_v1"
            or policy.get("target") != "ld"
            or policy.get("selected_method") != "candidate"
            or policy.get("candidate_id") != CANDIDATE_ID
            or policy.get("selection_basis") != "predeclared_five_fold_inner_validation_only"
            or policy.get("outer_test_used_for_selection") is not False
            or policy.get("image_count") != 36):
        raise ValueError("Final candidate fit requires a validation-locked LD candidate policy")
    project_root = Path(__file__).resolve().parent.parent
    if (policy.get("fold_map_sha256") != _sha256(fold_map_path)
            or policy.get("label_manifest_sha256") != _sha256(label_manifest_path)
            or policy.get("policy_protocol_sha256") != _sha256(project_root / PROTOCOL_PATH)):
        raise ValueError("Production policy differs from the frozen LD data or selection rule")
    checks = policy.get("global_checks")
    gates = policy.get("fold_gate_records")
    if (not isinstance(checks, dict)
            or set(checks) != set(REQUIRED_GLOBAL_CHECKS)
            or not all(checks[key] is True for key in REQUIRED_GLOBAL_CHECKS)
            or not isinstance(gates, list) or len(gates) != 5
            or {record.get("fold") for record in gates} != set(range(5))
            or sum(record.get("gate_selected") == "candidate" for record in gates) < 4
            or policy.get("gate_wins") != sum(
                record.get("gate_selected") == "candidate" for record in gates
            )):
        raise ValueError("The locked global validation gate did not select this method")
    settings = policy.get("scientific_settings")
    if not isinstance(settings, dict):
        raise ValueError("Candidate policy lacks scientific settings")
    fixed_settings = {
        "architecture": "unet_2d_baseline", "loss": "bce_dice",
        "positive_center_policy": "uniform_8_connected_training_component_then_pixel",
        "seed_base": 42, "patch_size": 512, "overlap": 128,
        "batch_size": 4, "prediction_batch_size": 4,
        "steps_per_epoch": 100, "epochs": 100,
        "learning_rate": 1e-4, "positive_patch_probability": 0.6,
        "lower_percentile": 1.0, "upper_percentile": 99.8,
        "checkpoint_selection": "validation_full_image_dice_at_0.5",
        "threshold_selection": "validation_mean_per_image_dice",
    }
    if any(settings.get(key) != value for key, value in fixed_settings.items()):
        raise ValueError("Candidate policy changes a frozen scientific training setting")
    if settings.get("code_sha256") != _code_sha256_manifest():
        raise ValueError("Candidate source differs from validation-locked training code")
    records = policy.get("fold_validation_records")
    if (not isinstance(records, list) or len(records) != 5
            or {item.get("fold") for item in records} != set(range(5))):
        raise ValueError("Candidate policy requires all five validation fold records")
    epochs: list[int] = []
    thresholds: list[float] = []
    for item in records:
        epoch = item.get("validation_best_epoch_count")
        threshold = item.get("validation_selected_threshold")
        history_hash = item.get("training_history_sha256")
        if (type(epoch) is not int or not 1 <= epoch <= int(settings["epochs"])
                or not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                or not math.isfinite(float(threshold)) or not 0 < float(threshold) < 1
                or not isinstance(history_hash, str) or len(history_hash) != 64):
            raise ValueError("Candidate policy has an invalid validation epoch or threshold")
        epochs.append(epoch)
        thresholds.append(float(threshold))
    if (policy.get("final_epoch_count") != int(statistics.median(epochs))
            or policy.get("deployment_threshold") != float(statistics.median(thresholds))):
        raise ValueError("Final epochs or threshold differ from the five validation records")
    # Recompute the *entire* policy from the exact archived inner-validation
    # runs. A caller cannot forge a favorable global_checks field or swap a
    # fold record in the supplied JSON. build_method_policy never opens any
    # outer_test result; it verifies the derived labels and both CV locks.
    recomputed = build_method_policy(
        baseline_cv_root, candidate_cv_root, fold_map_path,
        label_manifest_path, label_dir,
    )
    if policy != recomputed:
        raise ValueError("Locked LD method policy differs from validation-only CV evidence")
    return policy


def _validate_all_pairs(
    pairs: list, fold_map_path: Path, label_manifest_path: Path
) -> list[dict[str, int | float | str]]:
    """Verify all input and derived-mask bytes before any final training."""
    names = {pair.filename for pair in pairs}
    frozen = _read_json(fold_map_path)
    _read_frozen_map(fold_map_path, names, 5)
    input_hashes = frozen.get("input_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != names:
        raise ValueError("Frozen LD map lacks exact hashes of all 36 inputs")
    derived = _read_json(label_manifest_path)
    if (derived.get("rule_id") != "ld_antialias_midpoint_v1"
            or derived.get("label_semantics") != {"background": 0, "foreground": 1}):
        raise ValueError("Expected frozen midpoint binary LD labels")
    files = derived.get("files")
    if not isinstance(files, list) or len(files) != 36:
        raise ValueError("Derived-label manifest must have exactly 36 records")
    by_name = {item["filename"]: item for item in files}
    if len(by_name) != 36 or set(by_name) != names:
        raise ValueError("Derived-label records differ from the exact 36 LD pairs")
    for pair in pairs:
        name = pair.filename
        if _sha256(pair.image_path) != input_hashes[name]:
            raise ValueError(f"LD input TIFF changed since fold-map freezing: {name}")
        if _sha256(pair.label_path) != by_name[name]["derived_mask_sha256"]:
            raise ValueError(f"Derived LD mask changed since label preparation: {name}")
    return validate_pairs(pairs)


def _validate_keras_archive(path: Path) -> None:
    """Reject incomplete Keras model archives before declaring completion."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Final LD candidate checkpoint was not saved")
    try:
        with zipfile.ZipFile(path) as archive:
            if not {"metadata.json", "config.json", "model.weights.h5"}.issubset(
                archive.namelist()
            ):
                raise RuntimeError("Final LD candidate checkpoint lacks Keras entries")
            member = archive.testzip()
            if member is not None:
                raise RuntimeError(f"Final LD candidate checkpoint has a corrupt member: {member}")
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Final LD candidate checkpoint is not a readable archive") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--baseline-cv-root", type=Path, required=True,
                        help="Verified five-fold baseline archive, used only for validation evidence.")
    parser.add_argument("--candidate-cv-root", type=Path, required=True,
                        help="Verified five-fold candidate archive, used only for validation evidence.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A fresh directory, independent of all CV/Mito runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    label_dir = args.label_dir.resolve()
    if input_dir == label_dir or label_dir == (input_dir.parent / "LD_output").resolve():
        raise ValueError("Use versioned derived LD labels, not original Photoshop masks")
    policy_path = args.policy.resolve()
    fold_map_path = args.fold_map.resolve()
    label_manifest_path = args.label_manifest.resolve()
    policy = _validated_policy(
        policy_path, args.baseline_cv_root.resolve(),
        args.candidate_cv_root.resolve(), fold_map_path,
        label_manifest_path, label_dir,
    )
    pairs = pair_tiff_directories(input_dir, label_dir)
    if len(pairs) != 36:
        raise ValueError("Candidate final fit requires exactly 36 LD image/mask pairs")
    dataset_report = _validate_all_pairs(pairs, fold_map_path, label_manifest_path)
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

    # Delay TensorFlow initialization until all small-file/data checks pass.
    import tensorflow as tf

    from ld_component_sequence import ComponentUniformPatchSequence
    from losses import binary_crossentropy_dice_loss, soft_dice_coefficient
    from model import build_unet_2d

    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != 1:
        raise RuntimeError("Final LD candidate fit requires exactly one visible GPU")
    with tf.device("/GPU:0"):
        gpu_probe = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
    if "GPU" not in gpu_probe.device.upper() or len(tf.config.list_logical_devices("GPU")) != 1:
        raise RuntimeError(f"TensorFlow GPU preflight failed: {gpu_probe.device}")

    output_dir.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parent.parent
    source_hashes: dict[str, str] = {}
    for name in FINAL_SOURCE_FILES:
        source = project_root / name
        destination = output_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[name] = _sha256(source)
        if _sha256(destination) != source_hashes[name]:
            raise RuntimeError(f"Source changed while archiving final LD fit: {name}")
    shutil.copyfile(policy_path, output_dir / "production_policy.json")
    _write_json(output_dir / "dataset_report.json", {"all_36": dataset_report})
    _write_json(output_dir / "final_fit_config.json", {
        "target": "ld", "candidate_id": CANDIDATE_ID, "image_count": 36,
        "policy_sha256": _sha256(policy_path),
        "fold_map_sha256": _sha256(fold_map_path),
        "label_manifest_sha256": _sha256(label_manifest_path),
        "input_dir": str(input_dir), "label_dir": str(label_dir),
        "final_source_sha256": source_hashes,
        "epochs": int(policy["final_epoch_count"]),
        "scientific_settings": settings,
    })
    seed = config.random_seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sequence = ComponentUniformPatchSequence(
        pairs, batch_size=int(settings["batch_size"]),
        patch_size=config.patch_size,
        steps_per_epoch=int(settings["steps_per_epoch"]),
        lower_percentile=config.lower_percentile,
        upper_percentile=config.upper_percentile,
        positive_patch_probability=float(settings["positive_patch_probability"]),
        random_seed=seed, augment=True,
    )
    _write_json(output_dir / "training_component_counts.json", {
        "scope": "all_36_final_training_masks",
        "eight_connected_components_per_image": {
            filename: table.count for filename, table in sequence.components.items()
        },
    })
    model = build_unet_2d()  # Fresh LD initialization, never Mito or CV weights.
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
        raise RuntimeError("Final LD candidate fit did not finish with finite metrics")
    model_path = output_dir / "final_ld_component_unet.keras"
    model.save(model_path)
    _validate_keras_archive(model_path)
    probe_image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(1, 32, 32, 1)
    original_output = np.asarray(model(probe_image, training=False))
    reloaded_model = tf.keras.models.load_model(model_path, compile=False)
    reloaded_output = np.asarray(reloaded_model(probe_image, training=False))
    if (reloaded_output.shape != probe_image.shape
            or not np.isfinite(reloaded_output).all()
            or not np.allclose(original_output, reloaded_output, rtol=1e-5, atol=1e-6)):
        raise RuntimeError("Final LD candidate checkpoint failed reload/inference verification")
    del reloaded_model
    threshold = float(policy["deployment_threshold"])
    _write_json(output_dir / "threshold.json", {
        "threshold": threshold, "selection_basis": policy["selection_basis"],
        "policy_sha256": _sha256(policy_path),
    })
    _write_json(output_dir / "environment.json", {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "numpy": np.__version__, "tifffile": tifffile.__version__,
        "gpu": [device.name for device in physical],
    })
    _write_json(output_dir / "final_model_manifest.json", {
        "target": "ld", "status": "complete", "candidate_id": CANDIDATE_ID,
        "model_file": model_path.name, "model_sha256": _sha256(model_path),
        "policy_sha256": _sha256(output_dir / "production_policy.json"),
        "training_history_sha256": _sha256(output_dir / "training_history.csv"),
        "threshold": threshold,
        "tile_size": config.patch_size, "overlap": config.overlap,
        "prediction_batch_size": int(settings["prediction_batch_size"]),
        "lower_percentile": config.lower_percentile,
        "upper_percentile": config.upper_percentile,
        "semantic_labels": {"background": 0, "foreground": 1},
        "label_rule": "ld_antialias_midpoint_v1", "training_images": 36,
        "positive_center_policy": settings["positive_center_policy"],
        "epochs_completed": len(history.epoch),
        "final_loss": float(history.history["loss"][-1]),
        "final_soft_dice_on_training_patches": float(
            history.history["soft_dice_coefficient"][-1]
        ),
    })
    print(f"LD_COMPONENT_FINAL_TRAINING_COMPLETE epochs={len(history.epoch)} "
          f"model={model_path}")


if __name__ == "__main__":
    main()
