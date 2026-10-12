"""Synthetic, TensorFlow-free checks for LD final-policy and TIFF inference."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_final_policy import _validation_best_epoch, build_production_policy
from ld_predict import _validate_model_manifest, predict_folder
from ld_train import CODE_RELATIVE_PATHS, _code_sha256_manifest
from ld_train_final import FINAL_SOURCE_FILES, _load_and_validate_policy, _validate_keras_archive


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IdentityModel:
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return batch


class LdProductionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_best_epoch_uses_earliest_validation_maximum(self) -> None:
        path = self.root / "training_history.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "val_full_dice"])
            writer.writeheader()
            writer.writerows([
                {"epoch": 0, "val_full_dice": 0.2},
                {"epoch": 1, "val_full_dice": 0.7},
                {"epoch": 2, "val_full_dice": 0.7},
            ])
        self.assertEqual(_validation_best_epoch(path), (2, 0.7))

    def test_final_snapshot_includes_locked_baseline_sources(self) -> None:
        self.assertTrue(set(CODE_RELATIVE_PATHS).issubset(FINAL_SOURCE_FILES))
        self.assertEqual(len(FINAL_SOURCE_FILES), len(set(FINAL_SOURCE_FILES)))
        self.assertIn("code/ld_train_final.py", FINAL_SOURCE_FILES)
        self.assertIn("code/ld_predict.py", FINAL_SOURCE_FILES)

    def _fake_cv(self) -> tuple[Path, Path, Path]:
        names = [f"ld_{index:03d}.tif" for index in range(36)]
        assignments = {name: index % 5 for index, name in enumerate(names)}
        fold_map = self.root / "fold_map.json"
        label_manifest = self.root / "label_manifest.json"
        _write_json(fold_map, {
            "target": "ld", "num_folds": 5, "fold_ids": assignments,
        })
        _write_json(label_manifest, {"rule_id": "synthetic"})
        cv = self.root / "cv"
        for fold in range(5):
            directory = cv / "ld" / f"fold_{fold}"
            train = [name for name in names
                     if assignments[name] not in (fold, (fold + 1) % 5)]
            val = [name for name in names if assignments[name] == (fold + 1) % 5]
            test = [name for name in names if assignments[name] == fold]
            _write_json(directory / "split.json", {
                "target": "ld", "fold": fold, "fold_ids": assignments,
                "train": train, "validation": val, "test": test,
            })
            _write_json(directory / "experiment_config.json", {
                "target": "ld", "fold": fold, "outer_test_deferred": True,
                "architecture": "unet_2d_baseline", "loss": "bce_dice",
                "seed_base": 42, "patch_size": 512, "overlap": 128,
                "batch_size": 4, "prediction_batch_size": 4,
                "steps_per_epoch": 100, "epochs": 100, "learning_rate": 1e-4,
                "positive_patch_probability": 0.6,
                "lower_percentile": 1.0, "upper_percentile": 99.8,
                "checkpoint_selection": "validation_full_image_dice_at_0.5",
                "threshold_selection": "validation_mean_per_image_dice",
                "label_manifest_sha256": _sha256(label_manifest),
                "fold_map_sha256": _sha256(fold_map),
                "code_sha256": _code_sha256_manifest(),
            })
            _write_json(directory / "threshold.json", {"threshold": 0.4 + 0.05 * fold})
            _write_json(directory / "validation_artifacts" / "manifest.json", {
                "image_count": len(val),
            })
            (directory / "baseline_unet.keras").write_bytes(b"synthetic")
            with (directory / "training_history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "val_full_dice"])
                writer.writeheader()
                writer.writerows([
                    {"epoch": 0, "val_full_dice": 0.1},
                    {"epoch": fold + 1, "val_full_dice": 0.8},
                ])
            _write_json(directory / "selection.lock.json", {
                "target": "ld", "fold": fold,
                "fold_map_sha256": _sha256(fold_map),
                "label_manifest_sha256": _sha256(label_manifest),
                "experiment_config_sha256": _sha256(directory / "experiment_config.json"),
                "split_sha256": _sha256(directory / "split.json"),
                "checkpoint_sha256": _sha256(directory / "baseline_unet.keras"),
                "threshold_sha256": _sha256(directory / "threshold.json"),
                "validation_artifacts_manifest_sha256": _sha256(
                    directory / "validation_artifacts" / "manifest.json"
                ),
                "code_sha256": _code_sha256_manifest(),
                "threshold": 0.4 + 0.05 * fold,
            })
        return cv, fold_map, label_manifest

    def test_production_policy_uses_only_five_validation_records(self) -> None:
        cv, fold_map, labels = self._fake_cv()
        # Corrupt outer-test files must not affect a validation-only policy.
        for fold in range(5):
            path = cv / "ld" / f"fold_{fold}" / "outer_test" / "test_summary.json"
            path.parent.mkdir()
            path.write_text("deliberately unreadable JSON", encoding="utf-8")
        policy = build_production_policy(cv, fold_map, labels, "ld_baseline_locked")
        self.assertEqual(policy["final_epoch_count"], 4)
        self.assertAlmostEqual(policy["deployment_threshold"], 0.5)
        self.assertFalse(policy["outer_test_used_for_selection"])
        self.assertEqual(len(policy["fold_validation_records"]), 5)
        self.assertTrue(all(len(record["training_history_sha256"]) == 64
                            for record in policy["fold_validation_records"]))
        with self.assertRaisesRegex(ValueError, "supports only candidate_id"):
            build_production_policy(cv, fold_map, labels, "unverified_candidate")

    def test_keras_archive_integrity_is_checked(self) -> None:
        checkpoint = self.root / "final_ld_unet.keras"
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("metadata.json", "{}")
            archive.writestr("config.json", "{}")
            archive.writestr("model.weights.h5", b"synthetic weights")
        _validate_keras_archive(checkpoint)
        checkpoint.write_bytes(b"truncated archive")
        with self.assertRaisesRegex(RuntimeError, "readable Keras archive"):
            _validate_keras_archive(checkpoint)

    def test_final_fit_rejects_policy_with_changed_dataset(self) -> None:
        cv, fold_map, labels = self._fake_cv()
        policy = build_production_policy(cv, fold_map, labels, "ld_baseline_locked")
        path = self.root / "policy.json"
        _write_json(path, policy)
        self.assertEqual(
            _load_and_validate_policy(path, fold_map, labels)["final_epoch_count"], 4
        )
        _write_json(labels, {"rule_id": "changed"})
        with self.assertRaisesRegex(ValueError, "different LD fold map or label version"):
            _load_and_validate_policy(path, fold_map, labels)

    def test_folder_inference_preserves_names_and_pixel_grid(self) -> None:
        input_dir = self.root / "input"
        input_dir.mkdir()
        image = np.arange(31 * 47, dtype=np.uint16).reshape(31, 47)
        tifffile.imwrite(input_dir / "ld_001.tif", image)
        tifffile.imwrite(input_dir / "ld_002.tif", image[::-1])
        (input_dir / "._ld_003.tif").write_bytes(b"AppleDouble metadata")
        manifest = {
            "threshold": 0.5, "tile_size": 16, "overlap": 4,
            "prediction_batch_size": 2,
            "lower_percentile": 1.0, "upper_percentile": 99.8,
            "model_sha256": "synthetic",
        }
        output = self.root / "output"
        result = predict_folder(
            IdentityModel(), input_dir, output, manifest, save_probabilities=True
        )
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["ignored_appledouble_files"], 1)
        for name in ("ld_001.tif", "ld_002.tif"):
            mask = tifffile.imread(output / name)
            probability = tifffile.imread(output / "probabilities" / name)
            self.assertEqual(mask.shape, image.shape)
            self.assertEqual(probability.shape, image.shape)
            self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
        self.assertFalse((output / "._ld_003.tif").exists())
        with self.assertRaises(FileExistsError):
            predict_folder(IdentityModel(), input_dir, output, manifest)

    def test_model_manifest_requires_matching_checkpoint_and_policy(self) -> None:
        directory = self.root / "final_model"
        directory.mkdir()
        (directory / "final_ld_unet.keras").write_bytes(b"synthetic model")
        for source_name in ("ld_predict.py", "sliding_window.py", "tiff_io.py"):
            source = directory / "source_snapshot" / "code" / source_name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(Path(__file__).resolve().parents[1].joinpath(
                source_name
            ).read_bytes())
        _write_json(directory / "production_policy.json", {
            "target": "ld", "deployment_threshold": 0.5,
            "scientific_settings": {
                "patch_size": 512, "overlap": 128,
                "prediction_batch_size": 4,
                "lower_percentile": 1.0, "upper_percentile": 99.8,
            },
        })
        _write_json(directory / "threshold.json", {"threshold": 0.5})
        _write_json(directory / "final_model_manifest.json", {
            "target": "ld", "status": "complete",
            "semantic_labels": {"background": 0, "foreground": 1},
            "model_file": "final_ld_unet.keras",
            "model_sha256": _sha256(directory / "final_ld_unet.keras"),
            "policy_sha256": _sha256(directory / "production_policy.json"),
            "threshold": 0.5, "tile_size": 512, "overlap": 128,
            "prediction_batch_size": 4,
            "lower_percentile": 1.0, "upper_percentile": 99.8,
        })
        model_path, _ = _validate_model_manifest(directory)
        self.assertEqual(model_path.name, "final_ld_unet.keras")
        (directory / "final_ld_unet.keras").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            _validate_model_manifest(directory)


if __name__ == "__main__":
    unittest.main()
