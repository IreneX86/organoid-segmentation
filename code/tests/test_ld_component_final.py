"""Lightweight, GPU-free checks for the candidate-only LD final fit gate."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_component_train import _code_sha256_manifest
from ld_component_train_final import (
    _validate_all_pairs, _validate_keras_archive, _validated_policy,
)
from tiff_io import pair_tiff_directories


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


class CandidateFinalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_dir = self.root / "LD_input"
        self.label_dir = self.root / "LD_output_binary"
        self.input_dir.mkdir()
        self.label_dir.mkdir()
        self.fold_map = self.root / "fold_map.json"
        self.label_manifest = self.root / "manifest.json"
        self.policy_path = self.root / "policy.json"
        input_hashes = {}
        label_files = []
        folds = {}
        for index in range(36):
            name = f"ld_{index + 1:03d}.tif"
            image = np.full((4, 5), index, dtype=np.uint16)
            mask = np.zeros((4, 5), dtype=np.uint8)
            mask[index % 4, index % 5] = 1
            image_path = self.input_dir / name
            mask_path = self.label_dir / name
            tifffile.imwrite(image_path, image)
            tifffile.imwrite(mask_path, mask)
            input_hashes[name] = _sha256(image_path)
            label_files.append({"filename": name, "derived_mask_sha256": _sha256(mask_path)})
            folds[name] = index % 5
        _write_json(self.fold_map, {
            "target": "ld", "num_folds": 5, "fold_ids": folds,
            "input_sha256": input_hashes,
        })
        _write_json(self.label_manifest, {
            "rule_id": "ld_antialias_midpoint_v1",
            "label_semantics": {"background": 0, "foreground": 1},
            "files": label_files,
        })
        project_root = Path(__file__).resolve().parents[2]
        self.policy = {
            "policy_version": "ld_global_validation_method_v1",
            "target": "ld", "selected_method": "candidate",
            "candidate_id": "component_uniform_positive_center_v1",
            "selection_basis": "predeclared_five_fold_inner_validation_only",
            "outer_test_used_for_selection": False, "image_count": 36,
            "fold_map_sha256": _sha256(self.fold_map),
            "label_manifest_sha256": _sha256(self.label_manifest),
            "policy_protocol_sha256": _sha256(
                project_root / "experiments/ld_production_selection_policy.md"
            ),
            "global_checks": {
                "candidate_gate_wins_at_least_4_of_5": True,
                "all_36_mean_paired_dice_delta_at_least_0_01": True,
                "each_fold_mean_precision_delta_at_least_minus_0_03": True,
                "no_individual_dice_delta_below_minus_0_10": True,
            },
            "gate_wins": 4,
            "fold_gate_records": [
                {"fold": index, "gate_selected": "candidate" if index < 4 else "baseline"}
                for index in range(5)
            ],
            "scientific_settings": {
                "architecture": "unet_2d_baseline", "loss": "bce_dice",
                "positive_center_policy": "uniform_8_connected_training_component_then_pixel",
                "seed_base": 42, "patch_size": 512, "overlap": 128,
                "batch_size": 4, "prediction_batch_size": 4,
                "steps_per_epoch": 100, "epochs": 100,
                "learning_rate": 1e-4, "positive_patch_probability": 0.6,
                "lower_percentile": 1.0, "upper_percentile": 99.8,
                "checkpoint_selection": "validation_full_image_dice_at_0.5",
                "threshold_selection": "validation_mean_per_image_dice",
                "code_sha256": _code_sha256_manifest(),
            },
            "fold_validation_records": [
                {"fold": index, "validation_best_epoch_count": (index + 1) * 3,
                 "validation_selected_threshold": 0.4 + 0.05 * index,
                 "training_history_sha256": "a" * 64}
                for index in range(5)
            ],
            "final_epoch_count": 9, "deployment_threshold": 0.5,
        }
        _write_json(self.policy_path, self.policy)
        self.locked_evidence = copy.deepcopy(self.policy)

    def _validated_policy(self) -> dict:
        # The real builder is covered by its own synthetic five-fold tests.
        # Here a stable builder result proves the final entry point rejects a
        # caller-edited policy before any TensorFlow training is possible.
        with mock.patch("ld_component_train_final.build_method_policy",
                        return_value=self.locked_evidence):
            return _validated_policy(
                self.policy_path, self.root / "baseline_cv",
                self.root / "candidate_cv", self.fold_map,
                self.label_manifest, self.label_dir,
            )

    def test_candidate_gate_and_all_36_hashes_pass(self) -> None:
        self.assertEqual(self._validated_policy()["final_epoch_count"], 9)
        pairs = pair_tiff_directories(self.input_dir, self.label_dir)
        report = _validate_all_pairs(pairs, self.fold_map, self.label_manifest)
        self.assertEqual(len(report), 36)

    def test_reject_baseline_or_unlocked_global_gate(self) -> None:
        self.policy["selected_method"] = "baseline"
        _write_json(self.policy_path, self.policy)
        with self.assertRaisesRegex(ValueError, "candidate policy"):
            self._validated_policy()
        self.policy["selected_method"] = "candidate"
        self.policy["global_checks"]["no_individual_dice_delta_below_minus_0_10"] = False
        _write_json(self.policy_path, self.policy)
        with self.assertRaisesRegex(ValueError, "global validation gate"):
            self._validated_policy()

    def test_reject_posthoc_epoch_or_threshold_change(self) -> None:
        self.policy["final_epoch_count"] = 12
        _write_json(self.policy_path, self.policy)
        with self.assertRaisesRegex(ValueError, "differ from the five validation records"):
            self._validated_policy()

    def test_reject_forged_but_internally_consistent_policy_field(self) -> None:
        # This field formerly had no independent verification in the final
        # trainer; recomputation from validation archives must now reject it.
        self.policy["mean_paired_dice_delta_050_all_36"] = 0.99
        _write_json(self.policy_path, self.policy)
        with self.assertRaisesRegex(ValueError, "differs from validation-only CV evidence"):
            self._validated_policy()

    def test_reject_changed_derived_mask(self) -> None:
        name = "ld_001.tif"
        tifffile.imwrite(self.label_dir / name, np.ones((4, 5), dtype=np.uint8))
        pairs = pair_tiff_directories(self.input_dir, self.label_dir)
        with self.assertRaisesRegex(ValueError, "Derived LD mask changed"):
            _validate_all_pairs(pairs, self.fold_map, self.label_manifest)

    def test_checkpoint_zip_integrity(self) -> None:
        path = self.root / "final_ld_component_unet.keras"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("metadata.json", "{}")
            archive.writestr("config.json", "{}")
            archive.writestr("model.weights.h5", b"synthetic")
        _validate_keras_archive(path)
        path.write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "readable archive"):
            _validate_keras_archive(path)


if __name__ == "__main__":
    unittest.main()
