"""Locking and split checks that run without importing TensorFlow."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_evaluate import _validate_locked_run
from ld_train import (
    CODE_RELATIVE_PATHS,
    _code_sha256_manifest,
    _read_frozen_map,
    _save_validation_artifacts,
)
from tiff_io import ImageLabelPair


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LdProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.names = [f"ld_{index:03d}.tif" for index in range(36)]
        self.map_path = self.root / "fold_map.json"
        self.manifest_path = self.root / "label_manifest.json"
        self.fold_ids = {name: index % 5 for index, name in enumerate(self.names)}
        _write_json(self.map_path, {
            "target": "ld", "num_folds": 5, "fold_ids": self.fold_ids,
        })
        _write_json(self.manifest_path, {"rule_id": "ld_antialias_midpoint_v1"})
        self.run = self.root / "fold_0"
        self.run.mkdir()
        self.code_hashes = _code_sha256_manifest()
        project_root = Path(__file__).resolve().parents[2]
        for name in CODE_RELATIVE_PATHS:
            snapshot = self.run / "source_snapshot" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            copyfile(project_root / name, snapshot)
        test = [name for name in self.names if self.fold_ids[name] == 0]
        validation = [name for name in self.names if self.fold_ids[name] == 1]
        training = [name for name in self.names if self.fold_ids[name] not in (0, 1)]
        _write_json(self.run / "split.json", {
            "target": "ld", "fold": 0, "num_folds": 5,
            "fold_ids": self.fold_ids,
            "train": training, "validation": validation, "test": test,
        })
        _write_json(self.run / "experiment_config.json", {
            "target": "ld", "fold": 0, "num_folds": 5,
            "outer_test_deferred": True,
            "fold_map_sha256": _sha256(self.map_path),
            "label_manifest_sha256": _sha256(self.manifest_path),
            "code_sha256": self.code_hashes,
        })
        _write_json(self.run / "threshold.json", {"threshold": 0.45})
        _write_json(self.run / "validation_artifacts" / "manifest.json", {
            "target": "ld", "image_count": len(validation),
            "files": [{"filename": name} for name in validation],
        })
        (self.run / "baseline_unet.keras").write_bytes(b"locked model")
        _write_json(self.run / "selection.lock.json", {
            "target": "ld", "fold": 0,
            "fold_map_sha256": _sha256(self.map_path),
            "label_manifest_sha256": _sha256(self.manifest_path),
            "checkpoint_sha256": _sha256(self.run / "baseline_unet.keras"),
            "threshold_sha256": _sha256(self.run / "threshold.json"),
            "experiment_config_sha256": _sha256(self.run / "experiment_config.json"),
            "split_sha256": _sha256(self.run / "split.json"),
            "code_sha256": self.code_hashes,
            "validation_artifacts_manifest_sha256": _sha256(
                self.run / "validation_artifacts" / "manifest.json"
            ),
            "threshold": 0.45,
        })

    def _refresh_split_lock(self) -> None:
        path = self.run / "selection.lock.json"
        lock = json.loads(path.read_text())
        lock["split_sha256"] = _sha256(self.run / "split.json")
        _write_json(path, lock)

    def test_accepts_frozen_map_and_locked_run(self) -> None:
        self.assertEqual(_read_frozen_map(self.map_path, set(self.names), 5), self.fold_ids)
        _, split, threshold = _validate_locked_run(
            self.run, self.map_path, self.manifest_path
        )
        self.assertEqual(split["fold"], 0)
        self.assertEqual(threshold, 0.45)

    def test_rejects_near_duplicate_group_across_folds(self) -> None:
        payload = json.loads(self.map_path.read_text())
        payload["groups"] = {name: name for name in self.names}
        payload["groups"][self.names[1]] = self.names[0]
        _write_json(self.map_path, payload)
        with self.assertRaisesRegex(ValueError, "crosses outer folds"):
            _read_frozen_map(self.map_path, set(self.names), 5)

    def test_rejects_changed_threshold_after_selection(self) -> None:
        _write_json(self.run / "threshold.json", {"threshold": 0.50})
        with self.assertRaisesRegex(ValueError, "Locked threshold_sha256"):
            _validate_locked_run(self.run, self.map_path, self.manifest_path)

    def test_rejects_test_names_from_another_fold(self) -> None:
        path = self.run / "split.json"
        payload = json.loads(path.read_text())
        payload["test"] = [self.names[1]]
        _write_json(path, payload)
        self._refresh_split_lock()
        with self.assertRaisesRegex(ValueError, "rotated fold assignment"):
            _validate_locked_run(self.run, self.map_path, self.manifest_path)

    def test_rejects_changed_split_without_relocking(self) -> None:
        path = self.run / "split.json"
        payload = json.loads(path.read_text())
        payload["validation"] = []
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "Locked split_sha256"):
            _validate_locked_run(self.run, self.map_path, self.manifest_path)

    def test_rejects_changed_source_snapshot(self) -> None:
        (self.run / "source_snapshot" / "code/model/unet.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source snapshot differs"):
            _validate_locked_run(self.run, self.map_path, self.manifest_path)

    def test_saves_original_grid_validation_probabilities_and_masks(self) -> None:
        source = self.root / "input.tif"
        mask = self.root / "mask.tif"
        tifffile.imwrite(source, np.zeros((13, 17), dtype=np.uint16))
        tifffile.imwrite(mask, np.ones((13, 17), dtype=np.uint8))
        probability = np.full((13, 17), 0.7, dtype=np.float32)
        truth = np.ones((13, 17), dtype=np.uint8)
        pair = ImageLabelPair(source, mask)
        destination = self.root / "new_validation_run"
        destination.mkdir()
        manifest = _save_validation_artifacts([(pair, probability, truth)], 0.6, destination)
        self.assertEqual(json.loads(manifest.read_text())["image_count"], 1)
        probability_path = destination / "validation_artifacts" / "probabilities" / "input.tif"
        self.assertEqual(tifffile.imread(probability_path).shape, (13, 17))
        self.assertTrue(np.array_equal(
            tifffile.imread(destination / "validation_artifacts" / "predictions" /
                            "validation" / "input.tif"),
            np.full((13, 17), 255, dtype=np.uint8),
        ))


if __name__ == "__main__":
    unittest.main()
