"""CPU-only checks of the isolated LD focal candidate and its test-data gate."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_focal_gate import _gate_checks, _validated_run
from ld_focal_evaluate import _verify_all_gate_decisions


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(name: str, *, dice: float, precision: float, count: int, hits: int) -> dict:
    return {
        "filename": name, "dice": dice, "precision": precision,
        "small_component_count_proxy": count,
        "small_component_detected_any_proxy": hits,
    }


class FocalGateTests(unittest.TestCase):
    def test_component_gain_and_precision_guard(self) -> None:
        baseline = [_row("a.tif", dice=.80, precision=.80, count=10, hits=4)]
        candidate = [_row("a.tif", dice=.795, precision=.78, count=10, hits=5)]
        checks, observed = _gate_checks(baseline, candidate)
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(observed["small_component_overlap_fraction_delta"], .10)
        candidate[0]["precision"] = .75
        checks, _ = _gate_checks(baseline, candidate)
        self.assertFalse(checks["mean_precision_drop_no_more_than_0_03"])

    def test_ceiling_and_missing_components_default_to_baseline(self) -> None:
        baseline = [_row("a.tif", dice=.70, precision=.80, count=7, hits=7)]
        candidate = [_row("a.tif", dice=.705, precision=.80, count=7, hits=7)]
        checks, _ = _gate_checks(baseline, candidate)
        self.assertFalse(checks["small_component_ceiling_aware_condition"])
        baseline[0]["small_component_count_proxy"] = 0
        baseline[0]["small_component_detected_any_proxy"] = 0
        candidate[0]["small_component_count_proxy"] = 0
        candidate[0]["small_component_detected_any_proxy"] = 0
        checks, _ = _gate_checks(baseline, candidate)
        self.assertFalse(checks["small_component_ceiling_aware_condition"])


class FocalSelectionLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.fold_map = root / "map.json"
        self.label_manifest = root / "labels.json"
        self.run_dir = root / "run"
        names = [f"ld_{index:03d}.tif" for index in range(36)]
        fold_ids = {name: index % 5 for index, name in enumerate(names)}
        _write_json(self.fold_map, {
            "target": "ld", "num_folds": 5, "fold_ids": fold_ids,
        })
        _write_json(self.label_manifest, {
            "rule_id": "ld_antialias_midpoint_v1",
            "files": [{"filename": name} for name in names],
        })
        partitions = {
            "train": [name for name in names if fold_ids[name] not in (0, 1)],
            "validation": [name for name in names if fold_ids[name] == 1],
            "test": [name for name in names if fold_ids[name] == 0],
        }
        config = {
            "target": "ld", "fold": 0, "num_folds": 5,
            "outer_test_deferred": True,
            "candidate": "ld_focal_dice_gamma2_v1",
            "loss": "focal_bce_gamma2_plus_soft_dice",
            "upper_percentile": 99.8,
            "protocol_sha256": _sha256(
                Path(__file__).resolve().parents[2]
                / "experiments" / "ld_focal_dice_protocol.md"),
            "fold_map_sha256": _sha256(self.fold_map),
            "label_manifest_sha256": _sha256(self.label_manifest),
        }
        _write_json(self.run_dir / "experiment_config.json", config)
        _write_json(self.run_dir / "split.json", {
            "target": "ld", "fold": 0, "num_folds": 5,
            "fold_ids": fold_ids, **partitions,
        })
        _write_json(self.run_dir / "threshold.json", {"threshold": .35})
        _write_json(self.run_dir / "validation_artifacts" / "manifest.json", {})
        checkpoint = self.run_dir / "focal_unet.keras"
        checkpoint.write_bytes(b"focal-checkpoint-fixture")
        source = self.run_dir / "source_snapshot" / "code" / "model" / "unet.py"
        source.parent.mkdir(parents=True)
        source.write_text("# test source\n", encoding="utf-8")
        code_hashes = {"code/model/unet.py": _sha256(source)}
        config["code_sha256"] = code_hashes
        _write_json(self.run_dir / "experiment_config.json", config)
        _write_json(self.run_dir / "selection.lock.json", {
            "target": "ld", "fold": 0, "threshold": .35,
            "code_sha256": code_hashes,
            "fold_map_sha256": _sha256(self.fold_map),
            "label_manifest_sha256": _sha256(self.label_manifest),
            "checkpoint_sha256": _sha256(checkpoint),
            "threshold_sha256": _sha256(self.run_dir / "threshold.json"),
            "experiment_config_sha256": _sha256(self.run_dir / "experiment_config.json"),
            "split_sha256": _sha256(self.run_dir / "split.json"),
            "validation_artifacts_manifest_sha256": _sha256(
                self.run_dir / "validation_artifacts" / "manifest.json"),
        })

    def test_accepts_locked_focal_candidate(self) -> None:
        config, split, _, threshold = _validated_run(
            self.run_dir, self.fold_map, self.label_manifest, candidate=True,
        )
        self.assertEqual(config["loss"], "focal_bce_gamma2_plus_soft_dice")
        self.assertEqual(split["test"], [name for name, index in split["fold_ids"].items()
                                         if index == 0])
        self.assertEqual(threshold, .35)

    def test_rejects_candidate_after_outer_test_appears(self) -> None:
        (self.run_dir / "outer_test").mkdir()
        with self.assertRaisesRegex(FileExistsError, "outer-test"):
            _validated_run(self.run_dir, self.fold_map, self.label_manifest,
                           candidate=True)

    def test_rejects_wrong_loss_even_with_valid_rehashed_lock(self) -> None:
        config_path = self.run_dir / "experiment_config.json"
        config = json.loads(config_path.read_text())
        config["loss"] = "bce_dice"
        _write_json(config_path, config)
        lock_path = self.run_dir / "selection.lock.json"
        lock = json.loads(lock_path.read_text())
        lock["experiment_config_sha256"] = _sha256(config_path)
        _write_json(lock_path, lock)
        with self.assertRaisesRegex(ValueError, "focal-loss protocol"):
            _validated_run(self.run_dir, self.fold_map, self.label_manifest,
                           candidate=True)


class AllFoldGateBeforeTestTests(unittest.TestCase):
    def test_requires_all_five_matching_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_root = root / "runs"
            map_path = root / "fold_map.json"
            label_path = root / "manifest.json"
            _write_json(map_path, {"target": "ld"})
            _write_json(label_path, {"rule_id": "ld_antialias_midpoint_v1"})
            protocol_path = (Path(__file__).resolve().parents[2]
                             / "experiments" / "ld_focal_dice_protocol.md")
            for fold in range(5):
                folder = runs_root / "ld" / f"fold_{fold}"
                lock_path = folder / "selection.lock.json"
                validation_manifest = folder / "validation_artifacts" / "manifest.json"
                _write_json(lock_path, {"fold": fold})
                _write_json(validation_manifest, {"target": "ld"})
                _write_json(folder / "decision.lock.json", {
                    "format": "ld_focal_fold_local_decision_v1",
                    "target": "ld", "candidate": "ld_focal_dice_gamma2_v1",
                    "fold": fold, "selected": "baseline",
                    "outer_test_labels_opened": False,
                    "protocol_sha256": _sha256(protocol_path),
                    "fold_map_sha256": _sha256(map_path),
                    "label_manifest_sha256": _sha256(label_path),
                    "candidate_selection_lock_sha256": _sha256(lock_path),
                    "candidate_validation_manifest_sha256": _sha256(validation_manifest),
                })
            current = runs_root / "ld" / "fold_0"
            _verify_all_gate_decisions(runs_root, map_path, label_path, current)
            (runs_root / "ld" / "fold_4" / "decision.lock.json").unlink()
            with self.assertRaises(FileNotFoundError):
                _verify_all_gate_decisions(runs_root, map_path, label_path, current)


if __name__ == "__main__":
    unittest.main()
