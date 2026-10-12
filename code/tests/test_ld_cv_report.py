"""Synthetic protocol checks for five-fold LD out-of-fold aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_cv_report import METRICS, SCIENTIFIC_KEYS, aggregate_cv
from ld_train import CODE_RELATIVE_PATHS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LdCvReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.names = [f"LD_{index:03d}.tif" for index in range(36)]
        self.fold_ids = {name: index % 5 for index, name in enumerate(self.names)}
        self._create_folds()

    def _create_folds(self) -> None:
        for fold in range(5):
            directory = self.root / "ld" / f"fold_{fold}"
            code_hashes = {}
            for name in CODE_RELATIVE_PATHS:
                snapshot = directory / "source_snapshot" / name
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(f"frozen source {name}".encode())
                code_hashes[name] = _sha256(snapshot)
            selected = [name for name in self.names if self.fold_ids[name] == fold]
            validation = [
                name for name in self.names if self.fold_ids[name] == (fold + 1) % 5
            ]
            training = [
                name for name in self.names
                if self.fold_ids[name] not in (fold, (fold + 1) % 5)
            ]
            _write_json(directory / "split.json", {
                "target": "ld", "fold": fold, "num_folds": 5,
                "fold_ids": self.fold_ids, "train": training,
                "validation": validation, "test": selected,
            })
            settings = {key: "same-across-folds" for key in SCIENTIFIC_KEYS}
            settings.update({"target": "ld", "fold": fold,
                             "seed_base": 42, "seed": 42 + fold,
                             "code_sha256": code_hashes})
            _write_json(directory / "experiment_config.json", settings)
            _write_json(directory / "threshold.json", {"threshold": 0.45})
            validation_files = []
            validation_dir = directory / "validation_artifacts"
            for name in validation:
                paths = {
                    "probability_sha256": validation_dir / "probabilities" / name,
                    "validation_mask_sha256": (
                        validation_dir / "predictions" / "validation" / name
                    ),
                    "fixed_050_mask_sha256": (
                        validation_dir / "predictions" / "fixed_050" / name
                    ),
                }
                item = {"filename": name}
                for field, path in paths.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"synthetic {field} {name}".encode())
                    item[field] = _sha256(path)
                validation_files.append(item)
            _write_json(validation_dir / "manifest.json", {
                "target": "ld", "image_count": len(validation),
                "files": validation_files,
            })
            (directory / "baseline_unet.keras").write_bytes(b"synthetic checkpoint")
            lock = {
                "target": "ld", "fold": fold, "threshold": 0.45,
                "label_manifest_sha256": "same-across-folds",
                "fold_map_sha256": "same-across-folds",
                "checkpoint_sha256": _sha256(directory / "baseline_unet.keras"),
                "threshold_sha256": _sha256(directory / "threshold.json"),
                "experiment_config_sha256": _sha256(directory / "experiment_config.json"),
                "split_sha256": _sha256(directory / "split.json"),
                "code_sha256": code_hashes,
                "validation_artifacts_manifest_sha256": _sha256(
                    validation_dir / "manifest.json"
                ),
            }
            _write_json(directory / "selection.lock.json", lock)
            rows = [{"filename": name, **{metric: 0.8 for metric in METRICS}}
                    for name in selected]
            report_dir = directory / "outer_test"
            report_dir.mkdir()
            with (report_dir / "test_per_image.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            policy_rows = [{
                "filename": name,
                **{f"validation_{metric}": 0.8 for metric in (
                    "dice", "iou", "precision", "recall"
                )},
                **{f"fixed_050_{metric}": 0.75 for metric in (
                    "dice", "iou", "precision", "recall"
                )},
            } for name in selected]
            with (report_dir / "threshold_policy_per_image.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(policy_rows[0]))
                writer.writeheader()
                writer.writerows(policy_rows)
            _write_json(report_dir / "test_summary.json", {
                "image_count": len(selected),
                **{f"{metric}_mean": 0.8 for metric in METRICS},
            })
            _write_json(report_dir / "evaluation_manifest.json", {
                "fold": fold, "test_filenames": selected,
                "selection_lock_sha256": _sha256(directory / "selection.lock.json"),
            })

    def _refresh_lock(self, fold: int) -> None:
        directory = self.root / "ld" / f"fold_{fold}"
        path = directory / "selection.lock.json"
        lock = json.loads(path.read_text())
        lock["split_sha256"] = _sha256(directory / "split.json")
        lock["experiment_config_sha256"] = _sha256(directory / "experiment_config.json")
        _write_json(path, lock)
        manifest_path = directory / "outer_test" / "evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["selection_lock_sha256"] = _sha256(path)
        _write_json(manifest_path, manifest)

    def test_aggregates_each_image_exactly_once(self) -> None:
        rows, report = aggregate_cv(self.root)
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({row["filename"] for row in rows}), 36)
        self.assertAlmostEqual(report["oof"]["dice"]["mean"], 0.8)
        self.assertAlmostEqual(report["fixed_050_oof"]["dice"]["mean"], 0.75)

    def test_rejects_duplicate_or_wrong_test_assignment(self) -> None:
        path = self.root / "ld" / "fold_1" / "split.json"
        payload = json.loads(path.read_text())
        payload["test"][0] = self.names[0]
        _write_json(path, payload)
        self._refresh_lock(1)
        with self.assertRaisesRegex(ValueError, "rotated fold assignment"):
            aggregate_cv(self.root)

    def test_rejects_changed_scientific_configuration(self) -> None:
        path = self.root / "ld" / "fold_3" / "experiment_config.json"
        payload = json.loads(path.read_text())
        payload["architecture"] = "changed-after-lock"
        _write_json(path, payload)
        self._refresh_lock(3)
        with self.assertRaisesRegex(ValueError, "different scientific settings"):
            aggregate_cv(self.root)

    def test_rejects_modified_checkpoint(self) -> None:
        path = self.root / "ld" / "fold_2" / "baseline_unet.keras"
        path.write_bytes(b"altered checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            aggregate_cv(self.root)

    def test_rejects_changed_split_without_relocking(self) -> None:
        path = self.root / "ld" / "fold_0" / "split.json"
        payload = json.loads(path.read_text())
        payload["train"] = []
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "split differs from its selection lock"):
            aggregate_cv(self.root)

    def test_rejects_modified_frozen_source(self) -> None:
        path = self.root / "ld" / "fold_4" / "source_snapshot" / "code/losses.py"
        path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "frozen source differs"):
            aggregate_cv(self.root)

    def test_rejects_missing_validation_prediction(self) -> None:
        path = self.root / "ld" / "fold_0" / "validation_artifacts" / "probabilities"
        (path / self.names[1]).unlink()
        with self.assertRaises(FileNotFoundError):
            aggregate_cv(self.root)


if __name__ == "__main__":
    unittest.main()
