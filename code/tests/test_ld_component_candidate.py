"""Synthetic leakage and single-factor checks for LD component sampling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_split import FoldSplit
from ld_component_gate import _validation_metrics, compute_gate, validate_candidate
from ld_component_evaluate import _checked_probability
from ld_component_sampling import (
    component_pixels,
    sample_component_center,
    sample_component_uniform_patch,
)
from ld_component_train import (
    CODE_RELATIVE_PATHS,
    _code_sha256_manifest,
    _validate_train_validation_only,
)
from tiff_io import ImageLabelPair


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ComponentSamplingTests(unittest.TestCase):
    def test_outer_probability_must_match_grid_and_unit_interval(self) -> None:
        truth = np.zeros((2, 3), dtype=np.uint8)
        valid = np.full((2, 3), 0.25, dtype=np.float32)
        np.testing.assert_array_equal(_checked_probability(valid, truth, "x.tif"), valid)
        for invalid in (
            np.zeros((3, 2), dtype=np.float32),
            np.full((2, 3), np.nan, dtype=np.float32),
            np.full((2, 3), np.inf, dtype=np.float32),
            np.full((2, 3), -0.01, dtype=np.float32),
            np.full((2, 3), 1.01, dtype=np.float32),
            np.ones((2, 3), dtype=np.int32),
        ):
            with self.subTest(dtype=str(invalid.dtype), shape=invalid.shape,
                              first=str(invalid.flat[0])):
                with self.assertRaises((RuntimeError, ValueError)):
                    _checked_probability(invalid, truth, "x.tif")

    def test_eight_connected_regions_are_sampled_uniformly(self) -> None:
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[1, 1] = 1
        mask[2, 2] = 1  # Diagonal connection belongs to the first region.
        mask[5:8, 5:8] = 1  # Nine pixels, but same component weight.
        components = component_pixels(mask)
        self.assertEqual(components.count, 2)
        self.assertEqual(sorted(np.diff(components.offsets).tolist()), [2, 9])
        rng = np.random.default_rng(413)
        visits = np.zeros(2, dtype=np.int64)
        for _ in range(10_000):
            y, x = sample_component_center(components, rng)
            self.assertEqual(mask[y, x], 1)
            visits[0 if y < 4 else 1] += 1
        self.assertGreater(visits[0], 4_700)
        self.assertLess(visits[0], 5_300)

    def test_uniform_branch_retains_baseline_origin_geometry(self) -> None:
        class ChosenRng:
            def __init__(self) -> None:
                self.draws = iter((2, 3))

            def random(self) -> float:
                return 0.9  # Existing uniform branch at probability 0.60.

            def integers(self, upper: int) -> int:
                return next(self.draws)

        image = np.arange(63, dtype=np.float32).reshape(7, 9)
        mask = np.zeros((7, 9), dtype=np.uint8)
        mask[1, 1] = 1
        image_patch, mask_patch = sample_component_uniform_patch(
            image, mask, component_pixels(mask), 4, ChosenRng(), 0.6
        )
        np.testing.assert_array_equal(image_patch, image[2:6, 3:7])
        np.testing.assert_array_equal(mask_patch, mask[2:6, 3:7])
        self.assertEqual(image_patch.shape, (4, 4))

    def test_empty_foreground_falls_back_to_uniform_crop(self) -> None:
        image = np.arange(49, dtype=np.float32).reshape(7, 7)
        mask = np.zeros((7, 7), dtype=np.uint8)
        image_patch, mask_patch = sample_component_uniform_patch(
            image, mask, component_pixels(mask), 4,
            np.random.default_rng(9), 0.6,
        )
        self.assertEqual(image_patch.shape, (4, 4))
        self.assertFalse(mask_patch.any())

    def test_training_preflight_does_not_decode_outer_test_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = []
            records: dict[str, dict[str, object]] = {}
            for name in ("train", "validation", "outer_test"):
                image = root / f"{name}_input.tif"
                label = root / f"{name}_mask.tif"
                tifffile.imwrite(image, np.zeros((8, 8), dtype=np.uint16))
                if name == "outer_test":
                    label.write_bytes(b"deliberately unreadable outer-test mask")
                else:
                    tifffile.imwrite(label, np.ones((8, 8), dtype=np.uint8))
                    records[image.name] = {"derived_mask_sha256": _sha256(label)}
                pairs.append(ImageLabelPair(image, label))
            split = FoldSplit(train=[pairs[0]], validation=[pairs[1]], test=[pairs[2]])
            report = _validate_train_validation_only(split, records)
            self.assertEqual([item["filename"] for item in report],
                             [pairs[0].filename, pairs[1].filename])


class ComponentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.baseline.mkdir()
        self.candidate.mkdir()
        self.label_dir = self.root / "derived_labels"
        self.label_dir.mkdir()
        self.map = self.root / "fold_map.json"
        self.manifest = self.root / "manifest.json"
        self.names = [f"ld_{index:03d}.tif" for index in range(36)]
        self.fold_ids = {name: index % 5 for index, name in enumerate(self.names)}
        self.validation = {name for name in self.names if self.fold_ids[name] == 1}
        self.test = {name for name in self.names if self.fold_ids[name] == 0}
        self.train = set(self.names) - self.validation - self.test
        self.split = {
            "target": "ld", "fold": 0, "num_folds": 5,
            "fold_ids": self.fold_ids,
            "train": sorted(self.train),
            "validation": sorted(self.validation),
            "test": sorted(self.test),
        }
        self.map.write_text(json.dumps({"fold_ids": self.fold_ids}))
        self.manifest.write_text(json.dumps({
            "rule_id": "ld_antialias_midpoint_v1",
            "files": [{"filename": name} for name in self.names],
        }))
        for run in (self.baseline, self.candidate):
            (run / "selection.lock.json").write_text("{}")
            artifacts = run / "validation_artifacts"
            artifacts.mkdir()
            (artifacts / "manifest.json").write_text("{}")

    def test_gate_uses_only_this_folds_inner_validation(self) -> None:
        baseline_metrics = {
            name: {"dice": 0.60, "precision": 0.75, "recall": 0.55}
            for name in self.validation
        }
        candidate_metrics = {
            name: {"dice": 0.62, "precision": 0.75, "recall": 0.59}
            for name in self.validation
        }
        observed: list[set[str]] = []

        def metrics_stub(run: Path, names: set[str], *_: object) -> dict:
            observed.append(set(names))
            return candidate_metrics if run == self.candidate.resolve() else baseline_metrics

        config = {"input_dir": str(self.root / "inputs"),
                  "label_dir": str(self.label_dir.resolve()),
                  "protocol_sha256": "frozen"}
        with (mock.patch("ld_component_gate.validate_baseline",
                         return_value=(config, self.split, 0.5)),
              mock.patch("ld_component_gate.validate_candidate",
                         return_value=(config, self.split, 0.5)),
              mock.patch("ld_component_gate._validation_metrics",
                         side_effect=metrics_stub)):
            result = compute_gate(
                self.baseline, self.candidate, self.map,
                self.manifest, self.label_dir,
            )
        self.assertEqual(result["selected"], "candidate")
        self.assertEqual(observed, [self.validation, self.validation])
        self.assertTrue(self.test.isdisjoint(observed[0]))
        self.assertFalse((self.label_dir / next(iter(self.test))).exists())

    def test_gate_refuses_single_large_validation_regression(self) -> None:
        baseline = {
            name: {"dice": 0.50, "precision": 0.75, "recall": 0.50}
            for name in self.validation
        }
        candidate = {
            name: {"dice": 0.65, "precision": 0.75, "recall": 0.60}
            for name in self.validation
        }
        candidate[sorted(self.validation)[0]]["dice"] = 0.35
        config = {"input_dir": str(self.root / "inputs"),
                  "label_dir": str(self.label_dir.resolve()),
                  "protocol_sha256": "frozen"}
        with (mock.patch("ld_component_gate.validate_baseline",
                         return_value=(config, self.split, 0.5)),
              mock.patch("ld_component_gate.validate_candidate",
                         return_value=(config, self.split, 0.5)),
              mock.patch("ld_component_gate._validation_metrics",
                         side_effect=[baseline, candidate])):
            result = compute_gate(
                self.baseline, self.candidate, self.map,
                self.manifest, self.label_dir,
            )
        self.assertEqual(result["selected"], "baseline")
        self.assertFalse(result["gate_checks"]["no_single_dice_drop_more_than_0_10"])

    def test_gate_rejects_changes_to_other_scientific_factors(self) -> None:
        baseline_config = {"input_dir": str(self.root / "inputs"),
                           "label_dir": str(self.label_dir.resolve()),
                           "learning_rate": 1e-4}
        candidate_config = {**baseline_config, "learning_rate": 2e-4,
                            "protocol_sha256": "frozen"}
        with (mock.patch("ld_component_gate.validate_baseline",
                         return_value=(baseline_config, self.split, 0.5)),
              mock.patch("ld_component_gate.validate_candidate",
                         return_value=(candidate_config, self.split, 0.5)),
              mock.patch("ld_component_gate._validation_metrics",
                         side_effect=AssertionError("science mismatch must stop early"))):
            with self.assertRaisesRegex(ValueError, "scientific factor"):
                compute_gate(
                    self.baseline, self.candidate, self.map,
                    self.manifest, self.label_dir,
                )

    def test_validation_artifacts_recompute_without_outer_label(self) -> None:
        name = sorted(self.validation)[0]
        truth = np.zeros((4, 4), dtype=np.uint8)
        truth[1, 1] = 1
        truth_path = self.label_dir / name
        tifffile.imwrite(truth_path, truth)
        artifacts = self.baseline / "validation_artifacts"
        predicted = artifacts / "predictions" / "fixed_050" / name
        selected = artifacts / "predictions" / "validation" / name
        probability = artifacts / "probabilities" / name
        for target in (predicted, selected, probability):
            target.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(predicted, truth * 255)
        tifffile.imwrite(selected, truth * 255)
        tifffile.imwrite(probability, truth.astype(np.float32))
        (artifacts / "manifest.json").write_text(json.dumps({
            "image_count": 1,
            "files": [{
                "filename": name,
                "fixed_050_mask_sha256": _sha256(predicted),
                "validation_mask_sha256": _sha256(selected),
                "probability_sha256": _sha256(probability),
            }],
        }))
        with (artifacts / "threshold_policy_per_image.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "filename", "fixed_050_dice", "fixed_050_iou",
                "fixed_050_precision", "fixed_050_recall",
            ])
            writer.writeheader()
            writer.writerow({"filename": name, "fixed_050_dice": 1,
                             "fixed_050_iou": 1, "fixed_050_precision": 1,
                             "fixed_050_recall": 1})
        metrics = _validation_metrics(
            self.baseline, {name}, self.label_dir,
            {name: {"derived_mask_sha256": _sha256(truth_path)}},
        )
        self.assertEqual(metrics[name]["dice"], 1.0)
        self.assertFalse((self.label_dir / next(iter(self.test))).exists())

    def test_baseline_gate_short_circuits_before_any_outer_evaluation(self) -> None:
        from ld_component_evaluate import main

        decision = {"fold": 0, "selected": "baseline"}
        (self.candidate / "decision.lock.json").write_text(json.dumps(decision))
        args = argparse.Namespace(
            baseline_run_dir=self.baseline,
            candidate_run_dir=self.candidate,
            input_dir=self.root / "nonexistent_input",
            label_dir=self.label_dir,
            label_manifest=self.manifest,
            fold_map=self.map,
        )
        with (mock.patch("ld_component_evaluate.parse_args", return_value=args),
              mock.patch("ld_component_evaluate.compute_gate", return_value=decision),
              mock.patch("ld_component_evaluate.pair_tiff_directories",
                         side_effect=AssertionError("test labels must stay closed"))):
            main()
        self.assertFalse((self.candidate / "outer_test").exists())


class ComponentSelectionLockTests(unittest.TestCase):
    def test_candidate_lock_checks_source_and_training_only_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "candidate" / "ld" / "fold_0"
            run.mkdir(parents=True)
            names = [f"ld_{index:03d}.tif" for index in range(36)]
            folds = {name: index % 5 for index, name in enumerate(names)}
            training = [name for name in names if folds[name] not in (0, 1)]
            validation = [name for name in names if folds[name] == 1]
            test = [name for name in names if folds[name] == 0]
            map_path = root / "fold_map.json"
            labels_path = root / "label_manifest.json"
            map_path.write_text(json.dumps({
                "target": "ld", "num_folds": 5, "fold_ids": folds,
            }))
            labels_path.write_text(json.dumps({"rule_id": "ld_antialias_midpoint_v1"}))
            hashes = _code_sha256_manifest()
            project_root = Path(__file__).resolve().parents[2]
            for name in CODE_RELATIVE_PATHS:
                destination = run / "source_snapshot" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project_root / name, destination)
            split = {"target": "ld", "fold": 0, "num_folds": 5,
                     "fold_ids": folds, "train": training,
                     "validation": validation, "test": test}
            config = {
                "target": "ld", "fold": 0, "num_folds": 5,
                "candidate": "component_uniform_positive_center_v1",
                "positive_center_policy": "uniform_8_connected_training_component_then_pixel",
                "positive_patch_probability": 0.6,
                "outer_test_deferred": True,
                "fold_map_sha256": _sha256(map_path),
                "label_manifest_sha256": _sha256(labels_path),
                "protocol_sha256": hashes[
                    "experiments/ld_component_uniform_sampling_protocol.md"
                ],
                "code_sha256": hashes,
            }
            (run / "split.json").write_text(json.dumps(split))
            (run / "experiment_config.json").write_text(json.dumps(config))
            (run / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
            (run / "component_unet.keras").write_bytes(b"synthetic checkpoint")
            component_counts = run / "training_component_counts.json"
            component_counts.write_text(json.dumps({
                "scope": "training_masks_only",
                "eight_connected_components_per_image": {name: 1 for name in training},
            }))
            validation_manifest = run / "validation_artifacts" / "manifest.json"
            validation_manifest.parent.mkdir()
            validation_manifest.write_text(json.dumps({
                "image_count": len(validation),
                "files": [{"filename": name} for name in validation],
            }))
            lock = {
                "target": "ld", "fold": 0, "threshold": 0.5,
                "code_sha256": hashes,
                "fold_map_sha256": _sha256(map_path),
                "label_manifest_sha256": _sha256(labels_path),
                "checkpoint_sha256": _sha256(run / "component_unet.keras"),
                "threshold_sha256": _sha256(run / "threshold.json"),
                "experiment_config_sha256": _sha256(run / "experiment_config.json"),
                "split_sha256": _sha256(run / "split.json"),
                "validation_artifacts_manifest_sha256": _sha256(validation_manifest),
                "training_component_counts_sha256": _sha256(component_counts),
            }
            (run / "selection.lock.json").write_text(json.dumps(lock))
            self.assertEqual(
                validate_candidate(run, map_path, labels_path)[1]["test"], test
            )
            # This validator has no reason to open or even require test TIFFs.
            self.assertFalse((root / test[0]).exists())
            component_counts.write_text(json.dumps({"scope": "tampered"}))
            with self.assertRaisesRegex(ValueError, "training_component_counts_sha256"):
                validate_candidate(run, map_path, labels_path)


if __name__ == "__main__":
    unittest.main()
