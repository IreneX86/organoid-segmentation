"""Synthetic, TensorFlow-free tests for the predeclared global LD method rule."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_method_policy import build_method_policy
from segmentation_metrics import calculate_binary_metrics


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LdMethodPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.names = [f"ld_{index:03d}.tif" for index in range(36)]
        self.ids = {name: index % 5 for index, name in enumerate(self.names)}
        self.map = self.root / "fold_map.json"
        self.labels = self.root / "label_manifest.json"
        self.label_dir = self.root / "derived_labels"
        self.label_dir.mkdir()
        self.truth = np.zeros((40, 50), dtype=np.uint8)
        self.truth.flat[:1000] = 1
        _json(self.map, {"target": "ld", "num_folds": 5, "fold_ids": self.ids})
        label_records = []
        for name in self.names:
            path = self.label_dir / name
            tifffile.imwrite(path, self.truth, metadata=None)
            label_records.append({"filename": name, "derived_mask_sha256": _sha(path)})
        _json(self.labels, {
            "rule_id": "ld_antialias_midpoint_v1",
            "label_semantics": {"background": 0, "foreground": 1},
            "derived_mask_dir": "/archived/mac/ld_binary_midpoint_v1/LD_output",
            "files": label_records,
        })

    def _run(self, fold: int, candidate: bool,
             counts: dict[str, tuple[int, int]]) -> tuple[Path, dict[str, dict[str, float]]]:
        root = self.candidate if candidate else self.baseline
        directory = root / "ld" / f"fold_{fold}"
        validation = sorted(name for name in self.names
                            if self.ids[name] == (fold + 1) % 5)
        test = sorted(name for name in self.names if self.ids[name] == fold)
        train = sorted(set(self.names) - set(test) - set(validation))
        _json(directory / "split.json", {
            "target": "ld", "fold": fold, "num_folds": 5,
            "fold_ids": self.ids, "train": train,
            "validation": validation, "test": test,
        })
        code = {"code/synthetic.py": "a" * 64}
        snapshot = directory / "source_snapshot" / "code" / "synthetic.py"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("synthetic source", encoding="utf-8")
        code["code/synthetic.py"] = _sha(snapshot)
        if candidate:
            protocol_name = "experiments/ld_component_uniform_sampling_protocol.md"
            protocol = directory / "source_snapshot" / protocol_name
            protocol.parent.mkdir(parents=True, exist_ok=True)
            protocol.write_text("locked synthetic component protocol", encoding="utf-8")
            code[protocol_name] = _sha(protocol)
        config = {
            "target": "ld", "fold": fold, "num_folds": 5,
            "outer_test_deferred": True,
            "input_dir": "/synthetic/input",
            "label_dir": "/workspace/ld_study/LD_output_binary",
            "fold_map_sha256": _sha(self.map),
            "label_manifest_sha256": _sha(self.labels),
            "code_sha256": code,
            "architecture": "unet_2d_baseline", "loss": "bce_dice",
            "seed_base": 42, "seed": 42 + fold,
            "patch_size": 512, "overlap": 128,
            "batch_size": 4, "prediction_batch_size": 4,
            "steps_per_epoch": 100, "epochs": 100,
            "learning_rate": 1e-4, "positive_patch_probability": 0.6,
            "lower_percentile": 1.0, "upper_percentile": 99.8,
            "checkpoint_selection": "validation_full_image_dice_at_0.5",
            "threshold_selection": "validation_mean_per_image_dice",
        }
        if candidate:
            config.update({
                "candidate": "component_uniform_positive_center_v1",
                "positive_center_policy": "uniform_8_connected_training_component_then_pixel",
                "protocol_sha256": code[
                    "experiments/ld_component_uniform_sampling_protocol.md"
                ],
            })
        _json(directory / "experiment_config.json", config)
        threshold = (0.45 + fold * 0.05) if candidate else (0.35 + fold * 0.05)
        _json(directory / "threshold.json", {"threshold": threshold})
        artifacts = directory / "validation_artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        manifest_files = []
        metrics_by_name = {}
        csv_path = directory / "validation_artifacts" / "threshold_policy_per_image.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "filename", "fixed_050_dice", "fixed_050_iou",
                "fixed_050_precision", "fixed_050_recall",
            ])
            writer.writeheader()
            for name in validation:
                true_positive, false_positive = counts[name]
                predicted = np.zeros(self.truth.shape, dtype=np.uint8)
                predicted.flat[:true_positive] = 1
                predicted.flat[1000:1000 + false_positive] = 1
                metrics = calculate_binary_metrics(self.truth, predicted)
                metrics_by_name[name] = {
                    "dice": metrics.dice,
                    "iou": metrics.iou,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                }
                fixed = artifacts / "predictions" / "fixed_050" / name
                selected = artifacts / "predictions" / "validation" / name
                probability = artifacts / "probabilities" / name
                for path in (fixed, selected, probability):
                    path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(fixed, predicted * 255, metadata=None)
                tifffile.imwrite(selected, predicted * 255, metadata=None)
                tifffile.imwrite(probability,
                                 np.where(predicted, 0.9, 0.1).astype(np.float32),
                                 metadata=None)
                manifest_files.append({
                    "filename": name,
                    "fixed_050_mask_sha256": _sha(fixed),
                    "validation_mask_sha256": _sha(selected),
                    "probability_sha256": _sha(probability),
                })
                writer.writerow({
                    "filename": name,
                    "fixed_050_dice": metrics.dice,
                    "fixed_050_iou": metrics.iou,
                    "fixed_050_precision": metrics.precision,
                    "fixed_050_recall": metrics.recall,
                })
        _json(artifacts / "manifest.json", {
            "target": "ld", "threshold": threshold,
            "image_count": len(validation), "files": manifest_files,
        })
        mean_dice = sum(metrics_by_name[name]["dice"] for name in validation) / len(validation)
        best_count = (fold + 1) * (3 if candidate else 2)
        with (directory / "training_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "val_full_dice"])
            writer.writeheader()
            writer.writerows([
                {"epoch": 0, "val_full_dice": 0.4},
                {"epoch": best_count - 1, "val_full_dice": mean_dice},
                {"epoch": best_count, "val_full_dice": mean_dice},
            ])
        checkpoint = directory / ("component_unet.keras" if candidate else "baseline_unet.keras")
        checkpoint.write_bytes(b"synthetic checkpoint")
        lock = {
            "target": "ld", "fold": fold,
            "fold_map_sha256": _sha(self.map),
            "label_manifest_sha256": _sha(self.labels),
            "code_sha256": code,
            "experiment_config_sha256": _sha(directory / "experiment_config.json"),
            "split_sha256": _sha(directory / "split.json"),
            "checkpoint_sha256": _sha(checkpoint),
            "threshold_sha256": _sha(directory / "threshold.json"),
            "validation_artifacts_manifest_sha256": _sha(
                directory / "validation_artifacts" / "manifest.json"
            ),
            "threshold": threshold,
        }
        if candidate:
            component_counts = directory / "training_component_counts.json"
            _json(component_counts, {
                "scope": "training_masks_only",
                "eight_connected_components_per_image": {name: 1 for name in train},
            })
            lock["training_component_counts_sha256"] = _sha(component_counts)
        _json(directory / "selection.lock.json", lock)
        # Deliberately malformed outer-test content proves it is never parsed.
        outer = directory / "outer_test" / "test_summary.json"
        outer.parent.mkdir(exist_ok=True)
        outer.write_text("not JSON and not selection evidence", encoding="utf-8")
        return directory, metrics_by_name

    def _fixture(self, candidate_counts) -> None:
        for fold in range(5):
            names = sorted(name for name in self.names
                           if self.ids[name] == (fold + 1) % 5)
            baseline, base_metrics = self._run(
                fold, False, {name: (700, 300) for name in names}
            )
            candidate, cand_metrics = self._run(
                fold, True, {name: candidate_counts(fold, name) for name in names}
            )
            rows = []
            for name in names:
                base, cand = base_metrics[name], cand_metrics[name]
                delta = cand["dice"] - base["dice"]
                pdelta = cand["precision"] - base["precision"]
                rows.append({
                    "filename": name,
                    "baseline_dice_050": base["dice"],
                    "candidate_dice_050": cand["dice"],
                    "paired_dice_delta_050": delta,
                    "baseline_precision_050": base["precision"],
                    "candidate_precision_050": cand["precision"],
                    "paired_precision_delta_050": pdelta,
                    "baseline_recall_050": base["recall"],
                    "candidate_recall_050": cand["recall"],
                })
            dice_values = [row["paired_dice_delta_050"] for row in rows]
            precision_values = [row["paired_precision_delta_050"] for row in rows]
            mean_dice = sum(dice_values) / len(rows)
            mean_precision = sum(precision_values) / len(rows)
            checks = {
                "mean_dice_delta_at_least_0_01": mean_dice >= 0.01,
                "strict_majority_dice_improved": sum(x > 0 for x in dice_values) > len(rows) / 2,
                "no_single_dice_drop_more_than_0_10": all(x >= -0.10 for x in dice_values),
                "mean_precision_drop_no_more_than_0_03": mean_precision >= -0.03,
            }
            _json(candidate / "decision.lock.json", {
                "target": "ld", "candidate": "component_uniform_positive_center_v1",
                "fold": fold,
                "protocol_sha256": _json_read(candidate / "experiment_config.json")[
                    "protocol_sha256"
                ],
                "fold_map_sha256": _sha(self.map),
                "label_manifest_sha256": _sha(self.labels),
                "baseline_selection_lock_sha256": _sha(baseline / "selection.lock.json"),
                "candidate_selection_lock_sha256": _sha(candidate / "selection.lock.json"),
                "baseline_validation_manifest_sha256": _sha(
                    baseline / "validation_artifacts" / "manifest.json"
                ),
                "candidate_validation_manifest_sha256": _sha(
                    candidate / "validation_artifacts" / "manifest.json"
                ),
                "validation_filenames": names,
                "paired_validation_rows": rows,
                "mean_paired_dice_delta_050": mean_dice,
                "mean_paired_precision_delta_050": mean_precision,
                "gate_checks": checks,
                "selected": "candidate" if all(checks.values()) else "baseline",
                "outer_test_labels_opened": False,
            })

    def _policy(self) -> dict:
        return build_method_policy(
            self.baseline, self.candidate, self.map, self.labels, self.label_dir
        )

    def test_four_gate_wins_select_candidate_and_median_uses_gate_loser(self) -> None:
        self._fixture(lambda fold, _: (740, 300) if fold < 4 else (700, 300))
        policy = self._policy()
        self.assertEqual(policy["selected_method"], "candidate")
        self.assertEqual(policy["gate_wins"], 4)
        self.assertEqual(policy["final_epoch_count"], 9)
        self.assertAlmostEqual(policy["deployment_threshold"], 0.55)
        self.assertEqual(len(policy["fold_validation_records"]), 5)
        self.assertEqual(sum(item["validation_count"] for item in policy["fold_gate_records"]), 36)
        self.assertFalse(policy["outer_test_used_for_selection"])

    def test_hash_verified_local_label_relocation_is_allowed(self) -> None:
        self._fixture(lambda *_: (740, 300))
        self.assertNotEqual(_json_read(self.labels)["derived_mask_dir"], str(self.label_dir))
        self.assertNotEqual(_json_read(
            self.baseline / "ld" / "fold_0" / "experiment_config.json"
        )["label_dir"], str(self.label_dir))
        self.assertEqual(self._policy()["image_count"], 36)

    def test_three_gate_wins_retain_baseline(self) -> None:
        self._fixture(lambda fold, _: (760, 300) if fold < 3 else (700, 300))
        policy = self._policy()
        self.assertEqual(policy["selected_method"], "baseline")
        self.assertFalse(policy["global_checks"]["candidate_gate_wins_at_least_4_of_5"])
        self.assertEqual(policy["final_epoch_count"], 6)
        self.assertAlmostEqual(policy["deployment_threshold"], 0.45)

    def test_all_36_mean_is_required_even_with_four_gate_wins(self) -> None:
        self._fixture(lambda fold, _: (720, 300) if fold < 4 else (640, 360))
        policy = self._policy()
        self.assertEqual(policy["gate_wins"], 4)
        self.assertFalse(policy["global_checks"]["all_36_mean_paired_dice_delta_at_least_0_01"])
        self.assertEqual(policy["selected_method"], "baseline")

    def test_each_fold_precision_safety_applies_to_gate_loser(self) -> None:
        self._fixture(lambda fold, _: (760, 300) if fold < 4 else (850, 450))
        policy = self._policy()
        self.assertEqual(policy["gate_wins"], 4)
        self.assertFalse(policy["global_checks"][
            "each_fold_mean_precision_delta_at_least_minus_0_03"
        ])
        self.assertEqual(policy["selected_method"], "baseline")

    def test_one_large_dice_loss_rejects_candidate(self) -> None:
        damaged = next(name for name in self.names if self.ids[name] == 1)
        self._fixture(lambda fold, name: (550, 450) if name == damaged else (760, 300))
        policy = self._policy()
        self.assertEqual(policy["gate_wins"], 4)
        self.assertFalse(policy["global_checks"][
            "no_individual_dice_delta_below_minus_0_10"
        ])
        self.assertEqual(policy["selected_method"], "baseline")

    def test_tampered_gate_or_validation_csv_fails_closed(self) -> None:
        self._fixture(lambda *_: (740, 300))
        gate_path = self.candidate / "ld" / "fold_0" / "decision.lock.json"
        gate = _json_read(gate_path)
        gate["paired_validation_rows"][0]["candidate_dice_050"] = 0.99
        _json(gate_path, gate)
        with self.assertRaisesRegex(ValueError, "Validation evidence differs"):
            self._policy()
        self._fixture(lambda *_: (740, 300))
        csv_path = self.candidate / "ld" / "fold_0" / "validation_artifacts" / "threshold_policy_per_image.csv"
        with csv_path.open("a", encoding="utf-8") as handle:
            handle.write("ld_duplicate.tif,0.5,0.5,0.5\n")
        with self.assertRaisesRegex(ValueError, "Validation CSV does not cover"):
            self._policy()

    def test_prediction_and_derived_label_tampering_fail_closed(self) -> None:
        self._fixture(lambda *_: (740, 300))
        name = next(name for name in self.names if self.ids[name] == 1)
        prediction = (self.candidate / "ld" / "fold_0" / "validation_artifacts"
                      / "predictions" / "fixed_050" / name)
        tifffile.imwrite(prediction, np.zeros(self.truth.shape, dtype=np.uint8),
                         metadata=None)
        with self.assertRaisesRegex(ValueError, "Validation artifacts changed"):
            self._policy()

        self._fixture(lambda *_: (740, 300))
        tifffile.imwrite(self.label_dir / name, np.zeros(self.truth.shape, dtype=np.uint8),
                         metadata=None)
        with self.assertRaisesRegex(ValueError, "Relocated derived label differs"):
            self._policy()

    def test_best_epoch_must_match_recomputed_validation_mean(self) -> None:
        self._fixture(lambda *_: (740, 300))
        path = self.candidate / "ld" / "fold_0" / "training_history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "val_full_dice"])
            writer.writeheader()
            writer.writerow({"epoch": 0, "val_full_dice": 0.99})
        with self.assertRaisesRegex(ValueError, "Best-epoch Dice differs"):
            self._policy()

    def test_label_semantics_and_baseline_candidate_original_path_agree(self) -> None:
        self._fixture(lambda *_: (740, 300))
        manifest = _json_read(self.labels)
        original_manifest = json.loads(json.dumps(manifest))
        manifest["files"][-1] = manifest["files"][0]
        _json(self.labels, manifest)
        with self.assertRaisesRegex(ValueError, "wrong semantics or nonunique records"):
            self._policy()

        _json(self.labels, original_manifest)
        self._fixture(lambda *_: (740, 300))
        config_path = self.candidate / "ld" / "fold_0" / "experiment_config.json"
        config = _json_read(config_path)
        config["label_dir"] = "/wrong/labels"
        _json(config_path, config)
        candidate_dir = config_path.parent
        lock_path = candidate_dir / "selection.lock.json"
        lock = _json_read(lock_path)
        lock["experiment_config_sha256"] = _sha(config_path)
        _json(lock_path, lock)
        gate_path = candidate_dir / "decision.lock.json"
        gate = _json_read(gate_path)
        gate["candidate_selection_lock_sha256"] = _sha(lock_path)
        _json(gate_path, gate)
        with self.assertRaisesRegex(ValueError, "altered another scientific factor"):
            self._policy()


def _json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
