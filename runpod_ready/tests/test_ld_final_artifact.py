"""GPU-free publication checks for an all-36 LD final-model archive."""

from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ld_final_artifact as artifact
from ld_final_artifact import (ARCHIVE_NAME, SHA_NAME, sha256,
                               validate_global_selection, validate_locked_baseline,
                               validate_policy,
                               verify_archive)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class FinalArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fold_map = self.root / "fold_map.json"
        self.label_manifest = self.root / "manifest.json"
        write_json(self.fold_map, {"target": "ld", "num_folds": 5})
        write_json(self.label_manifest, {"rule_id": "ld_antialias_midpoint_v1"})
        self.policy_path = self.root / "policy.json"
        self.global_policy_path = self.root / "global_method_policy.json"
        self.tree = self.root / "remote_run"
        self.model = self.tree / "model"
        self.logs = self.tree / "logs"
        self.model.mkdir(parents=True)
        self.logs.mkdir()
        self.archive = self.root / ARCHIVE_NAME
        self.sha_file = self.root / SHA_NAME
        self.destination = self.root / "verified"
        self.make_artifact("baseline")
        for name, value in (
            ("LOCKED_BASELINE_FINAL_POLICY_SHA256", sha256(self.policy_path)),
            ("LOCKED_BASELINE_GLOBAL_POLICY_SHA256", sha256(self.global_policy_path)),
        ):
            lock_patch = patch.object(artifact, name, value)
            lock_patch.start()
            self.addCleanup(lock_patch.stop)

    def make_artifact(self, method: str) -> None:
        settings = {"patch_size": 512, "overlap": 128,
                    "prediction_batch_size": 4, "lower_percentile": 1.0,
                    "upper_percentile": 99.8}
        policy = {
            "target": "ld", "image_count": 36,
            "outer_test_used_for_selection": False,
            "fold_map_sha256": sha256(self.fold_map),
            "label_manifest_sha256": sha256(self.label_manifest),
            "final_epoch_count": 2, "deployment_threshold": 0.35,
            "scientific_settings": settings,
            "fold_validation_records": [
                {"fold": fold, "validation_best_epoch_count": 2,
                 "validation_selected_threshold": 0.35,
                 "training_history_sha256": "a" * 64}
                for fold in range(5)
            ],
        }
        if method == "baseline":
            policy.update(policy_version="ld_validation_median_v1",
                          candidate_id="ld_baseline_locked",
                          selection_basis="five_fold_inner_validation_only")
            model_name = "final_ld_unet.keras"
            complete = "LD_FINAL_TRAINING_COMPLETE"
        else:
            policy.update(policy_version="ld_global_validation_method_v1",
                          selected_method="candidate",
                          candidate_id="component_uniform_positive_center_v1",
                          selection_basis="predeclared_five_fold_inner_validation_only",
                          global_checks={
                              "candidate_gate_wins_at_least_4_of_5": True,
                              "all_36_mean_paired_dice_delta_at_least_0_01": True,
                              "each_fold_mean_precision_delta_at_least_minus_0_03": True,
                              "no_individual_dice_delta_below_minus_0_10": True,
                          })
            settings["positive_center_policy"] = "uniform_8_connected_training_component_then_pixel"
            model_name = "final_ld_component_unet.keras"
            complete = "LD_COMPONENT_FINAL_TRAINING_COMPLETE"
        write_json(self.policy_path, policy)
        policy_digest = sha256(self.policy_path)
        global_policy = dict(policy)
        if method == "baseline":
            global_policy["policy_version"] = "ld_global_validation_method_v1"
            global_policy["selected_method"] = "baseline"
            global_policy["selection_basis"] = "predeclared_five_fold_inner_validation_only"
        write_json(self.global_policy_path, global_policy)
        write_json(self.model / "production_policy.json", policy)
        snapshot = self.model / "source_snapshot/code/ld_predict.py"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("# frozen source\n", encoding="utf-8")
        config = {
            "target": "ld", "image_count": 36, "epochs": 2,
            "policy_sha256": policy_digest,
            "fold_map_sha256": sha256(self.fold_map),
            "label_manifest_sha256": sha256(self.label_manifest),
            "scientific_settings": settings,
            "final_source_sha256": {"code/ld_predict.py": sha256(snapshot)},
        }
        if method == "candidate":
            config["candidate_id"] = policy["candidate_id"]
        write_json(self.model / "final_fit_config.json", config)
        write_json(self.model / "dataset_report.json", {"all_36": [
            {"filename": f"ld_{i:03d}.tif"} for i in range(1, 37)]})
        write_json(self.model / "environment.json", {
            "python": "3.11.16", "tensorflow": "2.21.0",
            "numpy": "2.4.6", "tifffile": "2026.3.3",
        })
        write_json(self.model / "threshold.json", {
            "threshold": 0.35, "policy_sha256": policy_digest})
        with (self.model / "training_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle,
                                    fieldnames=["epoch", "loss", "soft_dice_coefficient"])
            writer.writeheader()
            writer.writerows([
                {"epoch": 0, "loss": 0.4, "soft_dice_coefficient": 0.5},
                {"epoch": 1, "loss": 0.3, "soft_dice_coefficient": 0.6},
            ])
        checkpoint = self.model / model_name
        with zipfile.ZipFile(checkpoint, "w", compression=zipfile.ZIP_STORED) as bundle:
            bundle.writestr("metadata.json", "{}")
            bundle.writestr("config.json", "{}")
            bundle.writestr("model.weights.h5", b"synthetic weights")
        manifest = {
            "target": "ld", "status": "complete", "model_file": model_name,
            "model_sha256": sha256(checkpoint), "policy_sha256": policy_digest,
            "training_history_sha256": sha256(self.model / "training_history.csv"),
            "training_images": 36, "epochs_completed": 2,
            "semantic_labels": {"background": 0, "foreground": 1},
            "label_rule": "ld_antialias_midpoint_v1", "threshold": 0.35,
            "tile_size": 512, "overlap": 128, "prediction_batch_size": 4,
            "lower_percentile": 1.0, "upper_percentile": 99.8,
            "final_loss": 0.3, "final_soft_dice_on_training_patches": 0.6,
        }
        if method == "candidate":
            manifest["candidate_id"] = policy["candidate_id"]
            manifest["positive_center_policy"] = settings["positive_center_policy"]
        write_json(self.model / "final_model_manifest.json", manifest)
        write_json(self.logs / "selected_method.json", {
            "method": method, "policy_sha256": policy_digest,
            "global_policy_sha256": sha256(self.global_policy_path),
            "fold_map_sha256": sha256(self.fold_map),
            "label_manifest_sha256": sha256(self.label_manifest),
        })
        write_json(self.logs / "global_method_policy.json", global_policy)
        if method == "baseline":
            preflight_name = "global_policy_preflight.log"
            preflight_text = (
                f"LD_GLOBAL_METHOD_LOCK_PASS method=baseline "
                f"final_policy_sha256={policy_digest} "
                f"global_policy_sha256={sha256(self.global_policy_path)}\n")
        else:
            preflight_name = "global_selection_preflight.log"
            preflight_text = (
                f"LD_GLOBAL_METHOD_SELECTION_PASS method=candidate "
                f"global_policy_sha256={sha256(self.global_policy_path)}\n")
        for name in ("global_policy_preflight.log", "global_selection_preflight.log"):
            (self.logs / name).unlink(missing_ok=True)
        for name, text in {
            "runner_exit_status.txt": "0\n", "start_utc.txt": "start\n",
            "end_utc.txt": "end\n", "hostname.txt": "pod\n",
            "runner.sh": "#!/bin/bash\n", "gpu_usage.csv": "time,util\n",
            "gpu_preflight.log": "LD_GPU_PREFLIGHT_PASS\n",
            preflight_name: preflight_text,
            "nvidia_smi_list.txt": "GPU 0: NVIDIA A40\n",
            "training.log": complete + "\n",
        }.items():
            (self.logs / name).write_text(text, encoding="utf-8")
        self.repack()

    def repack(self) -> None:
        with tarfile.open(self.archive, "w:gz") as bundle:
            bundle.add(self.model, arcname="model")
            bundle.add(self.logs, arcname="logs")
        self.sha_file.write_text(f"{sha256(self.archive)}  {ARCHIVE_NAME}\n", encoding="ascii")

    def verify(self) -> dict:
        return verify_archive(self.archive, self.sha_file, self.destination,
                              "baseline", self.policy_path, self.fold_map,
                              self.label_manifest, self.global_policy_path)

    def test_valid_archive_and_no_overwrite(self) -> None:
        result = self.verify()
        self.assertEqual(result["method"], "baseline")
        self.assertTrue((self.destination / "model/final_ld_unet.keras").is_file())
        self.assertTrue((self.destination / "archives" / ARCHIVE_NAME).is_file())
        with self.assertRaises(FileExistsError):
            self.verify()

    def test_candidate_policy_and_model(self) -> None:
        self.make_artifact("candidate")
        self.assertEqual(validate_policy("candidate", self.policy_path,
                                         self.fold_map, self.label_manifest)["selected_method"],
                         "candidate")
        with self.assertRaisesRegex(ValueError, "does not select"):
            validate_policy("baseline", self.policy_path, self.fold_map,
                            self.label_manifest)
        result = verify_archive(self.archive, self.sha_file, self.destination,
                                "candidate", self.policy_path, self.fold_map,
                                self.label_manifest, self.global_policy_path)
        self.assertEqual(result["method"], "candidate")

    def test_baseline_locked_preflight_needs_no_cv_trees(self) -> None:
        result = validate_locked_baseline(
            self.policy_path, self.global_policy_path,
            self.fold_map, self.label_manifest,
        )
        self.assertEqual(result["method"], "baseline")
        self.assertEqual(result["global_policy_sha256"], sha256(self.global_policy_path))

    def test_baseline_locked_preflight_rejects_forged_same_contract(self) -> None:
        selected = json.loads(self.global_policy_path.read_text())
        selected["gate_wins"] = 5
        write_json(self.global_policy_path, selected)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_locked_baseline(
                self.policy_path, self.global_policy_path,
                self.fold_map, self.label_manifest,
            )

    def test_baseline_archive_rejects_recomputation_marker(self) -> None:
        (self.logs / "global_policy_preflight.log").write_text(
            f"LD_GLOBAL_METHOD_SELECTION_PASS method=baseline "
            f"global_policy_sha256={sha256(self.global_policy_path)}\n",
            encoding="utf-8",
        )
        self.repack()
        with self.assertRaisesRegex(ValueError, "Global method preflight marker"):
            self.verify()

    def test_wrong_branch_rejected_after_recomputing_global_choice(self) -> None:
        selected = json.loads(self.global_policy_path.read_text())
        selected["selected_method"] = "candidate"
        selected["candidate_id"] = "component_uniform_positive_center_v1"
        write_json(self.global_policy_path, selected)
        with self.assertRaisesRegex(ValueError, "differs from the locked global selection"):
            validate_global_selection(
                "baseline", self.policy_path, self.global_policy_path,
                self.root / "baseline_cv", self.root / "candidate_cv",
                self.fold_map, self.label_manifest, self.root / "labels", self.root,
                builder=lambda *args: selected,
            )

    def test_forged_global_policy_rejected_against_recomputed_evidence(self) -> None:
        recomputed = json.loads(self.global_policy_path.read_text())
        forged = dict(recomputed)
        forged["gate_wins"] = 5
        write_json(self.global_policy_path, forged)
        with self.assertRaisesRegex(ValueError, "differs from recomputed validation evidence"):
            validate_global_selection(
                "baseline", self.policy_path, self.global_policy_path,
                self.root / "baseline_cv", self.root / "candidate_cv",
                self.fold_map, self.label_manifest, self.root / "labels", self.root,
                builder=lambda *args: recomputed,
            )

    def test_baseline_final_policy_must_match_global_threshold(self) -> None:
        selected = json.loads(self.global_policy_path.read_text())
        selected["deployment_threshold"] = 0.6
        write_json(self.global_policy_path, selected)
        with self.assertRaisesRegex(ValueError, "epoch, threshold or settings"):
            validate_global_selection(
                "baseline", self.policy_path, self.global_policy_path,
                self.root / "baseline_cv", self.root / "candidate_cv",
                self.fold_map, self.label_manifest, self.root / "labels", self.root,
                builder=lambda *args: selected,
            )

    def test_archive_sha_mismatch_leaves_destination_absent(self) -> None:
        self.archive.write_bytes(self.archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.verify()
        self.assertFalse(self.destination.exists())

    def test_keras_crc_failure_even_with_matching_outer_sha(self) -> None:
        checkpoint = self.model / "final_ld_unet.keras"
        data = bytearray(checkpoint.read_bytes())
        offset = data.index(b"synthetic weights")
        data[offset] ^= 1
        checkpoint.write_bytes(data)
        manifest = json.loads((self.model / "final_model_manifest.json").read_text())
        manifest["model_sha256"] = sha256(checkpoint)
        write_json(self.model / "final_model_manifest.json", manifest)
        self.repack()
        with self.assertRaisesRegex(ValueError, "corrupt member"):
            self.verify()
        self.assertFalse(self.destination.exists())

    def test_changed_local_policy_rejected(self) -> None:
        policy = json.loads(self.policy_path.read_text())
        policy["deployment_threshold"] = 0.6
        write_json(self.policy_path, policy)
        with self.assertRaisesRegex(ValueError, "validation medians|policy record|Locked policy|Locked baseline final policy"):
            self.verify()
        self.assertFalse(self.destination.exists())

    def test_model_manifest_threshold_mismatch_rejected(self) -> None:
        manifest_path = self.model / "final_model_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["threshold"] = 0.6
        write_json(manifest_path, manifest)
        self.repack()
        with self.assertRaisesRegex(ValueError, "manifest or threshold"):
            self.verify()
        self.assertFalse(self.destination.exists())

    def test_unsafe_archive_member_rejected(self) -> None:
        with tarfile.open(self.archive, "w:gz") as bundle:
            payload = b"escape"
            info = tarfile.TarInfo("../escape")
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
        self.sha_file.write_text(f"{sha256(self.archive)}  {ARCHIVE_NAME}\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.verify()
        self.assertFalse(self.destination.exists())
        self.assertFalse((self.root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
