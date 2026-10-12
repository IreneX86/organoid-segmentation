"""Synthetic checks for the locked-policy LD OOF error audit.

No test opens a real outer-test file or reads a real biological TIFF.
"""

from __future__ import annotations

import hashlib
import io
import csv
import json
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from ld_validation_error_analysis import analyse_case
    HAS_IMAGE_ANALYSIS_DEPS = True
except ModuleNotFoundError as exc:
    if exc.name not in {"PIL", "scipy"}:
        raise
    # The guard and artifact-integrity tests can still run on a small local
    # Python without SciPy/Pillow. Production audit execution requires them.
    HAS_IMAGE_ANALYSIS_DEPS = False
    placeholder = types.ModuleType("ld_validation_error_analysis")
    placeholder.analyse_case = None
    sys.modules["ld_validation_error_analysis"] = placeholder

import ld_oof_error_audit as audit
from segmentation_metrics import calculate_binary_metrics


class LockedPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = (
            self.root / "policy.json", self.root / "baseline", self.root / "candidate",
            self.root / "archives",
            self.root / "map.json", self.root / "labels.json",
            self.root / "inputs", self.root / "derived",
            self.root / "downloaded" / "final_metadata" / "gated_cv_report",
        )

    def test_missing_lock_rejects_before_outer_reader(self) -> None:
        with mock.patch.object(audit, "load_verified_gated_report") as outer_reader:
            with self.assertRaisesRegex(FileNotFoundError, "not locked"):
                audit.audit_oof(*self.args)
        outer_reader.assert_not_called()

    def test_archive_check_runs_after_policy_but_before_outer_reader(self) -> None:
        events: list[str] = []
        def reject_archive(*args: object) -> None:
            events.append("archives")
            raise ValueError("archive mismatch")

        with (mock.patch.object(
            audit, "verify_method_policy_before_outer_read",
            side_effect=lambda *args: events.append("policy") or {"selected_method": "baseline"},
        ), mock.patch.object(
            audit, "verify_baseline_outer_archives", side_effect=reject_archive,
        ), mock.patch.object(audit, "load_verified_gated_report") as outer_reader):
            with self.assertRaisesRegex(ValueError, "archive mismatch"):
                audit.audit_oof(*self.args)
        self.assertEqual(events, ["policy", "archives"])
        outer_reader.assert_not_called()

    def test_successful_guards_precede_outer_reader(self) -> None:
        events: list[str] = []
        def stop_after_reader(*args: object) -> None:
            events.append("outer_reader")
            raise RuntimeError("synthetic stop")

        with (mock.patch.object(
            audit, "verify_method_policy_before_outer_read",
            side_effect=lambda *args: events.append("policy") or {"selected_method": "baseline"},
        ), mock.patch.object(
            audit, "verify_baseline_outer_archives", side_effect=lambda *args: events.append("archives"),
        ), mock.patch.object(audit, "load_verified_gated_report", side_effect=stop_after_reader)):
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                audit.audit_oof(*self.args)
        self.assertEqual(events, ["policy", "archives", "outer_reader"])

    def test_policy_that_used_outer_test_is_rejected_first(self) -> None:
        self.args[0].write_text(json.dumps({
            "policy_version": "ld_global_validation_method_v1", "target": "ld",
            "selected_method": "candidate", "image_count": 36,
            "selection_basis": "predeclared_five_fold_inner_validation_only",
            "outer_test_used_for_selection": True,
        }), encoding="utf-8")
        with (mock.patch.object(audit, "build_method_policy") as builder,
              mock.patch.object(audit, "load_verified_gated_report") as outer_reader):
            with self.assertRaisesRegex(ValueError, "used outer-test evidence"):
                audit.audit_oof(*self.args)
        builder.assert_not_called()
        outer_reader.assert_not_called()

    def test_changed_saved_policy_rejects_before_outer_reader(self) -> None:
        policy = {
            "policy_version": "ld_global_validation_method_v1", "target": "ld",
            "selected_method": "candidate", "image_count": 36,
            "selection_basis": "predeclared_five_fold_inner_validation_only",
            "outer_test_used_for_selection": False,
            "policy_protocol_sha256": "control", "fold_map_sha256": "control",
            "label_manifest_sha256": "control",
            "fold_gate_records": [{"fold": fold} for fold in range(5)],
        }
        self.args[0].write_text(json.dumps(policy), encoding="utf-8")
        with (mock.patch.object(audit, "_sha256", return_value="control"),
              mock.patch.object(audit, "build_method_policy", return_value={**policy, "selected_method": "baseline"}),
              mock.patch.object(audit, "load_verified_gated_report") as outer_reader):
            with self.assertRaisesRegex(ValueError, "reconstructed validation-only evidence"):
                audit.audit_oof(*self.args)
        outer_reader.assert_not_called()


class SyntheticArchiveIntegrityTests(unittest.TestCase):
    """Use tiny synthetic tarballs; never open downloaded biological outputs."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archives = self.root / "archives"
        self.baseline = self.root / "baseline"
        self.archives.mkdir()
        for fold in range(5):
            name = f"fold_{fold}/outer_test/test_per_image.csv"
            payload = f"synthetic fold {fold}\n".encode()
            extracted = self.baseline / "ld" / f"fold_{fold}" / "outer_test" / "test_per_image.csv"
            extracted.parent.mkdir(parents=True)
            extracted.write_bytes(payload)
            self._archive(fold, [(name, payload)])

    def _archive(self, fold: int, entries: list[tuple[str, bytes]]) -> None:
        archive = self.archives / f"fold_{fold}.tar.gz"
        with tarfile.open(archive, "w:gz") as package:
            for name, payload in entries:
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                package.addfile(member, io.BytesIO(payload))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.archives / f"fold_{fold}.sha256").write_text(
            f"{digest}  /workspace/original/fold_{fold}.tar.gz\n", encoding="ascii"
        )

    def test_five_original_archives_match_extracted_outer_bytes(self) -> None:
        audit.verify_baseline_outer_archives(self.baseline, self.archives)

    def test_rejects_changed_original_archive_sha(self) -> None:
        with (self.archives / "fold_2.tar.gz").open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "archive SHA256 mismatch"):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)

    def test_rejects_changed_extracted_bytes(self) -> None:
        path = self.baseline / "ld" / "fold_3" / "outer_test" / "test_per_image.csv"
        path.write_bytes(b"changed after extraction")
        with self.assertRaisesRegex(ValueError, "outer-test bytes differ"):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)

    def test_rejects_missing_archive_or_extracted_file(self) -> None:
        (self.archives / "fold_1.tar.gz").unlink()
        with self.assertRaises(FileNotFoundError):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)
        self._archive(1, [("fold_1/outer_test/test_per_image.csv", b"synthetic fold 1\n")])
        (self.baseline / "ld" / "fold_1" / "outer_test" / "test_per_image.csv").unlink()
        with self.assertRaisesRegex(ValueError, "file set differs"):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)

    def test_rejects_missing_sha_record(self) -> None:
        (self.archives / "fold_4.sha256").unlink()
        with self.assertRaises(FileNotFoundError):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)

    def test_rejects_extra_extracted_outer_file(self) -> None:
        (self.baseline / "ld" / "fold_0" / "outer_test" / "unexpected.csv").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "file set differs"):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)

    def test_rejects_unsafe_tar_member(self) -> None:
        self._archive(0, [
            ("fold_0/outer_test/test_per_image.csv", b"synthetic fold 0\n"),
            ("fold_0/outer_test/../escape.csv", b"escape"),
        ])
        with self.assertRaisesRegex(ValueError, "Unsafe baseline archive member"):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)

    def test_rejects_duplicate_tar_member(self) -> None:
        name = "fold_0/outer_test/test_per_image.csv"
        self._archive(0, [(name, b"synthetic fold 0\n"), (name, b"later duplicate")])
        with self.assertRaisesRegex(ValueError, "Duplicate baseline archive member"):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)

    def test_rejects_tar_symlink_member(self) -> None:
        archive = self.archives / "fold_0.tar.gz"
        with tarfile.open(archive, "w:gz") as package:
            member = tarfile.TarInfo("fold_0/outer_test/test_per_image.csv")
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/not_a_saved_result"
            package.addfile(member)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.archives / "fold_0.sha256").write_text(
            f"{digest}  /workspace/original/fold_0.tar.gz\n", encoding="ascii"
        )
        with self.assertRaisesRegex(ValueError, "Unsafe baseline archive member"):
            audit.verify_baseline_outer_archives(self.baseline, self.archives)


class SyntheticGatedReportArchiveTests(unittest.TestCase):
    """Exercise the relocated final archive chain with fabricated report bytes."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Path(temporary.name) / "relocated_study"
        self.report = self.study / "downloaded" / "final_metadata" / "gated_cv_report"
        self.report.mkdir(parents=True)
        self.archives = self.study / "archives"
        self.archives.mkdir()
        self.archive = self.archives / "final_metadata.tar.gz"
        self.sha_file = self.archives / "final_metadata.sha256"
        self.marker = self.study / "downloaded" / "verified_final_metadata.json"
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=audit.GATED_ROW_FIELDS)
        writer.writeheader()
        for index in range(36):
            row = {field: "0" for field in audit.GATED_ROW_FIELDS}
            row.update(fold=str(index % 5), filename=f"synthetic_{index:03d}.tif",
                       selected="baseline")
            writer.writerow(row)
        (self.report / "paired_oof_per_image.csv").write_text(output.getvalue(), encoding="utf-8")
        (self.report / "gated_oof_summary.json").write_text(
            json.dumps({"target": "ld", "image_count": 36,
                        "candidate_fold_count": 1}), encoding="utf-8"
        )
        self._make_archive()

    def _make_archive(self, extra: list[tuple[str, bytes]] | None = None) -> None:
        members = [(f"gated_cv_report/{path.name}", path.read_bytes())
                   for path in self.report.iterdir()]
        members.append(("runner_exit_status.txt", b"0\n"))
        members.extend(extra or [])
        with tarfile.open(self.archive, "w:gz") as package:
            for name, payload in members:
                entry = tarfile.TarInfo(name)
                entry.size = len(payload)
                package.addfile(entry, io.BytesIO(payload))
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.sha_file.write_text(
            f"{digest}  /workspace/remote/archives/final_metadata.tar.gz\n",
            encoding="ascii",
        )
        self.marker.write_text(json.dumps({
            "archive_sha256": digest, "image_count": 36,
            "candidate_fold_count": 1, "runner_exit_status": 0,
        }), encoding="utf-8")

    def test_relocated_report_is_bound_to_checked_archive(self) -> None:
        rows, summary, provenance = audit.load_verified_gated_report(self.report)
        self.assertEqual(len(rows), 36)
        self.assertEqual(summary["target"], "ld")
        self.assertEqual(provenance["archive_sha256"],
                         hashlib.sha256(self.archive.read_bytes()).hexdigest())

    def test_rejects_changed_extracted_report_bytes(self) -> None:
        with (self.report / "paired_oof_per_image.csv").open("a", encoding="utf-8") as handle:
            handle.write("tampered\n")
        with self.assertRaisesRegex(ValueError, "bytes differ from final archive"):
            audit.load_verified_gated_report(self.report)

    def test_rejects_changed_archive_and_verification_marker(self) -> None:
        with self.archive.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "archive SHA256 mismatch"):
            audit.load_verified_gated_report(self.report)
        self._make_archive()
        marker = json.loads(self.marker.read_text())
        marker["archive_sha256"] = "0" * 64
        self.marker.write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification marker differs"):
            audit.load_verified_gated_report(self.report)

    def test_rejects_duplicate_or_unsafe_tar_member_even_with_matching_sha(self) -> None:
        self._make_archive([("gated_cv_report/paired_oof_per_image.csv", b"duplicate")])
        with self.assertRaisesRegex(ValueError, "Duplicate final-metadata archive member"):
            audit.load_verified_gated_report(self.report)
        self._make_archive([("gated_cv_report/../escape", b"unsafe")])
        with self.assertRaisesRegex(ValueError, "Unsafe final-metadata archive member"):
            audit.load_verified_gated_report(self.report)


class SyntheticCandidateArchiveIntegrityTests(unittest.TestCase):
    """Bind selected outer artifacts to fabricated candidate fold tarballs."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Path(temporary.name) / "relocated_study"
        self.archives = self.study / "archives"
        self.archives.mkdir(parents=True)
        self.downloaded = self.study / "downloaded"
        self.candidate = self.downloaded / "full"
        self.report = self.downloaded / "final_metadata" / "gated_cv_report"
        self.report.mkdir(parents=True)
        self.fold_ids = {f"synthetic_{index:03d}.tif": index % 5
                         for index in range(36)}
        self.policy = {"fold_gate_records": [
            {"fold": fold, "gate_selected": "candidate" if fold == 2 else "baseline"}
            for fold in range(5)
        ]}
        for fold in range(5):
            lock = self.candidate / "ld" / f"fold_{fold}" / "decision.lock.json"
            lock.parent.mkdir(parents=True)
            lock.write_text(json.dumps({"fold": fold, "selected":
                self.policy["fold_gate_records"][fold]["gate_selected"]}), encoding="utf-8")
            self.policy["fold_gate_records"][fold]["gate_sha256"] = hashlib.sha256(
                lock.read_bytes()
            ).hexdigest()
        for fold in range(5):
            self._archive(fold)

    def _archive(self, fold: int, *, unexpected_outer: bool = False) -> None:
        selected = self.policy["fold_gate_records"][fold]["gate_selected"]
        lock = self.candidate / "ld" / f"fold_{fold}" / "decision.lock.json"
        members = [(f"fold_{fold}/decision.lock.json", lock.read_bytes())]
        if selected == "candidate" or unexpected_outer:
            payload = f"synthetic fold {fold}\n".encode()
            members.append((f"fold_{fold}/outer_test/test_per_image.csv", payload))
            extracted = self.candidate / "ld" / f"fold_{fold}" / "outer_test" / "test_per_image.csv"
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted.write_bytes(payload)
        archive = self.archives / f"fold_{fold}.tar.gz"
        with tarfile.open(archive, "w:gz") as package:
            for name, payload in members:
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                package.addfile(item, io.BytesIO(payload))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.archives / f"fold_{fold}.sha256").write_text(
            f"{digest}  /workspace/remote/archives/fold_{fold}.tar.gz\n", encoding="ascii"
        )
        (self.downloaded / f"verified_fold_{fold}.json").write_text(json.dumps({
            "fold": fold, "archive_sha256": digest, "selected": selected,
            "outer_images": sum(assigned == fold for assigned in self.fold_ids.values())
            if selected == "candidate" else 0,
        }), encoding="utf-8")

    def test_selected_outer_bytes_match_five_fold_archives(self) -> None:
        digests = audit.verify_candidate_outer_archives(
            self.candidate, self.report, self.policy, self.fold_ids
        )
        self.assertEqual(set(digests), {str(fold) for fold in range(5)})

    def test_rejects_changed_selected_outer_artifact(self) -> None:
        path = self.candidate / "ld" / "fold_2" / "outer_test" / "test_per_image.csv"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes differ from archive"):
            audit.verify_candidate_outer_archives(
                self.candidate, self.report, self.policy, self.fold_ids
            )

    def test_rejects_baseline_selected_candidate_outer_artifact(self) -> None:
        self._archive(0, unexpected_outer=True)
        with self.assertRaisesRegex(ValueError, "outer-test files"):
            audit.verify_candidate_outer_archives(
                self.candidate, self.report, self.policy, self.fold_ids
            )

    def test_rejects_extracted_decision_lock_different_from_fold_archive(self) -> None:
        lock = self.candidate / "ld" / "fold_2" / "decision.lock.json"
        lock.write_text('{"fold": 2, "selected": "baseline"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "decision lock differs from verified archive"):
            audit.verify_candidate_outer_archives(
                self.candidate, self.report, self.policy, self.fold_ids
            )


class SyntheticGatedReportPairingTests(unittest.TestCase):
    """Check all 36 score rows against fabricated per-fold artifact records."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.fold_ids = {f"synthetic_{index:03d}.tif": index % 5
                         for index in range(36)}
        self.baseline_rows = []
        self.candidate_rows = []
        self.report_rows = []
        self.decisions = []
        policy_gates = []
        for fold in range(5):
            selected = "candidate" if fold == 2 else "baseline"
            path = self.candidate / "ld" / f"fold_{fold}" / "decision.lock.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "fold": fold, "selected": selected,
                "mean_paired_dice_delta_050": 0.02,
            }), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            policy_gates.append({"fold": fold, "gate_selected": selected,
                                 "gate_sha256": digest})
            self.decisions.append({
                "fold": fold, "selected": selected,
                "decision_lock_sha256": digest,
                "mean_paired_validation_dice_delta_050": 0.02,
            })
        self.policy = {"fold_gate_records": policy_gates}
        for name, fold in self.fold_ids.items():
            baseline = {"fold": fold, "filename": name,
                        "foreground_fraction_true": 0.01,
                        "foreground_fraction_predicted": 0.02}
            baseline.update({metric: 0.4 for metric in audit.SCORE_FIELDS})
            baseline.update({f"fixed_050_{metric}": 0.45 for metric in audit.SCORE_FIELDS})
            self.baseline_rows.append(baseline)
            chosen = dict(baseline)
            if fold == 2:
                chosen.update({metric: 0.6 for metric in audit.SCORE_FIELDS})
                chosen.update({f"fixed_050_{metric}": 0.65 for metric in audit.SCORE_FIELDS})
                self.candidate_rows.append({key: value for key, value in chosen.items()
                                            if key != "fold"})
            report = {"fold": str(fold), "filename": name,
                      "selected": "candidate" if fold == 2 else "baseline",
                      "foreground_fraction_true": str(baseline["foreground_fraction_true"])}
            for metric in audit.SCORE_FIELDS:
                report[f"baseline_{metric}"] = str(baseline[metric])
                report[f"selected_{metric}"] = str(chosen[metric])
                report[f"paired_{metric}_delta"] = str(chosen[metric] - baseline[metric])
                report[f"selected_fixed_050_{metric}"] = str(chosen[f"fixed_050_{metric}"])
            self.report_rows.append(report)
        self.baseline_report = {"oof": {
            metric: audit._describe([float(row[metric]) for row in self.baseline_rows])
            for metric in (*audit.SCORE_FIELDS, "foreground_fraction_true",
                           "foreground_fraction_predicted")
        }}
        self.summary = {
            "target": "ld", "procedure":
            "predeclared_fold_local_validation_gated_component_sampling",
            "fold_count": 5, "image_count": 36, "candidate_fold_count": 1,
            "fold_decisions": self.decisions,
            "baseline_oof": self.baseline_report["oof"],
            "selected_oof": {metric: audit._describe([
                float(row[f"selected_{metric}"]) for row in self.report_rows
            ]) for metric in audit.SCORE_FIELDS},
            "paired_delta": {metric: audit._describe([
                float(row[f"paired_{metric}_delta"]) for row in self.report_rows
            ]) for metric in audit.SCORE_FIELDS},
        }

    def _verify(self) -> dict[str, dict[str, str]]:
        with (mock.patch.object(audit, "aggregate_cv", return_value=(
                self.baseline_rows, self.baseline_report)),
              mock.patch.object(audit, "_candidate_test_rows",
                                return_value=self.candidate_rows) as candidate_reader):
            result = audit.verify_gated_report_against_folds(
                self.report_rows, self.summary, self.policy,
                self.baseline, self.candidate, self.fold_ids,
            )
        candidate_reader.assert_called_once()
        return result

    def test_36_rows_match_locked_remote_report_and_local_artifacts(self) -> None:
        self.assertEqual(set(self._verify()), set(self.fold_ids))

    def test_rejects_changed_per_image_score_or_delta(self) -> None:
        candidate_row = next(row for row in self.report_rows if row["selected"] == "candidate")
        candidate_row["selected_dice"] = "0.99"
        with self.assertRaisesRegex(ValueError, "selected dice differs"):
            self._verify()
        candidate_row["selected_dice"] = "0.6"
        candidate_row["paired_dice_delta"] = "0.99"
        with self.assertRaisesRegex(ValueError, "paired dice delta differs"):
            self._verify()

    def test_rejects_wrong_gate_hash_or_selected_fixed_score(self) -> None:
        self.decisions[2]["decision_lock_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "locked validation gate"):
            self._verify()
        self.decisions[2]["decision_lock_sha256"] = self.policy["fold_gate_records"][2]["gate_sha256"]
        candidate_row = next(row for row in self.report_rows if row["selected"] == "candidate")
        candidate_row["selected_fixed_050_dice"] = "0.99"
        with self.assertRaisesRegex(ValueError, "fixed-0.50 dice differs"):
            self._verify()


class SyntheticErrorGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((40, 50), dtype=np.uint16)
        self.truth = np.zeros((40, 50), dtype=np.uint8)
        self.truth[5:8, 5:8] = 1  # one nine-pixel semantic component
        self.image[self.truth == 1] = 1000

    @unittest.skipUnless(HAS_IMAGE_ANALYSIS_DEPS, "SciPy and Pillow required for geometry checks")
    def test_paired_fp_fn_small_component_and_stratum(self) -> None:
        baseline_probability = np.zeros(self.truth.shape, dtype=np.float32)
        baseline_probability[5, 5] = 0.9
        baseline_probability[5, 8] = 0.9  # FP at GT boundary
        baseline_probability[30, 30] = 0.9  # distant FP
        selected_probability = np.zeros(self.truth.shape, dtype=np.float32)
        selected_probability[5:8, 5:8] = 0.9
        selected_probability[7, 7] = 0.1  # one FN near boundary
        baseline = analyse_case(self.image, self.truth, baseline_probability, 0.5)
        selected = analyse_case(self.image, self.truth, selected_probability, 0.5)
        row = audit.compare_case(0, "ld_000.tif", "candidate", baseline, selected)
        self.assertEqual(row["foreground_stratum"], "above_0_1_to_0_5_percent")
        self.assertEqual(row["baseline_fp_within_2px_of_gt"], 1)
        self.assertEqual(row["baseline_fp_farther_than_10px_from_gt"], 1)
        self.assertEqual(row["selected_fp_farther_than_10px_from_gt"], 0)
        self.assertEqual(row["selected_fn_within_2px_of_gt_boundary"], 1)
        self.assertEqual(row["selected_small_component_count_proxy"], 1)
        self.assertEqual(row["baseline_small_component_detected_any_proxy"], 1)
        self.assertEqual(row["baseline_small_component_detected_half_proxy"], 0)
        self.assertEqual(row["selected_small_component_detected_half_proxy"], 1)
        self.assertGreater(row["paired_dice_delta"], 0)

    def test_baseline_gate_produces_exact_zero_paired_deltas(self) -> None:
        metrics = _synthetic_metrics(self.truth, self.truth)
        row = audit.compare_case(0, "ld_000.tif", "baseline", metrics, metrics)
        for field in (*audit.SCORE_FIELDS, *audit.COUNT_FIELDS):
            self.assertEqual(row[f"paired_{field}_delta"], 0)

    def test_36_image_summary_keeps_strata_and_fold_counts_descriptive(self) -> None:
        low_truth = np.zeros((100, 100), dtype=np.uint8)
        low_truth[1:3, 1:3] = 1
        high_truth = np.zeros((100, 100), dtype=np.uint8)
        high_truth[1:20, 1:20] = 1
        rows = []
        for index in range(36):
            truth = low_truth if index < 18 else high_truth
            metrics = _synthetic_metrics(truth, truth)
            rows.append(audit.compare_case(
                index % 5, f"ld_{index:03d}.tif", "baseline", metrics, metrics
            ))
        summary = audit.summarise_rows(rows, {"selected_method": "baseline"})
        self.assertEqual(summary["all"]["image_count"], 36)
        self.assertEqual(summary["by_foreground_stratum"]["at_most_0_1_percent"]["image_count"], 18)
        self.assertEqual(summary["by_foreground_stratum"]["above_0_5_percent"]["image_count"], 18)
        self.assertEqual(sum(group["image_count"] for group in summary["by_fold"].values()), 36)
        self.assertEqual(summary["all"]["paired_dice_delta"]["mean"], 0.0)

    def test_mismatched_truth_fraction_is_rejected(self) -> None:
        baseline = _synthetic_metrics(self.truth, self.truth)
        selected = dict(baseline, foreground_fraction_true=0.0)
        with self.assertRaisesRegex(ValueError, "same truth pixel grid"):
            audit.compare_case(0, "ld_000.tif", "candidate", baseline, selected)

    def test_saved_mask_must_match_probability_and_threshold(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name)
        name = "ld_001.tif"
        (run / "outer_test" / "probabilities").mkdir(parents=True)
        for policy in ("validation", "fixed_050"):
            (run / "outer_test" / "predictions" / policy).mkdir(parents=True)
        (run / "threshold.json").write_text('{"threshold": 0.5}', encoding="utf-8")
        (run / "selection.lock.json").write_text('{"threshold": 0.5}', encoding="utf-8")
        probability = self.truth.astype(np.float32)
        tifffile.imwrite(run / "outer_test" / "probabilities" / name, probability, metadata=None)
        correct = self.truth * 255
        tifffile.imwrite(run / "outer_test" / "predictions" / "validation" / name, correct, metadata=None)
        tifffile.imwrite(run / "outer_test" / "predictions" / "fixed_050" / name, correct, metadata=None)
        expected = _synthetic_metrics(self.truth, self.truth)
        scores = {field: expected[field] for field in
                  (*audit.SCORE_FIELDS, "foreground_fraction_true")}
        with mock.patch.object(audit, "analyse_case", return_value=expected):
            actual = audit._read_locked_case(
                run, name, self.image, self.truth, scores,
                {field: expected[field] for field in audit.SCORE_FIELDS},
            )
        self.assertEqual(actual["dice"], 1.0)
        with (mock.patch.object(audit, "analyse_case", return_value=expected),
              self.assertRaisesRegex(ValueError, "fixed-0.50 dice differs")):
            audit._read_locked_case(
                run, name, self.image, self.truth, scores,
                {**{field: expected[field] for field in audit.SCORE_FIELDS}, "dice": 0.0},
            )
        corrupted = correct.copy()
        corrupted[30, 30] = 255
        tifffile.imwrite(run / "outer_test" / "predictions" / "validation" / name, corrupted, metadata=None)
        with (mock.patch.object(audit, "analyse_case", return_value=expected),
              self.assertRaisesRegex(ValueError, "mask differs from probability")):
            audit._read_locked_case(run, name, self.image, self.truth, scores)


def _synthetic_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Supply complete synthetic audit fields for dependency-light contract tests."""
    result = calculate_binary_metrics(truth, prediction).to_dict()
    result.update({field: 0 for field in audit.COUNT_FIELDS})
    result.update({field: None for field in audit.OPTIONAL_FIELDS})
    return result


if __name__ == "__main__":
    unittest.main()
