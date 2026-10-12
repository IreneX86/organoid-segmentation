"""Synthetic checks for the locked-policy LD OOF error audit.

No test opens a real outer-test file or reads a real biological TIFF.
"""

from __future__ import annotations

import hashlib
import io
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
        )

    def test_missing_lock_rejects_before_outer_reader(self) -> None:
        with mock.patch.object(audit, "aggregate_gated_candidate") as outer_reader:
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
        ), mock.patch.object(audit, "aggregate_gated_candidate") as outer_reader):
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
        ), mock.patch.object(audit, "aggregate_gated_candidate", side_effect=stop_after_reader)):
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
              mock.patch.object(audit, "aggregate_gated_candidate") as outer_reader):
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
              mock.patch.object(audit, "aggregate_gated_candidate") as outer_reader):
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
        scores = {field: expected[field] for field in audit.SCORE_FIELDS}
        with mock.patch.object(audit, "analyse_case", return_value=expected):
            actual = audit._read_locked_case(run, name, self.image, self.truth, scores)
        self.assertEqual(actual["dice"], 1.0)
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
