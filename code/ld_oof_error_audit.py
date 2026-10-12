"""Exploratory LD OOF error audit, runnable only after the method policy is locked.

The global policy is reconstructed exclusively from saved inner-validation
evidence and compared with the supplied lock *before* opening any ``outer_test``
artifact, original input TIFF, or derived outer-test label. The later OOF audit
does not select a model, threshold, or hyperparameter. Because the rotated
validation folds contain these same 36 images, this OOF description is not an
independent confirmation of the globally selected method.

Example, after ``ld_method_policy.py`` has written its immutable policy::

    python code/ld_oof_error_audit.py \
      --method-policy experiments/results/ld_method_policy.lock.json \
      --baseline-cv-root runpod_results/ld_baseline_20261012/downloaded/full \
      --candidate-cv-root runpod_results/ld_component_20261012/downloaded/full \
      --baseline-archives runpod_results/ld_baseline_20261012/archives \
      --gated-report-dir runpod_results/ld_component_20261012/downloaded/final_metadata/gated_cv_report \
      --fold-map experiments/results/ld_five_fold_map_v1.json \
      --label-manifest data/ld_binary_midpoint_v1/LD_output/manifest.json \
      --input-dir /path/to/LD_input \
      --label-dir data/ld_binary_midpoint_v1/LD_output \
      --output-dir outputs/ld_oof_error_audit_YYYYMMDD

The output directory must not exist. Only small CSV/JSON descriptions are
written; image, label, probability, and mask TIFFs remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from ld_component_report import _candidate_test_rows
from ld_cv_report import aggregate_cv
from ld_method_policy import POLICY_PATH, _fold_dir, _sha256, build_method_policy
from ld_validation_error_analysis import analyse_case
from segmentation_metrics import calculate_binary_metrics
from tiff_io import read_tiff_2d


# These strata are descriptive and fixed before viewing OOF errors. LD occupies
# much less of an image than Mito; the first boundary matches the existing
# validation-only LD error audit (0.10% foreground).
STRATA = (
    ("at_most_0_1_percent", 0.001),
    ("above_0_1_to_0_5_percent", 0.005),
    ("above_0_5_percent", math.inf),
)
SCORE_FIELDS = ("dice", "iou", "precision", "recall")
COUNT_FIELDS = (
    "tp_pixels", "fp_pixels", "fn_pixels", "fp_within_2px_of_gt",
    "fp_farther_than_10px_from_gt", "fn_within_2px_of_gt_boundary",
    "fn_farther_than_2px_inside_gt", "fn_farther_than_10px_from_prediction",
    "small_component_count_proxy", "small_component_detected_any_proxy",
    "small_component_detected_half_proxy",
)
OPTIONAL_FIELDS = (
    "small_component_recall_any_proxy", "small_component_recall_half_proxy",
    "fp_intensity_median_normalised", "gt_intensity_median_normalised",
    "tn_intensity_median_normalised",
)
GATED_ROW_FIELDS = (
    "fold", "filename", "selected",
    *(field for metric in SCORE_FIELDS for field in (
        f"baseline_{metric}", f"selected_{metric}", f"paired_{metric}_delta",
        f"selected_fixed_050_{metric}",
    )),
    "foreground_fraction_true",
)


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _safe_name(name: object) -> str:
    if (not isinstance(name, str) or not name or Path(name).name != name
            or Path(name).suffix.lower() not in {".tif", ".tiff"}):
        raise ValueError(f"Unsafe LD TIFF filename: {name!r}")
    return name


def verify_method_policy_before_outer_read(
    policy_path: Path, baseline_root: Path, candidate_root: Path,
    fold_map: Path, label_manifest: Path, label_dir: Path,
) -> dict[str, Any]:
    """Fail closed until a complete, unchanged validation-only policy exists.

    This function never resolves an ``outer_test`` path. It first checks the
    policy file and its frozen control-file hashes without reading TIFF pixels;
    only then does it reconstruct the policy from inner-validation evidence.
    That reconstruction verifies derived-label hashes and validation pixels.
    """
    if not policy_path.is_file():
        raise FileNotFoundError(f"Global validation-only method policy is not locked: {policy_path}")
    saved = _json(policy_path)
    if (saved.get("policy_version") != "ld_global_validation_method_v1"
            or saved.get("target") != "ld"
            or saved.get("selected_method") not in {"baseline", "candidate"}
            or saved.get("selection_basis") != "predeclared_five_fold_inner_validation_only"
            or saved.get("outer_test_used_for_selection") is not False
            or saved.get("image_count") != 36):
        raise ValueError("Global method policy is absent, incomplete, or used outer-test evidence")
    if (saved.get("policy_protocol_sha256") != _sha256(POLICY_PATH)
            or saved.get("fold_map_sha256") != _sha256(fold_map)
            or saved.get("label_manifest_sha256") != _sha256(label_manifest)):
        raise ValueError("Global method policy does not match frozen protocol/data manifests")
    gates = saved.get("fold_gate_records")
    if (not isinstance(gates, list) or len(gates) != 5
            or {record.get("fold") for record in gates if isinstance(record, dict)}
            != set(range(5))):
        raise ValueError("Global method policy lacks five locked fold gates")
    reconstructed = build_method_policy(
        baseline_root, candidate_root, fold_map, label_manifest, label_dir
    )
    if saved != reconstructed:
        raise ValueError("Global method policy differs from reconstructed validation-only evidence")
    return saved


def _archived_outer_hashes(
    archive: Path, fold: int, *, kind: str = "baseline", allow_empty: bool = False,
) -> tuple[dict[str, str], str | None]:
    """Hash original tar member bytes without extracting or trusting tar paths."""
    prefix = f"fold_{fold}/outer_test/"
    decision_name = f"fold_{fold}/decision.lock.json"
    seen: set[tuple[str, ...]] = set()
    outer_hashes: dict[str, str] = {}
    decision_hash: str | None = None
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            raw_name = member.name
            parts = raw_name.rstrip("/").split("/")
            if (not raw_name or raw_name.startswith("/") or "\\" in raw_name
                    or any(part in {"", ".", ".."} for part in parts)
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe {kind} archive member: {archive}: {raw_name!r}")
            canonical = tuple(part.casefold() for part in parts)
            if canonical in seen:
                raise ValueError(f"Duplicate {kind} archive member: {archive}: {raw_name!r}")
            seen.add(canonical)
            if "outer_test" in parts and parts[0] != f"fold_{fold}":
                raise ValueError(f"Unexpected fold's outer-test member: {archive}: {raw_name!r}")
            if member.isdir() or (not raw_name.startswith(prefix) and raw_name != decision_name):
                continue
            source = package.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable {kind} archive member: {archive}: {raw_name!r}")
            digest = hashlib.sha256()
            with source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if raw_name == decision_name:
                decision_hash = digest.hexdigest()
            else:
                outer_hashes[raw_name[len(prefix):]] = digest.hexdigest()
    if not outer_hashes and not allow_empty:
        raise ValueError(f"{kind.capitalize()} archive lacks fold {fold} outer-test files: {archive}")
    return outer_hashes, decision_hash


def verify_baseline_outer_archives(baseline_root: Path, archives: Path) -> None:
    """Bind each extracted outer-test file to a checked original fold archive.

    Call only after the validation-only global method policy has passed. This
    guard rejects changed archive bytes, missing/extra extracted files, and
    mismatched file bytes before the OOF report reads any test metrics.
    """
    for fold in range(5):
        archive = archives / f"fold_{fold}.tar.gz"
        sha_file = archives / f"fold_{fold}.sha256"
        lines = sha_file.read_text(encoding="ascii").splitlines()
        record = re.fullmatch(r"([0-9a-f]{64})  (\S+)", lines[0]) if len(lines) == 1 else None
        if record is None or Path(record.group(2)).name != archive.name:
            raise ValueError(f"Invalid baseline archive SHA256 record: {sha_file}")
        if _sha256(archive) != record.group(1):
            raise ValueError(f"Baseline archive SHA256 mismatch: {archive}")
        archived, _ = _archived_outer_hashes(archive, fold)
        outer_root = _fold_dir(baseline_root, fold) / "outer_test"
        if not outer_root.is_dir() or outer_root.is_symlink():
            raise ValueError(f"Missing or unsafe extracted outer-test directory: {outer_root}")
        extracted: dict[str, Path] = {}
        for path in outer_root.rglob("*"):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError(f"Unsafe extracted outer-test path: {path}")
            if path.is_file():
                extracted[path.relative_to(outer_root).as_posix()] = path
        if set(extracted) != set(archived):
            raise ValueError(f"Extracted outer-test file set differs from archive: fold {fold}")
        for relative, path in extracted.items():
            if _sha256(path) != archived[relative]:
                raise ValueError(f"Extracted outer-test bytes differ from archive: {path}")


def verify_candidate_outer_archives(
    candidate_root: Path, report_dir: Path, policy: dict[str, Any],
    fold_ids: dict[str, int],
) -> dict[str, str]:
    """Bind candidate outer files to verified fold archives after policy lock."""
    study = report_dir.parent.parent.parent
    archives = study / "archives"
    downloaded = study / "downloaded"
    gates = {int(record["fold"]): record for record in policy["fold_gate_records"]}
    digests: dict[str, str] = {}
    for fold in range(5):
        archive = archives / f"fold_{fold}.tar.gz"
        sha_file = archives / f"fold_{fold}.sha256"
        marker_path = downloaded / f"verified_fold_{fold}.json"
        if any(path.is_symlink() for path in (archive, sha_file, marker_path)):
            raise ValueError(f"Unsafe candidate archive provenance path: fold {fold}")
        lines = sha_file.read_text(encoding="ascii").splitlines()
        record = re.fullmatch(r"([0-9a-fA-F]{64})  (\S+)", lines[0]) if len(lines) == 1 else None
        if record is None or Path(record.group(2)).name != archive.name:
            raise ValueError(f"Invalid candidate archive SHA256 record: {sha_file}")
        digest = _sha256(archive)
        if digest != record.group(1).lower():
            raise ValueError(f"Candidate archive SHA256 mismatch: {archive}")
        marker = _json(marker_path)
        selected = gates[fold]["gate_selected"]
        names = {name for name, assigned in fold_ids.items() if assigned == fold}
        if (marker.get("fold") != fold or marker.get("archive_sha256") != digest
                or marker.get("selected") != selected
                or marker.get("outer_images") != (len(names) if selected == "candidate" else 0)):
            raise ValueError(f"Candidate fold {fold} verification marker differs from lock/archive")
        archived, archived_decision_hash = _archived_outer_hashes(
            archive, fold, kind="candidate", allow_empty=selected == "baseline"
        )
        decision_path = _fold_dir(candidate_root, fold) / "decision.lock.json"
        if (archived_decision_hash is None
                or archived_decision_hash != gates[fold]["gate_sha256"]
                or decision_path.is_symlink()
                or _sha256(decision_path) != archived_decision_hash):
            raise ValueError(f"Candidate fold {fold} decision lock differs from verified archive")
        outer_root = _fold_dir(candidate_root, fold) / "outer_test"
        if selected == "baseline":
            if archived or outer_root.exists() or outer_root.is_symlink():
                raise ValueError(f"Baseline-selected candidate fold {fold} has outer-test files")
        else:
            if not outer_root.is_dir() or outer_root.is_symlink():
                raise ValueError(f"Missing or unsafe candidate outer-test directory: {outer_root}")
            extracted: dict[str, Path] = {}
            for path in outer_root.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValueError(f"Unsafe extracted candidate outer-test path: {path}")
                if path.is_file():
                    extracted[path.relative_to(outer_root).as_posix()] = path
            if set(extracted) != set(archived):
                raise ValueError(f"Extracted candidate outer-test file set differs from archive: fold {fold}")
            for relative, path in extracted.items():
                if _sha256(path) != archived[relative]:
                    raise ValueError(f"Extracted candidate outer-test bytes differ from archive: {path}")
        digests[str(fold)] = digest
    return digests


def load_verified_gated_report(
    report_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Bind the extracted remote report to its checked final-metadata archive.

    The directory may move with its study tree, but retains the downloaded /
    final_metadata / gated_cv_report layout. No fold gate is recomputed here.
    """
    if (report_dir.name != "gated_cv_report"
            or report_dir.parent.name != "final_metadata"
            or report_dir.parent.parent.name != "downloaded"):
        raise ValueError("Gated report must be inside downloaded/final_metadata/gated_cv_report")
    downloaded = report_dir.parent.parent
    archives = downloaded.parent / "archives"
    archive = archives / "final_metadata.tar.gz"
    sha_file = archive.with_suffix("").with_suffix(".sha256")
    marker_path = downloaded / "verified_final_metadata.json"
    for path in (report_dir, report_dir.parent, downloaded, archives,
                 archive, sha_file, marker_path):
        if path.is_symlink():
            raise ValueError(f"Unsafe gated-report provenance path: {path}")
    if not report_dir.is_dir():
        raise FileNotFoundError(f"Missing downloaded gated report: {report_dir}")
    lines = sha_file.read_text(encoding="ascii").splitlines()
    record = re.fullmatch(r"([0-9a-fA-F]{64})  (\S+)", lines[0]) if len(lines) == 1 else None
    if record is None or Path(record.group(2)).name != archive.name:
        raise ValueError(f"Invalid final-metadata archive SHA256 record: {sha_file}")
    archive_digest = _sha256(archive)
    if archive_digest != record.group(1).lower():
        raise ValueError(f"Final-metadata archive SHA256 mismatch: {archive}")
    marker = _json(marker_path)
    if (marker.get("archive_sha256") != archive_digest
            or marker.get("image_count") != 36
            or marker.get("runner_exit_status") != 0
            or type(marker.get("candidate_fold_count")) is not int
            or marker["candidate_fold_count"] not in range(6)):
        raise ValueError("Final-metadata verification marker differs from archive")

    report_names = {"gated_oof_summary.json", "paired_oof_per_image.csv"}
    archived: dict[str, str] = {}
    runner_exit: str | None = None
    seen: set[tuple[str, ...]] = set()
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            raw_name = member.name
            parts = raw_name.rstrip("/").split("/")
            if (not raw_name or raw_name.startswith("/") or "\\" in raw_name
                    or any(part in {"", ".", ".."} for part in parts)
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe final-metadata archive member: {raw_name!r}")
            canonical = tuple(part.casefold() for part in parts)
            if canonical in seen:
                raise ValueError(f"Duplicate final-metadata archive member: {raw_name!r}")
            seen.add(canonical)
            if parts[0] == "gated_cv_report" and member.isdir() and len(parts) != 1:
                raise ValueError(f"Unexpected gated-report archive directory: {raw_name!r}")
            if not member.isfile():
                continue
            if parts[0] == "gated_cv_report" and (
                len(parts) != 2 or parts[1] not in report_names
            ):
                raise ValueError(f"Unexpected gated-report archive member: {raw_name!r}")
            if raw_name not in {"runner_exit_status.txt", *(
                f"gated_cv_report/{name}" for name in report_names
            )}:
                continue
            source = package.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable final-metadata archive member: {raw_name!r}")
            digest = hashlib.sha256()
            with source:
                if raw_name == "runner_exit_status.txt":
                    runner_exit = source.read().decode("ascii").strip()
                else:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                    archived[parts[1]] = digest.hexdigest()
    if set(archived) != report_names or runner_exit != "0":
        raise ValueError("Final-metadata archive lacks a complete successful gated report")
    extracted = list(report_dir.iterdir())
    if ({path.name for path in extracted} != report_names
            or any(path.is_symlink() or not path.is_file() for path in extracted)):
        raise ValueError("Extracted gated-report file set differs from final archive")
    for name in report_names:
        if _sha256(report_dir / name) != archived[name]:
            raise ValueError(f"Extracted gated-report bytes differ from final archive: {name}")
    with (report_dir / "paired_oof_per_image.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != GATED_ROW_FIELDS:
            raise ValueError("Gated-report CSV has an unexpected field schema")
        rows = list(reader)
    if len(rows) != 36 or any(None in row or None in row.values() for row in rows):
        raise ValueError("Gated-report CSV lacks 36 complete image rows")
    summary = _json(report_dir / "gated_oof_summary.json")
    if summary.get("candidate_fold_count") != marker["candidate_fold_count"]:
        raise ValueError("Gated-report fold count differs from final-metadata marker")
    return rows, summary, {
        "archive_sha256": archive_digest,
        "paired_oof_per_image_sha256": archived["paired_oof_per_image.csv"],
        "gated_oof_summary_sha256": archived["gated_oof_summary.json"],
    }


def _finite_report_number(value: object, description: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid gated-report {description}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid gated-report {description}") from exc
    if not math.isfinite(number) or (unit_interval and not 0 <= number <= 1):
        raise ValueError(f"Invalid gated-report {description}")
    return number


def _same_report_number(actual: object, expected: object, description: str,
                        *, unit_interval: bool = False) -> None:
    number = _finite_report_number(actual, description, unit_interval=unit_interval)
    reference = _finite_report_number(expected, description, unit_interval=unit_interval)
    if not math.isclose(number, reference, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Gated-report {description} differs from saved fold artifact")


def verify_gated_report_against_folds(
    report_rows: list[dict[str, str]], summary: dict[str, Any], policy: dict[str, Any],
    baseline_root: Path, candidate_root: Path, fold_ids: dict[str, int],
) -> dict[str, dict[str, str]]:
    """Check every reported score against locked per-fold outer-test evidence."""
    if (summary.get("target") != "ld"
            or summary.get("procedure")
            != "predeclared_fold_local_validation_gated_component_sampling"
            or summary.get("fold_count") != 5 or summary.get("image_count") != 36):
        raise ValueError("Gated-report summary has the wrong LD procedure or coverage")
    baseline_rows, baseline_report = aggregate_cv(baseline_root)
    baseline_by_name = {str(row["filename"]): row for row in baseline_rows}
    if (len(baseline_by_name) != 36 or set(baseline_by_name) != set(fold_ids)
            or any(int(row["fold"]) != fold_ids[name]
                   for name, row in baseline_by_name.items())):
        raise ValueError("Baseline OOF artifacts differ from frozen LD fold assignments")
    policy_gates = {int(record["fold"]): record for record in policy["fold_gate_records"]}
    decisions = summary.get("fold_decisions")
    if not isinstance(decisions, list) or len(decisions) != 5:
        raise ValueError("Gated-report summary lacks five fold decisions")
    selected_by_name: dict[str, dict[str, Any]] = {}
    selected_folds = 0
    for fold, reported in enumerate(decisions):
        if not isinstance(reported, dict):
            raise ValueError(f"Invalid gated-report fold {fold} decision")
        run = _fold_dir(candidate_root, fold)
        lock_path = run / "decision.lock.json"
        locked = _json(lock_path)
        selected = policy_gates[fold]["gate_selected"]
        if (selected not in {"baseline", "candidate"}
                or policy_gates[fold]["gate_sha256"] != _sha256(lock_path)
                or locked.get("fold") != fold or locked.get("selected") != selected
                or reported.get("fold") != fold or reported.get("selected") != selected
                or reported.get("decision_lock_sha256") != _sha256(lock_path)):
            raise ValueError(f"Gated-report fold {fold} differs from locked validation gate")
        _same_report_number(
            reported.get("mean_paired_validation_dice_delta_050"),
            locked.get("mean_paired_dice_delta_050"), f"fold {fold} validation Dice delta",
        )
        names = {name for name, assigned in fold_ids.items() if assigned == fold}
        if selected == "candidate":
            selected_folds += 1
            candidates = _candidate_test_rows(run, fold, names)
            if len(candidates) != len(names):
                raise ValueError(f"Candidate fold {fold} has incomplete outer-test artifacts")
            selected_by_name.update({str(row["filename"]): {"fold": fold, **row}
                                     for row in candidates})
        else:
            if (run / "outer_test").exists() or (run / "outer_test").is_symlink():
                raise ValueError(f"Baseline-selected fold {fold} has candidate outer-test artifacts")
            selected_by_name.update({name: baseline_by_name[name] for name in names})
    if (summary.get("candidate_fold_count") != selected_folds
            or set(selected_by_name) != set(fold_ids)):
        raise ValueError("Gated-report selected-fold coverage differs from locked gates")
    by_name = {row.get("filename"): row for row in report_rows}
    if len(by_name) != 36 or set(by_name) != set(fold_ids):
        raise ValueError("Gated-report rows differ from frozen LD image coverage")
    for name, row in by_name.items():
        _safe_name(name)
        fold = fold_ids[name]
        selected = str(decisions[fold]["selected"])
        if row["fold"] != str(fold) or row["selected"] != selected:
            raise ValueError(f"Gated-report fold/method differs from lock: {name}")
        baseline, chosen = baseline_by_name[name], selected_by_name[name]
        if int(chosen["fold"]) != fold:
            raise ValueError(f"Gated-report selected artifact has wrong fold: {name}")
        _same_report_number(row["foreground_fraction_true"],
                            baseline["foreground_fraction_true"],
                            f"{name} foreground fraction", unit_interval=True)
        _same_report_number(chosen["foreground_fraction_true"],
                            baseline["foreground_fraction_true"],
                            f"{name} paired truth fraction", unit_interval=True)
        for metric in SCORE_FIELDS:
            a, b = float(baseline[metric]), float(chosen[metric])
            _same_report_number(row[f"baseline_{metric}"], a,
                                f"{name} baseline {metric}", unit_interval=True)
            _same_report_number(row[f"selected_{metric}"], b,
                                f"{name} selected {metric}", unit_interval=True)
            _same_report_number(row[f"paired_{metric}_delta"], b - a,
                                f"{name} paired {metric} delta")
            _same_report_number(row[f"selected_fixed_050_{metric}"],
                                chosen[f"fixed_050_{metric}"],
                                f"{name} fixed-0.50 {metric}", unit_interval=True)

    def check_statistics(actual: object, expected: object, description: str) -> None:
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise ValueError(f"Gated-report {description} statistics are incomplete")
        for statistic in ("mean", "median", "std", "min", "max"):
            _same_report_number(actual.get(statistic), expected.get(statistic),
                                f"{description} {statistic}")

    for metric, baseline_stats in baseline_report["oof"].items():
        if not isinstance(summary.get("baseline_oof"), dict):
            raise ValueError("Gated-report baseline summary is missing")
        check_statistics(summary["baseline_oof"].get(metric), baseline_stats,
                         f"baseline {metric}")
    for metric in SCORE_FIELDS:
        for field, column in (("selected_oof", f"selected_{metric}"),
                              ("paired_delta", f"paired_{metric}_delta")):
            values = [float(row[column]) for row in report_rows]
            expected = _describe(values)
            actual_group = summary.get(field)
            check_statistics(actual_group.get(metric) if isinstance(actual_group, dict) else None,
                             expected, f"{field} {metric}")
    return by_name


def _stratum(fraction: float) -> str:
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Foreground fraction must be finite and in [0, 1]")
    for name, maximum in STRATA:
        if fraction <= maximum:
            return name
    raise AssertionError("Unreachable foreground stratum")


def _manifest_records(fold_map: Path, label_manifest: Path) -> tuple[dict[str, int], dict[str, str], dict[str, dict]]:
    frozen, manifest = _json(fold_map), _json(label_manifest)
    fold_ids, input_hashes = frozen.get("fold_ids"), frozen.get("input_sha256")
    records = manifest.get("files")
    if (frozen.get("target") != "ld" or frozen.get("num_folds") != 5
            or not isinstance(fold_ids, dict) or len(fold_ids) != 36
            or any(type(fold) is not int or fold not in range(5) for fold in fold_ids.values())
            or not isinstance(input_hashes, dict) or set(input_hashes) != set(fold_ids)
            or manifest.get("rule_id") != "ld_antialias_midpoint_v1"
            or manifest.get("label_semantics") != {"background": 0, "foreground": 1}
            or not isinstance(records, list) or len(records) != 36):
        raise ValueError("Frozen LD fold map or derived-label manifest is incomplete")
    by_name = {item.get("filename"): item for item in records if isinstance(item, dict)}
    if len(by_name) != 36 or set(by_name) != set(fold_ids):
        raise ValueError("Derived-label records do not cover each frozen image exactly once")
    for name in fold_ids:
        _safe_name(name)
        if (by_name[name].get("source_image_sha256") != input_hashes[name]
                or not isinstance(by_name[name].get("derived_mask_sha256"), str)):
            raise ValueError(f"Original input/derived label provenance differs: {name}")
    return fold_ids, input_hashes, by_name


def _read_locked_case(
    run: Path, name: str, image: np.ndarray, truth: np.ndarray,
    expected: dict[str, Any], expected_fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a saved outer-test probability, both masks, and recorded scores."""
    threshold_record = _json(run / "threshold.json")
    lock = _json(run / "selection.lock.json")
    threshold = float(threshold_record.get("threshold", -1))
    if (not 0 < threshold < 1 or lock.get("threshold") != threshold):
        raise ValueError(f"Invalid validation-selected threshold: {run}")
    outer = run / "outer_test"
    probability = read_tiff_2d(outer / "probabilities" / name)
    if (probability.shape != truth.shape
            or not np.issubdtype(probability.dtype, np.floating)
            or not np.isfinite(probability).all()
            or np.any(probability < 0) or np.any(probability > 1)):
        raise ValueError(f"Invalid saved outer-test probability: {run}/{name}")
    result = analyse_case(image, truth, probability, threshold)
    for policy, cutoff in (("validation", threshold), ("fixed_050", 0.5)):
        saved = read_tiff_2d(outer / "predictions" / policy / name)
        if (saved.dtype != np.uint8 or saved.shape != image.shape
                or not np.array_equal(saved, (probability >= cutoff).astype(np.uint8) * 255)):
            raise ValueError(f"Saved {policy} outer-test mask differs from probability: {run}/{name}")
    for field in (*SCORE_FIELDS, "foreground_fraction_true"):
        recorded = float(expected[field])
        if not math.isclose(float(result[field]), recorded, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Saved outer-test {field} differs from TIFF pixels: {run}/{name}")
    if expected_fixed is not None:
        fixed = calculate_binary_metrics(truth, probability >= 0.5)
        for field in SCORE_FIELDS:
            if not math.isclose(float(getattr(fixed, field)), float(expected_fixed[field]),
                                rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Saved fixed-0.50 {field} differs from TIFF pixels: {run}/{name}")
    return result


def compare_case(
    fold: int, name: str, gate_selected: str,
    baseline: dict[str, Any], selected: dict[str, Any],
) -> dict[str, Any]:
    """Form one purely descriptive paired row from already-verified cases."""
    if gate_selected not in {"baseline", "candidate"}:
        raise ValueError("Fold gate must be baseline or candidate")
    if not math.isclose(
        float(baseline["foreground_fraction_true"]),
        float(selected["foreground_fraction_true"]), rel_tol=0, abs_tol=1e-12,
    ):
        raise ValueError("Paired results must use the same truth pixel grid")
    row: dict[str, Any] = {
        "fold": fold, "filename": _safe_name(name), "fold_gate_selected": gate_selected,
        "foreground_stratum": _stratum(float(baseline["foreground_fraction_true"])),
        "foreground_fraction_true": float(baseline["foreground_fraction_true"]),
    }
    for field in (*SCORE_FIELDS, *COUNT_FIELDS, *OPTIONAL_FIELDS):
        row[f"baseline_{field}"] = baseline[field]
        row[f"selected_{field}"] = selected[field]
        if field in (*SCORE_FIELDS, *COUNT_FIELDS):
            row[f"paired_{field}_delta"] = selected[field] - baseline[field]
    return row


def _describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Descriptive metric array must be finite and nonempty")
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()), "max": float(array.max()),
    }


def _describe_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"image_count": 0}
    result: dict[str, Any] = {"image_count": len(rows)}
    for field in SCORE_FIELDS:
        for prefix in ("baseline", "selected", "paired"):
            suffix = "_delta" if prefix == "paired" else ""
            key = f"{prefix}_{field}{suffix}"
            result[key] = _describe([float(row[key]) for row in rows])
    for field in COUNT_FIELDS:
        for prefix in ("baseline", "selected", "paired"):
            suffix = "_delta" if prefix == "paired" else ""
            key = f"{prefix}_{field}{suffix}"
            result[f"{key}_total"] = int(sum(int(row[key]) for row in rows))
    for prefix in ("baseline", "selected"):
        total = result[f"{prefix}_small_component_count_proxy_total"]
        for overlap in ("any", "half"):
            detected = result[f"{prefix}_small_component_detected_{overlap}_proxy_total"]
            result[f"{prefix}_small_component_recall_{overlap}_proxy_micro"] = (
                detected / total if total else None
            )
    return result


def summarise_rows(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != 36 or len({row["filename"] for row in rows}) != 36:
        raise ValueError("Exploratory OOF audit requires 36 unique image rows")
    return {
        "target": "ld", "image_count": 36,
        "scope": "exploratory internal five-fold OOF error description",
        "model_selection": "none; fold decisions and thresholds were locked before this audit",
        "independence_limit": (
            "The global method policy used five inner-validation appearances of "
            "these same 36 images. This OOF analysis is descriptive and is not "
            "an untouched or independent confirmation of that policy. File/hash "
            "checks establish content integrity, not wall-clock proof of when "
            "the policy was first written."
        ),
        "global_method_policy_selected": policy["selected_method"],
        "fold_gated_procedure": (
            "Use candidate outer-test prediction only where the pre-test fold "
            "gate selected candidate; otherwise reuse baseline outer-test prediction. "
            "This descriptive fold-gated procedure may differ from the global "
            "production method policy."
        ),
        "foreground_strata": {
            "at_most_0_1_percent": "foreground_fraction_true <= 0.001",
            "above_0_1_to_0_5_percent": "0.001 < foreground_fraction_true <= 0.005",
            "above_0_5_percent": "foreground_fraction_true > 0.005",
        },
        "small_component_proxy": (
            "8-connected semantic GT component <=25 pixels; any-overlap and "
            ">=50%-overlap are proxies, not instance segmentation metrics"
        ),
        "distance_units": "native image pixels",
        "all": _describe_group(rows),
        "by_foreground_stratum": {
            name: _describe_group([row for row in rows if row["foreground_stratum"] == name])
            for name, _ in STRATA
        },
        "by_fold": {
            str(fold): _describe_group([row for row in rows if row["fold"] == fold])
            for fold in range(5)
        },
    }


def audit_oof(
    method_policy: Path, baseline_root: Path, candidate_root: Path, baseline_archives: Path,
    fold_map: Path, label_manifest: Path, input_dir: Path, label_dir: Path,
    gated_report_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify policy and source archives, then inspect saved OOF cases."""
    policy = verify_method_policy_before_outer_read(
        method_policy, baseline_root, candidate_root, fold_map, label_manifest, label_dir
    )
    # The global policy must pass before archive member names/bytes or any
    # extracted outer-test file are opened. This second guard binds the
    # extracted baseline evidence to the original checked fold archives.
    verify_baseline_outer_archives(baseline_root, baseline_archives)
    paired_rows, gated, report_provenance = load_verified_gated_report(gated_report_dir)
    fold_ids, input_hashes, label_records = _manifest_records(fold_map, label_manifest)
    report_provenance["candidate_fold_archive_sha256"] = verify_candidate_outer_archives(
        candidate_root, gated_report_dir, policy, fold_ids
    )
    by_name = verify_gated_report_against_folds(
        paired_rows, gated, policy, baseline_root, candidate_root, fold_ids
    )
    rows: list[dict[str, Any]] = []
    for name in sorted(fold_ids):
        fold = fold_ids[name]
        image_path, label_path = input_dir / name, label_dir / name
        record = label_records[name]
        if (_sha256(image_path) != input_hashes[name]
                or _sha256(label_path) != record["derived_mask_sha256"]):
            raise ValueError(f"Input or derived label differs from frozen SHA256: {name}")
        image, truth = read_tiff_2d(image_path), read_tiff_2d(label_path)
        if (image.ndim != 2 or image.dtype.kind not in "ui" or truth.shape != image.shape
                or truth.dtype != np.uint8 or not np.isin(truth, (0, 1)).all()
                or record.get("shape") != list(image.shape)):
            raise ValueError(f"Input/derived label has invalid native grid or semantics: {name}")
        expected = by_name[name]
        baseline = _read_locked_case(
            _fold_dir(baseline_root, fold), name, image, truth,
            {**{field: expected[f"baseline_{field}"] for field in SCORE_FIELDS},
             "foreground_fraction_true": expected["foreground_fraction_true"]},
            {field: expected[f"selected_fixed_050_{field}"] for field in SCORE_FIELDS}
            if expected["selected"] == "baseline" else None,
        )
        gate_selected = str(expected["selected"])
        if gate_selected == "baseline":
            selected = baseline
        else:
            selected = _read_locked_case(
                _fold_dir(candidate_root, fold), name, image, truth,
                {**{field: expected[f"selected_{field}"] for field in SCORE_FIELDS},
                 "foreground_fraction_true": expected["foreground_fraction_true"]},
                {field: expected[f"selected_fixed_050_{field}"] for field in SCORE_FIELDS},
            )
        rows.append(compare_case(fold, name, gate_selected, baseline, selected))
    summary = summarise_rows(rows, policy)
    summary["gated_report_provenance"] = report_provenance
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-policy", type=Path, required=True)
    parser.add_argument("--baseline-cv-root", type=Path, required=True)
    parser.add_argument("--candidate-cv-root", type=Path, required=True)
    parser.add_argument("--baseline-archives", type=Path, required=True)
    parser.add_argument("--gated-report-dir", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing OOF audit: {args.output_dir}")
    rows, summary = audit_oof(
        args.method_policy, args.baseline_cv_root, args.candidate_cv_root,
        args.baseline_archives,
        args.fold_map, args.label_manifest, args.input_dir, args.label_dir,
        args.gated_report_dir,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired_oof_error_per_image.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "exploratory_oof_error_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output / "audit_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "target": "ld", "scope": summary["scope"],
            "method_policy_sha256": _sha256(args.method_policy),
            "fold_map_sha256": _sha256(args.fold_map),
            "label_manifest_sha256": _sha256(args.label_manifest),
            "policy_protocol_sha256": _sha256(POLICY_PATH),
            "gated_report_provenance": summary["gated_report_provenance"],
            "image_count": 36,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"LD_EXPLORATORY_OOF_AUDIT_COMPLETE images={len(rows)} output={output}")


if __name__ == "__main__":
    main()
